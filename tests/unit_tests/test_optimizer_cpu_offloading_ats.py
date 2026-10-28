# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GH200/GB200 ATS residency test using actual HybridDeviceOptimizer state.

GPU reads/writes of CPU moments are deliberate mechanism controls, not normal
CPU AdamW behavior. No explicit prefetch is used. Run the dedicated
cpu-offloading-ats unit group.
"""

import ctypes
import errno
import json
import os
from collections import Counter
from pathlib import Path

import pytest
import torch

from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

# Increment the original CPU allocation through its pointer, not a CUDA copy.
_TOUCH_PTX = b"""
.version 7.0
.target sm_70
.address_size 64
.visible .entry touch(.param .u64 base, .param .u64 count)
{
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    .reg .f32 %f;
    ld.param.u64 %rd1, [base];
    ld.param.u64 %rd2, [count];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    cvt.u64.u32 %rd3, %r4;
    mov.u32 %r1, %nctaid.x;
    mul.wide.u32 %rd4, %r1, %r2;
LOOP:
    setp.ge.u64 %p, %rd3, %rd2;
    @%p bra DONE;
    shl.b64 %rd5, %rd3, 2;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.f32 %f, [%rd6];
    add.f32 %f, %f, 0f3F800000;
    st.global.f32 [%rd6], %f;
    add.u64 %rd3, %rd3, %rd4;
    bra LOOP;
DONE:
    ret;
}
"""


def _cuda_call(cuda, name, *args):
    result = getattr(cuda, name)(*args)
    assert result == 0, f"{name} failed with CUDA driver error {result}"


def _ats_libraries():
    if not torch.cuda.is_available():
        pytest.skip('Requires CUDA')
    gpu_name = torch.cuda.get_device_name()
    if not any(model in gpu_name for model in ('GH200', 'GB200')):
        pytest.skip(f'Requires a GH200 or GB200 GPU, found {gpu_name}')
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
        numa = ctypes.CDLL('libnuma.so.1', use_errno=True)
    except OSError as error:
        pytest.skip(f'Requires CUDA driver and libnuma: {error}')
    _cuda_call(cuda, 'cuInit', 0)
    # cudaDevAttrPageableMemoryAccess and PageableMemoryAccessUsesHostPageTables.
    for attribute in (88, 100):
        enabled = ctypes.c_int()
        _cuda_call(
            cuda,
            'cuDeviceGetAttribute',
            ctypes.byref(enabled),
            attribute,
            torch.cuda.current_device(),
        )
        if not enabled.value:
            pytest.skip('Requires direct pageable-memory access through host page tables (ATS)')
    numa.numa_move_pages.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    numa.numa_move_pages.restype = ctypes.c_int
    return cuda, numa


def _page_locations(numa, tensor):
    page_size = os.sysconf('SC_PAGE_SIZE')
    first = tensor.data_ptr() // page_size * page_size
    last = (tensor.data_ptr() + tensor.nbytes - 1) // page_size * page_size
    pages = (last - first) // page_size + 1
    samples = min(pages, 512)
    addresses = (ctypes.c_void_p * samples)(
        *(first + (index * (pages - 1) // (samples - 1)) * page_size for index in range(samples))
    )
    status = (ctypes.c_int * samples)(*([-9999] * samples))
    # nodes=NULL and flags=0 QUERY residency; this call does not move pages.
    result = numa.numa_move_pages(0, samples, addresses, None, status, 0)
    if result < 0:
        error = ctypes.get_errno()
        if error in (errno.EPERM, errno.EACCES, errno.ENOSYS):
            pytest.skip(f'Page residency query unavailable: {os.strerror(error)}')
        raise OSError(error, os.strerror(error))
    assert all(node >= 0 for node in status), f'Unknown page locations: {list(status)}'
    return list(status)


def _gpu_touch(cuda, tensor, rounds):
    module, kernel = ctypes.c_void_p(), ctypes.c_void_p()
    _cuda_call(cuda, 'cuModuleLoadData', ctypes.byref(module), ctypes.c_char_p(_TOUCH_PTX))
    try:
        _cuda_call(cuda, 'cuModuleGetFunction', ctypes.byref(kernel), module, b'touch')
        pointer = ctypes.c_uint64(tensor.data_ptr())
        count = ctypes.c_uint64(tensor.numel())
        parameters = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.pointer(pointer), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(count), ctypes.c_void_p),
        )
        for _ in range(rounds):
            _cuda_call(
                cuda,
                'cuLaunchKernel',
                kernel,
                4096,
                1,
                1,
                256,
                1,
                1,
                0,
                ctypes.c_void_p(),
                parameters,
                None,
            )
            _cuda_call(cuda, 'cuCtxSynchronize')
    finally:
        _cuda_call(cuda, 'cuModuleUnload', module)


@pytest.mark.experimental
@pytest.mark.parametrize('pin_state', [False, True])
def test_hybrid_optimizer_state_ats_residency(pin_state, record_property):
    """Contrast actual HDO pageable/pinned moments under artificial GPU access."""
    cuda, numa = _ats_libraries()
    nodes = list(Path('/sys/devices/system/node').glob('node[0-9]*'))
    cpu_nodes = {int(node.name[4:]) for node in nodes if (node / 'cpulist').read_text().strip()}
    hbm_nodes = {int(node.name[4:]) for node in nodes} - cpu_nodes
    if not cpu_nodes or not hbm_nodes:
        pytest.skip('Requires GH200/GB200 CPU and memory-only HBM NUMA nodes')

    # One GiB exp_avg, matching the controlled migration reproducer.
    count = 1024**3 // 4
    param = torch.nn.Parameter(torch.ones(count, device='cuda', dtype=torch.float32))
    optimizer = HybridDeviceOptimizer(
        [param],
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        offload_fraction=1.0,
        pin_cpu_optimizer_state=pin_state,
        overlap_cpu_optimizer_d2h_h2d=False,
        fused=True,
        lr=1e-3,
    )
    param.grad = torch.ones_like(param)
    optimizer.step()
    torch.cuda.synchronize()
    moment = optimizer.state[param]['exp_avg']
    assert moment.device.type == 'cpu'
    assert moment.is_pinned() == pin_state
    pointer = moment.data_ptr()

    # Measure a normal HDO update first; it must not be confused with GPU stress.
    before = _page_locations(numa, moment)
    assert all(node in cpu_nodes for node in before)
    _gpu_touch(cuda, moment, rounds=100)
    after = _page_locations(numa, moment)
    driver_version = ctypes.c_int()
    _cuda_call(cuda, 'cuDriverGetVersion', ctypes.byref(driver_version))
    result = {
        'host': os.uname().nodename,
        'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'cuda_driver_api': driver_version.value,
        'pin_state': pin_state,
        'pages_before': dict(Counter(before)),
        'pages_after': dict(Counter(after)),
        'sampled_pages': len(after),
        'hbm_pages_before': sum(node in hbm_nodes for node in before),
        'hbm_pages_after': sum(node in hbm_nodes for node in after),
        'note': 'Artificial GPU access to real HDO exp_avg, no explicit prefetch',
    }
    record_property('ats_residency', json.dumps(result))
    print(json.dumps(result), flush=True)
    assert moment.data_ptr() == pointer
    assert moment.device.type == 'cpu'
    if pin_state:
        assert result['hbm_pages_after'] == 0, result
    else:
        assert result['hbm_pages_after'] > 0, (
            'Pageable control did not migrate; access-counter migration was not reproduced',
            result,
        )
    # CPU reads happen only after residency queries, since they can change placement.
    assert moment[0].item() == pytest.approx(100.1, abs=1e-3)
