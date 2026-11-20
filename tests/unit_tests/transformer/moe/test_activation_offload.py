# Copyright (c) 2026, Swiss AI Institute
"""Unit tests for the MoE expert activation-offload primitives (moe_offload_activations).

These exercise the machinery that offloads the saved expert input activation (fp8_x) to pinned
host memory during the combine phase and reloads it before the expert backward:

* ``MoEOffloadManager.offload_activation`` D2H-copies fp8_x and frees its GPU storage;
* ``MoEOffloadManager.reload`` restores it bit-for-bit into a fresh GPU tensor;
* ``MoEReloadTrigger`` is a forward identity whose backward triggers the reload;
* ``MoEOffloadMemoryPool`` recycles pinned buffers within its budget with an event guard.

Numeric grad-parity of the full offloading-experts path is covered separately as an integration
test (it needs DeepGEMM/grouped_gemm + distributed init); here we validate the offload plumbing
in isolation, which is the part introduced by moe_offload_activations.
"""
import os

import pytest
import torch

from megatron.core.transformer.moe.moe_offload import (
    MoEOffloadHandle,
    MoEOffloadManager,
    MoEOffloadMemoryPool,
    MoEReloadTrigger,
    StreamManager,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="activation offload requires CUDA (streams/events/pinned)"
)

FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def _select_cuda_device():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())


def _fresh_pool():
    MoEOffloadMemoryPool._instance = None
    return MoEOffloadMemoryPool.get_instance()


def _make_fp8(m, n):
    # Random bf16 values cast to fp8 so the bit pattern is representative.
    return (torch.randn(m, n, device="cuda", dtype=torch.bfloat16)).to(FP8)


def _bits(t):
    return t.view(torch.uint8)


def test_offload_frees_gpu_storage_and_reload_restores_bitexact():
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)

    fp8_x = _make_fp8(256, 128)
    reference = fp8_x.detach().clone().cpu()  # snapshot exact bytes (on CPU) before offload

    handle = MoEOffloadManager.offload_activation(fp8_x, sm, "fp8_x")
    slot = handle.slots["fp8_x"]
    torch.cuda.synchronize()

    # The activation slot owns the pinned copy while GPU storage is freed.
    assert handle.active
    assert slot.shape == (256, 128) and slot.dtype == FP8
    assert fp8_x.untyped_storage().size() == 0, "fp8_x GPU storage should be freed after offload"
    assert slot.host_base.is_pinned() and slot.host_buf.is_pinned()
    assert slot.host_buf.numel() == 256 * 128
    assert torch.equal(_bits(slot.host_buf), _bits(reference.reshape(-1))), "D2H must be bit-exact"

    MoEOffloadManager.reload(handle)
    torch.cuda.synchronize()
    assert slot.gpu_buf is not None
    assert slot.gpu_buf.shape == (256, 128) and slot.gpu_buf.dtype == FP8
    restored = MoEOffloadManager.get(handle, "fp8_x")
    assert torch.equal(_bits(restored.cpu()), _bits(reference)), "reload must be bit-exact"


def test_reload_trigger_forward_identity_backward_reloads():
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)

    fp8_x = _make_fp8(128, 64)
    reference = fp8_x.detach().clone().cpu()
    handle = MoEOffloadManager.offload_activation(fp8_x, sm, "fp8_x")
    slot = handle.slots["fp8_x"]
    torch.cuda.synchronize()

    # Forward is an identity on the (unrelated) combine-output tensor.
    combine_out = torch.randn(8, device="cuda", requires_grad=True)
    out = MoEReloadTrigger.apply(combine_out, handle)
    assert torch.equal(out, combine_out)
    assert slot.gpu_buf is None, "reload must not run in forward"

    # Backward triggers the reload.
    out.sum().backward()
    torch.cuda.synchronize()
    assert slot.gpu_buf is not None, "reload must run in MoEReloadTrigger.backward"
    restored = MoEOffloadManager.get(handle, "fp8_x")
    assert torch.equal(_bits(restored.cpu()), _bits(reference))


def test_reload_trigger_inactive_handle_is_noop():
    combine_out = torch.randn(8, device="cuda", requires_grad=True)
    # Inactive/None handles must be pass-through with no reload attempted.
    for handle in (None, MoEOffloadHandle(active=False)):
        out = MoEReloadTrigger.apply(combine_out, handle)
        out.sum().backward()
        assert torch.equal(out, combine_out)


def test_pinned_pool_capacity_based_recycle_and_growth():
    pool = _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)
    MB = 1024 * 1024

    # Flat uint8 buffers with capacity >= request; distinct shapes with the same byte size share a
    # buffer, so shape variance does not fragment the pool.
    buf1 = pool.allocate(3 * MB, torch.uint8, "cpu", sm.act_d2h_stream)
    assert buf1.dtype == torch.uint8 and buf1.dim() == 1 and buf1.is_pinned()
    assert buf1.numel() >= 3 * MB

    # A recycled buffer serves any request that fits (best-fit by capacity), regardless of shape.
    pool.free(buf1, ready_event=None)
    buf2 = pool.allocate(1 * MB, torch.uint8, "cpu", sm.act_d2h_stream)
    assert buf2 is buf1, "a smaller request should reuse the larger recycled buffer"

    # With buf1 still checked out, a concurrent request must allocate a distinct buffer.
    buf3 = pool.allocate(1 * MB, torch.uint8, "cpu", sm.act_d2h_stream)
    assert buf3 is not buf1

    # A request larger than every free buffer's capacity grows the pool.
    pool.free(buf2, ready_event=None)  # buf2 is buf1
    big = buf1.numel() + 1
    buf4 = pool.allocate(big, torch.uint8, "cpu", sm.act_d2h_stream)
    assert buf4 is not buf1 and buf4.numel() >= big


def test_offload_reload_roundtrip_reuses_pooled_buffer_across_shapes_bitexact():
    """Two offload/reload cycles with *different* fp8_x shapes must reuse the same capacity buffer
    (shape variance must not fragment the pool) and must not corrupt data (event-guarded recycle).
    """
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)

    # Reload now recycles the host buffer automatically; retain its identity before reload.
    x1 = _make_fp8(200, 128)
    ref1 = x1.detach().clone().cpu()
    h1 = MoEOffloadManager.offload_activation(x1, sm, "fp8_x")
    slot1 = h1.slots["fp8_x"]
    cpu_base = slot1.host_base
    MoEOffloadManager.reload(h1)
    torch.cuda.synchronize()
    restored1 = MoEOffloadManager.get(h1, "fp8_x")
    assert torch.equal(_bits(restored1.cpu()), _bits(ref1))

    # Second cycle, DIFFERENT shape but same capacity class -> reuse the pooled buffer.
    x2 = _make_fp8(177, 128)
    ref2 = x2.detach().clone().cpu()
    h2 = MoEOffloadManager.offload_activation(x2, sm, "fp8_x")
    slot2 = h2.slots["fp8_x"]
    assert (
        slot2.host_base is cpu_base
    ), "differently-shaped offload should reuse the capacity buffer"
    assert slot2.numel != slot1.numel, "the two cycles should have different element counts"
    MoEOffloadManager.reload(h2)
    torch.cuda.synchronize()
    restored2 = MoEOffloadManager.get(h2, "fp8_x")
    assert torch.equal(_bits(restored2.cpu()), _bits(ref2))


# ---------------------------------------------------------------------------
# fc1_output offload (moe_offload_fc1_output): a second, larger slot on the same handle.
# ---------------------------------------------------------------------------


def test_offload_fc1_output_both_slots_bitexact_and_storage_freed():
    """Offloading fp8_x then fp8_fc1 onto one handle parks both on pinned host (bit-exact) and frees
    both GPU storages; reload restores both into fresh GPU tensors with distinct h2d_done events."""
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)

    fp8_x = _make_fp8(128, 64)
    ref_x = fp8_x.detach().clone().cpu()
    handle = MoEOffloadManager.offload_activation(fp8_x, sm, "fp8_x")

    # fc1_output is ~5x larger; add a second named activation to the same handle.
    fp8_fc1 = _make_fp8(128, 320)
    ref_fc1 = fp8_fc1.detach().clone().cpu()
    MoEOffloadManager.offload_activation(fp8_fc1, sm, "fp8_fc1", handle)
    slot_x, slot_fc1 = handle.slots["fp8_x"], handle.slots["fp8_fc1"]
    torch.cuda.synchronize()

    assert handle.active and set(handle.slots) == {"fp8_x", "fp8_fc1"}
    assert fp8_x.untyped_storage().size() == 0, "fp8_x GPU storage must be freed"
    assert fp8_fc1.untyped_storage().size() == 0, "fp8_fc1 GPU storage must be freed"
    assert slot_x.host_buf.numel() == 128 * 64
    assert slot_fc1.host_buf.numel() == 128 * 320
    assert torch.equal(
        _bits(slot_x.host_buf), _bits(ref_x.reshape(-1))
    ), "fp8_x D2H must be bit-exact"
    assert torch.equal(
        _bits(slot_fc1.host_buf), _bits(ref_fc1.reshape(-1))
    ), "fc1 D2H must be bit-exact"

    MoEOffloadManager.reload(handle)
    torch.cuda.synchronize()
    assert slot_x.gpu_buf is not None and slot_fc1.gpu_buf is not None
    assert slot_x.h2d_done is not None and slot_fc1.h2d_done is not None
    assert slot_x.h2d_done is not slot_fc1.h2d_done
    assert slot_x.gpu_buf.shape == (128, 64) and slot_x.gpu_buf.dtype == FP8
    assert slot_fc1.gpu_buf.shape == (128, 320) and slot_fc1.gpu_buf.dtype == FP8
    restored_x = MoEOffloadManager.get(handle, "fp8_x")
    restored_fc1 = MoEOffloadManager.get(handle, "fp8_fc1")
    assert torch.equal(_bits(restored_x.cpu()), _bits(ref_x)), "fp8_x reload must be bit-exact"
    assert torch.equal(_bits(restored_fc1.cpu()), _bits(ref_fc1)), "fc1 reload must be bit-exact"


def test_fc1_disabled_keeps_handle_fp8_only():
    """Without a second offload, the handle stays fp8_x-only; reload adds no fc1 slot."""
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)
    fp8_x = _make_fp8(64, 64)
    ref = fp8_x.detach().clone().cpu()
    handle = MoEOffloadManager.offload_activation(fp8_x, sm, "fp8_x")

    assert "fp8_fc1" not in handle.slots
    MoEOffloadManager.reload(handle)
    torch.cuda.synchronize()
    assert handle.slots["fp8_x"].gpu_buf is not None
    assert "fp8_fc1" not in handle.slots
    restored = MoEOffloadManager.get(handle, "fp8_x")
    assert torch.equal(_bits(restored.cpu()), _bits(ref))


def test_reload_trigger_reloads_both_slots_via_backward():
    """MoEReloadTrigger.backward must reload BOTH the fp8_x and fc1 slots of a handle that carries
    fc1, so a single trigger node covers the two-tensor offload."""
    _fresh_pool()
    sm = StreamManager(num_h2d_streams=1, num_compute_streams=1)
    fp8_x = _make_fp8(64, 64)
    ref_x = fp8_x.detach().clone().cpu()
    fp8_fc1 = _make_fp8(64, 256)
    ref_fc1 = fp8_fc1.detach().clone().cpu()
    handle = MoEOffloadManager.offload_activation(fp8_x, sm, "fp8_x")
    MoEOffloadManager.offload_activation(fp8_fc1, sm, "fp8_fc1", handle)
    slot_x, slot_fc1 = handle.slots["fp8_x"], handle.slots["fp8_fc1"]

    combine_out = torch.randn(8, device="cuda", requires_grad=True)
    out = MoEReloadTrigger.apply(combine_out, handle)
    assert slot_x.gpu_buf is None and slot_fc1.gpu_buf is None, "reload must not run in forward"
    assert torch.equal(out, combine_out)

    out.sum().backward()
    torch.cuda.synchronize()
    assert slot_x.gpu_buf is not None and slot_fc1.gpu_buf is not None
    restored_x = MoEOffloadManager.get(handle, "fp8_x")
    restored_fc1 = MoEOffloadManager.get(handle, "fp8_fc1")
    assert torch.equal(_bits(restored_x.cpu()), _bits(ref_x))
    assert torch.equal(_bits(restored_fc1.cpu()), _bits(ref_fc1))
