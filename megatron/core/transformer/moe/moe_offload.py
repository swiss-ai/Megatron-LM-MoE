# Copyright (c) 2026, Swiss AI Institute
"""Shared MoE offloading primitives."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import weakref
import torch


class StreamManager:
    """Manage CUDA streams and events shared by MoE offloading paths."""

    _instance = None

    def __init__(
        self,
        num_h2d_streams,
        num_compute_streams=4,
    ):
        self.num_compute_streams = num_compute_streams
        self.num_h2d_streams = num_h2d_streams
        self.h2d_streams = [torch.cuda.Stream() for _ in range(num_h2d_streams)]
        self.compute_streams = [torch.cuda.Stream() for _ in range(self.num_compute_streams)]
        self.compute_cuda_streams = [stream.cuda_stream for stream in self.compute_streams]

        # Dedicated copy streams for activation offload D2H/H2D.
        self.act_d2h_stream = torch.cuda.Stream()
        self.act_h2d_stream = torch.cuda.Stream()

        # Dedicated copy streams for main-grad offload D2H/H2D.
        self.mgrad_d2h_stream = torch.cuda.Stream()
        self.mgrad_h2d_stream = torch.cuda.Stream()

        # Dedicated copy streams for moe weight offload D2H/H2D.
        self.param_d2h_stream = torch.cuda.Stream()
        self.param_h2d_stream = torch.cuda.Stream()

    @classmethod
    def get_instance(
        cls,
        num_h2d_streams=2,
        num_compute_streams=4,
    ):
        if cls._instance is None:
            cls._instance = StreamManager(num_h2d_streams, num_compute_streams)
        return cls._instance

    def get_h2d_stream(self, idx) -> torch.cuda.Stream:
        return self.h2d_streams[idx]

    def get_compute_streams(self) -> list[int]:
        return self.compute_cuda_streams

    def get_launch_streams(self) -> list[torch.cuda.Stream]:
        # VPP can execute a model chunk on a non-default current stream.
        current_stream = torch.cuda.current_stream()
        default_stream = torch.cuda.default_stream()
        if current_stream.cuda_stream == default_stream.cuda_stream:
            return [current_stream]
        return [current_stream, default_stream]

    def launch_streams_wait_compute_streams(self):
        launch_streams = self.get_launch_streams()
        for i in range(self.num_compute_streams):
            for launch_stream in launch_streams:
                launch_stream.wait_stream(self.compute_streams[i])

    def default_stream_wait_h2d_stream(self, idx):
        torch.cuda.default_stream().wait_stream(self.get_h2d_stream(idx))

    def compute_streams_wait_launch_streams(self):
        launch_streams = self.get_launch_streams()
        for i in range(self.num_compute_streams):
            for launch_stream in launch_streams:
                self.compute_streams[i].wait_stream(launch_stream)

    def h2d_stream_wait_consumer_streams(self, idx):
        h2d_stream = self.get_h2d_stream(idx)
        for launch_stream in self.get_launch_streams():
            h2d_stream.wait_stream(launch_stream)
        for i in range(self.num_compute_streams):
            h2d_stream.wait_stream(self.compute_streams[i])

    def compute_streams_wait_h2d_stream(self, idx):
        h2d_stream = self.get_h2d_stream(idx)
        for i in range(self.num_compute_streams):
            self.compute_streams[i].wait_stream(h2d_stream)

    def consumer_streams_wait_event(self, event):
        for launch_stream in self.get_launch_streams():
            launch_stream.wait_event(event)
        for i in range(self.num_compute_streams):
            self.compute_streams[i].wait_event(event)

    def h2d_stream_wait_default_stream(self, idx):
        self.get_h2d_stream(idx).wait_stream(torch.cuda.default_stream())

    def consumer_streams_record(self, tensor):
        for launch_stream in self.get_launch_streams():
            tensor.record_stream(launch_stream)
        for i in range(self.num_compute_streams):
            tensor.record_stream(self.compute_streams[i])

    def d2h_stream_wait_producers(self, d2h_stream):
        """Make an arbitrary D2H stream wait for whoever produced the tensors it is about to read.

        :class:`MoEOffloadManager` calls this with the stream its handle's policy names.
        """
        for launch_stream in self.get_launch_streams():
            d2h_stream.wait_stream(launch_stream)
        for i in range(self.num_compute_streams):
            d2h_stream.wait_stream(self.compute_streams[i])

@dataclass(frozen=True)
class MoEOffloadPolicy:
    kind: str
    host_pool: type = None       # None => host tensor is permanent, i.e. param.main_grad
    device_pool: type = None     # None => device tensor comes from on-demand allocation, i.e. activation
    writeback: bool = False      # device copy works an accumulator, i.e. param.main_grad. D2H it back before releasing
    d2h_stream_attr: str = None  # attribute on StreamManager naming this kind's D2H stream
    h2d_stream_attr: str = None

class SlotState(Enum):
    """Where a slot's ground-truth copy lives."""

    HOST = auto()  # on the host -> the device side owns nothing
    DEVICE = auto()  # staged on the device -> the host copy is redundant (act) or stale (main grad)
    RELEASED = auto()  # consumed and handed to the caller; the slot owns nothing


@dataclass
class MoEOffloadSlot:
    """One offloadable tensor: its host home, its device home, and the events between them.

    Exactly one of ``param`` / ``host_base`` is set. Main-grad slots read their host home lazily as
    ``param.main_grad`` -- DDP rebinds that attribute to a view of the grad buffer, so it must
    never be cached. Activation slots instead rent a pinned buffer from
    :class:`MoEOffloadMemoryPool`, whose host branch hands back the full ``uint8`` base; the slot
    keeps the base (that is what ``free`` takes) alongside the typed prefix view the copies use.
    """

    shape: tuple = None
    dtype: torch.dtype = None
    numel: int = 0
    element_size: int = 0  # bytes per element, to size the pinned slice
    device: torch.device = None  # where the device-side copy lives

    param: torch.nn.Parameter = None  # main grad: host home owner, read as ``param.main_grad``
    host_base: torch.Tensor = None  # activation: full pinned uint8 buffer owned by the pool
    # activation: typed view sliced out of host_base
    # parameter: permanent packed FP8 source
    host_buf: torch.Tensor = None
    gpu_buf: torch.Tensor = None  # device-side copy

    d2h_done: torch.cuda.Event = None
    h2d_done: torch.cuda.Event = None

    state: SlotState = SlotState.RELEASED
    # Host copy is known to be all-zero (right after ``zero_grad_buffer``), so the next reload can
    # zero the device buffer instead of paying an H2D. Accumulator slots only.
    host_is_zero: bool = False # main grad

    @property
    def host_tensor(self) -> torch.Tensor:
        """The host-side copy, however this slot happens to own it."""
        return self.host_buf if self.param is None else self.param.main_grad


@dataclass(eq=False)
class MoEOffloadHandle:
    """A named set of tensors that live off-device between their producer and their consumer.

    Per-slot state machine the driver maintains::

        RELEASED --offload--> HOST --reload--> DEVICE --get--> RELEASED   (activation)
                                ^                |
                                +-----offload----+                        (main grad)
                                +-----release----+                        (parameter)
    """

    policy: MoEOffloadPolicy = None
    stream_manager: object = None  # StreamManager carrying this policy's D2H/H2D streams
    slots: dict = field(default_factory=dict)  # name -> MoEOffloadSlot
    active: bool = False  # set by the driver once the handle owns at least one slot


class MoEOffloadMemoryPool:
    """Unified recycling pool of both pinned host buffers for activation offload
        and device buffer for main gradient reload."""

    _instance = None
    _CPU_POOL_GRANULARITY_BYTES = 2 * 1024 * 1024  # 2 MiB

    @classmethod
    def get_instance(cls) -> MoEOffloadMemoryPool:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # List of [buffer, ready_event_or_None]; scanned best-fit by capacity.
        self._free_cpu: list = []
        # strong refs so buffers are never GC'd / freed (avoids cudaFreeHost syncs).
        self._all_cpu: list = []
        self._total_bytes_cpu: int = 0

        # (numel, dtype, device) -> list of [buffer, ready_event_or_None].
        self._free_gpu: dict = {}
        # strong refs of buffers so the memory never returned to the allocator.
        self._all_gpu: list = []
        self._total_bytes_gpu: int = 0

    def stats(self) -> dict:
        """(num_buffers, total_bytes, free_bytes)"""
        results = {
            "cpu": (len(self._all_cpu), self._total_bytes_cpu, sum(b.numel() for b, _ev in self._free_cpu)),
            "gpu": (len(self._all_gpu), self._total_bytes_gpu, sum(b.numel() * b.element_size() for fl in self._free_gpu.values() for b, _ev in fl))
        }
        return results

    def allocate(
        self,
        numel: int,
        dtype: torch.dtype,
        device: str,
        wait_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        """Return a flat buffer of exactly ``numel`` elements of ``dtype`` on ``device``"""
        # allocate pinned host memory for activation offload
        if device == "cpu":
            best_idx = -1
            best_cap = None
            nbytes = numel * torch.tensor([], dtype=dtype).element_size()
            for i, (buf, _ev) in enumerate(self._free_cpu):
                cap = buf.numel()
                if cap >= nbytes and (best_cap is None or cap < best_cap):
                    best_cap, best_idx = cap, i
            if best_idx >= 0:
                buf, ready_event = self._free_cpu.pop(best_idx)
                if ready_event is not None:
                    wait_stream.wait_event(ready_event)
                return buf

            # round up to the nearest granularity
            cap = (
                (nbytes + self._CPU_POOL_GRANULARITY_BYTES - 1)
                // self._CPU_POOL_GRANULARITY_BYTES
                * self._CPU_POOL_GRANULARITY_BYTES
            )
            buf = torch.empty(cap, dtype=torch.uint8, device="cpu", pin_memory=True)
            self._all_cpu.append(buf)
            self._total_bytes_cpu += cap
            return buf
        # allocate GPU memory for main gradient reload
        else:
            free_list = self._free_gpu.setdefault((numel, dtype, device), [])
            if free_list:
                buf, ready_event = free_list.pop()
                if ready_event is not None:
                    # a previous writeback D2H is pending
                    # wait until the copy has landed.
                    wait_stream.wait_event(ready_event)
                return buf

            # if there is no free buffer, allocate a new one and add into reference
            buf = torch.empty(numel, dtype=dtype, device=device)
            self._all_gpu.append(buf)
            self._total_bytes_gpu += buf.numel() * buf.element_size()
            return buf

    def free(self, buf: torch.Tensor, ready_event: torch.cuda.Event = None):
        """Return ``buf`` to the pool, tagged with the event after which it is safe to overwrite."""
        if buf.device.type == "cpu":
            self._free_cpu.append([buf, ready_event])
        else:
            self._free_gpu.setdefault((buf.numel(), buf.dtype, buf.device), []).append([buf, ready_event])

class MoEOffloadManager:
    """ Unified **stateless** driver for offloading/reloading MoE weights (WIP), activations and main gradients. """

    PARAM_OFFLOAD = MoEOffloadPolicy(
        kind="parameter",
        host_pool=None,
        device_pool=MoEOffloadMemoryPool,
        writeback=False,
        d2h_stream_attr="param_d2h_stream",
        h2d_stream_attr="param_h2d_stream",
    )

    ACTIVATION_OFFLOAD = MoEOffloadPolicy(
        kind="activation",
        host_pool=MoEOffloadMemoryPool,
        device_pool=None,
        writeback=False,
        d2h_stream_attr="act_d2h_stream",
        h2d_stream_attr="act_h2d_stream",
    )

    MAIN_GRAD_OFFLOAD = MoEOffloadPolicy(
        kind="main_grad",
        host_pool=None,
        device_pool=MoEOffloadMemoryPool,
        writeback=True,
        d2h_stream_attr="mgrad_d2h_stream",
        h2d_stream_attr="mgrad_h2d_stream",
    )

    # un-awaited writeback event per D2H stream: cuda stream -> torch.Event
    # host buffer must not be read or writed while one of these is outstanding.
    _pending_writebacks: dict = {}

    _live_persistent_handles: weakref.WeakSet = weakref.WeakSet()

    @staticmethod
    def _d2h(
        handle: MoEOffloadHandle,
        slot: MoEOffloadSlot,
        src: torch.Tensor,
    ) -> None:
        """One slot's D2H: device copy to the host.

        The caller is responsible for ordering the D2H stream behind whoever produced ``src``
        (``d2h_stream_wait_producers``) before the first slot of a batch.
        """
        policy = handle.policy
        pool = MoEOffloadMemoryPool.get_instance()
        d2h_stream = getattr(handle.stream_manager, policy.d2h_stream_attr)
        if slot.h2d_done is not None:
            d2h_stream.wait_event(slot.h2d_done)

        with torch.cuda.stream(d2h_stream):
            if policy.host_pool is not None:
                # host side buffer on demand, for activation offload
                slot.host_base = pool.allocate(slot.numel, slot.dtype, "cpu", d2h_stream)
                slot.host_buf = slot.host_base[: slot.numel * slot.element_size].view(slot.dtype)
            slot.host_tensor.reshape(-1).copy_(src.reshape(-1), non_blocking=True)
        slot.d2h_done = torch.cuda.Event()
        slot.d2h_done.record(d2h_stream)

        if policy.device_pool is not None:
            # device side buffer on demand for main-grad offload
            pool.free(src.reshape(-1), slot.d2h_done)
        else:
            src.record_stream(d2h_stream)
            src.untyped_storage().resize_(0)
        if policy.writeback:
            MoEOffloadManager._pending_writebacks[d2h_stream.cuda_stream] = slot.d2h_done

        slot.gpu_buf = None
        slot.h2d_done = None
        slot.host_is_zero = False
        slot.state = SlotState.HOST

    @staticmethod
    def _h2d(
        handle: MoEOffloadHandle,
        slot: MoEOffloadSlot,
    ) -> None:
        """One slot's H2D: host copy to the device."""
        policy = handle.policy
        pool = MoEOffloadMemoryPool.get_instance()
        h2d_stream = getattr(handle.stream_manager, policy.h2d_stream_attr)
        if slot.d2h_done is not None:
            h2d_stream.wait_event(slot.d2h_done)

        with torch.cuda.stream(h2d_stream):
            if policy.device_pool is not None:
                # device side buffer fetch from pool for main-grad offload
                gpu_flat = pool.allocate(slot.numel, slot.dtype, slot.device, h2d_stream)
            else:
                # device side buffer allocated on demand for activation offload
                gpu_flat = torch.empty(slot.numel, dtype=slot.dtype, device=slot.device)

            if slot.host_is_zero:
                # skip the H2D entirely for main-grad slot
                gpu_flat.zero_()
            else:
                gpu_flat.copy_(slot.host_tensor.reshape(-1), non_blocking=True)
        slot.gpu_buf = gpu_flat.view(slot.shape)
        slot.h2d_done = torch.cuda.Event()
        slot.h2d_done.record(h2d_stream)
        slot.state = SlotState.DEVICE

        if policy.host_pool is not None:
            # recycle host buffer
            # the host copy is redundant the moment the device copy exists.
            pool.free(slot.host_base, slot.h2d_done)
            # release the reference
            slot.host_base = None
            slot.host_buf = None

    @staticmethod
    def offload_activation(
        x: torch.Tensor,
        stream_manager,
        key: str = "activations",
        handle: MoEOffloadHandle = None,
    ) -> MoEOffloadHandle:
        """Offload activation to pinned host memory and free the GPU storage."""
        if handle is None:
            handle = MoEOffloadHandle(policy=MoEOffloadManager.ACTIVATION_OFFLOAD, stream_manager=stream_manager)
        if handle.policy.kind != "activation":
            raise ValueError("Handle must be of kind 'activation' for offloading activations.")

        # check if the slot is already live
        prev = handle.slots.get(key)
        assert prev is None or prev.state is SlotState.RELEASED, (
            f"activation slot '{key}' is still live -- parking over it would leak its host buffer."
        )
        slot = MoEOffloadSlot(
            shape=tuple(x.shape),
            dtype=x.dtype,
            numel=x.numel(),
            element_size=x.element_size(),
            device=x.device,
        )
        handle.slots[key] = slot
        handle.active = True

        # schedule the offload
        stream_manager.d2h_stream_wait_producers(stream_manager.act_d2h_stream)
        MoEOffloadManager._d2h(handle, slot, x)
        return handle

    @staticmethod
    def offload_main_grad(
        handle: MoEOffloadHandle,
    ) -> None:
        """Write accumulated main grad back to host and free the GPU storage."""
        if handle is None or not handle.active:
            return
        if handle.policy.kind != "main_grad":
            raise ValueError("Handle must be of kind 'main_grad' for offloading main gradients.")

        stream_manager = handle.stream_manager
        stream_manager.d2h_stream_wait_producers(stream_manager.mgrad_d2h_stream)
        for slot in handle.slots.values():
            if slot.state is SlotState.DEVICE:
                MoEOffloadManager._d2h(handle, slot, slot.gpu_buf)

    @staticmethod
    def reload(
        handle: MoEOffloadHandle,
    ) -> None:
        """Stage every host-resident slot of ``handle`` on the device."""
        if handle is None or not handle.active:
            return
        for name, slot in handle.slots.items():
            if slot.state is SlotState.RELEASED:
                continue
            assert slot.state is SlotState.HOST, (
                f"offload slot '{name}' is already staged -- reload ran twice without an "
                "intervening offload, which for an accumulator would drop a microbatch's wgrad."
            )
            MoEOffloadManager._h2d(handle, slot)

    @staticmethod
    def get(
        handle: MoEOffloadHandle,
        name: str,
    ) -> torch.Tensor:
        """Return the staged device tensor for ``name``, ordered after its reload H2D."""
        slot = handle.slots.get(name)
        assert slot is not None and slot.state is SlotState.DEVICE, (
            f"offload slot '{name}' was not staged before its consumer ran -- the reload trigger "
            "must be wired on the combine output whenever this handle is active."
        )
        stream_manager = handle.stream_manager
        stream_manager.consumer_streams_wait_event(slot.h2d_done)
        gpu_buf = slot.gpu_buf
        if handle.policy.device_pool is None:
            # the device buffer is on demand allocated inside by H2D stream context
            # guarantee the consumer streams record the buffer
            stream_manager.consumer_streams_record(gpu_buf)
        if handle.policy.kind == "activation":
            # the device buffer is read-only, so the slot reference can be released
            # to save memory
            slot.gpu_buf = None
            slot.state = SlotState.RELEASED
        return gpu_buf

    @staticmethod
    def release_parameters(handle: MoEOffloadHandle) -> None:
        """Release read-only parameters after their final expert consumer launches."""
        if handle is None or not handle.active:
            return
        if handle.policy.kind != "parameter":
            raise ValueError("Handle must be of kind 'parameter' for releasing parameters.")

        # the D2H here does not actually happen
        # the wait here is to guarantee the GEMM is finished before the buffer is recycled
        stream_manager = handle.stream_manager
        release_stream = stream_manager.param_d2h_stream
        stream_manager.d2h_stream_wait_producers(release_stream)
        for slot in handle.slots.values():
            if slot.state is SlotState.DEVICE and slot.h2d_done is not None:
                release_stream.wait_event(slot.h2d_done)
        release_done = torch.cuda.Event()
        release_done.record(release_stream)

        pool = MoEOffloadMemoryPool.get_instance()
        for slot in handle.slots.values():
            if slot.state is not SlotState.DEVICE:
                continue
            pool.free(slot.gpu_buf.reshape(-1), release_done)
            slot.gpu_buf = None
            slot.h2d_done = None
            slot.d2h_done = release_done
            slot.state = SlotState.HOST


    # -------------- Active Offload Methods --------------

    @staticmethod
    def register(
        params: dict[str, torch.nn.Parameter],
        stream_manager,
        offload_param: bool = False,
        offload_main_grad: bool = True,
        parameter_tensors: dict[str, torch.Tensor] = None,
    ) -> tuple[MoEOffloadHandle, MoEOffloadHandle]:
        """Register persistent coarse-weight and main-gradient handles."""
        p_handle = None
        g_handle = None
        if offload_main_grad:
            g_handle = MoEOffloadHandle(policy=MoEOffloadManager.MAIN_GRAD_OFFLOAD, stream_manager=stream_manager)
            for name, param in params.items():
                main_grad = getattr(param, "main_grad", None)
                if main_grad is None or main_grad.device.type != "cpu":
                    # NOTE (fuguan): rethink this design
                    # all-or-nothing: if any param.main_grad is not on CPU, don't register any of them
                    g_handle.slots.clear()
                    break
                slot = MoEOffloadSlot(
                    param=param,
                    shape=tuple(param.shape),
                    dtype=param.main_grad.dtype,
                    numel=param.main_grad.numel(),
                    element_size=param.main_grad.element_size(),
                    device=torch.device(torch.cuda.current_device()),
                    host_is_zero=True,  # must be zero upon initialization
                    state=SlotState.HOST, # already on host
                )
                g_handle.slots[name] = slot
            g_handle.active = len(g_handle.slots) > 0
            if g_handle.active:
                MoEOffloadManager._live_persistent_handles.add(g_handle)

        if offload_param:
            assert parameter_tensors is not None, (
                "coarse-grained parameter offload requires packed FP8 parameter tensors"
            )
            p_handle = MoEOffloadHandle(policy=MoEOffloadManager.PARAM_OFFLOAD, stream_manager=stream_manager)
            for name, tensor in parameter_tensors.items():
                assert tensor.device.type == "cpu" and tensor.is_pinned(), (
                    f"coarse parameter '{name}' must be a pinned host tensor"
                )
                slot = MoEOffloadSlot(
                    host_buf=tensor,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    numel=tensor.numel(),
                    element_size=tensor.element_size(),
                    device=torch.device(torch.cuda.current_device()),
                    state=SlotState.HOST, # already on host
                )
                p_handle.slots[name] = slot
            p_handle.active = len(p_handle.slots) > 0
            if p_handle.active:
                MoEOffloadManager._live_persistent_handles.add(p_handle)

        return p_handle, g_handle

    @staticmethod
    def synchronize():
        """Synchronize all pending writebacks, then clear the pending list."""
        pending_writebacks = MoEOffloadManager._pending_writebacks
        for event in pending_writebacks.values():
            event.synchronize()
        pending_writebacks.clear()

    @staticmethod
    def mark_main_grad_zero():
        assert not MoEOffloadManager._pending_writebacks, (
            "there are pending writebacks of main grad when they are zeroed"
        )
        for handle in MoEOffloadManager._live_persistent_handles:
            if handle.active and handle.policy.kind == "main_grad":
                for slot in handle.slots.values():
                    slot.host_is_zero = True

class MoEReloadTrigger(torch.autograd.Function):
    """Backward trigger for reloading offloaded tensors."""

    @staticmethod
    def forward(ctx, output, handle: MoEOffloadHandle):
        ctx.handle = handle
        return output

    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.handle
        if handle is not None and handle.active:
            MoEOffloadManager.reload(handle)
        return grad_output, None
