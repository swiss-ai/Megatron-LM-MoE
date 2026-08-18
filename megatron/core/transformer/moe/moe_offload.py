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

    def act_d2h_stream_wait_producers(self):
        """Make the activation-offload D2H stream wait for activation producers."""
        for launch_stream in self.get_launch_streams():
            self.act_d2h_stream.wait_stream(launch_stream)
        for i in range(self.num_compute_streams):
            self.act_d2h_stream.wait_stream(self.compute_streams[i])

    def consumer_streams_wait_act_reload(self, h2d_done_event):
        """Make backward consumer streams wait until activation reload H2D completes."""
        self.consumer_streams_wait_event(h2d_done_event)

    def consumer_streams_record(self, tensor):
        for launch_stream in self.get_launch_streams():
            tensor.record_stream(launch_stream)
        for i in range(self.num_compute_streams):
            tensor.record_stream(self.compute_streams[i])

    def mgrad_d2h_stream_wait_producers(self):
        """Make the main-grad writeback D2H stream wait for the wgrad GEMMs."""
        for launch_stream in self.get_launch_streams():
            self.mgrad_d2h_stream.wait_stream(launch_stream)
        for i in range(self.num_compute_streams):
            self.mgrad_d2h_stream.wait_stream(self.compute_streams[i])

    def consumer_streams_wait_mgrad_reload(self, h2d_done_event):
        """Make the wgrad consumer streams wait until the main-grad staging H2D completes."""
        self.consumer_streams_wait_event(h2d_done_event)

    def d2h_stream_wait_producers(self, d2h_stream):
        """Make an arbitrary D2H stream wait for whoever produced the tensors it is about to read.

        The ``act_``/``mgrad_`` variants above are this method with the stream hard-coded;
        :class:`MoEOffloadHandle` calls this one with the stream its policy names.
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
    host_buf: torch.Tensor = None  # activation: typed view sliced out of the base
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

    Pure state; the transitions are driven by :class:`MoEOffloadManager`.

    Activation offload and main-grad offload are the same mechanism with the two sides swapped:

    ==================  =============================  ==================================
    ..                  activation offload             main-grad offload
    ==================  =============================  ==================================
    permanent side      device (the live activation)   host (``param.main_grad``)
    transient side      host (rented pinned buffer)    device (rented staging buffer)
    ``offload`` (D2H)   park, in forward               write back, after the wgrad GEMMs
    ``reload`` (H2D)    restore, before expert bwd     stage, before the wgrad GEMMs
    payload             read-only                      accumulator (read-modify-write)
    handle lifetime     one per forward instance       one per module, reused forever
    ==================  =============================  ==================================

    Everything else -- the pool's ready-event protocol, the ``d2h_done``/``h2d_done`` handshake,
    "wait producers before D2H, wait the H2D before consuming" -- is common to both. The
    differences above are declared once in :class:`MoEOffloadPolicy`, so the driver branches on
    meaning (``if policy.writeback``) rather than on kind.

    Per-slot state machine the driver maintains::

        RELEASED --offload(name, t)--> HOST --reload()--> DEVICE --get(name)--> RELEASED   (act)
                                        ^                    |
                                        +-----offload()------+                       (main grad)

    ``get`` is terminal for a read-only payload and non-mutating for an accumulator: that single
    bit is what stops a consumed activation being pinned across the optimizer step, while keeping
    a staged main grad alive until its writeback.

    Instances are identity-keyed (``eq=False``) so they can live in a ``WeakSet``. Slot names are
    the caller's vocabulary: ``"moe_input"`` / ``"moe_fc1_output"`` for activations (matching the
    ``moe_offload_activations`` config spellings), ``"cpu_w1"`` / ``"cpu_w2"`` for main grads.
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

    WEIGHT_OFFLOAD = MoEOffloadPolicy(
        kind="weight",
        host_pool=None,
        device_pool=MoEOffloadMemoryPool,
        writeback=False,
        d2h_stream_attr="weight_d2h_stream",
        h2d_stream_attr="weight_h2d_stream",
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
        """Stage every host-resident slot of ``handle`` back onto the device. Serves both kinds.

        Driven from the reload trigger's backward so the H2D overlaps the combine-backward
        all-to-all. Activation slots already consumed by ``get`` are skipped.
        """
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
        if not handle.policy.writeback:
            # the device buffer is read-only, so the slot reference can be released
            # to save memory
            slot.gpu_buf = None
            slot.state = SlotState.RELEASED
        return gpu_buf


    # -------------- Active Offload Methods --------------

    @staticmethod
    def register(
        params: dict[str, torch.nn.Parameter],
        stream_manager,
    ) -> MoEOffloadHandle:
        handle = MoEOffloadHandle(policy=MoEOffloadManager.MAIN_GRAD_OFFLOAD, stream_manager=stream_manager)
        for name, param in params.items():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is None or main_grad.device.type != "cpu":
                # NOTE (fuguan): rethink this design
                # all-or-nothing: if any param.main_grad is not on CPU, don't register any of them
                return MoEOffloadHandle(policy=MoEOffloadManager.MAIN_GRAD_OFFLOAD, stream_manager=stream_manager)
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
            handle.slots[name] = slot
        handle.active = len(handle.slots) > 0
        if handle.active:
            MoEOffloadManager._live_persistent_handles.add(handle)
        return handle

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
        

@dataclass
class ActivationOffloadHandle:
    """Handle carrying an offloaded expert activation through autograd.

    The handle is attached to both the expert autograd node and the reload-trigger node, so each
    forward instance owns its matching offload state even under VPP interleaving.
    """

    active: bool = False
    cpu_base: torch.Tensor = None  # full pinned uint8 buffer (owned by the pool); returned on free
    cpu_flat: torch.Tensor = None  # length-``numel`` fp8 view sliced out of ``cpu_base``
    d2h_done: torch.cuda.Event = None
    h2d_done: torch.cuda.Event = None
    gpu_slot: torch.Tensor = None  # reloaded fp8_x_full, filled in the reload backward
    shape: tuple = None
    dtype: torch.dtype = None
    device: torch.device = None
    numel: int = 0
    stream_manager: object = None  # StreamManager carrying the act_d2h/act_h2d streams

    # Slot for fp8_fc1_output (the first-linear gated pre-activation).
    fc1_active: bool = False
    cpu_base_fc1: torch.Tensor = None
    cpu_flat_fc1: torch.Tensor = None
    d2h_done_fc1: torch.cuda.Event = None
    h2d_done_fc1: torch.cuda.Event = None
    gpu_slot_fc1: torch.Tensor = None
    shape_fc1: tuple = None
    dtype_fc1: torch.dtype = None
    numel_fc1: int = 0


class PinnedActBufferPool:
    """Recycling pool of flat pinned host buffers for activation offload, keyed by capacity.

    ``x``'s token dimension varies every microbatch, so keying by shape would spawn an
    unbounded number of single-use pools. Instead we hold flat ``uint8`` buffers whose capacity is
    rounded up to ``_ACT_POOL_GRANULARITY_BYTES``, and each offload slices the exact bytes it needs
    out of a buffer (best-fit by capacity). This bounds the pool to a handful of capacity classes
    regardless of shape variance.

    Pinned host alloc/free (``cudaHostAlloc``/``cudaFreeHost``) synchronize the device, so buffers
    are never freed once created -- only recycled. A recycled buffer carries the ``h2d_done`` event
    of its previous reload; the next D2H write (issued on ``act_d2h_stream``) waits that event so it
    cannot overwrite a buffer whose reload H2D is still reading it.
    """

    _instance = None
    _ACT_POOL_GRANULARITY_BYTES = 2 * 1024 * 1024  # 2 MiB

    @classmethod
    def get_instance(cls) -> PinnedActBufferPool:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # List of [buffer, ready_event_or_None]; scanned best-fit by capacity.
        self._free: list = []
        # Keep strong refs so buffers are never GC'd / freed (avoids cudaFreeHost syncs).
        self._all: list = []
        self._total_bytes: int = 0

    def stats(self) -> tuple:
        """(num_buffers, total_bytes, free_bytes)"""
        return len(self._all), self._total_bytes, sum(b.numel() for b, _ev in self._free)

    def allocate(self, nbytes: int, wait_stream: torch.cuda.Stream) -> torch.Tensor:
        """Return a flat pinned ``uint8`` buffer with capacity >= ``nbytes``."""
        best_idx = -1
        best_cap = None
        for i, (buf, _ev) in enumerate(self._free):
            cap = buf.numel()
            if cap >= nbytes and (best_cap is None or cap < best_cap):
                best_cap, best_idx = cap, i
        if best_idx >= 0:
            buf, ready_event = self._free.pop(best_idx)
            if ready_event is not None:
                wait_stream.wait_event(ready_event)
            return buf

        cap = (
            (nbytes + self._ACT_POOL_GRANULARITY_BYTES - 1)
            // self._ACT_POOL_GRANULARITY_BYTES
            * self._ACT_POOL_GRANULARITY_BYTES
        )
        buf = torch.empty(cap, dtype=torch.uint8, device="cpu", pin_memory=True)
        self._all.append(buf)
        self._total_bytes += cap
        return buf

    def free(self, buf: torch.Tensor, ready_event: torch.cuda.Event = None):
        """Return ``buf`` to the pool, tagged with the event after which it is safe to overwrite."""
        self._free.append([buf, ready_event])


class MoEActivationOffloadManager:
    """Stateless driver for offloading/reloading FP8 expert activations."""

    @staticmethod
    def _offload_tensor(
        x: torch.Tensor,
        stream_manager,
        pool: PinnedActBufferPool,
    ) -> tuple:
        """Copy a contiguous GPU tensor into pinned host memory and free its GPU storage."""
        d2h_stream = stream_manager.act_d2h_stream
        x_flat = x.reshape(-1)
        numel = x_flat.numel()
        nbytes = numel * x.element_size()
        with torch.cuda.stream(d2h_stream):
            cpu_base = pool.allocate(nbytes, d2h_stream)
            cpu_flat = cpu_base[:nbytes].view(x.dtype)
            cpu_flat.copy_(x_flat, non_blocking=True)
        d2h_done = torch.cuda.Event()
        d2h_done.record(d2h_stream)

        x.record_stream(d2h_stream)
        x.untyped_storage().resize_(0)
        return cpu_base, cpu_flat, d2h_done, tuple(x.shape), x.dtype, numel

    @staticmethod
    def offload(
        x: torch.Tensor,
        stream_manager,
    ) -> ActivationOffloadHandle:
        """Park ``x`` on pinned host memory and free its GPU storage."""
        pool = PinnedActBufferPool.get_instance()
        stream_manager.act_d2h_stream_wait_producers()
        cpu_base, cpu_flat, d2h_done, shape, dtype, numel = MoEActivationOffloadManager._offload_tensor(
            x, stream_manager, pool
        )
        return ActivationOffloadHandle(
            active=True,
            shape=shape,
            dtype=dtype,
            device=x.device,
            numel=numel,
            stream_manager=stream_manager,
            cpu_base=cpu_base,
            cpu_flat=cpu_flat,
            d2h_done=d2h_done,
        )

    @staticmethod
    def offload_fc1(
        handle: ActivationOffloadHandle,
        x: torch.Tensor,
    ) -> None:
        """Park ``fp8_fc1_output`` in the handle's fc1 slot."""
        pool = PinnedActBufferPool.get_instance()
        stream_manager = handle.stream_manager
        stream_manager.act_d2h_stream_wait_producers()
        cpu_base, cpu_flat, d2h_done, shape, dtype, numel = MoEActivationOffloadManager._offload_tensor(
            x, stream_manager, pool
        )
        handle.fc1_active = True
        handle.cpu_base_fc1 = cpu_base
        handle.cpu_flat_fc1 = cpu_flat
        handle.d2h_done_fc1 = d2h_done
        handle.shape_fc1 = shape
        handle.dtype_fc1 = dtype
        handle.numel_fc1 = numel

    @staticmethod
    def reload(
        handle: ActivationOffloadHandle,
    ) -> None:
        """Reload offloaded activation slots from host into fresh GPU tensors."""
        stream_manager = handle.stream_manager
        h2d_stream = stream_manager.act_h2d_stream
        h2d_stream.wait_event(handle.d2h_done)
        with torch.cuda.stream(h2d_stream):
            gpu_slot = torch.empty(handle.shape, dtype=handle.dtype, device=handle.device)
            gpu_slot.reshape(-1).copy_(handle.cpu_flat, non_blocking=True)
        handle.gpu_slot = gpu_slot
        handle.h2d_done = torch.cuda.Event()
        handle.h2d_done.record(h2d_stream)

        if handle.fc1_active:
            h2d_stream.wait_event(handle.d2h_done_fc1)
            with torch.cuda.stream(h2d_stream):
                gpu_slot_fc1 = torch.empty(
                    handle.shape_fc1, dtype=handle.dtype_fc1, device=handle.device
                )
                gpu_slot_fc1.reshape(-1).copy_(handle.cpu_flat_fc1, non_blocking=True)
            handle.gpu_slot_fc1 = gpu_slot_fc1
            handle.h2d_done_fc1 = torch.cuda.Event()
            handle.h2d_done_fc1.record(h2d_stream)


class MoEActReloadTrigger(torch.autograd.Function):
    """Identity node whose backward launches activation reload."""

    @staticmethod
    def forward(ctx, output, handle):
        ctx.act_offload_handle = handle
        return output

    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.act_offload_handle
        if handle is not None and handle.active:
            MoEActivationOffloadManager.reload(handle)
        return grad_output, None


@dataclass
class MainGradSlot:
    """One offloaded ``param.main_grad``: its host home plus its transient GPU staging tensor.

    Unlike an activation slot, the shape is fixed by the config at module construction, so
    ``shape``/``dtype``/``numel`` are filled in once at registration and never re-derived.
    """

    param: torch.nn.Parameter = None  # host home is read lazily as ``param.main_grad``
    shape: tuple = None
    dtype: torch.dtype = None
    numel: int = 0
    device: torch.device = None
    gpu_slot: torch.Tensor = None  # staged fp32 accumulator, live between reload and offload
    h2d_done: torch.cuda.Event = None
    d2h_done: torch.cuda.Event = None
    # when the host copy is known to be all-zero (right after ``zero_grad_buffer``) for first microbatch
    # lets the next reload zero the GPU slot instead of paying an H2D.
    host_is_zero: bool = False


@dataclass(eq=False)
class MainGradOffloadHandle:
    """Handle carrying a layer's host-resident expert main-grads through the backward pass.

    The mirror of :class:`ActivationOffloadHandle`. Because the shapes are determined, one handle is built per module at construction time and
    reused every microbatch.
    """

    active: bool = False
    slots: dict = None  # name -> MainGradSlot
    stream_manager: object = None  # StreamManager carrying the mgrad_d2h/mgrad_h2d streams

    def __post_init__(self):
        if self.slots is None:
            self.slots = {}


class GpuMainGradSlotPool:
    """Recycling pool of flat GPU staging buffers for main-grad offload, keyed by exact capacity.

    The counterpart of :class:`PinnedActBufferPool`, with the two sides swapped: for main grads the
    host buffer is permanent (it is the DDP grad buffer) and the *device* buffer is the transient
    one, so this pool hands out GPU tensors.

    Activation shapes vary per microbatch, which is why the pinned pool rounds capacities up to a
    granularity and searches best-fit. Main-grad shapes are fixed by the config, so there are only
    as many capacity classes as there are distinct expert-weight sizes (two per rank: fc1 and fc2)
    and an exact ``(numel, dtype)`` key is enough -- no rounding, no best-fit scan, no slack.

    Buffers are never released back to the caching allocator, only recycled, so the pool's
    high-water mark is the whole GPU cost of the feature: ``concurrent_slots * slot_bytes``.
    Because the storage is retained, consumers do not need ``record_stream`` on a slot; reuse is
    ordered explicitly instead, by the ``d2h_done`` event a slot carries when it is freed.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> GpuMainGradSlotPool:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # (numel, dtype, device) -> list of [buffer, ready_event_or_None].
        self._free: dict = {}
        # strong refs of buffers so the memory never returned to the allocator.
        self._all: list = []
        self._total_bytes: int = 0

    def stats(self) -> tuple:
        """(num_buffers, total_bytes, free_bytes)"""
        return len(self._all), self._total_bytes, sum(
            b.numel() * b.element_size() for fl in self._free.values() for b, _ev in fl
        )

    def allocate(
        self,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
        wait_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        """Return a flat GPU buffer of exactly ``numel`` elements of ``dtype``."""
        free_list = self._free.setdefault((numel, dtype, device), [])
        if free_list:
            buf, ready_event = free_list.pop()
            if ready_event is not None:
                # a previous writeback D2H is pending
                # wait until the copy has landed.
                wait_stream.wait_event(ready_event)
            return buf

        # if there is no free buffer, allocate a new one and add into reference
        buf = torch.empty(numel, dtype=dtype, device=device)
        self._all.append(buf)
        self._total_bytes += buf.numel() * buf.element_size()
        return buf

    def free(self, buf: torch.Tensor, ready_event: torch.cuda.Event = None):
        """Return ``buf`` to the pool, tagged with the event after which it is safe to overwrite."""
        self._free.setdefault((buf.numel(), buf.dtype, buf.device), []).append([buf, ready_event])


class MoEMainGradOffloadManager:
    """Stateless driver for staging/writing back host-resident expert main-grads.

    Similar to :class:`MoEActivationOffloadManager`: ``reload`` performs H2D ``offload`` performs D2H. 
    Main-grad is reloaded before backward and offloaded once the wgrad GEMMs completed.
    """

    # un-awaited writeback event per D2H stream: cuda stream -> torch.Event
    # host buffer must not be read or writed while one of these is outstanding.
    _pending_writebacks: dict = {}

    # all live handles for all modules. DDP can mark the whole model's main-grads zero in one call 
    _live_handles: weakref.WeakSet = weakref.WeakSet()

    @staticmethod
    def register(
        params: dict,
        stream_manager,
    ) -> MainGradOffloadHandle:
        """Build the persistent handle for a module's expert weights. Called **once per module**.

        ``params``: ``{"w1": self.weight1, "w2": self.weight2}``. 
        """
        handle = MainGradOffloadHandle(stream_manager=stream_manager)
        for name, param in params.items():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is None or main_grad.device.type != "cpu":
                return MainGradOffloadHandle(stream_manager=stream_manager)
            handle.slots[name] = MainGradSlot(
                param=param,
                shape=tuple(main_grad.shape),
                dtype=main_grad.dtype,
                numel=main_grad.numel(),
                device=torch.device(torch.cuda.current_device()),
            )
        handle.active = len(handle.slots) > 0
        if handle.active:
            # upon register the buffer DDP allocated is zeroed
            # and stays zeroed until the first writeback
            MoEMainGradOffloadManager.mark_host_zero(handle)
            MoEMainGradOffloadManager._live_handles.add(handle)
        return handle

    @staticmethod
    def reload(
        handle: MainGradOffloadHandle,
    ) -> None:
        """Stage every slot's host main-grad into a fresh GPU accumulator."""
        if handle is None or not handle.active:
            return
        stream_manager = handle.stream_manager
        h2d_stream = stream_manager.mgrad_h2d_stream
        pool = GpuMainGradSlotPool.get_instance()
        for name, slot in handle.slots.items():
            assert slot.gpu_slot is None, (
                f"main-grad slot '{name}' is already staged -- reload ran twice without an "
                "intervening offload, which would drop a microbatch's wgrad."
            )
            if slot.d2h_done is not None:
                # wait until the previous d2h write is done
                h2d_stream.wait_event(slot.d2h_done)
            with torch.cuda.stream(h2d_stream):
                gpu_flat = pool.allocate(slot.numel, slot.dtype, slot.device, h2d_stream)
                if slot.host_is_zero:
                    # skip the H2D.
                    gpu_flat.zero_()
                else:
                    gpu_flat.copy_(slot.param.main_grad.reshape(-1), non_blocking=True)
            slot.gpu_slot = gpu_flat.view(slot.shape)
            slot.h2d_done = torch.cuda.Event()
            slot.h2d_done.record(h2d_stream)

    @staticmethod
    def get(
        handle: MainGradOffloadHandle,
        name: str,
    ) -> torch.Tensor:
        """Return the staged GPU accumulator for ``name``, ordered after its staging H2D.
        """
        slot = handle.slots[name]
        assert slot.gpu_slot is not None, (
            f"main-grad slot '{name}' was not staged before the wgrad GEMM. "
            "MoEMainGradReloadTrigger must be wired on the combine output when "
            "moe_offload_main_grads is enabled."
        )
        handle.stream_manager.consumer_streams_wait_mgrad_reload(slot.h2d_done)
        return slot.gpu_slot

    @staticmethod
    def offload(
        handle: MainGradOffloadHandle,
    ) -> None:
        """Write every staged accumulator back to its host main-grad and release the GPU slot."""
        if handle is None or not handle.active:
            return
        stream_manager = handle.stream_manager
        d2h_stream = stream_manager.mgrad_d2h_stream
        pool = GpuMainGradSlotPool.get_instance()
        stream_manager.mgrad_d2h_stream_wait_producers()
        for slot in handle.slots.values():
            if slot.gpu_slot is None:
                # no wgrad ran for this slot this microbatch (e.g. no tokens routed here).
                continue
            # Only ``get`` orders the GEMMs behind the staging H2D, so a slot staged but never
            # handed to a GEMM never took that dependency -- wait the H2D directly too.
            d2h_stream.wait_event(slot.h2d_done)
            gpu_flat = slot.gpu_slot.reshape(-1)
            with torch.cuda.stream(d2h_stream):
                slot.param.main_grad.reshape(-1).copy_(gpu_flat, non_blocking=True)
            slot.d2h_done = torch.cuda.Event()
            slot.d2h_done.record(d2h_stream)
            slot.host_is_zero = False
            pool.free(gpu_flat, slot.d2h_done)
            slot.gpu_slot = None
            slot.h2d_done = None
            MoEMainGradOffloadManager._pending_writebacks[d2h_stream.cuda_stream] = slot.d2h_done

    @staticmethod
    def mark_host_zero(handle: MainGradOffloadHandle) -> None:
        """Record that the host main-grads are all-zero, letting the next reload skip its H2D."""
        for slot in handle.slots.values():
            slot.host_is_zero = True

    @staticmethod
    def mark_all_host_zero() -> None:
        """Mark every live handle's host main-grads zero, after DDP cleared the grad buffers.
        """
        assert not MoEMainGradOffloadManager._pending_writebacks, (
            "grad buffers were zeroed while a main-grad writeback was still in flight -- the D2H "
            "would land after the zero_() and resurrect last iteration's gradients. "
            "MoEMainGradOffloadManager.synchronize() must run before zero_grad_buffer()."
        )
        for handle in MoEMainGradOffloadManager._live_handles:
            MoEMainGradOffloadManager.mark_host_zero(handle)

    @staticmethod
    def synchronize() -> None:
        """Block until every outstanding writeback D2H has landed in host memory.

        Must be called before operations on ``main_grad``: the DDP grad reduce / zero_grad
        """
        pending = MoEMainGradOffloadManager._pending_writebacks
        for event in pending.values():
            event.synchronize()
        pending.clear()


class MoEMainGradReloadTrigger(torch.autograd.Function):
    """Identity node whose backward stages the host main-grads onto the GPU.
    """

    @staticmethod
    def forward(ctx, output, handle):
        ctx.main_grad_offload_handle = handle
        return output

    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.main_grad_offload_handle
        if handle is not None and handle.active:
            MoEMainGradOffloadManager.reload(handle)
        return grad_output, None

