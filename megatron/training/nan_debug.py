# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""NaN localization hooks — env-gated, inert unless NAN_DEBUG is set.

Purpose
-------
When the rerun-state-machine flags a non-reproducible NaN, it tells you *when*
(the iteration) but not *where* (which module). These hooks report the FIRST
module whose output (forward) or input-gradient (backward) goes non-finite in a
step — i.e. the origin, before the downstream cascade — so you can localize the
NaN to a specific layer/submodule and then read only that code.

Usage
-----
Set one or more of these in the run env (they do nothing unless set):

    NAN_DEBUG=1            # register fwd+bwd hooks; print first non-finite site/step
    NAN_DEBUG_ANOMALY=1    # also enable torch.autograd anomaly detection (SLOW,
                           # ~2x) — pinpoints the exact backward op with a stack trace
    NAN_DEBUG_EVERY=N      # only check every N steps (default 1). Reduces the
                           # per-module isfinite overhead on healthy steps.

Wiring: `nan_debug_new_step(iteration, model)` is called once per step at the top
of train_step(); it lazily registers the hooks on the first call and resets the
per-step "already reported" latch on every call. No effect when NAN_DEBUG is unset.

Notes
-----
* Overhead: when active, every hooked module runs one `isfinite().all()` reduction
  per forward/backward until the first non-finite is seen that step (then it
  short-circuits). This is a *debug* tool — run it only to localize, not in
  production. It is a complete no-op when NAN_DEBUG is unset.
* The first line printed in a step (forward order) is the origin; later lines are
  the cascade. For backward, hooks fire in reverse layer order, so prefer
  NAN_DEBUG_ANOMALY to pinpoint the true backward origin.
"""

import os

import torch


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


_ENABLED = _flag("NAN_DEBUG")
_ANOMALY = _flag("NAN_DEBUG_ANOMALY")
# Independent of NAN_DEBUG: a production fix, not diagnostics. Zeroes non-finite
# grad elements so a rare spurious NaN (e.g. fp8-offloading wgrad GEMM on a
# near-dead expert) can't poison the grad-norm and trigger reruns.
_SANITIZE = _flag("NAN_DEBUG_SANITIZE")
try:
    _EVERY = max(1, int(os.environ.get("NAN_DEBUG_EVERY", "1")))
except ValueError:
    _EVERY = 1

# Finite grad-norm spike localizer (independent of NAN_DEBUG). On a spike, prints
# the top params by grad-norm — the finite analog of the non-finite scan.
_SPIKE = _flag("NAN_DEBUG_SPIKE")
try:
    _SPIKE_THRESH = float(os.environ.get("NAN_DEBUG_SPIKE_THRESH", "2.0"))
except ValueError:
    _SPIKE_THRESH = 2.0
try:
    _SPIKE_WINDOW = max(2, int(os.environ.get("NAN_DEBUG_SPIKE_WINDOW", "20")))
except ValueError:
    _SPIKE_WINDOW = 20
try:
    _SPIKE_TOPK = max(1, int(os.environ.get("NAN_DEBUG_SPIKE_TOPK", "5")))
except ValueError:
    _SPIKE_TOPK = 5
_spike_hist: list = []

_hooks_registered = False
_reported_fwd = False
_reported_bwd = False
_current_iter = -1
_active_step = False  # whether we check this step (per NAN_DEBUG_EVERY)


def _rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return 0


def _stats(t: torch.Tensor) -> str:
    finite = torch.isfinite(t)
    n_bad = int((~finite).sum().item())
    f = t[finite].float()
    if f.numel():
        body = f"min={f.min().item():.3e} max={f.max().item():.3e} amax={f.abs().max().item():.3e}"
    else:
        body = "all-nonfinite"
    return f"n_bad={n_bad}/{t.numel()} {body}"


def _check(name: str, t, which: str) -> None:
    """Report the first non-finite floating tensor seen this step (per direction)."""
    global _reported_fwd, _reported_bwd
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        return
    if which == "fwd" and _reported_fwd:
        return
    if which == "bwd" and _reported_bwd:
        return
    if torch.isfinite(t).all():
        return
    if which == "fwd":
        _reported_fwd = True
    else:
        _reported_bwd = True
    print(
        f"[NAN-DEBUG] rank={_rank()} iter={_current_iter} {which.upper()} "
        f"first non-finite in '{name}' | dtype={t.dtype} shape={tuple(t.shape)} {_stats(t)}",
        flush=True,
    )


def _iter_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield "", obj
    elif isinstance(obj, (tuple, list)):
        for i, o in enumerate(obj):
            if isinstance(o, torch.Tensor):
                yield f"[{i}]", o


def _fwd_hook(name):
    def hook(module, inputs, output):
        if not _active_step or _reported_fwd:
            return
        for sfx, t in _iter_tensors(output):
            _check(name + sfx, t, "fwd")
    return hook


def _bwd_hook(name):
    def hook(module, grad_input, grad_output):
        if not _active_step or _reported_bwd:
            return
        for sfx, t in _iter_tensors(grad_input):
            _check(name + sfx, t, "bwd")
    return hook


def register_nan_hooks(model) -> None:
    """Register forward + full-backward NaN hooks on every submodule (once)."""
    global _hooks_registered
    if not _ENABLED or _hooks_registered:
        return
    chunks = model if isinstance(model, (list, tuple)) else [model]
    n = 0
    for ci, chunk in enumerate(chunks):
        for name, module in chunk.named_modules():
            if name == "":  # skip the root wrapper; its output == last child's
                continue
            tag = f"chunk{ci}.{name}"
            module.register_forward_hook(_fwd_hook(tag))
            module.register_full_backward_hook(_bwd_hook(tag))
            n += 1
    _hooks_registered = True
    if _ANOMALY:
        torch.autograd.set_detect_anomaly(True)
    if _rank() == 0:
        print(
            f"[NAN-DEBUG] active: registered fwd+bwd hooks on {n} modules/chunk-set; "
            f"every={_EVERY}; anomaly={_ANOMALY}. Reports first non-finite site per step.",
            flush=True,
        )


def nan_debug_new_step(iteration: int, model=None) -> None:
    """Call at the top of train_step: lazily register hooks; reset per-step latches."""
    global _reported_fwd, _reported_bwd, _current_iter, _active_step
    if not _ENABLED:
        return
    if model is not None:
        register_nan_hooks(model)
    _current_iter = iteration
    _reported_fwd = False
    _reported_bwd = False
    _active_step = (iteration % _EVERY == 0)


def nan_debug_check_grads(model, iteration: int) -> None:
    """Scan PARAMETER gradients for the first non-finite. Call after backward,
    before the optimizer step.

    The forward/backward module hooks only see activation gradients
    (grad_input/grad_output) — they CANNOT see weight gradients (wgrad). A NaN
    born in the wgrad (e.g. the fp8-offloading k-grouped wgrad GEMM) surfaces as
    a NaN grad-norm and is invisible to those hooks. This scans p.grad and
    p.main_grad and names the first offending parameter.
    """
    if not _ENABLED:
        return
    if iteration % _EVERY != 0:
        return
    chunks = model if isinstance(model, (list, tuple)) else [model]
    n_bad_params = 0
    first = None
    for ci, chunk in enumerate(chunks):
        for name, p in chunk.named_parameters():
            for gname in ("grad", "main_grad"):
                g = getattr(p, gname, None)
                if isinstance(g, torch.Tensor) and g.is_floating_point() and not torch.isfinite(g).all():
                    n_bad_params += 1
                    if first is None:
                        first = (f"chunk{ci}.{name}", gname, g)
                    break  # don't double-count grad vs main_grad for one param
    if first is not None:
        name, gname, g = first
        print(
            f"[NAN-DEBUG] rank={_rank()} iter={iteration} GRAD first non-finite param "
            f"'{name}'.{gname} ({n_bad_params} param-grads non-finite this rank) | "
            f"dtype={g.dtype} shape={tuple(g.shape)} {_stats(g)}",
            flush=True,
        )


def _median(xs: list) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def nan_debug_check_grad_spikes(model, iteration: int) -> None:
    """On a grad-norm SPIKE, report the top-K parameters by grad-norm — the finite
    analog of nan_debug_check_grads. Guarded by NAN_DEBUG_SPIKE.

    A spike is: total-grad-norm > NAN_DEBUG_SPIKE_THRESH x the median of the last
    NAN_DEBUG_SPIKE_WINDOW steps (defaults 2.0x / 20 — the 20 matches the rerun
    machine's num_samples). On a spike it prints the params contributing most to
    the norm, so you can see whether spikes come from the same weight/layer.

    Grads are per-rank/local, so the total here won't equal the rerun machine's
    global grad_norm — but the rank(s) holding the spiking param will show it in
    their top-K. Cost: one norm per param on GPU + one host sync for the total.
    Call after backward, before prepare_grad_norm().
    """
    if not _SPIKE:
        return
    if iteration % _EVERY != 0:
        return
    names = []
    norms = []
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for ci, chunk in enumerate(chunks):
        for name, p in chunk.named_parameters():
            g = getattr(p, "main_grad", None)
            if g is None:
                g = p.grad
            if isinstance(g, torch.Tensor) and g.is_floating_point():
                names.append(f"chunk{ci}.{name}")
                norms.append(g.detach().float().norm())
    if not norms:
        return
    norms_t = torch.stack(norms)
    total = norms_t.norm().item()  # one host sync
    if len(_spike_hist) >= min(_SPIKE_WINDOW, 5):
        med = _median(_spike_hist)
        if med > 0 and total > _SPIKE_THRESH * med:
            k = min(_SPIKE_TOPK, len(names))
            topv, topi = torch.topk(norms_t, k)
            top = ", ".join(
                f"{names[i]}={v:.3e}" for i, v in zip(topi.tolist(), topv.tolist())
            )
            print(
                f"[NAN-DEBUG] rank={_rank()} iter={iteration} SPIKE local_total_grad_norm={total:.3e} "
                f"(>{_SPIKE_THRESH}x recent median {med:.3e}); top-{k} params by grad-norm: {top}",
                flush=True,
            )
    _spike_hist.append(total)
    if len(_spike_hist) > _SPIKE_WINDOW:
        _spike_hist.pop(0)


def nan_debug_sanitize_grads(model) -> None:
    """Zero non-finite elements in parameter gradients (grad + main_grad), in place.

    Guarded by NAN_DEBUG_SANITIZE (independent of NAN_DEBUG — this is a fix, not a
    diagnostic). Unconditional ``nan_to_num_`` (nan/inf -> 0): no isfinite check,
    so no host sync, and a true no-op on finite grads. nan_to_num is element-wise,
    so a real gradient keeps all its finite values — only the spurious element(s)
    are zeroed. inf is mapped to 0 too (not to 3.4e38) so it can't re-inflate the
    grad-norm. Call after backward, before prepare_grad_norm() / the optimizer.

    Safe here because the target is a single ~0-magnitude artifact element from
    the fp8-offloading wgrad GEMM on a near-dead expert; zeroing it has no training
    impact. A genuine gradient spike would be handled by grad clipping, not this.
    """
    if not _SANITIZE:
        return
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for p in chunk.parameters():
            for gname in ("grad", "main_grad"):
                g = getattr(p, gname, None)
                if isinstance(g, torch.Tensor) and g.is_floating_point():
                    torch.nan_to_num_(g, nan=0.0, posinf=0.0, neginf=0.0)


def sanitize_enabled() -> bool:
    return _SANITIZE


def nan_debug_check_tensor(name: str, t) -> None:
    """Check one named intermediate tensor for non-finite values (guarded by
    NAN_DEBUG). Uses the iteration set by nan_debug_new_step. One host sync per
    call — place it OUTSIDE per-chunk loops, on the full tensor. For localizing
    NaNs born inside custom autograd Functions (e.g. the fp8-offloading backward),
    which the module fwd/bwd hooks never see.
    """
    if not _ENABLED:
        return
    if _current_iter % _EVERY != 0:
        return
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        return
    if not torch.isfinite(t).all():
        print(
            f"[NAN-DEBUG] rank={_rank()} iter={_current_iter} TENSOR non-finite in '{name}' | "
            f"dtype={t.dtype} shape={tuple(t.shape)} {_stats(t)}",
            flush=True,
        )


def enabled() -> bool:
    return _ENABLED
