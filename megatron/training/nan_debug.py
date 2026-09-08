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
try:
    _EVERY = max(1, int(os.environ.get("NAN_DEBUG_EVERY", "1")))
except ValueError:
    _EVERY = 1

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


def enabled() -> bool:
    return _ENABLED
