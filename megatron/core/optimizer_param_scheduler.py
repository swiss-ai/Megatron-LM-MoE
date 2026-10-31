# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Learning rate decay and weight decay incr functions."""
import logging
import math
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch

from megatron.core.utils import log_single_rank

if TYPE_CHECKING:
    # Avoid circular import.
    from megatron.core.optimizer import MegatronOptimizer

logger = logging.getLogger(__name__)


class ParamGroupOverride(TypedDict):
    """Override values for a parameter group. These values may be optimizer-state/scheduler related.

    These are the values you see later in param_group.get(...) calls in the
        OptimizerParamScheduler.get_lr and get_wd methods. If you use a custom optimizer
        or scheduler, you could override those variables instead.

    Example:
        >>> param_group_override = ParamGroupOverride(min_lr=1e-4, wd_mult=0.1)
        >>> param_group_override == ParamGroupOverride(newvar=3) # this is ok too

    """

    max_lr: float
    min_lr: float
    start_wd: float
    end_wd: float
    wd_mult: float


def get_canonical_lr_for_logging(param_groups: list[dict]) -> float | None:
    """Return the lr of the first ``default_config=True`` param group.

    All ``default_config`` groups share the same LR schedule, so the first one
    is representative.  This includes empty rank-alignment stub groups, which
    the scheduler still writes a valid lr onto.

    Args:
        param_groups (list[dict]): parameter groups from the optimizer.

    Returns:
        float | None: the canonical learning rate, or None if no
            ``default_config=True`` group is found.
    """
    for param_group in param_groups:
        if param_group.get('default_config', False):
            return param_group.get('lr')
    return None


def param_group_override_to_tuple(
    param_group_override: ParamGroupOverride | None,
) -> tuple[tuple[str, Any], ...] | None:
    """Convert a param group override to a tuple for use as a key in a dictionary.

    The tuple is sorted by the keys of the param group override to handle different orderings of
     the keys in different override dictionaries which still mean the same thing.
    """
    if param_group_override is None:
        return None
    return tuple(sorted(param_group_override.items()))


def combine_param_group_overrides(
    param_group_overrides: list[ParamGroupOverride | None],
) -> ParamGroupOverride:
    """Combine a list of param group overrides into a single param group override.

    This function ensures that the overrides are not conflicting as well.

    Args:
        param_group_overrides (list[ParamGroupOverride]): list of param group overrides to combine

    Returns:
        ParamGroupOverride: combined param group override
    """
    combined_override = ParamGroupOverride()
    for override in param_group_overrides:
        if override is None:
            continue
        for key, value in override.items():
            if key in combined_override:
                if combined_override[key] != value:
                    raise ValueError(
                        f"Conflicting overrides for {key}: {combined_override[key]} and {value}"
                    )
            combined_override[key] = value
    return combined_override


LOG_BRIDGE_BUDGETS = ("remaining", "total")


def build_log_bridge_schedule(
    high_lr: float,
    low_lr: float,
    final_lr: float,
    total_steps: float,
    warmup_steps: float,
    switch_step: float,
    log_start: float = 0.70,
    rho_ref: float = 0.05,
    budget: str = "remaining",
):
    """Build the 'log-bridge' LR schedule and return ``lr(step)``.

    Starts a run at a high peak LR (``high_lr``) and, from ``switch_step`` on, brings it down so
    that its LR area matches that of a linear schedule with a smaller peak (``low_lr``).

    * Before ``switch_step``: the original schedule (linear warmup from 0 to ``high_lr``, then
      linear decay to ``final_lr`` at ``total_steps``). Identical to the 'linear' decay style, so a
      'linear' run can be resumed with this style at any step up to ``switch_step``.
    * From ``switch_step`` to ``log_start * total_steps``: an nGPT-log-shaped bridge, continuous
      at both ends. Its curvature (rho) is solved by bisection so the area budget is met.
    * From ``log_start * total_steps`` on: the tail of a reference run (linear warmup + nGPT log
      decay with ``rho_ref``) whose total LR area equals that of the low-LR linear schedule.

    ``budget='remaining'``: the area from ``switch_step`` on equals the low-LR schedule's
    remaining area. ``budget='total'``: the whole-run area equals the low-LR schedule's total
    area (the excess spent before the switch is paid back).

    Steps can be in any unit (Megatron passes samples). The returned callable carries an ``info``
    dict with the solved quantities.
    """
    if budget not in LOG_BRIDGE_BUDGETS:
        raise ValueError(f"log-bridge budget must be one of {LOG_BRIDGE_BUDGETS}, got {budget!r}")
    T, W, f = float(total_steps), float(warmup_steps), float(final_lr)
    ts = float(int(round(log_start * T)))
    if not W <= switch_step < ts < T:
        raise ValueError(
            "log-bridge needs warmup <= switch_step < log_start*total_steps < total_steps, got "
            f"warmup={W}, switch_step={switch_step}, log_start*total={ts}, total={T}"
        )

    def linear(peak):
        return lambda t: peak * t / W if t < W else peak - (peak - f) * (t - W) / (T - W)

    def area(fn, a, b):  # exact for piecewise-linear fn with a kink only at W
        pts = sorted({a, b} | ({W} if a < W < b else set()))
        return sum((fn(x) + fn(y)) / 2 * (y - x) for x, y in zip(pts, pts[1:]))

    def shape(rho, x):
        return 1 - math.log1p(x / rho) / math.log1p(1 / rho)

    def mean(rho):  # average of shape on [0, 1]
        return 1 / math.log1p(1 / rho) - rho

    def G(rho, x):  # antiderivative of log1p(x / rho)
        return (rho + x) * math.log1p(x / rho) - x

    orig, low = linear(high_lr), linear(low_lr)

    # Reference log run, peak calibrated to the low-LR schedule's total area.
    S, D, M = area(low, 0.0, T), T - W, mean(rho_ref)
    peak_ref = (S - D * f * (1 - M)) / (W / 2 + D * M)

    def ref(t):
        if t < W:
            return peak_ref * t / W
        return f + (peak_ref - f) * shape(rho_ref, (t - W) / D)

    # Tail area (closed form) and bridge endpoints.
    x1 = (ts - W) / D
    tail = f * (T - ts) + (peak_ref - f) * D * (
        (1 - x1) - (G(rho_ref, 1) - G(rho_ref, x1)) / math.log1p(1 / rho_ref)
    )
    lr0, lr_ts, L = orig(switch_step), ref(ts), ts - switch_step

    target = (
        area(low, switch_step, T) if budget == "remaining" else S - area(orig, 0.0, switch_step)
    )
    need = ((target - tail) / L - lr_ts) / (lr0 - lr_ts)
    if not 0 < need < 0.5:
        raise ValueError(
            f"log-bridge budget infeasible for a log shape (mean shape {need:.3f}, must be in "
            f"(0, 0.5)); high_lr={high_lr}, low_lr={low_lr}, final_lr={final_lr}. "
            "Change log_start, switch_step, budget or the low factor."
        )

    lo, hi = math.log(1e-8), math.log(1e4)  # bisect on log(rho); mean() is increasing in rho
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean(math.exp(mid)) < need else (lo, mid)
    rho_b = math.exp((lo + hi) / 2)
    # The bisection range caps how sharp the bridge can be: mean(1e-8) ~= 0.054, so a smaller
    # `need` would silently saturate at rho=1e-8 and overshoot the area budget. Refuse instead.
    if abs(mean(rho_b) - need) > 1e-6:
        raise ValueError(
            f"log-bridge budget infeasible for a log shape: the bridge needs mean shape "
            f"{need:.4f}, but rho in [1e-8, 1e4] only reaches [{mean(1e-8):.4f}, {mean(1e4):.4f}] "
            "(it would have to be a near-step drop). Change log_start, switch_step, budget or the "
            "low factor."
        )

    def lr(step):
        if step < switch_step:
            return orig(step)
        if step < ts:
            return lr_ts + (lr0 - lr_ts) * shape(rho_b, (step - switch_step) / L)
        return ref(min(step, T))

    lr.info = dict(
        peak_ref=peak_ref,
        rho_bridge=rho_b,
        lr_at_switch=lr0,
        lr_at_log_start=lr_ts,
        tail_area=tail,
        target_area=target,
    )
    return lr


class OptimizerParamScheduler:
    """Anneals learning rate and weight decay

    Args:
        optimizer (MegatronOptimizer): the optimizer to be used
        init_lr (float): initial learning rate
        max_lr (float): maximum learning rate
        min_lr (float): minimum learning rate
        lr_warmup_steps (int): number of warmup steps
        lr_decay_steps (int): number of decay steps
        lr_decay_style (str): decay style for learning rate
        start_wd (float): initial weight decay
        end_wd (float): final weight decay
        wd_incr_steps (int): number of weight decay increment steps
        wd_incr_style (str): weight decay increment style
        use_checkpoint_opt_param_scheduler (bool, optional): whether to use the checkpoint values
            for the optimizer param scheduler
        override_opt_param_scheduler (bool, optional): whether to override the optimizer param
            scheduler values with the class values
        wsd_decay_steps (int, optional): number of weight decay decay steps
        lr_wsd_decay_style (str, optional): decay style for learning rate during weight decay decay
            steps
        lr_bridge_low_factor (float, optional): 'log-bridge' only. Low peak LR as a
            fraction of each param group's max_lr (the group's max_lr is the high, original peak).
        lr_bridge_switch_steps (int, optional): 'log-bridge' only. Step at which the run leaves
            the original linear schedule.
        lr_bridge_log_start (float, optional): 'log-bridge' only. Fraction of lr_decay_steps at
            which the bridge hands over to the reference log tail.
        lr_bridge_rho_ref (float, optional): 'log-bridge' only. Curvature of the reference log
            decay.
        lr_bridge_budget (str, optional): 'log-bridge' only. 'remaining' or 'total' (see
            build_log_bridge_schedule).

    """

    def __init__(
        self,
        optimizer: "MegatronOptimizer",
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        start_wd: float,
        end_wd: float,
        wd_incr_steps: int,
        wd_incr_style: str,
        use_checkpoint_opt_param_scheduler: Optional[bool] = True,
        override_opt_param_scheduler: Optional[bool] = False,
        wsd_decay_steps: Optional[int] = None,
        lr_wsd_decay_style: Optional[str] = None,
        lr_bridge_low_factor: Optional[float] = None,
        lr_bridge_switch_steps: Optional[int] = None,
        lr_bridge_log_start: float = 0.70,
        lr_bridge_rho_ref: float = 0.05,
        lr_bridge_budget: str = "remaining",
    ) -> None:

        # Class values.
        self.optimizer = optimizer

        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style
        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_style = lr_decay_style
        if self.lr_decay_style == "WSD":
            assert self.wsd_decay_steps is not None

        self.lr_bridge_low_factor = lr_bridge_low_factor
        self.lr_bridge_switch_steps = lr_bridge_switch_steps
        self.lr_bridge_log_start = lr_bridge_log_start
        self.lr_bridge_rho_ref = lr_bridge_rho_ref
        self.lr_bridge_budget = lr_bridge_budget
        # One solved schedule per distinct (max_lr, min_lr) param-group pair.
        self._log_bridge_cache: dict = {}
        if self.lr_decay_style == "log-bridge":
            assert self.lr_bridge_low_factor is not None and self.lr_bridge_low_factor > 0, (
                "log-bridge needs a positive lr_bridge_low_factor"
            )
            assert self.lr_bridge_switch_steps is not None, "log-bridge needs lr_bridge_switch_steps"
            assert self.init_lr == 0.0, "log-bridge warms up from 0 (lr_warmup_init must be 0)"

        self.start_wd = start_wd
        self.end_wd = end_wd
        assert self.start_wd >= 0.0
        assert self.end_wd >= self.start_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, (
                'both override and ' 'use-checkpoint are set.'
            )

        # Set the learning rate
        self.step(0)
        log_single_rank(logger, logging.INFO, f"> learning rate decay style: {self.lr_decay_style}")

    def get_wd(self, param_group: Optional[dict] = None) -> float:
        """Weight decay incr functions

        Args:
            param_group (dict): parameter group from the optimizer."""

        if param_group is not None:
            start_wd = param_group.get('start_wd', self.start_wd)
            end_wd = param_group.get('end_wd', self.end_wd)
        else:
            start_wd = self.start_wd
            end_wd = self.end_wd

        if self.num_steps > self.wd_incr_steps:
            return end_wd

        if self.wd_incr_style == 'constant':
            assert start_wd == end_wd
            return end_wd

        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        assert incr_ratio >= 0.0
        assert incr_ratio <= 1.0
        delta_wd = end_wd - start_wd

        if self.wd_incr_style == 'linear':
            coeff = incr_ratio
        elif self.wd_incr_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise Exception(f'{self.wd_incr_style} weight decay increment style is not supported.')

        return start_wd + coeff * delta_wd

    def get_lr(self, param_group: dict) -> float:
        """Learning rate decay functions from:
        https://openreview.net/pdf?id=BJYwwY9ll pg. 4

        Args:
            param_group (dict): parameter group from the optimizer.
        """

        max_lr = param_group.get('max_lr', self.max_lr)
        min_lr = param_group.get('min_lr', self.min_lr)

        if self.lr_decay_style == 'log-bridge':
            return self._get_log_bridge_lr(max_lr, min_lr)

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return self.init_lr + (
                (max_lr - self.init_lr) * float(self.num_steps) / float(self.lr_warmup_steps)
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == 'constant':
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == 'inverse-square-root':
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = max_lr - min_lr

        coeff = None
        if self.lr_decay_style == 'linear':
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.lr_decay_style == 'WSD':
            wsd_anneal_start_ = self.lr_decay_steps - self.wsd_decay_steps
            if self.num_steps <= wsd_anneal_start_:
                coeff = 1.0
            else:
                wsd_steps = self.num_steps - wsd_anneal_start_
                wsd_decay_ratio = float(wsd_steps) / float(self.wsd_decay_steps)
                if self.lr_wsd_decay_style == "linear":
                    coeff = 1.0 - wsd_decay_ratio
                elif self.lr_wsd_decay_style == "cosine":
                    coeff = 0.5 * (math.cos(math.pi * wsd_decay_ratio) + 1.0)
                elif self.lr_wsd_decay_style == "exponential":
                    coeff = (2.0 * math.pow(0.5, wsd_decay_ratio)) - 1.0
                elif self.lr_wsd_decay_style == "minus_sqrt":
                    coeff = 1.0 - math.sqrt(wsd_decay_ratio)

        else:
            raise Exception(f'{self.lr_decay_style} decay style is not supported.')
        assert coeff is not None

        return min_lr + coeff * delta_lr

    def _get_log_bridge_lr(self, max_lr: float, min_lr: float) -> float:
        """'log-bridge' LR for a param group with the given bounds (solved once per pair)."""
        if max_lr <= min_lr:
            # Degenerate (constant) group: nothing to bridge.
            return max_lr
        key = (max_lr, min_lr)
        fn = self._log_bridge_cache.get(key)
        if fn is None:
            fn = build_log_bridge_schedule(
                high_lr=max_lr,
                low_lr=self.lr_bridge_low_factor * max_lr,
                final_lr=min_lr,
                total_steps=self.lr_decay_steps,
                warmup_steps=self.lr_warmup_steps,
                switch_step=self.lr_bridge_switch_steps,
                log_start=self.lr_bridge_log_start,
                rho_ref=self.lr_bridge_rho_ref,
                budget=self.lr_bridge_budget,
            )
            self._log_bridge_cache[key] = fn
            info = ", ".join(f"{k}={v:.6g}" for k, v in fn.info.items())
            log_single_rank(
                logger,
                logging.INFO,
                f"> log-bridge schedule for group max_lr={max_lr:.6g} min_lr={min_lr:.6g} "
                f"(low peak {self.lr_bridge_low_factor * max_lr:.6g}): {info}",
            )
        return fn(self.num_steps)

    def step(self, increment: int) -> None:
        """Set lr for all parameters groups.

        Args:
            increment (int): number of steps to increment
        """
        self.num_steps += increment
        # Do not skip empty param groups: get_canonical_lr_for_logging reads lr
        # from default_config groups regardless of whether they hold parameters.
        # This is important for logging under model parallelism that may leave
        # some ranks with empty default_config parameter groups.
        for param_group in self.optimizer.param_groups:
            if isinstance(param_group.get('lr'), torch.Tensor):
                param_group['lr'].fill_(self.get_lr(param_group))
            else:
                param_group['lr'] = self.get_lr(param_group)
            param_group['weight_decay'] = self.get_wd(param_group) * param_group.get('wd_mult', 1.0)

    def state_dict(self) -> dict:
        """Return the state dict."""
        state_dict = {
            'max_lr': self.max_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'num_steps': self.num_steps,
            'lr_decay_style': self.lr_decay_style,
            'lr_decay_steps': self.lr_decay_steps,
            'min_lr': self.min_lr,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_style': self.wd_incr_style,
            'wd_incr_steps': self.wd_incr_steps,
        }
        if self.lr_decay_style == 'log-bridge':
            state_dict.update(
                lr_bridge_low_factor=self.lr_bridge_low_factor,
                lr_bridge_switch_steps=self.lr_bridge_switch_steps,
                lr_bridge_log_start=self.lr_bridge_log_start,
                lr_bridge_rho_ref=self.lr_bridge_rho_ref,
                lr_bridge_budget=self.lr_bridge_budget,
            )
        return state_dict

    def _check_and_set(self, cls_value: float, sd_value: float, name: str) -> float:
        """Auxiliary function for checking the values in the checkpoint and
        setting them.

        Args:
            cls_value (float): class value
            sd_value (float): checkpoint value
            name (str): name of the parameter
        """

        if self.override_opt_param_scheduler:
            log_single_rank(logger, logging.INFO, f" > overriding {name} value to {cls_value}")
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            assert cls_value == sd_value, (
                f'OptimizerParamScheduler: class input value {cls_value} and checkpoint'
                f'value {sd_value} for {name} do not match'
            )

        log_single_rank(logger, logging.INFO, f" > using checkpoint value {sd_value} for {name}")
        return sd_value

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dict.

        Args:
            state_dict (dict): state dict to be load
        """

        if 'start_lr' in state_dict:
            max_lr_ = state_dict['start_lr']
        else:
            max_lr_ = state_dict['max_lr']
        self.max_lr = self._check_and_set(self.max_lr, max_lr_, 'learning rate')

        self.min_lr = self._check_and_set(
            self.min_lr, state_dict['min_lr'], 'minimum learning rate'
        )

        if 'warmup_iter' in state_dict:
            lr_warmup_steps_ = state_dict['warmup_iter']
        elif 'warmup_steps' in state_dict:
            lr_warmup_steps_ = state_dict['warmup_steps']
        else:
            lr_warmup_steps_ = state_dict['lr_warmup_steps']
        self.lr_warmup_steps = self._check_and_set(
            self.lr_warmup_steps, lr_warmup_steps_, 'warmup iterations'
        )

        if 'end_iter' in state_dict:
            lr_decay_steps_ = state_dict['end_iter']
        elif 'decay_steps' in state_dict:
            lr_decay_steps_ = state_dict['decay_steps']
        else:
            lr_decay_steps_ = state_dict['lr_decay_steps']
        self.lr_decay_steps = self._check_and_set(
            self.lr_decay_steps, lr_decay_steps_, 'total number of iterations'
        )

        if 'decay_style' in state_dict:
            lr_decay_style_ = state_dict['decay_style']
        else:
            lr_decay_style_ = state_dict['lr_decay_style']

        if 'num_iters' in state_dict:
            num_steps = state_dict['num_iters']
        else:
            num_steps = state_dict['num_steps']

        if (
            self.lr_decay_style == 'log-bridge'
            and lr_decay_style_ == 'linear'
            and not self.use_checkpoint_opt_param_scheduler
        ):
            # 'log-bridge' equals 'linear' up to the switch step, so a linear checkpoint can be
            # resumed into it without --override-opt-param-scheduler.
            log_single_rank(
                logger,
                logging.INFO,
                " > switching learning rate decay style from checkpoint 'linear' to 'log-bridge'",
            )
            if num_steps > self.lr_bridge_switch_steps:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"log-bridge: resuming a 'linear' checkpoint at step {num_steps}, past the "
                    f"switch step {self.lr_bridge_switch_steps}; the LR will jump.",
                )
        else:
            self.lr_decay_style = self._check_and_set(
                self.lr_decay_style, lr_decay_style_, 'learning rate decay style'
            )

        # log-bridge knobs: taken from the checkpoint only with --use-checkpoint-opt-param-scheduler;
        # otherwise the current args win (changing them mid-run reshapes the remaining schedule).
        for name in (
            'lr_bridge_low_factor',
            'lr_bridge_switch_steps',
            'lr_bridge_log_start',
            'lr_bridge_rho_ref',
            'lr_bridge_budget',
        ):
            if name not in state_dict:
                continue
            if self.use_checkpoint_opt_param_scheduler:
                setattr(self, name, state_dict[name])
            elif getattr(self, name) != state_dict[name]:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f" > {name}: using {getattr(self, name)} (checkpoint had {state_dict[name]})",
                )
        # Bounds above may have changed; re-solve the bridge schedules lazily.
        self._log_bridge_cache.clear()
        self.step(increment=num_steps)

        if 'start_wd' in state_dict:
            self.start_wd = self._check_and_set(
                self.start_wd, state_dict['start_wd'], "start weight decay"
            )
            self.end_wd = self._check_and_set(self.end_wd, state_dict['end_wd'], "end weight decay")
            self.wd_incr_steps = self._check_and_set(
                self.wd_incr_steps,
                state_dict['wd_incr_steps'],
                "total number of weight decay iterations",
            )
            self.wd_incr_style = self._check_and_set(
                self.wd_incr_style, state_dict['wd_incr_style'], "weight decay incr style"
            )
