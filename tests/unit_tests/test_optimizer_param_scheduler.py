# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from unittest.mock import MagicMock

import pytest

from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,
    get_canonical_lr_for_logging,
)


@pytest.fixture
def mock_optimizer():
    optimizer = MagicMock()
    optimizer.param_groups = [{'lr': 0.0, 'weight_decay': 0.0}]
    return optimizer


def test_initialization(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    assert scheduler.init_lr == 0.01
    assert scheduler.max_lr == 0.1
    assert scheduler.min_lr == 0.001
    assert scheduler.lr_warmup_steps == 100
    assert scheduler.lr_decay_steps == 1000
    assert scheduler.lr_decay_style == 'linear'
    assert scheduler.start_wd == 0.0
    assert scheduler.end_wd == 0.1
    assert scheduler.wd_incr_steps == 1000
    assert scheduler.wd_incr_style == 'linear'


def test_get_wd_constant(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='constant',
    )

    scheduler.step(500)
    wd = scheduler.get_wd()
    assert wd == 0.1


def test_get_wd_linear(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    scheduler.step(500)
    wd = scheduler.get_wd()
    assert wd == 0.05


def test_get_wd_cosine(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='cosine',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='cosine',
    )

    scheduler.step(500)
    wd = scheduler.get_wd()
    expected_wd = 0.05 * (math.cos(math.pi * (1 - 0.5)) + 1.0)
    assert math.isclose(wd, expected_wd, rel_tol=1e-5)


def test_get_lr_linear(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    param_group = {'max_lr': 0.1, 'min_lr': 0.001}

    scheduler.step(50)
    lr = scheduler.get_lr(param_group)
    expected_lr = 0.01 + (0.1 - 0.01) * (50 / 100)
    assert math.isclose(lr, expected_lr, rel_tol=1e-5)

    scheduler.step(450)
    lr = scheduler.get_lr(param_group)
    expected_lr = 0.1 - ((0.1 - 0.001) * ((500 - 100) / (1000 - 100)))
    assert math.isclose(lr, expected_lr, rel_tol=1e-5)

    scheduler.step(501)
    lr = scheduler.get_lr(param_group)
    expected_lr = 0.001
    assert math.isclose(lr, expected_lr, rel_tol=1e-5)


def test_get_lr_cosine(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='cosine',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    scheduler.step(500)
    param_group = {'max_lr': 0.1, 'min_lr': 0.001}
    lr = scheduler.get_lr(param_group)
    expected_lr = 0.001 + (0.1 - 0.001) * 0.5 * (
        math.cos(math.pi * ((500 - 100) / (1000 - 100))) + 1.0
    )
    assert math.isclose(lr, expected_lr, rel_tol=1e-5)


def test_step_function(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    scheduler.step(100)
    assert scheduler.num_steps == 100
    param_group = mock_optimizer.param_groups[0]
    assert math.isclose(param_group['lr'], 0.01 + (0.1 - 0.01) * (100 / 100), rel_tol=1e-5)
    assert math.isclose(param_group['weight_decay'], 0.01, rel_tol=1e-5)


def test_step_updates_empty_param_groups():
    """Empty param groups (rank-alignment stubs) must still receive lr updates.

    get_canonical_lr_for_logging reads lr from default_config groups regardless
    of whether they hold parameters, so step() must not skip them.
    """
    optimizer = MagicMock()
    # lr and weight_decay are set by the scheduler's step() method
    optimizer.param_groups = [
        # Non-default group with its own max_lr override (lr will differ from the canonical schedule)
        {'params': [1, 2], "min_lr": 0.001, "max_lr": 0.2, "default_config": False},
        # Model parallelism may leave default_config groups empty on some ranks
        {'params': [], "wd_mult": 0.0, 'default_config': True},
    ]
    scheduler = OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    scheduler.step(100)
    non_empty, empty = optimizer.param_groups

    # Verify learning rates: at step 100 warmup is complete so lr == max_lr
    assert "lr" in non_empty, "non-empty param group must have an lr"
    assert "lr" in empty, "empty param group must have an lr"
    assert non_empty['lr'] == pytest.approx(0.2)  # warmup complete → this group's max_lr override
    assert empty['lr'] == pytest.approx(0.1)  # warmup complete → scheduler's default max_lr
    assert get_canonical_lr_for_logging(optimizer.param_groups) == pytest.approx(0.1)

    # Verify weight decay: linear from 0.0 to 0.1 over 1000 steps → base wd is 0.01 at step 100
    assert "weight_decay" in non_empty, "non-empty param group must have a weight decay"
    assert "weight_decay" in empty, "empty param group must have a weight decay"
    assert non_empty['weight_decay'] == pytest.approx(0.01)  # base wd, no wd_mult override
    assert empty['weight_decay'] == pytest.approx(0.0)  # base wd * wd_mult=0.0


def test_state_dict(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    state_dict = scheduler.state_dict()
    assert state_dict['max_lr'] == 0.1
    assert state_dict['lr_warmup_steps'] == 100
    assert state_dict['num_steps'] == 0
    assert state_dict['lr_decay_style'] == 'linear'
    assert state_dict['lr_decay_steps'] == 1000
    assert state_dict['min_lr'] == 0.001
    assert state_dict['start_wd'] == 0.0
    assert state_dict['end_wd'] == 0.1
    assert state_dict['wd_incr_style'] == 'linear'
    assert state_dict['wd_incr_steps'] == 1000


def test_load_state_dict(mock_optimizer):
    scheduler = OptimizerParamScheduler(
        optimizer=mock_optimizer,
        init_lr=0.01,
        max_lr=0.1,
        min_lr=0.001,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style='linear',
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=1000,
        wd_incr_style='linear',
    )

    state_dict = {
        'max_lr': 0.2,
        'min_lr': 0.0005,
        'lr_warmup_steps': 200,
        'lr_decay_steps': 2000,
        'lr_decay_style': 'cosine',
        'num_steps': 500,
        'start_wd': 0.01,
        'end_wd': 0.2,
        'wd_incr_steps': 500,
        'wd_incr_style': 'cosine',
    }

    scheduler.load_state_dict(state_dict)
    assert scheduler.max_lr == 0.2
    assert scheduler.min_lr == 0.0005
    assert scheduler.lr_warmup_steps == 200
    assert scheduler.lr_decay_steps == 2000
    assert scheduler.lr_decay_style == 'cosine'
    assert scheduler.num_steps == 500
    assert scheduler.start_wd == 0.01
    assert scheduler.end_wd == 0.2
    assert scheduler.wd_incr_steps == 500
    assert scheduler.wd_incr_style == 'cosine'


# ── get_canonical_lr_for_logging tests ──────────────────────────────────────
#
# Returns the lr of the first default_config=True param group.  In practice
# the scheduler always sets a valid lr on every group (including empty
# rank-alignment stubs), so a default_config=True group with a float lr is
# always present.


class TestGetCanonicalLrForLogging:
    """Tests for get_canonical_lr_for_logging."""

    def test_single_default_config_group(self):
        """Typical case: one default_config group with a valid lr."""
        param_groups = [{'lr': 0.05, 'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) == 0.05

    def test_default_config_with_non_default_groups(self):
        """default_config group is returned even when non-default groups are present."""
        param_groups = [{'lr': 0.001, 'default_config': True}, {'lr': 0.999}]
        assert get_canonical_lr_for_logging(param_groups) == 0.001

    def test_default_config_after_non_default(self):
        """default_config group is found even when it is not first in the list."""
        param_groups = [{'lr': 0.50}, {'lr': 0.01, 'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) == 0.01

    def test_no_default_config_groups(self):
        """Returns None when no group has default_config=True."""
        param_groups = [{'lr': 0.50}, {'lr': 0.01}]
        assert get_canonical_lr_for_logging(param_groups) is None

    def test_missing_lr_key(self):
        """Returns None (not KeyError) when the default_config group lacks an 'lr' key."""
        param_groups = [{'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) is None

    def test_empty_param_groups(self):
        """Returns None when there are no param groups at all."""
        assert get_canonical_lr_for_logging([]) is None

    def test_no_default_config_no_lr(self):
        """Returns None when groups exist but none are default_config."""
        param_groups = [{'params': []}]
        assert get_canonical_lr_for_logging(param_groups) is None

    def test_lr_zero_is_valid(self):
        """lr=0.0 is a legitimate value, not to be confused with None."""
        param_groups = [{'lr': 0.0, 'default_config': True}]
        assert get_canonical_lr_for_logging(param_groups) == 0.0


# ---------------------------------------------------------------------------
# log-bridge
# ---------------------------------------------------------------------------


def _reference_log_bridge(
    high_lr,
    low_lr,
    final_lr,
    total_steps,
    warmup_steps,
    switch_step,
    log_start=0.70,
    rho_ref=0.05,
    budget="remaining",
):
    """The original standalone implementation the scheduler is checked against."""
    T, W, f = total_steps, warmup_steps, final_lr
    ts = int(round(log_start * T))
    assert W <= switch_step < ts < T

    def linear(peak):
        return lambda t: peak * t / W if t < W else peak - (peak - f) * (t - W) / (T - W)

    def area(fn, a, b):
        pts = sorted({a, b} | ({W} if a < W < b else set()))
        return sum((fn(x) + fn(y)) / 2 * (y - x) for x, y in zip(pts, pts[1:]))

    shape = lambda rho, x: 1 - math.log1p(x / rho) / math.log1p(1 / rho)
    mean = lambda rho: 1 / math.log1p(1 / rho) - rho
    G = lambda rho, x: (rho + x) * math.log1p(x / rho) - x

    orig, low = linear(high_lr), linear(low_lr)
    S, D, M = area(low, 0, T), T - W, mean(rho_ref)
    peak_ref = (S - D * f * (1 - M)) / (W / 2 + D * M)
    ref = lambda t: peak_ref * t / W if t < W else f + (peak_ref - f) * shape(rho_ref, (t - W) / D)

    x1 = (ts - W) / D
    tail = f * (T - ts) + (peak_ref - f) * D * (
        (1 - x1) - (G(rho_ref, 1) - G(rho_ref, x1)) / math.log1p(1 / rho_ref)
    )
    lr0, lr_ts, L = orig(switch_step), ref(ts), ts - switch_step
    target = (
        area(low, switch_step, T) if budget == "remaining" else S - area(orig, 0, switch_step)
    )
    need = ((target - tail) / L - lr_ts) / (lr0 - lr_ts)
    if not 0 < need < 0.5:
        raise ValueError("infeasible")
    lo, hi = math.log(1e-8), math.log(1e4)
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean(math.exp(mid)) < need else (lo, mid)
    rho_b = math.exp((lo + hi) / 2)

    def lr(step):
        if step < switch_step:
            return orig(step)
        if step < ts:
            return lr_ts + (lr0 - lr_ts) * shape(rho_b, (step - switch_step) / L)
        return ref(min(step, T))

    return lr


# Two param groups with different bounds (e.g. scalar LR and matrix LR), absolute min_lr floor.
_BRIDGE_GROUPS = [(2.83e-3, 5.1e-5), (1.41e-2, 5.1e-5)]
_BRIDGE_KW = dict(T=100_000, W=1_000, switch=20_000, factor=0.5)


def _make_bridge_scheduler(budget="remaining", style="log-bridge", **overrides):
    kw = {**_BRIDGE_KW, **overrides}
    optimizer = MagicMock()
    optimizer.param_groups = [
        {'lr': 0.0, 'weight_decay': 0.0, 'max_lr': hi, 'min_lr': lo} for hi, lo in _BRIDGE_GROUPS
    ]
    return OptimizerParamScheduler(
        optimizer=optimizer,
        init_lr=0.0,
        max_lr=_BRIDGE_GROUPS[0][0],
        min_lr=_BRIDGE_GROUPS[0][1],
        lr_warmup_steps=kw['W'],
        lr_decay_steps=kw['T'],
        lr_decay_style=style,
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=kw['T'],
        wd_incr_style='constant',
        lr_bridge_low_factor=kw['factor'],
        lr_bridge_switch_steps=kw['switch'],
        lr_bridge_budget=budget,
        # As in training (args default): checkpoint scheduler values must match the args.
        use_checkpoint_opt_param_scheduler=False,
    )


# 'total' must pay back the pre-switch excess, so it needs an early switch to stay feasible.
_BUDGET_SWITCH = {"remaining": 20_000, "total": 5_000}


@pytest.mark.parametrize("budget", ["remaining", "total"])
def test_log_bridge_matches_reference(budget):
    T, W, factor = (_BRIDGE_KW[k] for k in ('T', 'W', 'factor'))
    switch = _BUDGET_SWITCH[budget]
    scheduler = _make_bridge_scheduler(budget=budget, switch=switch)
    refs = [
        _reference_log_bridge(hi, factor * hi, lo, T, W, switch, budget=budget)
        for hi, lo in _BRIDGE_GROUPS
    ]
    steps = [0, 1, 500, W, W + 1, 10_000, switch - 1, switch, switch + 1, 50_000, 69_999, 70_000,
             70_001, 90_000, T - 1, T, T + 1, 2 * T]
    for step in steps:
        scheduler.num_steps = step
        for group, ref in zip(scheduler.optimizer.param_groups, refs):
            assert scheduler.get_lr(group) == pytest.approx(ref(step), rel=1e-12, abs=1e-15)


def test_log_bridge_is_linear_before_switch_and_continuous():
    T, W, switch = _BRIDGE_KW['T'], _BRIDGE_KW['W'], _BRIDGE_KW['switch']
    bridge = _make_bridge_scheduler()
    linear = _make_bridge_scheduler(style='linear')
    for step in range(0, switch, 997):
        bridge.num_steps = linear.num_steps = step
        for g in bridge.optimizer.param_groups:
            assert bridge.get_lr(g) == pytest.approx(linear.get_lr(g), rel=1e-12)
    # Continuous at the switch and at the log start; ends at min_lr.
    for g in bridge.optimizer.param_groups:
        for knot in (switch, int(round(0.7 * T))):
            bridge.num_steps = knot - 1
            before = bridge.get_lr(g)
            bridge.num_steps = knot
            assert bridge.get_lr(g) == pytest.approx(before, rel=1e-3)
        bridge.num_steps = T
        assert bridge.get_lr(g) == pytest.approx(g['min_lr'], rel=1e-9)


@pytest.mark.parametrize("budget", ["remaining", "total"])
def test_log_bridge_area_budget(budget):
    T, W, factor = (_BRIDGE_KW[k] for k in ('T', 'W', 'factor'))
    switch = _BUDGET_SWITCH[budget]
    scheduler = _make_bridge_scheduler(budget=budget, switch=switch)
    for hi, lo in _BRIDGE_GROUPS:
        group = {'max_lr': hi, 'min_lr': lo}
        low = lambda t: factor * hi * t / W if t < W else (
            factor * hi - (factor * hi - lo) * (t - W) / (T - W)
        )

        def lr_at(t):
            scheduler.num_steps = t
            return scheduler.get_lr(group)

        start = switch if budget == "remaining" else 0
        got = sum(lr_at(t) for t in range(start, T))  # left Riemann sum, step = 1
        want = sum(low(t) for t in range(start, T))
        assert got == pytest.approx(want, rel=1e-3)


def test_log_bridge_infeasible_raises():
    # Low peak barely below the original: the bridge would need to stay above linear.
    with pytest.raises(ValueError, match="infeasible"):
        _make_bridge_scheduler(factor=0.999)
    # 'total' with a late switch needs a sharper-than-reachable drop (would overshoot the budget).
    with pytest.raises(ValueError, match="near-step drop"):
        _make_bridge_scheduler(budget="total", switch=20_000)


def test_log_bridge_resumes_linear_checkpoint():
    linear = _make_bridge_scheduler(style='linear')
    linear.step(15_000)
    sd = linear.state_dict()

    bridge = _make_bridge_scheduler()  # no override flag needed for linear -> log-bridge
    bridge.load_state_dict(sd)
    assert bridge.lr_decay_style == 'log-bridge'
    assert bridge.num_steps == 15_000
    for g_lin, g_br in zip(linear.optimizer.param_groups, bridge.optimizer.param_groups):
        assert g_br['lr'] == pytest.approx(g_lin['lr'], rel=1e-12)

    # Round-trip of a log-bridge checkpoint keeps the style and knobs.
    bridge.step(30_000)
    sd2 = bridge.state_dict()
    assert sd2['lr_bridge_switch_steps'] == _BRIDGE_KW['switch']
    resumed = _make_bridge_scheduler()
    resumed.load_state_dict(sd2)
    for g_a, g_b in zip(bridge.optimizer.param_groups, resumed.optimizer.param_groups):
        assert g_b['lr'] == pytest.approx(g_a['lr'], rel=1e-12)


# ---------------------------------------------------------------------------------------------
# air-bridge
# ---------------------------------------------------------------------------------------------


def _reference_air(step, *, switch_step, total_steps, start_lr, final_lr, alpha_frac, orig):
    """The AIR (affine 1/sqrt) decay as specified, with the pre-switch schedule passed in."""
    if step < switch_step:
        return orig(step)
    K = total_steps - switch_step
    alpha = alpha_frac * K
    k = min(step - switch_step, K)
    q = (1 + k / alpha) ** -0.5
    qK = (1 + K / alpha) ** -0.5
    return final_lr + (start_lr - final_lr) * (q - qK) / (1 - qK)


@pytest.mark.parametrize("budget", ["remaining", "total"])
def test_air_bridge_matches_reference_and_budget(budget):
    T, W, factor = (_BRIDGE_KW[k] for k in ('T', 'W', 'factor'))
    switch = _BUDGET_SWITCH[budget]
    scheduler = _make_bridge_scheduler(budget=budget, style='sqrt-bridge', switch=switch)
    linear = _make_bridge_scheduler(style='linear')
    for g, (hi, lo) in zip(scheduler.optimizer.param_groups, _BRIDGE_GROUPS):
        scheduler.num_steps = 0
        scheduler.get_lr(g)  # solve
        fn = scheduler._log_bridge_cache[(hi, lo)]
        orig = lambda t, g=g: (setattr(linear, 'num_steps', t), linear.get_lr(g))[1]
        ref = lambda t: _reference_air(
            t, switch_step=switch, total_steps=T, start_lr=orig(switch), final_lr=lo,
            alpha_frac=fn.info['alpha_frac'], orig=orig,
        )
        for step in [0, 500, W, 3_000, switch - 1, switch, switch + 1, 50_000, T - 1, T, 2 * T]:
            scheduler.num_steps = step
            assert scheduler.get_lr(g) == pytest.approx(ref(step), rel=1e-12, abs=1e-15)
        scheduler.num_steps = T
        assert scheduler.get_lr(g) == pytest.approx(lo, rel=1e-9)

        low = lambda t: factor * hi * t / W if t < W else (
            factor * hi - (factor * hi - lo) * (t - W) / (T - W)
        )
        start = switch if budget == "remaining" else 0

        def lr_at(t):
            scheduler.num_steps = t
            return scheduler.get_lr(g)

        got = sum(lr_at(t) for t in range(start, T))
        want = sum(low(t) for t in range(start, T))
        assert got == pytest.approx(want, rel=1e-3)


def test_air_bridge_infeasible_raises():
    with pytest.raises(ValueError, match="infeasible"):
        # Unlike log-bridge (fixed log tail), AIR can approach linear, so only a low peak above
        # the actual peak (bridge would have to sit above linear) is infeasible.
        _make_bridge_scheduler(style="sqrt-bridge", factor=1.2)


def test_air_bridge_resumes_log_bridge_checkpoint():
    # A log-bridge checkpoint from before the switch resumes into air-bridge without override.
    log = _make_bridge_scheduler()
    log.step(15_000)
    air = _make_bridge_scheduler(style='sqrt-bridge')
    air.load_state_dict(log.state_dict())
    assert air.lr_decay_style == 'sqrt-bridge'
    assert air.num_steps == 15_000
    for g_log, g_air in zip(log.optimizer.param_groups, air.optimizer.param_groups):
        assert g_air['lr'] == pytest.approx(g_log['lr'], rel=1e-12)
    air.step(30_000)
    resumed = _make_bridge_scheduler(style='sqrt-bridge')
    resumed.load_state_dict(air.state_dict())
    for g_a, g_b in zip(air.optimizer.param_groups, resumed.optimizer.param_groups):
        assert g_b['lr'] == pytest.approx(g_a['lr'], rel=1e-12)


def test_bridge_style_aliases():
    # The aliases map to 'log-bridge' / 'sqrt-bridge', both as args and in checkpoints.
    for old, new in (('log', 'log-bridge'), ('sqrt', 'sqrt-bridge'), ('air-bridge', 'sqrt-bridge')):
        sched = _make_bridge_scheduler(style=old)
        assert sched.lr_decay_style == new
        sched.step(25_000)
        sd = sched.state_dict()
        sd['lr_decay_style'] = old  # a checkpoint written under the old name
        resumed = _make_bridge_scheduler(style=new)
        resumed.load_state_dict(sd)
        assert resumed.lr_decay_style == new
        for g_a, g_b in zip(sched.optimizer.param_groups, resumed.optimizer.param_groups):
            assert g_b['lr'] == pytest.approx(g_a['lr'], rel=1e-12)
