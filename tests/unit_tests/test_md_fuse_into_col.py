"""Tests for --md-fuse-prenorm-col-gains-mode into-col: the pre-norm gamma is folded into every reader's own
column gain (a direct, signed multiplier afterwards) and held at 1."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.optimizer.md_decoupling as md_module
from megatron.core.optimizer.optimizer import _decay_to_target
from tests.unit_tests.test_md_fuse_prenorm import H, KEY, T, _Wrapper, _setup, _sp


def _fuse_into_col(md, adam):
    calls = []
    chained = SimpleNamespace(chained_optimizers=[_Wrapper(md, calls), _Wrapper(adam, calls)],
                              allgather_params=lambda: calls.append("allgather"))
    for p in md.param_groups[0]["params"]:
        p.md_col_gain_direct = True
    for p in adam.param_groups[0]["params"]:
        p.md_fuse_norm_frozen = True
    summary = md_module.fuse_prenorm_col_gains(chained, "none", 0.03, mode="into-col")
    assert calls == ["copy", "copy", "allgather"]
    return summary


def _grads_old(md, gamma, readers, X, R):
    gam = gamma.detach().clone().requires_grad_()
    rows, cols, bares, L = [], [], [], 0.0
    for k, p in enumerate(readers):
        st = md.state[p]
        r = st["row_gain"].clone().requires_grad_(); c = st["col_gain"].clone().requires_grad_()
        bare = (p.detach() / (_sp(r.detach())[:, None] * _sp(c.detach())[None, :])).clone().requires_grad_()
        rows.append(r); cols.append(c); bares.append(bare)
        L = L + (R[k] * ((_sp(r)[:, None] * bare * _sp(c)[None, :]) @ (gam[:, None] * X))).sum()
    L.backward()
    return float(L), gam.grad, [r.grad for r in rows], [c.grad for c in cols], [b.grad for b in bares]


def _grads_new(md, readers, X, R):
    """gamma is gone (held at 1): W = diag(softplus(r)) bare diag(d), d direct."""
    rows, cols, bares, L = [], [], [], 0.0
    for k, p in enumerate(readers):
        st = md.state[p]
        r = st["row_gain"].clone().requires_grad_(); d = st["col_gain"].clone().requires_grad_()
        dd = d.detach(); safe = torch.where(dd == 0, torch.ones_like(dd), dd)
        bare = (p.detach() / (_sp(r.detach())[:, None] * safe[None, :])).clone().requires_grad_()
        rows.append(r); cols.append(d); bares.append(bare)
        L = L + (R[k] * ((_sp(r)[:, None] * bare * d[None, :]) @ X)).sum()
    L.backward()
    return float(L), [r.grad for r in rows], [c.grad for c in cols], [b.grad for b in bares]


@pytest.mark.parametrize("K", [1, 3])
def test_into_col_preserves_function_and_maps_state(K):
    """Same loss; gamma = 1 with zero moments; d = phi(c) * gamma (signed); the row-gain and bare-weight
    moments (set to the old gradients) equal the new gradients unchanged; the column-gain moments map to
    the gradient w.r.t. d (zero where gamma == 0)."""
    md, adam, gamma, readers, g = _setup(K, seed=3)
    with torch.no_grad():
        gamma[0] = -0.7; gamma[2] = 0.0; gamma[4] = 1e-9
    X = torch.randn(H, T, generator=g, dtype=torch.float64)
    R = [torch.randn(p.shape[0], T, generator=g, dtype=torch.float64) for p in readers]
    L0, gg, gr, gc, gb = _grads_old(md, gamma, readers, X, R)
    adam.state[gamma] = {"step": torch.tensor(5.0), "exp_avg": gg.clone(), "exp_avg_sq": gg ** 2}
    for k, p in enumerate(readers):
        st = md.state[p]
        st["row_gain_m"].copy_(gr[k]); st["row_gain_v"].copy_(gr[k] ** 2)
        st["col_gain_m"].copy_(gc[k]); st["col_gain_v"].copy_(gc[k] ** 2)
        st["exp_avg"] = gb[k].clone(); st["exp_avg_sq"] = gb[k] ** 2
    c_old = [_sp(md.state[p]["col_gain"]).clone() for p in readers]
    gamma_old = gamma.detach().clone()

    summary = _fuse_into_col(md, adam)

    assert summary["norms"] == 1 and summary["readers"] == K and summary["max_rel_change_fp32"] < 1e-12
    assert summary["zero_gamma_col_gains"] == K and summary["restarted_col_gain_moments"] == 0
    torch.testing.assert_close(gamma.detach(), torch.ones(H, dtype=torch.float64))
    assert adam.state[gamma]["exp_avg"].abs().max() == 0 and adam.state[gamma]["exp_avg_sq"].abs().max() == 0
    L1, nr, nc, nb = _grads_new(md, readers, X, R)
    assert abs(L1 - L0) < 1e-10 * max(1.0, abs(L0))
    live = gamma_old != 0
    for k, p in enumerate(readers):
        s = md.state[p]
        torch.testing.assert_close(s["col_gain"], c_old[k] * gamma_old)
        torch.testing.assert_close(s["row_gain_m"], nr[k])
        torch.testing.assert_close(s["exp_avg"], nb[k])
        torch.testing.assert_close(s["col_gain_m"][live], nc[k][live])
        torch.testing.assert_close(s["col_gain_v"][live], nc[k][live] ** 2)
        assert s["col_gain_m"][~live].abs().max() == 0


def test_into_col_restarts_moments_that_overflow():
    """A vanishing gamma makes 1 / (phi' gamma)^2 overflow: those column-gain moments restart at 0
    (never inf / nan), the others are mapped, and the stored bare column is still recovered."""
    md, adam, gamma, readers, g = _setup(1, seed=6)
    with torch.no_grad():
        gamma[1] = 1e-200
    st = md.state[readers[0]]
    st["col_gain_m"].fill_(0.5); st["col_gain_v"].fill_(0.25)
    summary = _fuse_into_col(md, adam)
    assert summary["restarted_col_gain_moments"] == 1 and summary["zero_gamma_col_gains"] == 0
    assert summary["max_rel_change_fp32"] < 1e-12
    assert st["col_gain_m"][1] == 0 and st["col_gain_v"][1] == 0
    assert torch.isfinite(st["col_gain_m"]).all() and torch.isfinite(st["col_gain_v"]).all()
    assert (st["col_gain_v"][[0, 2, 3, 4, 5]] > 0).all()


def test_direct_col_gain_round_trip_is_exact():
    """_preprocess_gains / _apply_gains invert each other exactly for signed and tiny direct gains, and
    the gain gradient has no softplus factor."""
    md, adam, gamma, readers, g = _setup(3, seed=4)
    with torch.no_grad():
        gamma[1] = -2.0; gamma[3] = 1e-12
    _fuse_into_col(md, adam)
    for p in readers:
        before = p.detach().clone()
        grad = torch.randn(p.shape, generator=g, dtype=torch.float64)
        p.grad = grad.clone()
        grads = md._preprocess_gains(p)
        bare = p.detach().clone()
        r = _sp(md.state[p]["row_gain"])
        # dL/dd_j = sum_i dL/dW[i, j] * r_i * bare[i, j], no softplus factor
        torch.testing.assert_close(grads["col_gain"], (grad * r[:, None] * bare).sum(0))
        md._apply_gains(p)
        torch.testing.assert_close(p.detach(), before, rtol=1e-12, atol=0)


def test_direct_col_gain_pull_keeps_sign():
    """With a pull, a direct column gain moves its magnitude toward 1 and keeps its sign; the softplus row
    gain is pulled toward softplus_inv(1)."""
    md, adam, gamma, readers, g = _setup(1, seed=5)
    with torch.no_grad():
        gamma.copy_(torch.tensor([-0.2, 0.3, -3.0, 2.5, 0.5, -0.5], dtype=torch.float64))
    _fuse_into_col(md, adam)
    p = readers[0]
    p.md_rowcol_gain_pull_coeff = 0.5
    d0 = md.state[p]["col_gain"].clone()
    md.param_groups[0]["lr"] = 0.1
    md.gains_lr = None
    for _ in range(5):
        p.grad = torch.zeros_like(p)
        md.step()
    d1 = md.state[p]["col_gain"]
    assert torch.equal(torch.sign(d1), torch.sign(d0))
    assert ((d1.abs() - 1).abs() < (d0.abs() - 1).abs()).all()


def test_frozen_norm_held_at_one_and_not_pulled():
    gamma = torch.nn.Parameter(torch.tensor([0.3, -2.0, 1e-9]))
    gamma.md_fuse_norm_frozen = True
    gamma.md_decay_to_target = (0.5, 1.0, 0.0)
    other = torch.nn.Parameter(torch.tensor([0.3, -2.0]))
    other.md_decay_to_target = (0.5, 1.0, 0.0)
    sgd = torch.optim.SGD([gamma, other], lr=0.1)
    _decay_to_target(sgd)
    torch.testing.assert_close(gamma.detach(), torch.ones(3))
    torch.testing.assert_close(other.detach(), torch.tensor([0.3, -2.0]) - 0.05 * (torch.tensor([0.3, -2.0]) - torch.tensor([1.0, -1.0])))


def test_tagging_into_col():
    params = {n: torch.nn.Parameter(torch.ones(4, 6) if "norm" not in n else torch.ones(6)) for n in (
        "decoder.layers.3.pre_mlp_layernorm.weight", "decoder.layers.3.mlp.router.weight",
        "decoder.layers.3.mlp.fc1_latent_proj.weight", "decoder.layers.3.mlp.shared_experts.linear_fc1.weight",
        "decoder.layers.3.mlp.experts.linear_fc1.weight0")}
    for p in params.values():
        p.md_gain_log_layer = 3
    md_module._tag_fused_prenorm_readers(params, into_col=True)
    assert params["decoder.layers.3.pre_mlp_layernorm.weight"].md_fuse_norm_frozen
    for n in ("router", "fc1_latent_proj", "shared_experts.linear_fc1"):
        assert params[f"decoder.layers.3.mlp.{n}.weight"].md_col_gain_direct
    assert not hasattr(params["decoder.layers.3.mlp.experts.linear_fc1.weight0"], "md_col_gain_direct")
    q = {n: torch.nn.Parameter(p.detach().clone()) for n, p in params.items()}
    for p in q.values():
        p.md_gain_log_layer = 3
    md_module._tag_fused_prenorm_readers(q)
    assert not any(hasattr(p, "md_col_gain_direct") or hasattr(p, "md_fuse_norm_frozen") for p in q.values())


def test_mode_argument():
    import argparse

    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.training.arguments import _add_training_args

    parser = _add_training_args(argparse.ArgumentParser(allow_abbrev=False))
    assert parser.parse_args([]).md_fuse_prenorm_col_gains_mode == "into-norm"
    assert OptimizerConfig().md_fuse_prenorm_col_gains_mode == "into-norm"
    assert parser.parse_args(["--md-fuse-prenorm-col-gains-mode", "into-col"]).md_fuse_prenorm_col_gains_mode == "into-col"
    with pytest.raises(SystemExit):
        parser.parse_args(["--md-fuse-prenorm-col-gains-mode", "both"])
