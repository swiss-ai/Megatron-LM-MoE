"""Tests for the one-shot fusion of the readers' column gains into their pre-norm gamma
(--md-fuse-prenorm-col-gains-iteration, md_decoupling.fuse_prenorm_col_gains)."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.optimizer.md_decoupling as md_module
from megatron.core.optimizer.md_decoupling import MDDecoupling
from megatron.core.optimizer.optimizer_config import OptimizerConfig

H, T = 6, 5
KEY = "L3:pre_mlp_layernorm.weight"


class _NoProcessGroups:
    tp = None
    expt_tp = None


class _Wrapper:
    def __init__(self, inner, calls):
        self.optimizer = inner
        self.is_stub_optimizer = False
        self._calls = calls

    def _copy_main_params_to_model_params(self):
        self._calls.append("copy")


def _sp(x):
    return torch.nn.functional.softplus(x)


def _setup(K, seed=0, frozen=False, rows=(4, 6, 8)):
    """K readers (MDDecoupling, rowcol softplus gains, Adam branch so both moments exist) of one
    gamma (AdamW), all in float64 so the checks can be tight."""
    g = torch.Generator().manual_seed(seed)
    readers = []
    for k in range(K):
        p = torch.nn.Parameter(torch.empty(rows[k], H, dtype=torch.float64))
        p.md_fuse_norm, p.md_fuse_reader = KEY, k
        readers.append(p)
    md = MDDecoupling(params=readers, lr=0.01, pg_collection=_NoProcessGroups(), tp_mode="duplicated",
                      hypersphere_mode="row", hypersphere_gains_mode="rowcol",
                      gain_parametrization="softplus", use_orthogonal_updates=False,
                      fused_col_gains_frozen=frozen)
    for p in readers:
        md._maybe_init_gain_state(p)
        st = md.state[p]
        st["row_gain"] = torch.randn(p.shape[0], generator=g, dtype=torch.float64)
        st["col_gain"] = torch.randn(H, generator=g, dtype=torch.float64) * 2
        for n in ("row_gain", "col_gain"):
            st[f"{n}_m"] = torch.zeros_like(st[n]); st[f"{n}_v"] = torch.zeros_like(st[n])
        bare = torch.randn(p.shape, generator=g, dtype=torch.float64)
        bare = bare / bare.norm(dim=1, keepdim=True) * (0.5 + torch.rand(p.shape[0], 1, generator=g, dtype=torch.float64))
        p.data.copy_(bare * _sp(st["row_gain"])[:, None] * _sp(st["col_gain"])[None, :])
    gamma = torch.nn.Parameter(torch.randn(H, generator=g, dtype=torch.float64))
    gamma.md_fuse_norm, gamma.md_fuse_num_readers = KEY, K
    adam = torch.optim.AdamW([gamma], lr=0.0)
    return md, adam, gamma, readers, g


def _loss_terms(gamma, readers_W, X, R):
    return sum((R[k] * (W @ (gamma[:, None] * X))).sum() for k, W in enumerate(readers_W))


def _grads_old(md, gamma, readers, X, R):
    """Autograd gradients of a fixed linear loss w.r.t. gamma, the raw row gains and the bare
    weights in the old parametrization."""
    gam = gamma.detach().clone().requires_grad_()
    rows, bares, Ws = [], [], []
    for p in readers:
        st = md.state[p]
        r = st["row_gain"].clone().requires_grad_()
        c = st["col_gain"].clone()
        bare = (p.detach() / (_sp(r.detach())[:, None] * _sp(c)[None, :])).clone().requires_grad_()
        rows.append(r); bares.append(bare)
        Ws.append(_sp(r)[:, None] * bare * _sp(c)[None, :])
    L = _loss_terms(gam, Ws, X, R)
    L.backward()
    return float(L), gam.grad, [r.grad for r in rows], [b.grad for b in bares]


def _grads_new(md, gamma, readers, X, R):
    """The same in the fused parametrization (column gains fixed at 1)."""
    gam = gamma.detach().clone().requires_grad_()
    rows, bares, Ws = [], [], []
    for p in readers:
        st = md.state[p]
        r = st["row_gain"].clone().requires_grad_()
        bare = (p.detach() / _sp(r.detach())[:, None]).clone().requires_grad_()
        rows.append(r); bares.append(bare)
        Ws.append(_sp(r)[:, None] * bare)
    L = _loss_terms(gam, Ws, X, R)
    L.backward()
    return float(L), gam.grad, [r.grad for r in rows], [b.grad for b in bares]


def _fuse(md, adam, revive="none", thr=0.03):
    calls = []
    chained = SimpleNamespace(chained_optimizers=[_Wrapper(md, calls), _Wrapper(adam, calls)],
                              allgather_params=lambda: calls.append("allgather"))
    summary = md_module.fuse_prenorm_col_gains(chained, revive, thr)
    assert calls == ["copy", "copy", "allgather"]
    return summary


@pytest.mark.parametrize("K", [1, 3])
def test_fusion_preserves_function_and_maps_optimizer_state(K):
    """Same loss after the fusion; every stored moment that equals the old gradient (or its square)
    equals the new gradient (or its square) afterwards: gamma Adam, row-gain Adam, bare-weight
    moments. Column gains end at phi = 1 with zero moments, bare rows keep their sphere radius."""
    md, adam, gamma, readers, g = _setup(K)
    X = torch.randn(H, T, generator=g, dtype=torch.float64)
    R = [torch.randn(p.shape[0], T, generator=g, dtype=torch.float64) for p in readers]
    L0, gg, gr, gb = _grads_old(md, gamma, readers, X, R)
    adam.state[gamma] = {"step": torch.tensor(7.0), "exp_avg": gg.clone(), "exp_avg_sq": gg ** 2}
    radii = []
    for k, p in enumerate(readers):
        st = md.state[p]
        st["row_gain_m"].copy_(gr[k]); st["row_gain_v"].copy_(gr[k] ** 2)
        st["exp_avg"] = gb[k].clone(); st["exp_avg_sq"] = gb[k] ** 2
        st["col_gain_m"].fill_(0.3); st["col_gain_v"].fill_(0.2)
        radii.append((p.detach() / (_sp(st["row_gain"])[:, None] * _sp(st["col_gain"])[None, :])).norm(dim=1))
    c_old = torch.stack([_sp(md.state[p]["col_gain"]) for p in readers])
    gamma_old = gamma.detach().clone()

    summary = _fuse(md, adam)

    assert summary["norms"] == 1 and summary["readers"] == K
    assert summary["max_rel_change_fp32"] < 1e-12 and summary["max_rel_radius_change"] < 1e-12
    cbar = torch.exp(torch.log(c_old).mean(0))
    torch.testing.assert_close(gamma.detach(), gamma_old * cbar)
    L1, ng, nr, nb = _grads_new(md, gamma, readers, X, R)
    assert abs(L1 - L0) < 1e-10 * max(1.0, abs(L0))
    st = adam.state[gamma]
    torch.testing.assert_close(st["exp_avg"], ng)
    torch.testing.assert_close(st["exp_avg_sq"], ng ** 2)
    for k, p in enumerate(readers):
        s = md.state[p]
        torch.testing.assert_close(s["row_gain_m"], nr[k])
        torch.testing.assert_close(s["row_gain_v"], nr[k] ** 2)
        torch.testing.assert_close(s["exp_avg"], nb[k])
        torch.testing.assert_close(s["exp_avg_sq"], nb[k] ** 2)
        torch.testing.assert_close(_sp(s["col_gain"]), torch.ones(H, dtype=torch.float64))
        assert s["col_gain_m"].abs().max() == 0 and s["col_gain_v"].abs().max() == 0
        torch.testing.assert_close((p.detach() / _sp(s["row_gain"])[:, None]).norm(dim=1), radii[k])
        if K == 1:  # c_bar = c: weights scale by 1/c, row gains and the weight moments are unchanged
            torch.testing.assert_close(nb[k], gb[k])
            torch.testing.assert_close(nr[k], gr[k])


def test_fused_col_gains_stay_frozen_and_are_not_pulled():
    """After the fusion an MDDecoupling step (pull on every gain) leaves the fused column gains at
    phi = 1; the row gains and untagged matrices still move."""
    md, adam, gamma, readers, g = _setup(3, frozen=True)
    _fuse(md, adam)
    other = torch.nn.Parameter(torch.randn(4, H, dtype=torch.float64))
    md.add_param_group({"params": [other]})
    for p in readers + [other]:
        p.md_rowcol_gain_pull_coeff = 0.5
        p.grad = torch.randn(p.shape, generator=g, dtype=torch.float64)
    md._maybe_init_gain_state(other)
    rows_before = [md.state[p]["row_gain"].clone() for p in readers]
    other_col = md.state[other]["col_gain"].clone() + 0.0
    md.state[other]["col_gain"].add_(0.7)
    md.step()
    one = torch.ones(H, dtype=torch.float64)
    for p, rb in zip(readers, rows_before):
        torch.testing.assert_close(_sp(md.state[p]["col_gain"]), one)
        assert not torch.equal(md.state[p]["row_gain"], rb)
    assert not torch.allclose(md.state[other]["col_gain"], other_col + 0.7)


def test_fusion_revive_mean():
    """revive='mean': useless channels (read amplitude below 3% of the median) get the mean of the
    live gamma', their bare column becomes the mean of the live columns in every reader, and their
    gamma moments and weight-moment columns are zeroed. Kept channels keep the exact function."""
    md, adam, gamma, readers, g = _setup(3, seed=1)
    with torch.no_grad():
        gamma[1] = 1e-9
        gamma[4] = -2e-9
    adam.state[gamma] = {"step": torch.tensor(3.0), "exp_avg": torch.full((H,), 0.1, dtype=torch.float64),
                         "exp_avg_sq": torch.full((H,), 0.01, dtype=torch.float64)}
    for p in readers:
        md.state[p]["exp_avg"] = torch.ones_like(p)
        md.state[p]["exp_avg_sq"] = torch.ones_like(p)
    useless = torch.zeros(H, dtype=torch.bool); useless[[1, 4]] = True

    summary = _fuse(md, adam, revive="mean")

    assert summary["revived"] == {KEY: 2} and summary["max_rel_change_fp32"] < 1e-12
    gnew = gamma.detach()
    torch.testing.assert_close(gnew[useless], gnew[~useless].mean().expand(2))
    st = adam.state[gamma]
    assert st["exp_avg"][useless].abs().max() == 0 and st["exp_avg"][~useless].abs().min() > 0
    for p in readers:
        s = md.state[p]
        # new bare weight: the revived columns are the mean of the live ones (the per-row
        # re-projection is a row scaling, so it keeps that relation)
        A = p.detach() / _sp(s["row_gain"])[:, None]
        live_mean = A[:, ~useless].mean(1)
        torch.testing.assert_close(A[:, 1], live_mean)
        torch.testing.assert_close(A[:, 4], live_mean)
        assert s["exp_avg"][:, useless].abs().max() == 0 and s["exp_avg"][:, ~useless].abs().min() > 0


def _tag_names(names):
    params = {}
    for n in names:
        p = torch.nn.Parameter(torch.ones(4, 6) if n.endswith(".weight") and "norm" not in n else torch.ones(6))
        layer = int(n.split(".layers.")[1].split(".")[0]) if ".layers." in n else None
        p.md_gain_log_layer = layer
        params[n] = p
    return params


def test_tag_fused_prenorm_readers():
    names = [
        "decoder.layers.0.self_attention.in_proj.weight", "decoder.layers.0.self_attention.in_proj.layer_norm_weight",
        "decoder.layers.0.mlp.linear_fc1.weight", "decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
        "decoder.layers.0.mlp.linear_fc2.weight",
        "decoder.layers.3.self_attention.linear_qkv.weight", "decoder.layers.3.self_attention.linear_qkv.layer_norm_weight",
        "decoder.layers.3.self_attention.linear_proj.weight", "decoder.layers.3.pre_mlp_layernorm.weight",
        "decoder.layers.3.mlp.router.weight", "decoder.layers.3.mlp.fc1_latent_proj.weight",
        "decoder.layers.3.mlp.fc2_latent_proj.weight", "decoder.layers.3.mlp.shared_experts.linear_fc1.weight",
        "decoder.layers.3.mlp.shared_experts.linear_fc2.weight", "decoder.layers.3.mlp.experts.linear_fc1.weight0",
        "decoder.layers.3.post_mlp_layernorm.weight", "output_layer.weight",
    ]
    params = _tag_names(names)
    assert md_module._tag_fused_prenorm_readers(params) == 4
    tags = {n: (getattr(p, "md_fuse_norm", None), getattr(p, "md_fuse_reader", None)) for n, p in params.items()}
    assert tags["decoder.layers.0.self_attention.in_proj.weight"] == ("L0:self_attention.in_proj.layer_norm_weight", 0)
    assert tags["decoder.layers.0.mlp.linear_fc1.weight"] == ("L0:mlp.linear_fc1.layer_norm_weight", 0)
    assert tags["decoder.layers.3.self_attention.linear_qkv.weight"] == ("L3:self_attention.linear_qkv.layer_norm_weight", 0)
    moe = {n: tags[n] for n in names if tags[n][0] == "L3:pre_mlp_layernorm.weight"}
    assert moe == {"decoder.layers.3.pre_mlp_layernorm.weight": ("L3:pre_mlp_layernorm.weight", None),
                   "decoder.layers.3.mlp.fc1_latent_proj.weight": ("L3:pre_mlp_layernorm.weight", 0),
                   "decoder.layers.3.mlp.router.weight": ("L3:pre_mlp_layernorm.weight", 1),
                   "decoder.layers.3.mlp.shared_experts.linear_fc1.weight": ("L3:pre_mlp_layernorm.weight", 2)}
    assert params["decoder.layers.3.pre_mlp_layernorm.weight"].md_fuse_num_readers == 3
    for n in ("decoder.layers.0.mlp.linear_fc2.weight", "decoder.layers.3.mlp.fc2_latent_proj.weight",
              "decoder.layers.3.mlp.experts.linear_fc1.weight0", "decoder.layers.3.post_mlp_layernorm.weight",
              "output_layer.weight", "decoder.layers.3.self_attention.linear_proj.weight"):
        assert tags[n] == (None, None), n


def test_tag_refuses_plain_moe():
    """Without a latent projection the routed experts read pre_mlp_layernorm directly."""
    params = _tag_names(["decoder.layers.2.pre_mlp_layernorm.weight", "decoder.layers.2.mlp.router.weight",
                         "decoder.layers.2.mlp.experts.linear_fc1.weight0"])
    with pytest.raises(AssertionError, match="routed experts read this norm"):
        md_module._tag_fused_prenorm_readers(params)


def test_fuse_arguments():
    import argparse

    from megatron.training.arguments import _add_training_args

    parser = _add_training_args(argparse.ArgumentParser(allow_abbrev=False))
    a = parser.parse_args([])
    assert a.md_fuse_prenorm_col_gains_iteration is None and a.md_fuse_revive == "none"
    assert a.md_fuse_revive_threshold == 0.03
    assert OptimizerConfig().md_fuse_prenorm_col_gains_iteration is None
    a = parser.parse_args(["--md-fuse-prenorm-col-gains-iteration", "4000", "--md-fuse-revive", "mean"])
    assert a.md_fuse_prenorm_col_gains_iteration == 4000 and a.md_fuse_revive == "mean"
    with pytest.raises(SystemExit):
        parser.parse_args(["--md-fuse-revive", "random"])
