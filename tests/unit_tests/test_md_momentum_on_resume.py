"""A changed --muon-momentum must take effect when a checkpoint is resumed."""
import torch

from megatron.core.optimizer.md_decoupling import MDDecoupling
from tests.unit_tests.test_md_decoupling import _NoProcessGroups


def _opt(momentum):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 8))
    return p, MDDecoupling(params=[p], lr=0.01, hypersphere_mode="row", use_orthogonal_updates=False,
                           momentum_beta=momentum, pg_collection=_NoProcessGroups())


def test_muon_momentum_from_current_config_after_load():
    _, old = _opt(0.95)
    state = old.state_dict()
    assert state["param_groups"][0]["momentum_beta"] == 0.95
    _, new = _opt(0.0)
    new.load_state_dict(state)
    assert new.param_groups[0]["momentum_beta"] == 0.0
    _, same = _opt(0.95)
    same.load_state_dict(state)
    assert same.param_groups[0]["momentum_beta"] == 0.95
