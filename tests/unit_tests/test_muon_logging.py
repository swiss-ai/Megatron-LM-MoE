# Copyright (c) 2026, EPFL / Swiss AI Initiative.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.muon_logging import collect_muon_stats
from megatron.core.optimizer.muon import TensorParallelMuon


@pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers is not installed"
)
def test_collect_muon_stats_logs_layernorm_sparsity_and_param_rms():
    param = torch.nn.Parameter(torch.tensor([[0.0, 1.0], [1e-21, -1.0]]))
    param.md_gain_log_family = "attention-in"
    param.md_gain_log_layer = 2
    optimizer = TensorParallelMuon([param])
    optimizer.state[param]["momentum_buffer"] = torch.zeros_like(param)
    layernorm = torch.nn.Parameter(torch.zeros(2))
    layernorm.md_gain_log_family = "layernorm"
    layernorm.md_gain_log_layer = 2
    layernorm.md_layernorm_gain_offset = 1.0
    scalar_optimizer = torch.optim.Adam([layernorm])
    optimizer = SimpleNamespace(
        chained_optimizers=(
            SimpleNamespace(optimizer=optimizer),
            SimpleNamespace(optimizer=scalar_optimizer),
        )
    )

    stats = collect_muon_stats(optimizer, per_layer=True)

    assert stats[
        "muon/sparsity/attention-in/fraction-below-1e-20"
    ] == pytest.approx(0.5)
    assert stats["muon/params/attention-in/rms"] == pytest.approx(0.5**0.5)
    assert stats[
        "muon/layers/2/sparsity/attention-in/fraction-below-1e-20"
    ] == pytest.approx(0.5)
    assert stats["muon/layers/2/params/attention-in/rms"] == pytest.approx(0.5**0.5)
    assert stats["muon/gains/layernorm/flat/mean"] == pytest.approx(1.0)
    assert stats["muon/gains/layernorm/flat/effective-rms"] == pytest.approx(1.0)
