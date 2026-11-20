"""CPU tests of router z-loss, isolated from CUDA-only router imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def load_function(relative_path, name, namespace):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative_path, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("coeff", [None, 0.0, 1e-4])
@pytest.mark.parametrize("logging", [False, True])
def test_z_loss_logging_and_gradients(coeff, logging):
    records = []

    class Scaler:
        @staticmethod
        def apply(logits, loss):
            # Same forward identity; inject auxiliary loss into backward for this test.
            return logits + (loss - loss.detach()) / logits.numel()

    ns = dict(torch=torch, Optional=Optional, MoEAuxLossAutoScaler=Scaler,
              save_to_aux_losses_tracker=lambda *args: records.append(args))
    load_function("megatron/core/transformer/moe/moe_utils.py", "z_loss_func", ns)
    apply = load_function("megatron/core/transformer/moe/router.py", "apply_z_loss", ns)
    router = SimpleNamespace(
        config=SimpleNamespace(moe_z_loss_coeff=coeff, moe_router_log_z_loss=logging,
                               num_layers=2, mtp_num_layers=None),
        training=True, tp_cp_group=SimpleNamespace(size=lambda: 2),
        calculate_per_token_loss=False, is_mtp_layer=False, layer_number=1)
    logits = torch.tensor([[1., 2.], [100., 100.]], requires_grad=True)
    mask = torch.tensor([False, True])
    result = apply(router, logits, mask)
    result.sum().backward()
    penalty = coeff not in (None, 0.0)
    assert len(records) == int(logging or penalty)
    if records:
        torch.testing.assert_close(records[0][1].detach(), torch.logsumexp(logits[0].detach(), 0).square())
    if not penalty:
        assert result is logits
        torch.testing.assert_close(logits.grad, torch.ones_like(logits))
        if records:
            assert not records[0][1].requires_grad
    else:
        assert not torch.equal(logits.grad[0], torch.ones_like(logits.grad[0]))
    torch.testing.assert_close(logits.grad[1], torch.ones_like(logits.grad[1]))
    records.clear()
    with torch.no_grad():
        assert apply(router, logits, mask) is logits
    assert not records
