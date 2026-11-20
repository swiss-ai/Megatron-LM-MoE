import numpy as np
import pytest
import torch

from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from tests.unit_tests.test_utilities import Utils


class _FakeTPGroup:
    """Single-rank group for the standalone numerical checks."""

    def rank(self):
        return 0

    def size(self):
        return 1


def test_vocab_parallel_cross_entropy():
    Utils.initialize_model_parallel(4, 2)
    vocab_parallel_logits = torch.range(0, 7).repeat(16, 4).cuda()
    target = torch.arange(0, 32, 2).cuda()
    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target)
    expected_output = torch.tensor(
        [
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
        ]
    ).cuda()
    assert torch.equal(torch.round(expected_output), torch.round(output))
    Utils.destroy_model_parallel()


def _reference_cross_entropy(logits: torch.Tensor, target: torch.Tensor):
    """fp64 loss and gradient written as softplus(logsumexp(others) - x_target).

    Unlike log_softmax, this form never adds the target's exp(0) = 1 to the tiny sum of the
    others, so it stays exact when the target dominates."""
    z = logits.detach().double().clone().requires_grad_(True)
    target_logit = z.gather(-1, target.unsqueeze(-1))
    others = (z - target_logit).masked_fill(
        torch.nn.functional.one_hot(target, z.size(-1)).bool(), float("-inf")
    )
    loss = torch.nn.functional.softplus(torch.logsumexp(others, dim=-1))
    loss.sum().backward()
    return loss.detach(), z.grad


def _confident_logits(margins, vocab_size, targets):
    """Rows where the target leads the logsumexp of the other (all-zero) logits by ``margin``."""
    logits = torch.zeros(len(margins), vocab_size, dtype=torch.float64)
    for row, (margin, t) in enumerate(zip(margins, targets)):
        logits[row, t] = margin + np.log(vocab_size - 1)
    return logits


def test_vocab_parallel_cross_entropy_confident_target(monkeypatch):
    """Once the target leads by more than ~16.6 in fp32, log(sum_exp) - x and softmax - 1 used
    to round to exactly 0 while the other classes kept their gradients
    (https://arxiv.org/abs/2610.31642). The loss and the target gradient must stay accurate,
    and the row gradient must keep summing to (nearly) zero."""
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None, group=None: tensor)
    margins = [5.0, 10.0, 17.0, 20.0, 40.0, 80.0]
    vocab_size = 1024
    target = torch.tensor([3, 700, 1023, 0, 512, 9])
    logits = _confident_logits(margins, vocab_size, target.tolist())
    expected_loss, expected_grad = _reference_cross_entropy(logits, target)

    vocab_parallel_logits = logits.float().requires_grad_(True)
    loss = vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=_FakeTPGroup())
    torch.testing.assert_close(loss.double(), expected_loss, rtol=1e-5, atol=0)

    loss.sum().backward()
    grad = vocab_parallel_logits.grad.double()
    rows = torch.arange(len(margins))
    torch.testing.assert_close(grad[rows, target], expected_grad[rows, target], rtol=1e-5, atol=0)
    # Used to be +exp(-margin): the target's share was rounded away.
    assert (grad.sum(dim=-1).abs() <= 1e-4 * torch.exp(-torch.tensor(margins).double())).all()


def test_vocab_parallel_cross_entropy_label_smoothing_matches_reference(monkeypatch):
    """With label smoothing the target gradient is now written as target_grad + smoothing;
    it must keep matching the smoothed loss of the forward, differentiated in fp64."""
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None, group=None: tensor)
    torch.manual_seed(0)
    label_smoothing, vocab_size = 0.1, 512
    logits = torch.randn(64, vocab_size) * 4
    target = torch.randint(0, vocab_size, (64,))

    z = logits.double().requires_grad_(True)
    smoothing = label_smoothing * vocab_size / (vocab_size - 1)
    log_probs = torch.log_softmax(z, dim=-1)
    expected_loss = (1.0 - smoothing) * -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(
        -1
    ) - smoothing * log_probs.mean(dim=-1)
    expected_loss.sum().backward()

    vocab_parallel_logits = logits.clone().requires_grad_(True)
    loss = vocab_parallel_cross_entropy(
        vocab_parallel_logits, target, label_smoothing, _FakeTPGroup()
    )
    loss.sum().backward()
    torch.testing.assert_close(loss.double(), expected_loss.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(vocab_parallel_logits.grad.double(), z.grad, rtol=1e-4, atol=1e-7)


def test_vocab_parallel_cross_entropy_matches_torch_on_random_logits(monkeypatch):
    """Ordinary rows keep matching torch.nn.functional.cross_entropy."""
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None, group=None: tensor)
    torch.manual_seed(0)
    logits = torch.randn(64, 512) * 4
    target = torch.randint(0, 512, (64,))
    vocab_parallel_logits = logits.clone().requires_grad_(True)
    loss = vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=_FakeTPGroup())
    loss.sum().backward()
    expected_loss, expected_grad = _reference_cross_entropy(logits, target)
    torch.testing.assert_close(loss.double(), expected_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        vocab_parallel_logits.grad.double(), expected_grad, rtol=1e-4, atol=1e-7
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPUs")
def test_vocab_parallel_cross_entropy_confident_target_across_ranks():
    """Same check with the vocabulary split over every rank: the target, and the largest
    other logit, sit on different ranks, so the target-free sum is reduced across ranks."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    tp_size = Utils.world_size
    rank = torch.distributed.get_rank()
    partition = 256
    vocab_size = partition * tp_size
    margins = [5.0, 17.0, 20.0, 40.0]
    targets = [(k * partition + 7) % vocab_size for k in range(1, len(margins) + 1)]
    logits = _confident_logits(margins, vocab_size, targets)
    # A row whose target is not the largest logit, with the largest on another rank.
    logits = torch.cat((logits, torch.zeros(1, vocab_size, dtype=torch.float64)))
    logits[-1, 5] = 30.0
    targets.append(vocab_size - 1)
    target = torch.tensor(targets)
    expected_loss, expected_grad = _reference_cross_entropy(logits, target)

    local = logits[:, rank * partition : (rank + 1) * partition].float().cuda().requires_grad_(True)
    loss = vocab_parallel_cross_entropy(local.clone(), target.cuda())
    torch.testing.assert_close(loss.double().cpu(), expected_loss, rtol=1e-5, atol=0)

    local_for_grad = local.detach().clone().requires_grad_(True)
    vocab_parallel_cross_entropy(local_for_grad, target.cuda()).sum().backward()
    expected_local_grad = expected_grad[:, rank * partition : (rank + 1) * partition]
    torch.testing.assert_close(
        local_for_grad.grad.double().cpu(), expected_local_grad, rtol=1e-4, atol=1e-30
    )
    Utils.destroy_model_parallel()
