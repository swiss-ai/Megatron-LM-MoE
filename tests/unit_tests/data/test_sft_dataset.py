from types import SimpleNamespace

import numpy as np
import torch

from megatron.training.datasets.sft_dataset import IGNORE_INDEX, SFTDataset


class _Tokenizer:
    eod = 2
    pad = 0

    def tokenize_conversation(self, conversation, return_target, add_generation_prompt):
        assert return_target and not add_generation_prompt
        return torch.tensor([10, 11, 12]), torch.tensor([10, IGNORE_INDEX, 12])


def test_padding_labels_use_ignore_index():
    dataset = object.__new__(SFTDataset)
    dataset.config = SimpleNamespace(
        tokenizer=_Tokenizer(),
        sequence_length=5,
        context_parallel_size=1,
        reset_position_ids=False,
        reset_attention_mask=False,
        create_attention_mask=False,
    )
    dataset.dataset = [[{"role": "system", "content": "test"}]]
    dataset.indices = np.array([0])

    sample = dataset[0]

    assert torch.equal(sample["labels"], torch.tensor([IGNORE_INDEX, 12, -100, -100, -100]))
    assert torch.equal(sample["loss_mask"], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
