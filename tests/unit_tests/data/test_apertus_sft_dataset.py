"""CPU coverage using genuine Megatron indexed files and packed batch utilities.

Run without the GPU conftest: pytest --confcutdir=tests/unit_tests/data <this file>.
The dataset is loaded directly to avoid training-package GPU initialization.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
from megatron.core.datasets.utils import Split
from megatron.core.utils import flatten_batch_for_packed_sequences

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "apertus_sft_cpu", ROOT / "megatron/training/datasets/apertus_sft_dataset.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ApertusSFTDataset = module.ApertusSFTDataset
ApertusSFTLowLevelDataset = module.ApertusSFTLowLevelDataset


class Tokenizer:
    vocab_size = 128
    pad = 0
    unique_identifiers = {"fixture": "apertus2"}

    def __init__(self):
        self.tokenizer = SimpleNamespace(
            init_kwargs={},
            all_special_ids=[90, 91, 92],
            output_start_token_id=90,
            output_end_token_id=91,
            wait_token_id=92,
        )


def write_index(prefix, documents, dtype, grouped=False):
    builder = IndexedDatasetBuilder(str(prefix) + ".bin", dtype=dtype)
    if grouped:
        builder.add_document(np.concatenate(documents), list(map(len, documents)))
    else:
        for row in documents:
            builder.add_document(np.asarray(row), [len(row)])
    builder.finalize(str(prefix) + ".idx")


def make_dataset(tmp_path, rows, weights=None, *, requested=None, **kwargs):
    prefix = tmp_path / "tokens"
    write_index(prefix, rows, np.int32)
    if weights is not None:
        write_index(tmp_path / "loss_weights", weights, np.float64)
    config = GPTDatasetConfig(
        random_seed=123,
        sequence_length=kwargs.pop("sequence_length", 16),
        tokenizer=kwargs.pop("tokenizer", Tokenizer()),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        path_to_cache=str(tmp_path / "cache"),
        sft_load_loss_mask=weights is not None,
        **kwargs,
    )
    low = ApertusSFTLowLevelDataset(str(prefix), config)
    return ApertusSFTDataset(
        low, str(prefix), np.arange(len(rows), dtype=np.int32), requested, Split.train, config
    )


def test_authoritative_weights_shift_and_real_pad_token(tmp_path):
    # Positive input weights and a suppressed assistant turn must survive unchanged.
    data = make_dataset(tmp_path, [[1, 0, 90, 5, 91, 92]], [[0, 0.25, 0, 0, 2, 0.5]])
    row = data[0]
    assert row["tokens"][:6].tolist() == [1, 0, 90, 5, 91, 92]
    assert row["labels"][:6].tolist() == [0, 90, 5, 91, 92, 0]
    assert row["loss_mask"][:6].tolist() == [0.25, 0, 0, 2, 0.5, 0]
    assert not row["padding_mask"][:6].any()
    assert row["padding_mask"][6:].all()
    assert row["loss_mask"][6:].sum() == 0


def test_stored_weights_need_no_apertus_controls(tmp_path):
    tokenizer = Tokenizer()
    tokenizer.tokenizer = None
    data = make_dataset(tmp_path, [[1, 2]], [[0, 1]], tokenizer=tokenizer)
    assert data[0]["loss_mask"][0] == 1


def test_assistant_mask_is_independent_of_stored_weights(tmp_path):
    data = make_dataset(
        tmp_path,
        [[1, 90, 5, 91, 92, 2, 90, 6, 91, 92]],
        [[0, 0.25, 0, 0, 0, 1, 2, 0, 0, 0]],
        sft_report_assistant_loss=True,
        context_parallel_size=2,
    )
    row = data[0]
    assert row['assistant_mask'][:10].tolist() == [
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert row['assistant_mask'].dtype == torch.bool
    assert row['assistant_mask'][row['padding_mask']].sum() == 0
    assert not row['assistant_mask'][9]  # The conversation's final prediction has no target.
    assert row['loss_mask'][1] == 0 and row['assistant_mask'][1]  # Suppressed assistant.
    assert row['loss_mask'][4] == 1 and not row['assistant_mask'][4]  # Weighted input.
    assert not data[None]['assistant_mask'].any()


@pytest.mark.parametrize('strategy', ['greedy', 'bfd'])
def test_assistant_mask_shift_pack_padding_and_collation(tmp_path, strategy):
    data = make_dataset(
        tmp_path,
        [[1, 90, 2, 91, 92], [3, 90, 4, 91, 92]],
        sft_report_assistant_loss=True,
        sft_pack_samples=True,
        sft_packing_strategy=strategy,
        context_parallel_size=2,
    )
    row = data[0]
    expected = [True, True, True, True, False, False, False, False] * 2
    assert row['assistant_mask'].tolist() == expected
    assert row['assistant_mask'][row['padding_mask']].sum() == 0
    batch = flatten_batch_for_packed_sequences(default_collate([row, row]))
    assert batch['assistant_mask'].shape == (1, 32)
    assert batch['assistant_mask'][0].tolist() == expected * 2


def test_assistant_reporting_requires_control_metadata(tmp_path):
    tokenizer = Tokenizer()
    tokenizer.tokenizer = None
    with pytest.raises(ValueError, match='registered'):
        make_dataset(
            tmp_path, [[1, 2]], [[0, 1]], tokenizer=tokenizer, sft_report_assistant_loss=True
        )


def test_scan_all_output_types_repeated_waits_and_truncation(tmp_path):
    # Header/payload IDs are deliberately arbitrary: every output frame is eligible.
    tokens = [1, 2, 90, 11, 4, 91, 92, 2, 90, 12, 5, 91, 92, 90, 13, 6, 91, 92]
    data = make_dataset(tmp_path, [tokens], sequence_length=16)
    assert data[0]["loss_mask"].tolist() == [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    assert data[0]["tokens"].tolist() == tokens[:16]


def test_control_ids_from_nested_artifact_metadata():
    glyphs = {
        "output_start_token": "<|out|>",
        "output_end_token": "<|/out|>",
        "wait_token": "<|wait|>",
    }
    hf = SimpleNamespace(
        init_kwargs=glyphs,
        added_tokens_decoder={i: SimpleNamespace(special=True) for i in (90, 91, 92)},
        convert_tokens_to_ids=lambda glyph: {"<|out|>": 90, "<|/out|>": 91, "<|wait|>": 92}[glyph],
    )
    assert ApertusSFTDataset._read_controls(
        SimpleNamespace(_tokenizer=SimpleNamespace(tokenizer=hf))
    ) == {"output_start_token_id": 90, "output_end_token_id": 91, "wait_token_id": 92}
    hf.init_kwargs["wait_token"] = "missing"
    hf.convert_tokens_to_ids = lambda _: 0
    with pytest.raises(ValueError, match="registered"):
        ApertusSFTDataset._read_controls(hf)


@pytest.mark.parametrize("strategy", ["greedy", "bfd"])
def test_packing_shifts_before_padding_and_resets_by_length(tmp_path, strategy):
    data = make_dataset(
        tmp_path,
        [[1, 90, 2], [3, 4, 92, 5, 6]],
        [[0, 1, 2], [0, 0.5, 1, 0, 3]],
        sft_pack_samples=True,
        sft_packing_strategy=strategy,
        context_parallel_size=2,
        sequence_parallel_size=2,
    )
    assert len(data) == 1
    row = data[0]
    boundaries = torch.unique_consecutive(row["cu_seqlens"]).tolist()
    assert boundaries[0] == 0 and boundaries[-1] == 16
    assert all((end - start) % 4 == 0 for start, end in zip(boundaries, boundaries[1:]))
    for pos, doc in enumerate(data.document_index):
        start = boundaries[pos]
        tokens, weights, _ = data._document(doc)
        length = len(tokens)
        np.testing.assert_array_equal(row["tokens"][start : start + length], tokens)
        np.testing.assert_array_equal(row["labels"][start : start + length - 1], tokens[1:])
        np.testing.assert_array_equal(row["loss_mask"][start : start + length - 1], weights[1:])
        assert row["loss_mask"][start + length - 1] == 0
        assert row["labels"][start + length - 1] == 0
        assert row["position_ids"][start] == 0
    assert row["loss_mask"][row["padding_mask"]].sum() == 0


@pytest.mark.parametrize("strategy", ["greedy", "bfd"])
def test_long_docs_truncated_before_alignment(tmp_path, strategy):
    data = make_dataset(
        tmp_path,
        [[1] * 37, [2] * 3],
        sft_pack_samples=True,
        sft_packing_strategy=strategy,
        context_parallel_size=2,
    )
    assert sorted(data._packing_lengths.tolist()) == [4, 16]
    assert sorted(data.document_index.tolist()) == [0, 1]
    assert len(data) == 2
    for row in data:
        assert row["cu_seqlens"][-1] == 16


def test_default_collation_handles_different_conversation_counts(tmp_path):
    data = make_dataset(tmp_path, [[1] * 4, [2] * 4, [3] * 15], sft_pack_samples=True)
    batch = flatten_batch_for_packed_sequences(default_collate([data[0], data[1]]))
    assert batch["tokens"].shape == (1, 32)
    assert batch["padding_mask"].shape == (1, 32)
    assert batch["cu_seqlens"].shape == (1, 4)
    assert batch["cu_seqlens"][0, -1] == 32
    assert torch.equal(batch["cu_seqlens"], batch["cu_seqlens_padded"])


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), 1e100, 1e-100])
def test_invalid_weights_fail_with_document_context(tmp_path, bad):
    data = make_dataset(tmp_path, [[1, 2]], [[0, bad]])
    with pytest.raises(ValueError, match="document 0"):
        data[0]


@pytest.mark.parametrize("problem", ["dtype", "length", "boundaries", "empty", "ambiguous"])
def test_invalid_indices_rejected(tmp_path, problem):
    rows = [[]] if problem == "empty" else [[1, 2], [3]]
    prefix = tmp_path / "tokens"
    write_index(prefix, rows, np.int32)
    write_index(
        tmp_path / "loss_weights",
        [[0]] if problem == "length" else rows,
        np.int32 if problem == "dtype" else np.float64,
        grouped=problem == "boundaries",
    )
    if problem == "ambiguous":
        write_index(tmp_path / "tokens.loss_weights", rows, np.float64)
    with pytest.raises(ValueError):
        ApertusSFTLowLevelDataset(
            str(prefix), SimpleNamespace(mmap_bin_files=True, sft_load_loss_mask=True)
        )


def test_cache_reload_epoch_shuffle_and_source_invalidation(tmp_path, monkeypatch):
    rows = [[i] * 3 for i in range(1, 9)]
    data = make_dataset(tmp_path, rows, requested=24)
    first_hash = data.unique_description_hash
    assert sorted(data.shuffle_index[:8]) == list(range(8))
    assert sorted(data.shuffle_index[8:16]) == list(range(8))
    assert not np.array_equal(data.shuffle_index[:8], data.shuffle_index[8:16])
    # Reopen the same files without rewriting them so source timestamps stay fixed.
    with monkeypatch.context() as patch:
        patch.setattr(module.np.random, "RandomState", lambda _: pytest.fail("cache miss"))
        reopened = ApertusSFTDataset(
            data.dataset, data.dataset_path, data.indices, 24, Split.train, data.config
        )
    np.testing.assert_array_equal(reopened.shuffle_index, data.shuffle_index)
    changed = make_dataset(tmp_path, rows, requested=24)
    assert changed.unique_description_hash != first_hash
    np.testing.assert_array_equal(changed.shuffle_index, data.shuffle_index)


def test_alignment_and_zero_loss_placeholder(tmp_path):
    with pytest.raises(ValueError, match="alignment"):
        make_dataset(tmp_path, [[1, 2]], sequence_length=15, context_parallel_size=2)
    data = make_dataset(tmp_path, [[1, 2]], [[0, 0]])
    assert data[0]["loss_mask"].sum() == 0
    empty = data[None]
    assert empty["padding_mask"].all()
    assert empty["loss_mask"].sum() == 0
    assert torch.unique_consecutive(empty["cu_seqlens"]).tolist() == [0, 16]


def test_real_apertus_encoding_matches_runtime_and_stored_masks(tmp_path):
    if sys.version_info < (3, 13):
        pytest.skip('apertus-common requires Python 3.13')
    common = pytest.importorskip('apertus_common')
    transformers = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    controls = ['<|in|>', '<|/in|>', '<|out|>', '<|/out|>', '<|hdr|>', '<|wait|>']
    specials = controls + ['<|pad|>', '<s>']
    vocab = {
        token: index
        for index, token in enumerate(
            specials + sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
        )
    }
    raw = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    raw.decoder = tokenizers.decoders.ByteLevel()
    raw.add_special_tokens(
        [tokenizers.AddedToken(value, normalized=False, special=True) for value in specials]
    )
    roles = dict(
        zip(
            [
                'input_start_token',
                'input_end_token',
                'output_start_token',
                'output_end_token',
                'header_end_token',
                'wait_token',
            ],
            controls,
        )
    )
    hf = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw,
        extra_special_tokens=roles,
        pad_token='<|pad|>',
        bos_token='<s>',
        clean_up_tokenization_spaces=False,
    )
    artifact = tmp_path / 'artifact'
    hf.save_pretrained(artifact)
    encoding = common.load_encoding(artifact)
    conversation = common.Conversation(
        system=common.SystemPrompt.build(),
        items=[
            common.User(payload='literal <|out|> is input'),
            common.Think(payload='reasoning'),
            common.Reply(payload='answer'),
            common.Wait(),
            common.User(payload='continue'),
            common.Call(name='lookup', counter=0, payload='{}'),
            common.Claim(payload='claim'),
            common.Wait(),
        ],
    )
    result = encoding.encode_conversation(conversation, return_loss_weights=True)
    wrapper = SimpleNamespace(
        tokenizer=hf,
        vocab_size=len(hf),
        pad=hf.pad_token_id,
        unique_identifiers={'artifact': str(artifact)},
    )
    sequence_length = (len(result.token_ids) + 3) // 4 * 4
    data = make_dataset(
        tmp_path,
        [result.token_ids],
        tokenizer=wrapper,
        sequence_length=sequence_length,
        context_parallel_size=2,
        sft_report_assistant_loss=True,
    )
    expected = np.asarray(result.loss_weights[1:] + [0], dtype=np.float32)
    np.testing.assert_array_equal(data[0]['loss_mask'][: len(expected)], expected)
    np.testing.assert_array_equal(data[0]['assistant_mask'][: len(expected)], expected > 0)
    # Preserve customized weights through the same real encoding and indexed path.
    customized = common.Conversation(
        system=common.SystemPrompt.build(),
        items=[
            common.Reply(payload='outdated', loss_weight=0),
            common.Wait(),
            common.User(payload='correction'),
            common.Reply(payload='revised', loss_weight=0.125),
            common.Claim(payload='weighted', loss_weight=2.5),
            common.Wait(),
        ],
    )
    result = encoding.encode_conversation(customized, return_loss_weights=True)
    data = make_dataset(
        tmp_path,
        [result.token_ids],
        [result.loss_weights],
        tokenizer=wrapper,
        sequence_length=(len(result.token_ids) + 3) // 4 * 4,
        sft_report_assistant_loss=True,
    )
    expected = np.asarray(result.loss_weights[1:] + [0], dtype=np.float32)
    np.testing.assert_array_equal(data[0]['loss_mask'][: len(expected)], expected)
    output_mask = data._output_mask(np.asarray(result.token_ids))[1:]
    np.testing.assert_array_equal(
        data[0]['assistant_mask'][: len(expected)], np.concatenate([output_mask, [False]])
    )
    assert data[0]['assistant_mask'].sum() > (data[0]['loss_mask'] > 0).sum()
