"""Length-delimited Apertus-2 SFT with stored or token-scanned loss weights."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from typing import Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, _build_sample_idx_bfd
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset

logger = logging.getLogger(__name__)


class ApertusSFTLowLevelDataset:
    """One indexed sequence per conversation, optionally paired with float weights."""

    def __init__(self, path: str, config: GPTDatasetConfig):
        self.path_prefix = path
        if os.stat(path + '.bin').st_size == 0:
            raise ValueError(f'empty token dataset: {path}')
        self.tokens = IndexedDataset(path, mmap=config.mmap_bin_files)
        if not np.issubdtype(self.tokens.index.dtype, np.integer):
            raise ValueError(f"tokens must have integer dtype: {path}")
        if (self.tokens.sequence_lengths <= 0).any():
            raise ValueError(f"empty conversations are not supported: {path}")
        if not np.array_equal(
            self.tokens.document_indices, np.arange(len(self.tokens) + 1)
        ):
            raise ValueError(f"expected one sequence per conversation/document: {path}")
        self.weights = None
        source_paths = [path]
        if config.sft_load_loss_mask:
            candidates = [path + ".loss_weights", path + ".loss_mask"]
            if os.path.basename(path) == "tokens":
                candidates += [
                    os.path.join(os.path.dirname(path), name)
                    for name in ("loss_weights", "loss_mask")
                ]
            matches = [p for p in candidates if os.path.exists(p + ".idx")]
            if len(matches) != 1:
                raise ValueError(
                    f"expected exactly one paired loss_weights/loss_mask index for {path}; "
                    f"found {matches}"
                )
            weight_path = matches[0]
            self.weights = IndexedDataset(weight_path, mmap=config.mmap_bin_files)
            if self.weights.index.dtype not in (np.float32, np.float64):
                raise ValueError(f"loss weights must be float32 or float64: {weight_path}")
            if (
                len(self.tokens) != len(self.weights)
                or not np.array_equal(self.tokens.sequence_lengths, self.weights.sequence_lengths)
                or not np.array_equal(self.tokens.document_indices, self.weights.document_indices)
            ):
                raise ValueError(f"unaligned token/weight indices: {path} / {weight_path}")
            source_paths.append(weight_path)
        self.source_identity = {
            os.path.realpath(prefix + suffix): {
                "size": os.stat(prefix + suffix).st_size,
                "mtime_ns": os.stat(prefix + suffix).st_mtime_ns,
            }
            for prefix in source_paths
            for suffix in (".idx", ".bin")
        }

    def __len__(self):
        return len(self.tokens)

    @property
    def sequence_lengths(self):
        return self.tokens.sequence_lengths

    def __getitem__(self, index):
        return self.tokens[index]

    def get(self, index: int, offset: int = 0, length: Optional[int] = None):
        tokens = self.tokens.get(index, offset, length)
        if self.weights is None:
            return tokens, None
        raw = self.weights.get(index, offset, length)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            weights = np.asarray(raw, dtype=np.float32)
        if (
            not np.isfinite(raw).all()
            or (raw < 0).any()
            or not np.isfinite(weights).all()
            or ((raw > 0) & (weights == 0)).any()
        ):
            raise ValueError(
                f"invalid or unrepresentable loss weights: {self.path_prefix}, document {index}"
            )
        return tokens, weights


def _greedy_samples(lengths, order, capacity):
    boundaries = [0]
    used = 0
    for position, document_id in enumerate(order):
        length = int(lengths[document_id])
        if used and used + length > capacity:
            boundaries.append(position)
            used = 0
        used += length
    if len(order):
        boundaries.append(len(order))
    return order, np.asarray(boundaries, dtype=np.int64)


class ApertusSFTDataset(MegatronDataset):
    """Pack complete retained conversations and shift each target independently."""

    @staticmethod
    def numel_low_level_dataset(low_level_dataset):
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path, config):
        return ApertusSFTLowLevelDataset(dataset_path, config)

    @staticmethod
    def _key_config_attributes():
        return [
            "random_seed", "sequence_length", "split", "split_matrix", "tokenizer",
            "sft_load_loss_mask", "sft_packing_strategy", "sft_pack_samples",
            "sft_truncate_right", "max_docs_per_bin_sft",
            "context_parallel_size", "sequence_parallel_size",
        ]

    def __init__(self, dataset, dataset_path, indices, num_samples, index_split, config):
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)
        self.unique_identifiers["sources"] = dataset.source_identity
        self.unique_identifiers["sft_index_version"] = 1
        self.unique_description = json.dumps(
            self.unique_identifiers, indent=4,
            default=lambda obj: obj.unique_identifiers,
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8")
        ).hexdigest()
        self._control_ids = (
            self._read_controls(config.tokenizer)
            if not config.sft_load_loss_mask or config.sft_report_assistant_loss else None
        )
        self._vocab_size = config.tokenizer.vocab_size
        cp = max(1, int(config.context_parallel_size or 1))
        sp = max(1, int(config.sequence_parallel_size or 1))
        self._segment_granularity = math.lcm(2 * cp if cp > 1 else 1, cp * sp)
        if config.sequence_length % self._segment_granularity:
            raise ValueError("SFT sequence length must be divisible by CP/SP segment alignment")
        if config.sft_packing_strategy not in ("greedy", "bfd"):
            raise ValueError("unknown SFT packing strategy")
        if config.max_docs_per_bin_sft < 0:
            raise ValueError("max_docs_per_bin_sft must be nonnegative")
        self._packing_lengths = np.minimum(dataset.sequence_lengths, config.sequence_length)
        self._packing_lengths = (
            (self._packing_lengths.astype(np.int64) + self._segment_granularity - 1)
            // self._segment_granularity * self._segment_granularity
        )
        self._build_indices()
        if len(self.document_index):
            retained = np.minimum(dataset.sequence_lengths[self.document_index], config.sequence_length)
            truncated = int(np.count_nonzero(
                dataset.sequence_lengths[self.document_index] > config.sequence_length
            ))
            physical = (len(self.sample_index) - 1) * config.sequence_length
            logger.info(
                "Apertus SFT %s: %d conversations, %d packs, %d truncated, %d padding "
                "tokens, %.2f%% utilization",
                index_split.name, len(self.document_index), len(self.sample_index) - 1,
                truncated, physical - int(retained.sum()),
                100 * float(retained.sum()) / physical,
            )

    def _build_indices(self):
        cache_root = self.config.path_to_cache or os.path.join(
            os.path.dirname(self.dataset_path), "cache", type(self).__name__
        )
        cache_prefix = os.path.join(
            cache_root, self.unique_description_hash + "-" + type(self).__name__
        )
        names = ("document_index", "sample_index", "shuffle_index")
        marker = cache_prefix + "-description.txt"
        if os.path.isfile(marker) and all(
            os.path.isfile(cache_prefix + "-" + name + ".npy") for name in names
        ):
            for name in names:
                setattr(self, name, np.load(
                    cache_prefix + "-" + name + ".npy", mmap_mode="r", allow_pickle=False
                ))
            return
        rng = np.random.RandomState(self.config.random_seed)
        order = np.asarray(self.indices, dtype=np.int32).copy()
        rng.shuffle(order)
        if self.config.sft_pack_samples and len(order):
            if self.config.sft_packing_strategy == "bfd":
                self.document_index, boundaries = _build_sample_idx_bfd(
                    self._packing_lengths, order, self.config.sequence_length, 0,
                    self.config.max_docs_per_bin_sft,
                )
                self.sample_index = boundaries[:, 0]
            else:
                self.document_index, self.sample_index = _greedy_samples(
                    self._packing_lengths, order, self.config.sequence_length
                )
        else:
            self.document_index = order
            self.sample_index = np.arange(len(order) + 1, dtype=np.int64)
        available = len(self.sample_index) - 1
        requested = available if self.num_samples is None else self.num_samples
        if requested and not available:
            raise ValueError("Apertus SFT split contains no conversations")
        self.shuffle_index = (
            np.concatenate([
                rng.permutation(available)
                for _ in range(math.ceil(requested / available))
            ])[:requested]
            if requested else np.empty(0, dtype=np.int64)
        )
        os.makedirs(cache_root, exist_ok=True)
        for name in names:
            with tempfile.NamedTemporaryFile(dir=cache_root, delete=False) as handle:
                temporary = handle.name
                np.save(handle, getattr(self, name), allow_pickle=False)
            os.replace(temporary, cache_prefix + "-" + name + ".npy")
        with tempfile.NamedTemporaryFile(mode="w", dir=cache_root, delete=False) as handle:
            handle.write(self.unique_description)
            temporary = handle.name
        os.replace(temporary, marker)

    @staticmethod
    def _read_controls(tokenizer):
        from megatron.core.tokenizers.utils.tokenizer_extra_metadata import (
            _extract_special_token_ids,
            _find_hf_tokenizer,
        )

        hf = _find_hf_tokenizer(tokenizer) or tokenizer
        special_ids = _extract_special_token_ids(hf)
        controls = {}
        for name in ("output_start_token_id", "output_end_token_id", "wait_token_id"):
            value = getattr(hf, name, None)
            glyph_name = name[:-3]
            if value is None:
                glyph = getattr(hf, glyph_name, None)
                if glyph is None:
                    glyph = getattr(hf, "init_kwargs", {}).get(glyph_name)
                if glyph is None:
                    glyph = getattr(hf, "extra_special_tokens", {}).get(glyph_name)
                if glyph is not None:
                    value = hf.convert_tokens_to_ids(glyph)
            if value is None or value not in special_ids:
                raise ValueError(f"Apertus tokenizer lacks registered {name}")
            controls[name] = int(value)
        if len(set(controls.values())) != len(controls):
            raise ValueError("Apertus output control IDs must be distinct")
        return controls

    def __len__(self):
        return len(self.shuffle_index)

    def _output_mask(self, tokens):
        """Output membership, independent of stored training weights."""
        output_start = self._control_ids["output_start_token_id"]
        output_end = self._control_ids["output_end_token_id"]
        wait = self._control_ids["wait_token_id"]
        result = np.zeros(len(tokens), dtype=bool)
        active = False
        for index, token in enumerate(tokens):
            if token == output_start:
                active = True
            if active or token == wait:
                result[index] = True
            if token == output_end:
                active = False
        return result

    def _document(self, document_id):
        tokens, weights = self.dataset.get(int(document_id))
        tokens = np.asarray(tokens, dtype=np.int64)
        if (tokens < 0).any() or (tokens >= self._vocab_size).any():
            raise ValueError(f"out-of-vocabulary tokens: {self.dataset_path}, document {document_id}")
        assistant_mask = (
            self._output_mask(tokens) if self._control_ids is not None else None
        )
        if weights is None:
            weights = assistant_mask.astype(np.float32)
        window = (
            slice(None, self.config.sequence_length) if self.config.sft_truncate_right
            else slice(-self.config.sequence_length, None)
        )
        return (
            tokens[window], weights[window],
            assistant_mask[window] if self.config.sft_report_assistant_loss else None,
        )

    def __getitem__(self, idx):
        target = self.config.sequence_length
        tokens = np.zeros(target, dtype=np.int64)
        labels = np.zeros(target, dtype=np.int64)
        loss_mask = np.zeros(target, dtype=np.float32)
        assistant_mask = (
            np.zeros(target, dtype=bool) if self.config.sft_report_assistant_loss else None
        )
        positions = np.zeros(target, dtype=np.int64)
        padding = np.ones(target, dtype=bool)
        boundaries = [0]
        cursor = 0
        if idx is not None:
            sample = int(self.shuffle_index[idx])
            begin, end = self.sample_index[sample:sample + 2]
            for document_id in self.document_index[begin:end]:
                part, weights, output_mask = self._document(document_id)
                length = len(part)
                padded = int(self._packing_lengths[document_id])
                if cursor + padded > target:
                    raise RuntimeError("SFT packing index exceeds sequence length")
                tokens[cursor:cursor + length] = part
                labels[cursor:cursor + length - 1] = part[1:]
                loss_mask[cursor:cursor + length - 1] = weights[1:]
                if assistant_mask is not None:
                    assistant_mask[cursor:cursor + length - 1] = output_mask[1:]
                positions[cursor:cursor + length] = np.arange(length)
                padding[cursor:cursor + length] = False
                cursor += padded
                boundaries.append(cursor)
        if len(boundaries) == 1:
            boundaries.append(target)
        else:
            # Kernels consume the entire physical tensor, including its final padding.
            boundaries[-1] = target
        # Fixed-size metadata can use the default DataLoader collator. Flattening
        # removes the repeated terminal offsets before constructing PackedSeqParams.
        cu = np.full(target + 1, target, dtype=np.int32)
        cu[:len(boundaries)] = boundaries
        batch = {
            "tokens": torch.from_numpy(tokens),
            "labels": torch.from_numpy(labels),
            "loss_mask": torch.from_numpy(loss_mask),
            "position_ids": torch.from_numpy(positions),
            "padding_mask": torch.from_numpy(padding),
            "cu_seqlens": torch.from_numpy(cu),
            "cu_seqlens_padded": torch.from_numpy(cu.copy()),
            "max_seqlen": torch.tensor(max(np.diff(boundaries)), dtype=torch.int32),
        }
        if assistant_mask is not None:
            batch['assistant_mask'] = torch.from_numpy(assistant_mask)
        return batch
