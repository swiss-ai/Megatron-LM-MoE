#!/usr/bin/env python3
"""Keep only the conversations of at most MAX tokens in a Megatron SFT dataset, by writing a new index.

The trainer truncates a conversation longer than its sequence length (apertus_sft_dataset.py: lengths are clipped to
config.sequence_length), so a 16K run on a 131K render would train on cut-off conversations. This writes
<dst>/tokens.idx (and tokens.loss_weights.idx when present) with only the conversations of <= MAX tokens, in the stored
order, and links <dst>/tokens.bin (and the loss-weights .bin) to the originals: no data is copied. Index helpers as in
hanint05's make_subset_index.py (/appsdata/hanint05/sft-data/v0.3-chonk-131k-balanced/).
    python3 filter_long_index.py <src train dir> <dst dir> [--max-tokens 16384]
"""
import argparse
import os
import struct

import numpy as np

HEADER = b"MMIDIDX\x00\x00"
ITEMSIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}  # Megatron DType codes


def read_index(path):
    with open(path, "rb") as f:
        if f.read(9) != HEADER:
            raise ValueError("not a Megatron index: " + path)
        version, = struct.unpack("<Q", f.read(8))
        if version != 1:
            raise ValueError("index version %d: %s" % (version, path))
        code, = struct.unpack("<B", f.read(1))
        count, = struct.unpack("<Q", f.read(8))
        docs, = struct.unpack("<Q", f.read(8))
        offset = f.tell()
    buf = np.memmap(path, mode="r", order="C")
    lengths = np.frombuffer(buf, dtype=np.int32, count=count, offset=offset)
    pointers = np.frombuffer(buf, dtype=np.int64, count=count, offset=offset + lengths.nbytes)
    doc_index = np.frombuffer(buf, dtype=np.int64, count=docs, offset=offset + lengths.nbytes + pointers.nbytes)
    return code, lengths, pointers, doc_index


def write_index(path, code, lengths, pointers):
    count = len(lengths)
    with open(path + ".tmp", "wb") as f:
        f.write(HEADER)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", code))
        f.write(struct.pack("<Q", count))
        f.write(struct.pack("<Q", count + 1))
        f.write(np.ascontiguousarray(lengths, dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(pointers, dtype=np.int64).tobytes())
        f.write(np.arange(count + 1, dtype=np.int64).tobytes())
    os.replace(path + ".tmp", path)


def main(src_dir, dst_dir, max_tokens):
    os.makedirs(dst_dir, exist_ok=True)
    mask = kept = None
    for prefix in ("tokens", "tokens.loss_weights"):
        src = os.path.join(src_dir, prefix)
        if not os.path.exists(src + ".idx"):
            continue
        code, lengths, pointers, doc_index = read_index(src + ".idx")
        count, size = len(lengths), ITEMSIZE[code]
        if not np.array_equal(doc_index, np.arange(count + 1)):
            raise ValueError("expected one sequence per document: " + src)
        if int(pointers[-1]) + int(lengths[-1]) * size != os.path.getsize(src + ".bin"):
            raise ValueError(".idx does not match the size of .bin: " + src)
        if mask is None:
            mask = lengths <= max_tokens
        if len(mask) != count:
            raise ValueError("loss weights have a different sequence count: " + src)
        sub_lengths, sub_pointers = lengths[mask].copy(), pointers[mask].copy()
        if kept is not None and not np.array_equal(sub_lengths, kept):
            raise ValueError("loss weights do not match the tokens: " + src)
        kept = sub_lengths
        dst = os.path.join(dst_dir, prefix)
        write_index(dst + ".idx", code, sub_lengths, sub_pointers)
        if os.path.lexists(dst + ".bin"):
            os.remove(dst + ".bin")
        os.symlink(os.path.realpath(src + ".bin"), dst + ".bin")
        check = read_index(dst + ".idx")
        if not (check[0] == code and np.array_equal(check[1], sub_lengths) and np.array_equal(check[2], sub_pointers)):
            raise ValueError("re-read of the written index differs: " + dst)
        all_tok, kept_tok = int(lengths.sum(dtype=np.int64)), int(sub_lengths.sum(dtype=np.int64))
        print("%s: %d -> %d conversations (%.2f%% dropped), %d -> %d items (%.2f%% dropped), max kept %d" % (
            src, count, len(sub_lengths), 100 * (1 - len(sub_lengths) / count), all_tok, kept_tok,
            100 * (1 - kept_tok / all_tok), int(sub_lengths.max()) if len(sub_lengths) else 0))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src"); p.add_argument("dst"); p.add_argument("--max-tokens", type=int, default=16384)
    a = p.parse_args()
    main(a.src, a.dst, a.max_tokens)
