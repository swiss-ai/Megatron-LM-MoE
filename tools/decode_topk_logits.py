#!/usr/bin/env python3
"""Read the top-k logit files written by --save-topk-logits-dir. numpy only (no torch, no Megatron).

  python tools/decode_topk_logits.py /ritom/datacache/cscs/swissai/infra01/topk-logits
  python tools/decode_topk_logits.py <dir>/iter_0079501/rank_00000.npz

In Python:
  from decode_topk_logits import load
  d = load(".../iter_0079501/rank_00000.npz")
  d["indices"]  int64   [T, k]  token ids, ascending within each token
  d["logits"]   float32 [T, k]  dequantized logits (top-1 exact, the rest within clip/510 nats)
  d["lse"]      float32 [T]     log-sum-exp over the whole vocabulary
  d["labels"]   int32   [T]     the target token at that position
  d["position"] int32   [T]     position inside the sequence      (None in older files)
  d["sample"]   int32   [T]     microbatch * batch + index in it   (None in older files)
  d["loss_valid"] bool  [T]     True if the token counts in the training loss (None in older files)
  Every token is saved, loss-masked ones included, so rows of one sample are contiguous:
  rows = d["sample"] == s gives positions 0..seq-1 of that sample.
  log-probability of entry j of token t:  d["logits"][t, j] - d["lse"][t]
"""
import argparse
import glob
import os

import numpy as np


def ef_layout(vocab, k):
    """Elias-Fano parameters: (low bits per id, bytes of the low part, bytes of the upper part)."""
    low_bits = max(0, (vocab // k).bit_length() - 1)
    upper_bits = k + ((vocab - 1) >> low_bits) + 1
    return low_bits, (k * low_bits + 7) // 8, (upper_bits + 7) // 8


def decode_indices_ef(packed, vocab, k, chunk=16384):
    """uint8 [rows, bytes] Elias-Fano -> int64 ids [rows, k], ascending within a row."""
    low_bits, low_bytes, upper_bytes = ef_layout(vocab, k)
    out = np.empty((packed.shape[0], k), dtype=np.int64)
    for start in range(0, packed.shape[0], chunk):
        block = packed[start : start + chunk]
        rows = block.shape[0]
        if low_bits:
            bits = np.unpackbits(block[:, :low_bytes], axis=1, bitorder="little")[:, : k * low_bits]
            low = (bits.reshape(rows, k, low_bits).astype(np.int64) << np.arange(low_bits)).sum(-1)
        else:
            low = np.zeros((rows, k), dtype=np.int64)
        upper = np.unpackbits(block[:, low_bytes : low_bytes + upper_bytes], axis=1, bitorder="little")
        _, position = np.nonzero(upper)  # exactly k set bits per row, in row-major order
        out[start : start + chunk] = ((position.reshape(rows, k) - np.arange(k)) << low_bits) | low
    return out


def load(path, rows=None):
    """Decode one rank_*.npz. ``rows`` limits it to the first N tokens (the files are large)."""
    with np.load(path) as data:
        clip, k, vocab = float(data["clip"]), int(data["k"]), int(data["vocab"])
        n = data["top1"].shape[0] if rows is None else min(rows, data["top1"].shape[0])
        q = data["q"][:n].astype(np.float32)
        logits = data["top1"][:n, None] - q * (clip / 255.0)
        return {
            "indices": decode_indices_ef(data["idx_ef"][:n], vocab, k),
            "logits": logits.astype(np.float32),
            "lse": data["lse"][:n],
            "labels": data["labels"][:n],
            "position": data["position"][:n] if "position" in data.files else None,
            "sample": data["sample"][:n] if "sample" in data.files else None,
            "loss_valid": data["loss_valid"][:n].astype(bool) if "loss_valid" in data.files else None,
            "batch": int(data["batch"]) if "batch" in data.files else None,
            "step": int(data["step"]),
            "k": k,
            "vocab": vocab,
            "clip": clip,
            "tokens_in_file": int(data["top1"].shape[0]),
        }


def summarize(path, rows=20000):
    d = load(path, rows)
    prob = np.exp(d["logits"].astype(np.float64) - d["lse"][:, None].astype(np.float64))
    tail = 1.0 - prob.sum(axis=1)  # probability mass outside the saved top-k
    top1 = prob.max(axis=1)
    in_topk = (d["indices"] == d["labels"][:, None]).any(axis=1)
    if d["loss_valid"] is not None:  # statistics over the tokens that count in the loss
        keep = d["loss_valid"]
        print(f"  loss-valid tokens: {keep.mean():.1%} of the first {len(keep):,}")
        prob, tail, top1, in_topk = prob[keep], tail[keep], top1[keep], in_topk[keep]
    print(
        f"{os.path.relpath(path)}: step {d['step']}, {d['tokens_in_file']:,} tokens, k={d['k']}, "
        f"vocab={d['vocab']} | first {len(prob):,}: top-1 prob mean {top1.mean():.3f}, "
        f"tail mass mean {tail.mean():.2e} (max {tail.max():.2e}), label in top-k {in_topk.mean():.1%}, "
        f"ids ascending {bool(np.all(np.diff(d['indices'], axis=1) > 0))}"
    )
    return d


def show_token(d, t=0, n=5):
    order = np.argsort(-d["logits"][t])[:n]
    print(f"  token {t}: label {d['labels'][t]}, lse {d['lse'][t]:.3f}")
    for j in order:
        p = np.exp(float(d["logits"][t, j]) - float(d["lse"][t]))
        print(f"    id {d['indices'][t, j]:>7}  logit {d['logits'][t, j]:8.3f}  prob {p:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="the topk-logits directory, an iter_* directory, or one .npz file")
    parser.add_argument("--rows", type=int, default=20000, help="tokens to decode per file for the summary")
    parser.add_argument("--max-files", type=int, default=5, help="files to summarize")
    args = parser.parse_args()
    if os.path.isfile(args.path):
        files = [args.path]
    else:
        files = sorted(glob.glob(os.path.join(args.path, "**", "rank_*.npz"), recursive=True))
    print(f"{len(files)} files found")
    for i, path in enumerate(files[: args.max_files]):
        d = summarize(path, args.rows)
        if i == 0:
            show_token(d)


if __name__ == "__main__":
    main()
