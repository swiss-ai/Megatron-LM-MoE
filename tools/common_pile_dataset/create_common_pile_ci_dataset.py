#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Create CI test dataset from Common Pile (filtered).

Downloads a sample from the Common Pile filtered dataset and preprocesses it
into Megatron-LM's indexed binary format for GPT, BERT, and T5 models.

Output structure:
  <output_dir>/
  ├── my-gpt3_00_text_document.{bin,idx}
  ├── bpe/
  │   ├── vocab.json
  │   └── merges.txt
  ├── my-bert_00_text_sentence.{bin,idx}
  ├── vocab.txt
  ├── my-t5_00_text_document.{bin,idx}
  └── bert-large-cased-vocab.txt

Usage:
  # Small test run (streaming):
  python tools/common_pile_dataset/create_common_pile_ci_dataset.py \
    --output-dir /path/to/output \
    --num-documents 10000 \
    --copy-vocab-from /path/to/existing/the_pile

  # GPT-only production run (~24 GB, streaming):
  uv run python tools/common_pile_dataset/create_common_pile_ci_dataset.py \
    --output-dir /path/to/output \
    --work-dir /path/to/scratch \
    --num-documents 12000000 \
    --models gpt \
    --download-vocab

  # With vocab download (no existing dataset needed):
  python tools/common_pile_dataset/create_common_pile_ci_dataset.py \
    --output-dir /path/to/output \
    --num-documents 10000 \
    --download-vocab
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def _format_eta(seconds):
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def download_common_pile_sample(output_jsonl, num_documents, dataset_name, revision=None):
    """Download a sample from Common Pile using the datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found. Install with: pip install datasets")
        sys.exit(1)

    print(f"Downloading {num_documents} documents from {dataset_name}...")
    print("  (Using HF_TOKEN from environment for authentication if set)")

    ds = load_dataset(dataset_name, revision=revision, split="train", streaming=True)

    count = 0
    start_time = time.time()
    log_interval = max(1000, num_documents // 100)  # Log ~100 times, min every 1000

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for sample in ds:
            text = sample.get("text", "")
            # Skip very short documents that wouldn't be useful for CI tests
            if text and len(text.strip()) > 100:
                f.write(json.dumps({"text": text}) + "\n")
                count += 1
                if count >= num_documents:
                    break
                if count % log_interval == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    remaining = (num_documents - count) / rate if rate > 0 else 0
                    pct = count / num_documents * 100
                    file_size_mb = os.path.getsize(output_jsonl) / (1024 * 1024)
                    print(
                        f"  [{pct:5.1f}%] {count:,}/{num_documents:,} docs | "
                        f"{rate:,.0f} docs/s | "
                        f"{file_size_mb:,.1f} MB on disk | "
                        f"ETA: {_format_eta(remaining)}"
                    )

    elapsed = time.time() - start_time
    print(f"  Saved {count:,} documents to {output_jsonl} in {_format_eta(elapsed)}")
    file_size_mb = os.path.getsize(output_jsonl) / (1024 * 1024)
    print(f"  JSONL file size: {file_size_mb:,.2f} MB")
    return count


def download_common_pile_bulk(output_jsonl, num_documents, dataset_name, revision=None):
    """Download from Common Pile using bulk parquet loading for speed.

    This is faster than streaming for large downloads (>100K docs) because
    it downloads full parquet files and processes them locally.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found. Install with: pip install datasets")
        sys.exit(1)

    print(f"Bulk downloading {num_documents:,} documents from {dataset_name}...")
    print("  (Using non-streaming mode for faster throughput)")
    print("  (Using HF_TOKEN from environment for authentication if set)")

    # Load non-streaming — this downloads parquet shards to the HF cache
    print("  Loading dataset (downloading parquet shards)...")
    ds = load_dataset(dataset_name, revision=revision, split="train")
    total_available = len(ds)
    print(f"  Dataset loaded: {total_available:,} documents available")

    count = 0
    skipped = 0
    start_time = time.time()
    log_interval = max(1000, num_documents // 100)

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for i in range(min(total_available, num_documents + num_documents // 10)):
            text = ds[i].get("text", "")
            if text and len(text.strip()) > 100:
                f.write(json.dumps({"text": text}) + "\n")
                count += 1
                if count >= num_documents:
                    break
                if count % log_interval == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    remaining = (num_documents - count) / rate if rate > 0 else 0
                    pct = count / num_documents * 100
                    file_size_mb = os.path.getsize(output_jsonl) / (1024 * 1024)
                    print(
                        f"  [{pct:5.1f}%] {count:,}/{num_documents:,} docs | "
                        f"{rate:,.0f} docs/s | "
                        f"{file_size_mb:,.1f} MB on disk | "
                        f"ETA: {_format_eta(remaining)}"
                    )
            else:
                skipped += 1

    elapsed = time.time() - start_time
    print(
        f"  Saved {count:,} documents to {output_jsonl} in {_format_eta(elapsed)} "
        f"({skipped:,} short docs skipped)"
    )
    file_size_mb = os.path.getsize(output_jsonl) / (1024 * 1024)
    print(f"  JSONL file size: {file_size_mb:,.2f} MB")
    return count


def copy_vocab_files(output_dir, source_base, models=("gpt", "bert", "t5")):
    """Copy vocabulary files from existing the_pile dataset directories."""
    copies = [
        (
            "gpt",
            os.path.join(source_base, "shard00", "bpe", "vocab.json"),
            os.path.join(output_dir, "bpe", "vocab.json"),
        ),
        (
            "gpt",
            os.path.join(source_base, "shard00", "bpe", "merges.txt"),
            os.path.join(output_dir, "bpe", "merges.txt"),
        ),
        (
            "bert",
            os.path.join(source_base, "bert_shard00", "vocab.txt"),
            os.path.join(output_dir, "vocab.txt"),
        ),
        (
            "t5",
            os.path.join(source_base, "t5_shard00", "bert-large-cased-vocab.txt"),
            os.path.join(output_dir, "bert-large-cased-vocab.txt"),
        ),
    ]

    for model, src, dst in copies:
        if model not in models:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(src):
            print(f"  ERROR: Source vocab file not found: {src}")
            sys.exit(1)
        if os.path.exists(dst):
            print(f"  Already exists: {dst}")
            continue
        print(f"  Copying {src} -> {dst}")
        with open(src, 'rb') as f_in, open(dst, 'wb') as f_out:
            f_out.write(f_in.read())


def download_vocab_files(output_dir, models=("gpt", "bert", "t5"), gpt2_revision="main"):
    """Download tokenizer vocabulary files from HuggingFace."""
    downloads = [
        (
            "gpt",
            f"https://huggingface.co/openai-community/gpt2/resolve/{gpt2_revision}/vocab.json",
            os.path.join(output_dir, "bpe", "vocab.json"),
        ),
        (
            "gpt",
            f"https://huggingface.co/openai-community/gpt2/resolve/{gpt2_revision}/merges.txt",
            os.path.join(output_dir, "bpe", "merges.txt"),
        ),
        (
            "bert",
            "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt",
            os.path.join(output_dir, "vocab.txt"),
        ),
        (
            "t5",
            "https://huggingface.co/google-bert/bert-large-cased/resolve/main/vocab.txt",
            os.path.join(output_dir, "bert-large-cased-vocab.txt"),
        ),
    ]

    for model, url, dst in downloads:
        if model not in models:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print(f"  Already exists: {dst}")
            continue
        print(f"  Downloading {url}")
        print(f"         -> {dst}")
        hf_token = os.environ.get("HF_TOKEN", "")
        req = urllib.request.Request(url)
        if hf_token:
            req.add_header("Authorization", f"Bearer {hf_token}")
        with urllib.request.urlopen(req, timeout=60) as response, open(dst, 'wb') as f_out:
            f_out.write(response.read())


def run_preprocess(megatron_dir, jsonl_path, output_prefix, tokenizer_type,
                   vocab_file, merge_file=None, split_sentences=False,
                   append_eod=False, workers=4):
    """Run preprocess_data.py to create .bin/.idx files."""
    cmd = [
        sys.executable,
        os.path.join(megatron_dir, "tools", "preprocess_data.py"),
        "--input", jsonl_path,
        "--output-prefix", output_prefix,
        "--tokenizer-type", tokenizer_type,
        "--vocab-file", vocab_file,
        "--workers", str(workers),
    ]

    if merge_file:
        cmd.extend(["--merge-file", merge_file])
    if split_sentences:
        cmd.append("--split-sentences")
    if append_eod:
        cmd.append("--append-eod")

    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=megatron_dir)
    if result.returncode != 0:
        print(f"  ERROR: Preprocessing failed with return code {result.returncode}")
        sys.exit(1)


def verify_output(output_dir, models=("gpt", "bert", "t5")):
    """Verify the output files for the selected model families."""
    model_files = {
        "gpt": [
            "my-gpt3_00_text_document.bin",
            "my-gpt3_00_text_document.idx",
            "bpe/vocab.json",
            "bpe/merges.txt",
        ],
        "bert": [
            "my-bert_00_text_sentence.bin",
            "my-bert_00_text_sentence.idx",
            "vocab.txt",
        ],
        "t5": [
            "my-t5_00_text_document.bin",
            "my-t5_00_text_document.idx",
            "bert-large-cased-vocab.txt",
        ],
    }
    expected_files = [name for model in models for name in model_files[model]]

    all_ok = True
    for f in expected_files:
        full_path = os.path.join(output_dir, f)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  OK: {f} ({size_mb:.2f} MB)")
        else:
            print(f"  MISSING: {f}")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Create CI test dataset from Common Pile"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--megatron-dir", type=str, default=None,
        help="Path to Megatron-LM repo root (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--num-documents", type=int, default=10000,
        help="Number of documents to download (default: 10000)",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="common-pile/comma_v0.1_training_dataset",
        help="HuggingFace dataset name (default: common-pile/comma_v0.1_training_dataset)",
    )
    parser.add_argument(
        "--models", nargs="+", choices=("gpt", "bert", "t5"),
        default=["gpt", "bert", "t5"],
        help="Model families to prepare (default: gpt bert t5)",
    )
    parser.add_argument(
        "--dataset-revision", default=None,
        help="Hugging Face dataset revision or commit",
    )
    parser.add_argument(
        "--gpt2-revision", default="main",
        help="GPT-2 tokenizer revision or commit",
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Directory for temporary JSONL files (default: output directory)",
    )
    parser.add_argument(
        "--copy-vocab-from", type=str, default=None,
        help="Copy vocab files from existing the_pile base directory "
             "(e.g., /lustre/.../text/the_pile)",
    )
    parser.add_argument(
        "--download-vocab", action="store_true",
        help="Download vocab files from HuggingFace instead of copying",
    )
    parser.add_argument(
        "--existing-jsonl", type=str, default=None,
        help="Path to existing JSONL file (skip download step)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of worker processes for preprocessing (default: 4)",
    )
    parser.add_argument(
        "--keep-jsonl", action="store_true",
        help="Keep the intermediate JSONL file after preprocessing",
    )
    parser.add_argument(
        "--bulk-download", action="store_true",
        help="Use bulk (non-streaming) download for faster throughput at scale. "
             "Downloads full parquet shards to HF cache before writing JSONL. "
             "Recommended for --num-documents > 100000.",
    )
    args = parser.parse_args()

    # Auto-detect Megatron-LM directory
    if args.megatron_dir is None:
        args.megatron_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
        )
    print(f"Megatron-LM directory: {args.megatron_dir}")
    print(f"Output directory: {args.output_dir}")

    # Verify megatron dir has preprocess_data.py
    preprocess_script = os.path.join(args.megatron_dir, "tools", "preprocess_data.py")
    if not os.path.exists(preprocess_script):
        print(f"ERROR: preprocess_data.py not found at {preprocess_script}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    work_dir = args.work_dir or args.output_dir
    os.makedirs(work_dir, exist_ok=True)

    # ================================================================
    # Step 1: Get vocabulary files
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 1: Setting up vocabulary files...")
    print("=" * 60)

    if args.copy_vocab_from:
        print(f"  Copying from: {args.copy_vocab_from}")
        copy_vocab_files(args.output_dir, args.copy_vocab_from, args.models)
    elif args.download_vocab:
        print("  Downloading from HuggingFace...")
        download_vocab_files(args.output_dir, args.models, args.gpt2_revision)
    else:
        # Default: try to copy from standard CI location, fall back to download
        default_source = (
            "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore"
            "/mcore_ci/text/the_pile"
        )
        if os.path.exists(default_source):
            print(f"  Copying from default location: {default_source}")
            copy_vocab_files(args.output_dir, default_source, args.models)
        else:
            print("  Default vocab source not found, downloading from HuggingFace...")
            download_vocab_files(args.output_dir, args.models, args.gpt2_revision)

    # ================================================================
    # Step 2: Get raw text data
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 2: Preparing raw text data...")
    print("=" * 60)

    if args.existing_jsonl:
        jsonl_path = args.existing_jsonl
        print(f"  Using existing JSONL: {jsonl_path}")
    else:
        jsonl_path = os.path.join(work_dir, "common_pile_raw.jsonl")
        if os.path.exists(jsonl_path):
            raise FileExistsError(
                f"Raw JSONL already exists: {jsonl_path}. "
                "Use a new work directory or --existing-jsonl for explicit reuse."
            )
        partial_jsonl = jsonl_path + ".partial"
        if args.bulk_download:
            download = download_common_pile_bulk
        else:
            download = download_common_pile_sample
        count = download(
            partial_jsonl, args.num_documents, args.dataset_name, args.dataset_revision
        )
        if count != args.num_documents:
            raise RuntimeError(
                f"Expected {args.num_documents} documents, received {count}"
            )
        os.replace(partial_jsonl, jsonl_path)

    # ================================================================
    # Step 3: Preprocess for GPT (GPT2BPETokenizer)
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 3: Preprocessing for GPT (GPT2BPETokenizer)...")
    print("=" * 60)

    gpt_prefix = os.path.join(args.output_dir, "my-gpt3_00")
    gpt_bin = gpt_prefix + "_text_document.bin"
    if "gpt" not in args.models:
        print("  Skipping GPT preprocessing")
    elif os.path.exists(gpt_bin):
        print(f"  GPT data already exists: {gpt_bin}")
    else:
        run_preprocess(
            megatron_dir=args.megatron_dir,
            jsonl_path=jsonl_path,
            output_prefix=gpt_prefix,
            tokenizer_type="GPT2BPETokenizer",
            vocab_file=os.path.join(args.output_dir, "bpe", "vocab.json"),
            merge_file=os.path.join(args.output_dir, "bpe", "merges.txt"),
            append_eod=True,
            workers=args.workers,
        )

    # ================================================================
    # Step 4: Preprocess for BERT (BertWordPieceLowerCase + split-sentences)
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 4: Preprocessing for BERT (BertWordPieceLowerCase)...")
    print("=" * 60)

    bert_prefix = os.path.join(args.output_dir, "my-bert_00")
    bert_bin = bert_prefix + "_text_sentence.bin"
    if "bert" not in args.models:
        print("  Skipping BERT preprocessing")
    elif os.path.exists(bert_bin):
        print(f"  BERT data already exists: {bert_bin}")
    else:
        # BERT with --split-sentences requires two passes when partitions=1:
        #   Pass 1: splits sentences, creates <input>_ss.jsonl, then returns
        #   Pass 2: detects _ss.jsonl exists, encodes to binary .bin/.idx
        jsonl_base, jsonl_ext = os.path.splitext(jsonl_path)
        ss_file = jsonl_base + "_ss" + jsonl_ext
        if not os.path.exists(ss_file):
            print("  Pass 1: Splitting sentences...")
            run_preprocess(
                megatron_dir=args.megatron_dir,
                jsonl_path=jsonl_path,
                output_prefix=bert_prefix,
                tokenizer_type="BertWordPieceLowerCase",
                vocab_file=os.path.join(args.output_dir, "vocab.txt"),
                split_sentences=True,
                workers=args.workers,
            )
        else:
            print(f"  Sentence-split file already exists: {ss_file}")
        print("  Pass 2: Encoding split sentences to binary...")
        run_preprocess(
            megatron_dir=args.megatron_dir,
            jsonl_path=jsonl_path,
            output_prefix=bert_prefix,
            tokenizer_type="BertWordPieceLowerCase",
            vocab_file=os.path.join(args.output_dir, "vocab.txt"),
            split_sentences=True,
            workers=args.workers,
        )

    # ================================================================
    # Step 5: Preprocess for T5 (BertWordPieceCase)
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 5: Preprocessing for T5 (BertWordPieceCase)...")
    print("=" * 60)

    t5_prefix = os.path.join(args.output_dir, "my-t5_00")
    t5_bin = t5_prefix + "_text_document.bin"
    if "t5" not in args.models:
        print("  Skipping T5 preprocessing")
    elif os.path.exists(t5_bin):
        print(f"  T5 data already exists: {t5_bin}")
    else:
        run_preprocess(
            megatron_dir=args.megatron_dir,
            jsonl_path=jsonl_path,
            output_prefix=t5_prefix,
            tokenizer_type="BertWordPieceCase",
            vocab_file=os.path.join(args.output_dir, "bert-large-cased-vocab.txt"),
            append_eod=True,
            workers=args.workers,
        )

    # ================================================================
    # Step 6: Clean up and verify
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 6: Verifying output...")
    print("=" * 60)

    all_ok = verify_output(args.output_dir, args.models)
    if not all_ok:
        print("\nERROR: Some expected files are missing. Temporary text files were retained.")
        sys.exit(1)

    if not args.keep_jsonl and not args.existing_jsonl:
        intermediate = os.path.join(work_dir, "common_pile_raw.jsonl")
        if os.path.exists(intermediate):
            print(f"  Removing intermediate JSONL: {intermediate}")
            os.remove(intermediate)
        ss_intermediate = os.path.join(work_dir, "common_pile_raw_ss.jsonl")
        if os.path.exists(ss_intermediate):
            print(f"  Removing sentence-split JSONL: {ss_intermediate}")
            os.remove(ss_intermediate)

    print(f"\nDataset created successfully at: {args.output_dir}")
    print(f"Prepared model families: {', '.join(args.models)}")


if __name__ == "__main__":
    main()
