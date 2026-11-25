#!/bin/bash
#SBATCH --job-name=hf-fetch-fast
#SBATCH -A infra01
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:20:00
#SBATCH --output=/users/dmelikidze/transfer/logs/%j-%x.out
#SBATCH --error=/users/dmelikidze/transfer/logs/%j-%x.out
# Fast HF download on a GH200 compute node (xet ~1.1 GB/s per file there; the x86 xfer node gave ~165 MB/s).
#   REPO=<repo> DST=<dir> [PATTERN=<glob>] [WORKERS=8] sbatch hf_fetch_fast.sbatch
set -uo pipefail
export HF_HOME=/iopsstor/scratch/cscs/dmelikidze/huggingface HF_XET_HIGH_PERFORMANCE=1
export REPO=${REPO:?} DST=${DST:?} PATTERN=${PATTERN:-} WORKERS=${WORKERS:-8}
/users/dmelikidze/venvs/hfup-node/bin/python -u - <<'PY'
import os, time
from huggingface_hub import HfApi, snapshot_download
repo, dst, pat, w = os.environ["REPO"], os.environ["DST"], os.environ["PATTERN"], int(os.environ["WORKERS"])
t = time.time()
snapshot_download(repo_id=repo, repo_type="model", local_dir=dst, max_workers=w, allow_patterns=[pat] if pat else None)
dt = time.time() - t
remote = {e.path: e.size for e in HfApi().list_repo_tree(repo, repo_type="model", recursive=True) if hasattr(e, "size")}
want = {p: s for p, s in remote.items() if (not pat or __import__("fnmatch").fnmatch(p, pat))}
bad = [p for p, s in want.items() if not os.path.isfile(os.path.join(dst, p)) or os.path.getsize(os.path.join(dst, p)) != s]
tot = sum(want.values())
print(f"{len(want)} files, {tot / 2**30:.0f} GiB in {dt / 60:.1f} min ({tot / dt / 1e6:.0f} MB/s incl. skipped); {len(bad)} missing or wrong size {bad[:5]}")
print("FETCH_OK" if not bad else "FETCH_INCOMPLETE"); raise SystemExit(1 if bad else 0)
PY
rc=$?
[ $rc = 0 ] && rm -rf "$DST/.cache"; exit $rc
