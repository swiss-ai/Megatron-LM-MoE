"""Upload a large folder (e.g. a Megatron DCP checkpoint) to a PRIVATE Hub repo in batches of files, one commit per batch,
so an interrupted upload keeps what was committed. Re-running uploads only files missing or of the wrong size.
    REPO=<org/name> SRC=<folder> [BATCH_GIB=400] [PATH_PREFIX=iter_0001540] python hf_upload_batched.py
"""
import os
import time

from huggingface_hub import HfApi

REPO, SRC = os.environ["REPO"], os.environ["SRC"]
BATCH = float(os.environ.get("BATCH_GIB", "400")) * 2**30
PREFIX = os.environ.get("PATH_PREFIX", os.path.basename(SRC.rstrip("/")))
api = HfApi()
print("HF user:", api.whoami()["name"], flush=True)
local = {f: os.path.getsize(os.path.join(SRC, f)) for f in sorted(os.listdir(SRC)) if os.path.isfile(os.path.join(SRC, f))}
print(f"local: {len(local)} files, {sum(local.values()) / 2**40:.2f} TiB", flush=True)


def require_private():
    if api.repo_info(REPO, repo_type="model").private is not True:
        raise SystemExit(f"{REPO} is NOT private; refusing to continue")


def missing():
    have = {e.path: e.size for e in api.list_repo_tree(REPO, repo_type="model", recursive=True) if hasattr(e, "size")}
    return [f for f, s in local.items() if have.get(f"{PREFIX}/{f}") != s]


api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
require_private()
todo = missing()
print(f"repo {REPO} private OK; to upload: {len(todo)} files, {sum(local[f] for f in todo) / 2**30:.1f} GiB", flush=True)
batches, cur, size = [], [], 0
for f in todo:
    cur.append(f); size += local[f]
    if size >= BATCH:
        batches.append(cur); cur, size = [], 0
if cur:
    batches.append(cur)
for i, b in enumerate(batches, 1):
    t = time.time()
    gib = sum(local[f] for f in b) / 2**30
    api.upload_folder(repo_id=REPO, repo_type="model", folder_path=SRC, path_in_repo=PREFIX, allow_patterns=b,
                      commit_message=f"{PREFIX}: batch {i}/{len(batches)} ({len(b)} files, {gib:.0f} GiB)")
    dt = time.time() - t
    print(f"{time.strftime('%F %T')} batch {i}/{len(batches)}: {len(b)} files, {gib:.0f} GiB in {dt / 60:.1f} min "
          f"({gib * 2**30 / dt / 1e6:.0f} MB/s)", flush=True)
left = missing()
if left:
    raise SystemExit(f"INCOMPLETE: {len(left)} files missing or of the wrong size, e.g. {left[:5]}")
require_private()
print(f"UPLOAD_DONE: {len(local)} files under {PREFIX}/ in {REPO}, sizes match, private", flush=True)
