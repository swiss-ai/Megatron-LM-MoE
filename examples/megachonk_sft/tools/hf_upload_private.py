"""Upload one HF model folder to a PRIVATE Hub repo (refuses a public one); re-running uploads only the files missing or
of the wrong size on the Hub (as Clariden's megachonk_upload.py); checks every size at the end.
    REPO=<org/name> SRC=<folder> python hf_upload_private.py
"""
import os
import time

from huggingface_hub import HfApi

REPO, SRC = os.environ["REPO"], os.environ["SRC"]
api = HfApi()
print("HF user:", api.whoami()["name"], flush=True)
local = {f: os.path.getsize(os.path.join(SRC, f)) for f in sorted(os.listdir(SRC)) if os.path.isfile(os.path.join(SRC, f))}
print(f"local: {len(local)} files, {sum(local.values()) / 2**30:.1f} GiB", flush=True)


def require_private():
    if api.repo_info(REPO, repo_type="model").private is not True:
        raise SystemExit(f"{REPO} is NOT private; refusing to continue")


def missing():
    have = {e.path: e.size for e in api.list_repo_tree(REPO, repo_type="model", recursive=True) if hasattr(e, "size")}
    return [f for f, s in local.items() if have.get(f) != s]


api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
require_private()
print(f"repo {REPO}: private OK", flush=True)
todo = missing()
print(f"to upload: {len(todo)} files, {sum(local[f] for f in todo) / 2**30:.1f} GiB", flush=True)
if todo:
    start = time.time()
    api.upload_folder(repo_id=REPO, repo_type="model", folder_path=SRC, allow_patterns=todo,
                      commit_message=f"MegaChonk SFT v1 16K, {os.path.basename(SRC)} (HF format)")
    print(f"upload call finished in {(time.time() - start) / 60:.1f} min", flush=True)
left = missing()
if left:
    raise SystemExit(f"INCOMPLETE: {len(left)} files missing or of the wrong size, e.g. {left[:5]}")
require_private()
print(f"UPLOAD_DONE: {len(local)} files on the Hub, sizes match, {REPO} is private", flush=True)
