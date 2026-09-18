---
name: apertus-testing
description: Running Megatron-LM-MoE unit tests on CSCS Alps. Covers the srun shape, rank wiring, container choice, pytest flags this repo needs, and which tests to run for a given change.
when_to_use: Running or adding a unit test in this repo; verifying a change before opening a PR; 'run the tests', 'which tests cover this', 'pytest on Alps', 'test failed', 'how many GPUs do I need'.
---

# Testing on Alps

Upstream's `mcore-testing` describes a 1-node × 8-GPU runner. **Alps GH200 nodes have
4 GPUs**, so anything needing 8 ranks is a 2-node job. That single difference drives
most of what follows.

---

## Rule: run the tests that cover what you changed

Before opening a PR, run the test files covering every path the diff touches — not
just the one the change was "about".

**Do not work from a remembered list of test files.** It goes stale, and this repo
gains and moves tests on every upstream integration. Discover what exists first:

```bash
# what test areas exist at all
ls tests/unit_tests/

# every test file, or every one under the area you touched
find tests/unit_tests -name 'test_*.py' | sort
find tests/unit_tests/transformer/moe -name 'test_*.py' | sort

```

`ls` and `find` work on the login node. `--collect-only` does **not** — there is no
pytest outside the container — but it needs no GPU and takes about a second, so a
1-task step is enough to list a file's classes and parametrizations:

```bash
MEGACHONK=/capstor/store/cscs/swissai/infra01/users/gfu/img/alps-pytorch2512-megachonk.toml
srun --account=infra01 --partition=preemptable --nodes=1 --ntasks=1 \
  --cpus-per-task=72 --gres=gpu:4 --time=00:07:00 --environment="$MEGACHONK" \
  --container-mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor" \
  bash -lc 'cd <repo> && python -m pytest --collect-only -q -p no:cacheprovider <test_file>'
```

Then map the diff to tests:

1. **Mirror the source path.** `megatron/core/transformer/moe/router.py` →
   `tests/unit_tests/transformer/moe/`. The tree mostly parallels `megatron/core/`.
2. **Grep for the symbols you changed**, which catches tests that live elsewhere:
   ```bash
   git diff --name-only main...HEAD
   grep -rl "attach_and_log_load_balancing_loss\|valid_token_count" tests/
   ```
3. **Check the config knobs you touched**, since a flag is often exercised from an
   unrelated-looking file:
   ```bash
   grep -rl "moe_router_violation_metrics" tests/
   ```

Run the **whole file**, not a single `-k` selection: each test class in this repo
builds its own config, so a fix that unblocks one class often leaves others broken
(see Gotchas). A green `-k` selection is not evidence the file is green. Read the
summary line and compare it against `--collect-only` — if fewer tests ran than were
collected, something aborted early.

Note the GPU cost before launching: `--collect-only` tells you the largest
parametrization in the file, and that sets how many ranks (and therefore nodes) the
run needs.

## The srun shape

```bash
MEGACHONK=/capstor/store/cscs/swissai/infra01/users/gfu/img/alps-pytorch2512-megachonk.toml
srun --account=infra01 --partition=preemptable \
  --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=72 \
  --time=00:30:00 --network=disable_rdzv_get --mpi=pmix -l \
  --environment="$MEGACHONK" \
  --container-mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/capstor:/capstor,/iopsstor:/iopsstor" \
  bash $SCRATCH/tmp/<session>/run_tests.sh
```

- **2 nodes / 8 tasks** for any test parametrized at `tp=8`, `cp=8`, or `2×2×2`.
  4 ranks on one node only covers configs whose product is ≤ 4.
- **megachonk**, not `apertus2-alps4-temp`. See the root `CLAUDE.md`.
- Check `sinfo -a -o "%P %a"` first; `normal` is often down.

## The rank script

```bash
#!/bin/bash
set +e
cd /users/anowak/open_source/Megatron-LM-MoE
export WORLD_SIZE=$SLURM_NTASKS RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_PROCID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=23451
python -m pytest -p no:cacheprovider -p no:unraisableexception --no-header \
  -q --tb=line --maxfail=0 tests/unit_tests/<path> > "$LOGDIR/rank_$SLURM_PROCID.log" 2>&1
echo "RANK $SLURM_PROCID exit=$?"
```

**`LOCAL_RANK` must be the GLOBAL rank.** `tests/unit_tests/test_utilities.py:31`
reads `Utils.rank` from `LOCAL_RANK` and then does
`set_device(rank % device_count())`. Setting `LOCAL_RANK=$SLURM_LOCALID` makes both
nodes claim ranks 0-3 and the job hangs in rendezvous. `SLURM_PROCID` is correct and
the modulo maps it onto the right local GPU.

Do **not** wrap in `torchrun`; Slurm provides the layout.

---

## pytest flags this repo needs

- `--maxfail=0` — `pyproject.toml:237` sets `addopts = "... -x"`, so by default the
  run stops at the first failure and you get a misleading "1 error" summary.
- `-p no:unraisableexception` — `MoEModelTestContainer.__del__` calls
  `torch.distributed.barrier()` after teardown has destroyed the process group. The
  resulting `ValueError` is noise, but the plugin promotes it to a collected error
  and (with `-x`) aborts the run.
- `-p no:cacheprovider` — keeps `.pytest_cache` out of the repo.
- `--tb=line` for a sweep; `--tb=long -x` when chasing one failure. With `--tb=line`
  the tracebacks only print in the end-of-run summary, so a run that times out tells
  you nothing.

---

## Gotchas

**Never background the `srun`.** An interactive `srun` dies with the shell that
launched it — the step is cancelled and per-rank logs stop mid-write with no error.
Keep it in the foreground. If a run needs more than ~10 minutes, launch it detached
with `setsid nohup` and poll `squeue`, or use `sbatch`.

**`interactive-srun` reuse does not work for multi-rank steps.** The allocation's
holder step owns the Slingshot VNI, so an overlapping `--ntasks=8` step fails with
`Error configuring interconnect` regardless of `--network=` or how the allocation was
sized. 1-task steps overlap fine. For multi-rank test runs, submit a normal job.

**`moe_router_violation_metrics` defaults to `['mbs']`.** Every training-mode router
forward then reaches the fork-only `_record_expert_load_samples`, which sizes buffers
from `get_num_microbatches()` — a global unit tests never initialize. Result:
`AttributeError: 'NoneType' object has no attribute 'get'` on every MoE router test.
Any new test config needs `moe_router_violation_metrics=[]`. Three separate places in
`test_aux_loss.py` alone build configs and each needed it independently.

**A test file can be entirely red without anyone noticing.** `test_aux_loss.py` was
43 failed / 20 errors / 1 passed before this was found. Run the full file and read
the summary line; do not assume a file is green because CI is quiet.

---

## Logs

Per `CLAUDE.md`, write to `${SCRATCH}/tmp/<descriptive-session>/`, one file per rank.
Never `$HOME` — its inode quota is small and overrunning it makes Slurm fail to start
jobs with no usable error.
