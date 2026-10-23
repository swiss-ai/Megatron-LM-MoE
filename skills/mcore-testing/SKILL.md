---
name: mcore-testing
description: Run and diagnose Megatron-LM unit and functional tests using the repository's documented runners. Use when asked to run tests, reproduce a test failure, select a test group or recipe, or interpret test results. Read tests/README.md before planning or launching tests.
license: Apache-2.0
---

# Running Megatron-LM Tests

## Read first

1. Read [tests/README.md](../../tests/README.md) completely before proposing commands or launching tests. This branch's README is the source of truth; do not substitute remembered upstream commands.
2. Read the selected recipe under `tests/test_utils/recipes/` and any environment/package notes linked from the README that apply to the run. If documentation and behavior disagree, inspect the existing runner and report the discrepancy.
3. Confirm the requested scope: unit group, file/case, functional recipe, or full suite. Ask only for decisions that cannot be resolved from the request and repository.

All command paths below are relative to the repository root.

## Use the existing runners

```bash
# Discover available selections.
bash tests/run_tests.sh unit --list
bash tests/run_tests.sh functional --list

# One unit group.
bash tests/run_tests.sh unit moe

# Attempt all unit groups, retaining real failures and continuing after them.
UNIT_TEST_FAIL_FAST=false bash tests/run_tests.sh unit all

# One functional case.
bash tests/run_tests.sh functional gpt3_mcore_te_tp2_pp2_resume_torch_dist
```

For another functional family, select `FUNCTIONAL_RECIPE` as documented. `functional all` means the selected recipe, not every model family.

- Use the README's account, partition, EDF, node count, allocation budget, timeouts, and asset preparation. Honor an explicitly requested image; do not silently replace it with a personal default.
- On Alps, also read the `alps-evals` and `interactive-srun` skills when available. Four GH200 GPUs per node means eight ranks need two nodes.
- Use the public runner from the login node or an owned allocation. Its internal `--node` helper requires a correctly configured `srun` step; do not call it directly from a batch shell.
- Use a fresh `LOG_DIR` under scratch. Preserve the intended WIP source and record its revision/changes before execution; never edit a running snapshot.
- Use image-owned Python and dependencies as documented. Missing assets or packages are blockers, not permission to install overlays or change the workload.

## Verify and report

- Inspect the group and phase ledgers, every expected rank's JUnit and exit marker, and raw logs for the first failure. Slurm exit zero alone is not test success.
- Distinguish executed passes, failures, skips, collection-only runs, blocked cases, timeouts, and incomplete coverage. An empty experimental phase contributes zero tests; missing reports are incomplete coverage.
- Report counts per rank without multiplying distributed case counts. A targeted file pass is not a whole-group or full-suite pass.
- Do not hide failures by weakening assertions, tolerances, markers, data, or goldens. Deferred failures remain failures. Running tests does not authorize adding test files or cases.
- Keep logs and writable fixtures in scratch, preserve canonical assets, and release allocations created for the run. Leave unrelated jobs alone.
- Consult the README for report previews. Commit, push, baseline promotion, and GitHub publication require explicit user approval.
