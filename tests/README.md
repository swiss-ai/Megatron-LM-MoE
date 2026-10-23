# Megatron-LM Tests

## CSCS Alps

```bash
bash tests/run_tests.sh unit --list

bash tests/run_tests.sh unit all

bash tests/run_tests.sh unit moe

NODES=1 bash tests/run_tests.sh unit pipeline-two-ranks
```

### Already available on Alps

The shared assets are already provisioned at the launcher's default `DATA_PATH`:

```text
/capstor/store/cscs/swissai/infra01/megatron-test-assets/
├── unit-tests/                           # NVIDIA unit datasets and tokenizers
└── text/common_pile/v01_filtered_data/    # GPT/hybrid training data
    ├── my-gpt3_00_text_document.{bin,idx}  # 12 million documents
    └── bpe/{vocab.json,merges.txt}        # GPT-2 tokenizer
```

**No download or data preparation is needed for the default Alps runs.** The
launcher uses these paths automatically. Canonical assets stay unchanged; writable
unit fixtures, dataset caches and generated model checkpoints go into `LOG_DIR`.
The repository includes GPT and Mamba/hybrid training configurations, not
pretrained weights; see [model examples](#model-examples).

### Environment variables

Set these before a command to override its defaults:

| Variable | Default | Controls |
| --- | --- | --- |
| `ACCOUNT` | `csstaff` | Slurm account |
| `PARTITION` | `preemptable` | Slurm partition; use `normal` where available |
| `NODES` | `2` (or existing allocation size) | Number of nodes |
| `TIME` | `05:00:00` | Allocation time limit |
| `ENVIRONMENT` | `tests/test_utils/recipes/gh200/environment.toml` | Container environment (EDF) |
| `LOG_DIR` | New scratch directory; see below | Output directory for results, logs, JUnit and coverage |
| `PUBLISH_RESULTS` | `false` | Dispatch the GitHub report workflow after a unit or functional run |
| `UNIT_TEST_COLLECT_ONLY` | `false` | Collect unit tests without running their bodies |
| `UNIT_TEST_FAIL_FAST` | `true` | Stop a unit-test phase at its first failure |
| `UNIT_TEST_TIMEOUT` | `30` | Deadline in minutes for each unit production or experimental phase |
| `UNIT_TEST_TRACE` | `false` | Log per-rank test transitions and session cleanup; dump Python stacks after 120 seconds in a test and retain NCCL timeout traces |
| `UNIT_TEST_DISABLE_TRITON_NVLS` | `false` (`true` in GH200 EDF) | Select the public NCCL fallback in optimized dynamic-engine tests (retaining TP/SP/CUDA graphs) and skip NVLS-specific MoE buffer/collective tests. This is a profile switch, not automatic capability detection; NVLS kernels are not validated. |

Functional runs use `/ritom/scratch/cscs/$USER/tmp/` for their default output
directory, including generated checkpoints and dataset caches. Unit runs use
`$SCRATCH/tmp/`. Set `LOG_DIR` to select another location.
Shared source assets stay at `DATA_PATH`.

For example, choose the partition and save all results to a known directory:

```bash
ACCOUNT=csstaff \
PARTITION=normal \
LOG_DIR="$SCRATCH/tmp/my-unit-run" \
bash tests/run_tests.sh unit all
```

Use a new `LOG_DIR` for each run. Pass that same directory to
`tests/publish_results.sh` when previewing or publishing its GitHub report.

Unit groups are defined in
[`gh200/unit-tests.yaml`](test_utils/recipes/gh200/unit-tests.yaml).

Check collection and imports in all unit groups without running test bodies:

```bash
UNIT_TEST_COLLECT_ONLY=true \
bash tests/run_tests.sh unit all
```

A successful collection reports `COLLECTED`, not `COMPLETED`. Collection does not
check dependencies that are imported only inside a test body.

Existing allocations retain their account and partition.

Groups run in sequence within one allocation. Failures do not stop later groups.
Within a group, fail-fast is enabled by default. Disable it to continue through
ordinary test and collection failures and attempt both marker phases:

```bash
UNIT_TEST_FAIL_FAST=false bash tests/run_tests.sh unit all
```

The same marker filters and assertions apply. Failures still produce a nonzero
exit code. Production and experimental phases run in separate Slurm steps, so all
nodes finish or terminate a phase before the next phase starts. A production timeout
does not suppress the experimental phase. Slurm bounds each phase; a local watchdog
also bounds the launcher client. A timeout is not a passing test or complete coverage.

`TIME` limits the allocation; `UNIT_TEST_TIMEOUT` limits each phase. A five-hour
allocation may not cover all 42 phases at the 30-minute cap. Set `TIME` for the
full run. Interrupted runs retain `INCOMPLETE` and `NOT_RUN` rows.

An existing allocation is reused and left open; a new allocation is released on exit.
A shared-scratch checkout is required when dataset C++ helpers need rebuilding.
Unit fixtures are copied from `$DATA_PATH/unit-tests` to `$LOG_DIR/unit-test-data`
and mounted writable at `/opt/data`: dataset tests create index caches there.
The shared source fixtures are never modified. Missing source fixtures stop the run;
they are not downloaded at runtime.

**Topology:** four GH200 GPUs per node; eight-rank groups require two nodes.
Insufficient nodes produce `BLOCKED`. Tests requiring eight local GPUs may skip
or fail. File-group coverage does not imply coverage of every GPU layout.

| Location | Requirements |
| --- | --- |
| Login node | Slurm, GNU `timeout`, Go `yq` v4 |
| Test image | Selected by [`environment.toml`](test_utils/recipes/gh200/environment.toml), the image/PATH source of truth |
| Image tools | `uv`, Go `yq` 4.44.1, GNU `envsubst`, GNU `timeout` |
| Python and packages | Used directly from the image |

[`setup_environment.sh`](test_utils/setup_environment.sh) selects the image's
Python, disables uv project synchronization and user-site packages, and sets cache
paths. It also loads the repository's optional `.env` file. No virtual environment,
requirements export, runtime overlay, package installation or tool download is
performed. `IMAGE_PYTHON` can select another image-owned interpreter; the default
is `/usr/bin/python`.

The GH200 image includes DeepEP, but not NeMo, TensorRT-LLM or HybridEP.
Missing dependencies must be supplied by the image, not installed during a test run.

| Output | Location |
| --- | --- |
| Group status | `$LOG_DIR/results.tsv` |
| Phase status and missing artifacts | `$LOG_DIR/<group>/phase-results.tsv` |
| Source changes and recorded EDF | `$LOG_DIR/test-changes.patch`, `$LOG_DIR/environment.toml` |
| Launcher source snapshot | `$LOG_DIR/launcher-source/` |
| Writable unit fixtures and index caches | `$LOG_DIR/unit-test-data/` |
| Logs, per-rank JUnit, coverage | `$LOG_DIR/<group>/` |
| Compiler caches | Per-worker node-local `/tmp` |
| Download/tool caches | Ritom; session scratch as fallback |

`COMPLETED` requires successful steps and all expected per-rank JUnit/exit markers;
JUnit records test passes and skips. `TIMED_OUT` and `INCOMPLETE` never count as
passing coverage. Phase logs use `prod.node-*.{out,err}` and
`experimental.node-*.{out,err}`. Failures are printed immediately in per-rank stderr,
without waiting for pytest's final summary. Compiler-cache isolation does not
isolate UCCL proxy state between cases in the same pytest process.

### Unit fixtures

NVIDIA's public v2.5 unit datasets and tokenizers are already installed at
`/capstor/store/cscs/swissai/infra01/megatron-test-assets/unit-tests/`, alongside
`source.json` and `SHA256SUMS`. The launcher copies them to
`$LOG_DIR/unit-test-data` and mounts that **writable copy** at `/opt/data`.
The shared source fixtures are never modified.

Only when provisioning a **different, empty asset root**, download the fixtures
(about 13 MB compressed) once inside the test container:

```bash
ASSET_ROOT="$SCRATCH/tmp/my-test-assets"
source tests/test_utils/setup_environment.sh
uv run --no-project --python "$PYTHON_BIN" python tests/test_utils/python_scripts/download_unit_tests_dataset.py \
    --assets-dir "$ASSET_ROOT/unit-tests"
```

Then set `DATA_PATH="$ASSET_ROOT"` for runs using that alternate root.

## Functional tests

```bash
bash tests/run_tests.sh functional --list

bash tests/run_tests.sh functional gpt3_mcore_te_tp2_pp2_resume_torch_dist

bash tests/run_tests.sh functional all
```

[`gh200/gpt.yaml`](test_utils/recipes/gh200/gpt.yaml) lists 88 GPT selections.
`functional all` selects this recipe, not every model family in the repository.
Each case uses eight ranks across two nodes, with five repeats by default.
`N_REPEAT` overrides the repeat count. Select the five Mamba/hybrid pretraining
cases with another recipe:

```bash
FUNCTIONAL_RECIPE=tests/test_utils/recipes/gh200/mamba.yaml \
bash tests/run_tests.sh functional all
```

| Input | Default |
| --- | --- |
| `DATA_PATH` | `/capstor/store/cscs/swissai/infra01/megatron-test-assets` |
| `FUNCTIONAL_RECIPE` | `tests/test_utils/recipes/gh200/gpt.yaml` |
| Golden values | `<case>/golden_values_dev_gh200.json`, validated on the current image |

Missing data or GH200 baselines produce `BLOCKED`. If every selected case is
blocked, no allocation is requested. Library-uninstall and distillation cases
have explicit blockers in the recipe. Original configs and baselines remain
unchanged; copied configs, writable checkpoints, caches and logs use case-local
scratch directories. Resume checks synchronize checkpoint updates across nodes.

Candidate metrics can be collected without an external baseline:

```bash
RECORD_BASELINE=true \
N_REPEAT=1 \
bash tests/run_tests.sh functional gpt3_mcore_te_tp2_pp2_resume_torch_dist
```

`RECORDED` outputs `candidate-repeat-*.json` in the case log directory. Resume
comparisons remain enabled. After review, copy the selected candidate beside the
case's `model_config.yaml` as `golden_values_dev_gh200.json` and include it in the
reviewed code change, like NVIDIA's other platform baselines. Test runs never
update these files automatically.

```text
tests/functional_tests/test_cases/gpt/<case>/
├── model_config.yaml
├── golden_values_dev_dgx_h100.json
└── golden_values_dev_gh200.json
```

### Model examples

These configurations use the already available Common Pile data and GPT-2
tokenizer above. The runs initialize their own models; resume tests generate and
reload their own checkpoints under `LOG_DIR`, rather than requiring pretrained
weights in the shared asset directory.

| Example | Included configuration |
| --- | --- |
| GPT, 12 layers, hidden size 512, TP2/PP2 | [GPT resume model](functional_tests/test_cases/gpt/gpt3_mcore_te_tp2_pp2_resume_torch_dist/model_config.yaml) |
| Mamba/hybrid, hidden size 1024, TP1/PP1 | [Hybrid model](functional_tests/test_cases/hybrid/hybrid_mr_mcore_te_tp1_pp1_cp1_dgx_a100_1N8G/model_config.yaml) |

```bash
bash tests/run_tests.sh functional gpt3_mcore_te_tp2_pp2_resume_torch_dist

FUNCTIONAL_RECIPE=tests/test_utils/recipes/gh200/mamba.yaml \
bash tests/run_tests.sh functional hybrid_mr_mcore_te_tp1_pp1_cp1_dgx_a100_1N8G
```

Use `functional --list` with the appropriate recipe to see the other included
configurations. These examples are functional test workloads, not a claim that
all recipe entries currently pass.

### GPT data

The shared GPT data is **already prepared** at the default Capstor `DATA_PATH`;
do not rebuild it for normal test runs. Only when provisioning a new asset root,
run this once inside the test container:

```bash
DATA_ROOT="$SCRATCH/tmp/my-test-assets" \
bash tools/common_pile_dataset/prepare_gpt_ci_data.sh
```

The [Alps launch example](../tools/common_pile_dataset/README.md#prepare-data-on-cscs-alps)
has the full provisioning procedure. Test runs reuse the existing assets.

| Property | Value |
| --- | --- |
| Source | `common-pile/comma_v0.1_training_dataset` |
| Preparation | NVIDIA production settings; 12 million documents; GPT only |
| Tokenizer | GPT-2 BPE |
| Storage estimate | 24 GB final; 70–80 GB during streaming preparation |
| Provenance | Source revisions, preparation settings, output checksums |

```text
/capstor/store/cscs/swissai/infra01/megatron-test-assets/
└── text/common_pile/v01_filtered_data/
    ├── my-gpt3_00_text_document.bin
    ├── my-gpt3_00_text_document.idx
    └── bpe/{vocab.json,merges.txt}
```

Numerical baselines depend on the exact data,
config, image, and topology; matching preparation settings alone does not establish
identical NVIDIA fixtures. Generated checkpoints and dataset caches stay in scratch.

## Updating Functional Test Golden Values (NVIDIA only)

When adding new functional tests, it may be necessary to update the golden values used to verify if the test is
passing as expected.

1. Add the new functional test case with the scope set to `mr-github`
2. Open a PR with the new test. Ensure the label `Run functional tests` is added
3. Run the PR CI tests
4. Run the script to download golden values from a Github CI run
    a. Ensure click, requests, and python-gitlab are installed in your environment
    b. Ensure a Github access token is set as an environment variable `GITHUB_TOKEN`
    c. Run the script `python tests/test_utils/python_scripts/download_golden_values.py --source github --pipeline-id <github-workflow-run-id>`
    d. Optionally pass in `--only-failing` to only download golden values for failing tests only
    e. Ensure you are only checking-in golden values for tests are you updating

The Github CI infra may not be appropriate for Perf tests. Perf tests may be more appropriate for nightly jobs on other infra.

## GitHub reports

| Task | Location |
| --- | --- |
| Run tests | Alps, through Slurm |
| View reports | GitHub Actions |
| View commit status | Tested commit; links to the Actions run |
| Keep full logs and checkpoints | Scratch |

The [report workflow](../.github/workflows/alps-test-results.yml) uses the existing
`gh` login on the cluster. It does not need SSH access to Alps. `--publish` sends
saved results to GitHub; it does not run GPU tests.

The target repository follows the current branch's push remote, then
`remote.pushDefault`, then the branch's tracking remote. It falls back to `origin`
if none is set. It uses that remote's push URL. Use `--repo` to select another
repository.

The workflow runs from the current publisher branch, or the repository's default
branch for a detached checkout. Use `--ref` to select another workflow branch.
That branch must contain the reporting workflow and its trusted policy.

The tested commit and branch come from the run's `launch.txt`, not the publisher's
current checkout. The publisher identifies a PR only when one associated PR has
the same head commit and branch. The run title shows the suite, tested branch,
PR number when identified, and tested SHA. The summary links to the PR and shows
the reporting workflow ref separately. Old runs without branch metadata show an
unknown branch and no PR association; the publisher does not guess. PR lookup
runs only during publication. Offline previews show the recorded branch and SHA.
Use a clean checkout to validate a PR commit: the SHA does not include local
changes. Unit runs save tracked test changes in `test-changes.patch`.

Run tests:

```bash
export LOG_DIR="$SCRATCH/tmp/megatron-unit-results"
bash tests/run_tests.sh unit rl
```

Run tests and publish their results automatically:

```bash
PUBLISH_RESULTS=true bash tests/run_tests.sh unit all
PUBLISH_RESULTS=true bash tests/run_tests.sh functional all
```

This option supports both suites and individual selections. It uses the
publisher's repository and workflow branch defaults. The launcher host requires
`gh`, `uv`, and a GitHub login with workflow-dispatch permission. The reporting
branch must be pushed before the run.

Results are published on exit, including failed or blocked runs. Test failures
keep their exit code. If tests pass but dispatch fails, the launcher returns the
dispatch error. It does not wait for the workflow to finish. Forced termination
can prevent publication; saved results can be published separately.

Preview the report. This command does not change GitHub:

```bash
bash tests/publish_results.sh "$LOG_DIR" \
    --suite unit
```

Reports check only the groups or cases they contain. Unreported groups do not
fail a report. Failures or incomplete results within reported groups still do.

Send the report:

```bash
bash tests/publish_results.sh "$LOG_DIR" \
    --suite unit \
    --publish
```

Publish functional results through the same workflow:

```bash
bash tests/publish_results.sh "$LOG_DIR" \
    --suite functional \
    --publish
```

Both suites get colored status tables, JSON/Markdown artifacts and stable checks:
`alps/unit/regressions` and `alps/functional/regressions` for full-suite reports.
Selected reports use `alps/<suite>/selected`.
Large reports are compressed for transport; downloaded artifacts remain ordinary
JSON/Markdown.
The publisher reads the tested commit from `launch.txt`, not the current checkout.

| Option | Use |
| --- | --- |
| `--results FILE` | Read another results ledger |
| `--sha SHA` | Set the recorded tested commit when `launch.txt` is absent |
| `--repo OWNER/REPO` | Override the repository selected from Git configuration |
| `--ref BRANCH` | Override the automatically selected workflow branch |

Reports show outcomes, JUnit counts per rank and failing test IDs. They do not
contain raw logs, tokens or checkpoints. Actions keeps the JSON and Markdown
reports for 30 days. Candidates, blockers and incomplete runs are not green.
These are runner results, not an independent numerical audit.
