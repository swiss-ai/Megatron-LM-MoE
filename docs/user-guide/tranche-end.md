# Checkpoint and exit at the end of a data tranche

Add the following to the size file's `EXTRA_NETWORK_ARGS`:

```bash
--exit-on-tranche-end
```

The configured training data paths define the current tranche. The GPT dataset
builder creates a single pass over each contributing shard, with exhaustive
blending rather than repeating data to fill `--train-samples`. Explicit blend
weights are unsupported; pass unweighted shard prefixes, as the megachonk
launcher does when `DATA_SOURCES="."`.

The full-run training budget and LR decay are unchanged. All pipeline/tensor
parallel ranks receive the same sample boundary. At the last complete global
batch, the run writes a persistent checkpoint, finalizes any asynchronous save,
and exits successfully. The boundary save takes precedence over interval saves,
including non-persistent intervals.

Samples that cannot fill a complete global batch are left unused. Packing may
also discard the usual incomplete sequence tails. Startup logs report
`[TRANCHE] ... unused_tail_samples=...`; the exit logs `[TRANCHE-END]`.

## Resume and move to another tranche

For a crash/time-limit resume of the same tranche, keep the data paths, sequence
length and packing configuration unchanged. The checkpoint
consumed-sample counter determines the remaining batches. Replaying a rejected
step does not consume another batch.

At completion, rank zero atomically writes `tranche_end.txt` in the save
directory, after the final checkpoint (including any asynchronous save) is
finished. It records the tranche identity, its start/end global consumed-sample
counters, and the final checkpoint iteration.
The `.txt` file contains JSON-formatted metadata, including the tranche identity
needed to distinguish a resume from a new tranche.

When switching to a new tranche, point `DATA_ROOT` at its shards and keep:

```bash
--exit-on-tranche-end
```

The loader reads the marker from `--load` (or `--save` when load is unset).
Different resolved shard prefixes identify a new tranche: its start offset is
the previous tranche's end counter. The same prefixes identify the same tranche:
its original start offset is retained, so a completed tranche cannot silently
restart. Job-specific data-list filenames do not affect the identity.

This restarts only the new tranche's local data cursor while keeping the global
counter, optimizer state and LR schedule. Resuming a partially consumed next
tranche continues from the original offset in the previous completion marker.
Switch data paths only after completing the current tranche.

For the first tranche or checkpoints predating this feature, no marker exists.
The fallback is the existing phase-transition offset, or zero. An explicit
`--tranche-start-samples <counter>` still overrides automatic selection. Loading
a checkpoint older than the automatically selected start offset is rejected
instead of silently skipping samples.

Enabling this feature requests single-pass dataset indices. Enable it
from the start of a tranche: switching an existing multi-epoch run midway may
change that run's shuffled data order. New index caches may need to be
built; disable `--dataloader-fast-cache-load` for their first build.

Requires `--save`, `--dataloader-type single`, a fixed global batch size, and
GPT pretraining data. Batch rampup, dynamic batch-size decreases, mock data, SFT,
RL and `--rerun-strategy skip_iteration` are rejected.

The launcher `--auto-requeue` is separate from this training exit condition. Do
not use an unconditional `afterany` auto-requeue chain at the final tranche
boundary: it will submit another job for the already-finished tranche. Resume
time-limited jobs manually or make the launcher distinguish `[TRANCHE-END]`
from time-limit exits before scheduling a successor.

## CPU regression checks

```bash
python tests/unit_tests/test_tranche_end.py
```

These checks exercise production boundary, dataset-size, resume, rank broadcast,
and checkpoint-exit logic with distributed/model dependencies stubbed. Validate
the final checkpoint and exit on a short GPU run before a large allocation.
