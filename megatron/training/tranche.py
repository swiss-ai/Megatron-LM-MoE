"""Finite-tranche boundaries, independent of model and distributed execution."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Optional, Sequence


MARKER_NAME = "tranche_end.txt"


def tranche_identity(prefixes: Sequence[str], split: Optional[str]) -> str:
    """Identify resolved training shards, independent of a per-job data-list filename."""
    data = json.dumps({"prefixes": list(prefixes), "split": split}, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def resolve_tranche_start(
    checkpoint_dir: Optional[str],
    identity: str,
    consumed_total: int,
    explicit_start: Optional[int] = None,
    fallback_start: int = 0,
) -> int:
    """Resume the same tranche or start a different tranche after the last completed one."""
    if explicit_start is not None:
        return explicit_start
    if checkpoint_dir is None:
        return fallback_start
    path = Path(checkpoint_dir) / MARKER_NAME
    if not path.exists():
        return fallback_start
    record = json.loads(path.read_text(encoding="utf-8"))
    start, end = record["start_consumed_samples"], record["end_consumed_samples"]
    if record.get("version") != 1 or type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError(f"Invalid tranche marker: {path}")
    offset = start if record["tranche_id"] == identity else end
    if consumed_total < offset:
        raise ValueError(
            f"Checkpoint consumed samples ({consumed_total}) precede tranche offset ({offset}) "
            f"in {path}; load the boundary checkpoint or set --tranche-start-samples explicitly"
        )
    return offset


def write_tranche_end(
    checkpoint_dir: str, identity: str, start: int, end: int, iteration: int
) -> None:
    """Atomically publish completion only after the boundary checkpoint is durable."""
    record = dict(version=1, tranche_id=identity, start_consumed_samples=start,
                  end_consumed_samples=end, checkpoint_iteration=iteration)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".tranche_end-", delete=False) as output:
            temporary = output.name
            json.dump(record, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, directory / MARKER_NAME)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def tranche_end_samples(
    total_samples: int,
    consumed_in_tranche: int,
    consumed_total: int,
    global_batch_size: int,
) -> int:
    """Return the absolute boundary reachable without an incomplete global batch.

    A tranche-local counter can restart while the checkpoint's global counter
    and learning-rate schedule remain continuous. Replays of a cached batch do
    not advance either counter.
    """
    if total_samples <= 0 or global_batch_size <= 0:
        raise ValueError("Tranche size and global batch size must be positive")
    if consumed_in_tranche < 0 or consumed_total < consumed_in_tranche:
        raise ValueError("Invalid tranche consumption counters")
    remaining = max(0, total_samples - consumed_in_tranche)
    return consumed_total + remaining // global_batch_size * global_batch_size


def tranche_finished(enabled: bool, boundary: Optional[int], consumed: int) -> bool:
    """Whether training must stop before fetching another batch."""
    return enabled and boundary is not None and consumed >= boundary
