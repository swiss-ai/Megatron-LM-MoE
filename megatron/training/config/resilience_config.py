# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import Literal

@dataclass(kw_only=True)
class RerunStateMachineConfig:
    """Configuration for the rerun state machine used for result validation or stats."""

    error_injection_rate: int = 0
    """Rate at which to inject unexpected results, e.g. 1000 means
    once every 1000 result validations"""

    error_injection_type: Literal["correct_result", "transient_error", "persistent_error"] = "transient_error"
    """Type of error to inject. """

    rerun_mode: Literal["disabled", "validate_results", "report_stats"] = "validate_results"
    """Use re-run engine to validate results (default) or to emit stats
    on variability of computations due to non-deterministic algorithms."""

    rerun_strategy: Literal["rerun_in_place", "skip_iteration"] = "rerun_in_place"
    """What to do when a result is rejected. 'rerun_in_place' (default) replays the iteration.
    'skip_iteration' instead discards the current global batch and reruns the
    forward-backward pass on the next batch."""

    check_for_nan_in_loss: bool = True
    """Check for NaN in the loss."""

    check_for_spiky_loss: bool = False
    """Check for spiky loss."""

    check_grad_norm: bool = False
    """Check for spiky grad norm. Only supported by the md_decoupling optimizer."""

    check_grad_norm_threshold: float = 5.0
    """Threshold for spiky grad norm detection."""


@dataclass(kw_only=True)
class StragglerDetectionConfig:
    """Configuration settings for detecting and logging GPU stragglers."""

    log_straggler: bool = False
    """If set, tracks and logs straggler per GPU."""

    straggler_ctrlr_port: int = 65535
    """Port number to toggle StragglerDetector on/off at runtime"""

    straggler_minmax_count: int = 1
    """Number of ranks to report with high/low estimated throughput"""

    disable_straggler_on_startup: bool = False
    """If set, StragglerDetector is disabled on startup."""

