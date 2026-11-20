# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch
import torch.distributed

from megatron.core import config
from megatron.core.utils import is_te_min_version
from tests.test_utils.python_scripts.download_unit_tests_dataset import download_and_extract_asset
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def pytest_addoption(parser):
    """
    Additional command-line arguments passed to pytest.
    For now:
        --experimental: Enable the Mcore experimental flag (DEFAULT: False)
    """
    parser.addoption(
        '--experimental',
        action='store_true',
        help="pass that argument to enable experimental flag during testing (DEFAULT: False)",
    )


def pytest_sessionstart(session):
    # Bind before collection can allocate CUDA tensors or cache capture streams.
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


@pytest.fixture(autouse=True)
def experimental(request):
    """Simple fixture setting the experimental flag [CPU | GPU]"""
    config.ENABLE_EXPERIMENTAL = request.config.getoption("--experimental") is True


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


def _trace_event(event):
    if os.environ.get('UNIT_TEST_TRACE') == 'true':
        memory = ''
        if torch.cuda.is_initialized():
            memory = (
                f' cuda_allocated={torch.cuda.memory_allocated()}'
                f' cuda_reserved={torch.cuda.memory_reserved()}'
            )
        print(
            f"[unit-trace {datetime.now(timezone.utc).isoformat(timespec='milliseconds')}] "
            f"rank={os.environ.get('RANK', '0')} monotonic={time.monotonic():.6f} {event}{memory}",
            file=sys.stderr,
            flush=True,
        )


def pytest_runtest_logstart(nodeid, location):
    _trace_event(f"start {nodeid}")


def pytest_runtest_logreport(report):
    _trace_event(f"{report.when} {report.outcome} {report.nodeid}")
    # Log failures before distributed cleanup can block the final summary.
    if report.failed:
        print(
            f"\nRank {os.environ.get('RANK', '0')}: {report.nodeid} ({report.when})\n"
            f"{report.longreprtext}",
            file=sys.stderr,
            flush=True,
        )


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    initialized = torch.distributed.is_initialized()
    _trace_event(f"session cleanup initialized={initialized}")
    if initialized:
        _trace_event("session cleanup barrier enter")
        try:
            torch.distributed.barrier(timeout=timedelta(seconds=300))
        except Exception as error:
            _trace_event(f"session cleanup barrier error: {type(error).__name__}: {error}")
            return
        _trace_event("session cleanup barrier exit")
        _trace_event("session cleanup destroy enter")
        torch.distributed.destroy_process_group()
        _trace_event("session cleanup destroy exit")


@pytest.fixture(scope="function", autouse=True)
def set_env():
    if is_te_min_version("1.3"):
        os.environ['NVTE_FLASH_ATTN'] = '0'
        os.environ['NVTE_FUSED_ATTN'] = '0'


@pytest.fixture(scope="session")
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.

    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """

    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'

    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
            yield tmp_dir
            _trace_event("shared checkpoint fixture cleanup enter")
        _trace_event("shared checkpoint fixture cleanup exit")

    else:
        yield tmp_dir
        _trace_event("shared checkpoint fixture non-owner exit")


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Ensure test data is available at /opt/data by downloading if necessary."""
    data_path = Path("/opt/data")

    # Check if data directory exists and has content
    if not data_path.exists() or not any(data_path.iterdir()):
        print("Test data not found at /opt/data. Downloading...")

        try:
            # Download assets to /opt/data
            download_and_extract_asset(assets_dir=data_path)

            print("Test data downloaded successfully.")

        except ImportError as e:
            print(f"Failed to import download function: {e}")
            # Don't fail the tests, just warn
        except Exception as e:
            print(f"Failed to download test data: {e}")
            # Don't fail the tests, just warn
    else:
        print("Test data already available at /opt/data")


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables"""
    # Store the original environment variables before the test
    original_env = dict(os.environ)

    # Run the test
    yield

    # After the test, restore the original environment
    os.environ.clear()
    os.environ.update(original_env)
