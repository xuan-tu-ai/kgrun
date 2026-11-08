"""
Utility functions for sample project.
"""

from __future__ import annotations

import os
import sys


def get_environment_info() -> dict[str, str]:
    """
    Collect execution runtime variables and paths.
    """
    return {
        "python_version": sys.version,
        "kgrun_out": os.environ.get("KGRUN_OUT", "outputs"),
        "kgrun_data": os.environ.get("KGRUN_DATA", "not_mounted"),
    }


def compute_epoch_metric(epoch: int, base_value: float) -> float:
    """
    Calculate simulated training metric for an epoch.
    """
    return round(base_value / (epoch + 1), 4)

