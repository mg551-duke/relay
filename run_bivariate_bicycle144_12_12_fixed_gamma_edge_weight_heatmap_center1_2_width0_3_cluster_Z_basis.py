#!/usr/bin/env python3
"""Preset wrapper for the fixed-gamma edge-weight circuit-level heatmap run.

This reuses the general Relay-BP-1 fixed-gamma edge-weight heatmap runner, but
sets the experiment defaults to:

- gamma values: 0.125, 0.250, 0.375
- center range: [1, 2] with 11 grid points
- width range: [0, 3] with 11 grid points
"""

from __future__ import annotations

import sys
from pathlib import Path

import run_bivariate_bicycle144_12_12_fixed_gamma_edge_weight_heatmap_cluster_Z_basis as base


DEFAULT_ARG_PAIRS = [
    ("--gamma-values", "0.125,0.25,0.375"),
    ("--center-min", "1.0"),
    ("--center-max", "2.0"),
    ("--center-count", "11"),
    ("--width-min", "0.0"),
    ("--width-max", "3.0"),
    ("--width-count", "11"),
]

DEFAULT_OUTPUT_STEM = (
    "bicycle_bivariate_144_12_12_relay_bp1_r31_fixed_gamma_"
    "edge_weight_heatmap_center1_2_width0_3_cluster_Z_basis"
)


def _has_flag(argv: list[str], flag: str) -> bool:
    return flag in argv


def _with_preset_defaults(argv: list[str]) -> list[str]:
    updated = list(argv)
    for flag, value in DEFAULT_ARG_PAIRS:
        if not _has_flag(updated, flag):
            updated.extend([flag, value])
    if not _has_flag(updated, "--output-dir"):
        repo_root = base.find_repo_root(Path.cwd().resolve())
        output_dir = repo_root / "examples" / "notebook_data" / DEFAULT_OUTPUT_STEM
        updated.extend(["--output-dir", str(output_dir)])
    return updated


def main() -> None:
    sys.argv = [sys.argv[0], *_with_preset_defaults(sys.argv[1:])]
    base.main()


if __name__ == "__main__":
    main()
