#!/usr/bin/env python3
"""Run Relay-BP-5 performance for the best refined message-mix result.

This is a thin wrapper around
``run_bivariate_bicycle144_12_12_relay_bp5_performance_cluster_Z_basis.py``.
It selects the best ``best_params.json`` under the message-mix refinement output
base, then runs the no-gamma and fixed-gamma message-mix Relay-BP-5 decoders
over the standard p sweep. The wrapped runner records mean iteration counts
through the same detailed Relay-BP sampler used by the other BP5 performance
runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DECODER_NAMES = [
    "relay-bp5-r601-message-mix-no-gamma",
    "relay-bp5-r601-message-mix-gamma0p125",
]
DEFAULT_P_VALUES = "0.001,0.002,0.003,0.004,0.005"
DEFAULT_REFINEMENT_BASE = (
    "examples/notebook_data/"
    "bicycle_bivariate_144_12_12_message_mix_grid_refine_Z_p003_training"
)
DEFAULT_OUTPUT_DIR = (
    "examples/notebook_data/"
    "bicycle_bivariate_144_12_12_message_mix_grid_refine_"
    "relay_bp5_performance_cluster_Z_basis"
)
DEFAULT_RUNNER = (
    "run_bivariate_bicycle144_12_12_relay_bp5_performance_cluster_Z_basis.py"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. Defaults to auto-detection from this script path.",
    )
    parser.add_argument(
        "--refinement-base",
        type=Path,
        default=None,
        help=(
            "Message-mix refinement output base containing seed_*/best_params.json. "
            f"Defaults to {DEFAULT_REFINEMENT_BASE}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Performance output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(DEFAULT_RUNNER),
        help="Underlying Relay-BP5 performance runner.",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--p-values", default=DEFAULT_P_VALUES)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--reset-data", action="store_true")
    parser.add_argument("--print-progress", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the wrapped command without running it.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    repo_root = (
        args.repo_root.resolve()
        if args.repo_root is not None
        else find_repo_root(Path(__file__).resolve())
    )
    runner = resolve_under_repo(args.runner, repo_root)
    refinement_base = resolve_under_repo(
        args.refinement_base or Path(DEFAULT_REFINEMENT_BASE), repo_root
    )
    output_dir = resolve_under_repo(
        args.output_dir or Path(DEFAULT_OUTPUT_DIR), repo_root
    )

    command = [
        args.python_bin,
        str(runner),
        "--repo-root",
        str(repo_root),
        "--output-dir",
        str(output_dir),
        "--message-mix-best-params-path",
        str(refinement_base),
        "--decoders",
        *DECODER_NAMES,
        "--p-values",
        args.p_values,
        "--max-batch-size",
        str(args.max_batch_size),
    ]
    if args.reset_data:
        command.append("--reset-data")
    if args.print_progress:
        command.append("--print-progress")
    command.extend(passthrough)

    print(
        json.dumps(
            {
                "repo_root": str(repo_root),
                "runner": str(runner),
                "refinement_base": str(refinement_base),
                "output_dir": str(output_dir),
                "decoders": DECODER_NAMES,
                "p_values": args.p_values,
                "command": command,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    subprocess.run(command, check=True)


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "tests" / "testdata" / "bicycle_bivariate").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root containing tests/testdata/bicycle_bivariate "
        f"starting from {start}."
    )


def resolve_under_repo(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":
    main()
