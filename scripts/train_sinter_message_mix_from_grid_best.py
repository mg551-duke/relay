#!/usr/bin/env python3
"""Start message-mix CEM training from the best grid-search regions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PARAMETER_NAMES = [
    "fresh_negative",
    "fresh_positive",
    "fresh_p_positive",
    "previous_negative",
    "previous_positive",
    "previous_p_positive",
]

BOUNDS = {
    "fresh_negative": (-1.0, 0.0),
    "fresh_positive": (0.0, 1.0),
    "fresh_p_positive": (0.001, 0.999),
    "previous_negative": (-1.0, 0.0),
    "previous_positive": (0.0, 1.0),
    "previous_p_positive": (0.001, 0.999),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select one of the top grid-search message-mix candidates and launch "
            "the standard CEM trainer with narrowed bounds around it."
        )
    )
    parser.add_argument("--grid-output-base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--rank-count", type=int, default=5)
    parser.add_argument("--runs-per-candidate", type=int, default=2)
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("scripts/train_sinter_bernoulli_message_mix_distribution.py"),
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--warm-start-raw-std", type=float, default=0.7)
    parser.add_argument("--fresh-negative-radius", type=float, default=0.12)
    parser.add_argument("--fresh-positive-radius", type=float, default=0.12)
    parser.add_argument("--fresh-probability-radius", type=float, default=0.08)
    parser.add_argument("--previous-negative-radius", type=float, default=0.18)
    parser.add_argument("--previous-positive-radius", type=float, default=0.18)
    parser.add_argument("--previous-probability-radius", type=float, default=0.20)
    parser.add_argument(
        "--force-warm-start",
        action="store_true",
        help="Overwrite an existing checkpoint_latest.json before launching training.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    _validate_args(args)

    grid_rows = load_grid_rows(args.grid_output_base)
    selected_rows = select_best_unique(grid_rows, top_k=args.rank_count)
    if len(selected_rows) < args.rank_count:
        raise SystemExit(
            f"Only found {len(selected_rows)} unique grid candidates; "
            f"need {args.rank_count}."
        )

    candidate_rank = args.task_index // args.runs_per_candidate
    repeat_index = args.task_index % args.runs_per_candidate
    if candidate_rank >= args.rank_count:
        raise SystemExit(
            f"task-index {args.task_index} is outside rank-count "
            f"{args.rank_count} with runs-per-candidate {args.runs_per_candidate}."
        )

    selected = selected_rows[candidate_rank]
    params = {name: float(selected[name]) for name in PARAMETER_NAMES}
    narrowed = narrowed_bounds(params, args)
    raw_mean = [
        raw_from_bounded(params[name], narrowed[name][0], narrowed[name][1])
        for name in PARAMETER_NAMES
    ]
    raw_std = [float(args.warm_start_raw_std)] * len(PARAMETER_NAMES)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_warm_start_files(
        args=args,
        selected=selected,
        params=params,
        narrowed=narrowed,
        raw_mean=raw_mean,
        raw_std=raw_std,
        candidate_rank=candidate_rank,
        repeat_index=repeat_index,
    )

    train_script = args.train_script
    if not train_script.is_absolute():
        train_script = Path.cwd() / train_script
    command = [
        args.python_bin,
        str(train_script),
        "--output-dir",
        str(args.output_dir),
        "--fresh-negative-min",
        str(narrowed["fresh_negative"][0]),
        "--fresh-negative-max",
        str(narrowed["fresh_negative"][1]),
        "--fresh-positive-min",
        str(narrowed["fresh_positive"][0]),
        "--fresh-positive-max",
        str(narrowed["fresh_positive"][1]),
        "--fresh-probability-min",
        str(narrowed["fresh_p_positive"][0]),
        "--fresh-probability-max",
        str(narrowed["fresh_p_positive"][1]),
        "--previous-negative-min",
        str(narrowed["previous_negative"][0]),
        "--previous-negative-max",
        str(narrowed["previous_negative"][1]),
        "--previous-positive-min",
        str(narrowed["previous_positive"][0]),
        "--previous-positive-max",
        str(narrowed["previous_positive"][1]),
        "--previous-probability-min",
        str(narrowed["previous_p_positive"][0]),
        "--previous-probability-max",
        str(narrowed["previous_p_positive"][1]),
        *passthrough,
    ]
    print(
        json.dumps(
            {
                "candidate_rank": candidate_rank,
                "repeat_index": repeat_index,
                "selected_candidate_index": int(selected["candidate_index"]),
                "selected_params": params,
                "narrowed_bounds": narrowed,
                "output_dir": str(args.output_dir),
                "command": command,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.run(command, check=True)


def load_grid_rows(base: Path) -> list[dict[str, Any]]:
    base = base.resolve()
    if not base.exists():
        raise FileNotFoundError(f"Grid output base does not exist: {base}")
    candidates: list[Path] = []
    if base.is_file():
        candidates.append(base)
    else:
        candidates.extend(sorted(base.glob("grid_results_sorted.csv")))
        candidates.extend(sorted(base.glob("grid_results.csv")))
        candidates.extend(sorted(base.glob("seed_*/grid_results_sorted.csv")))
        candidates.extend(sorted(base.glob("seed_*/grid_results.csv")))
    rows: list[dict[str, Any]] = []
    seen_files: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen_files or not resolved.exists():
            continue
        seen_files.add(resolved)
        rows.extend(read_grid_csv(resolved))
    if not rows:
        raise FileNotFoundError(f"No grid result rows found under {base}")
    return rows


def read_grid_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {"source_csv": str(path)}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                if key in {"stopped_early"}:
                    parsed[key] = value.lower() == "true"
                elif key in {
                    "candidate_index",
                    "local_index",
                    "shots",
                    "repeats",
                    "trials",
                    "logical_failures",
                    "target_logical_errors",
                    "converged",
                }:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def select_best_unique(
    rows: list[dict[str, Any]], *, top_k: int
) -> list[dict[str, Any]]:
    unique: dict[tuple[float, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(round(float(row[name]), 12) for name in PARAMETER_NAMES)
        current = unique.get(key)
        if current is None or rank_key(row) < rank_key(current):
            unique[key] = row
    return sorted(unique.values(), key=rank_key)[:top_k]


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row.get("logical_failure_rate", float("inf"))),
        float(row.get("mean_iterations", float("inf"))),
        -float(row.get("convergence_rate", 0.0)),
        int(row.get("candidate_index", 0)),
    )


def narrowed_bounds(
    params: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, tuple[float, float]]:
    radii = {
        "fresh_negative": args.fresh_negative_radius,
        "fresh_positive": args.fresh_positive_radius,
        "fresh_p_positive": args.fresh_probability_radius,
        "previous_negative": args.previous_negative_radius,
        "previous_positive": args.previous_positive_radius,
        "previous_p_positive": args.previous_probability_radius,
    }
    narrowed = {}
    for name in PARAMETER_NAMES:
        global_low, global_high = BOUNDS[name]
        radius = float(radii[name])
        low = max(global_low, float(params[name]) - radius)
        high = min(global_high, float(params[name]) + radius)
        if high <= low:
            raise ValueError(f"Invalid narrowed bound for {name}: {(low, high)}")
        narrowed[name] = (low, high)
    return narrowed


def raw_from_bounded(value: float, low: float, high: float) -> float:
    fraction = (float(value) - float(low)) / (float(high) - float(low))
    fraction = min(max(fraction, 1.0e-9), 1.0 - 1.0e-9)
    return math.log(fraction / (1.0 - fraction))


def write_warm_start_files(
    *,
    args: argparse.Namespace,
    selected: dict[str, Any],
    params: dict[str, float],
    narrowed: dict[str, tuple[float, float]],
    raw_mean: list[float],
    raw_std: list[float],
    candidate_rank: int,
    repeat_index: int,
) -> None:
    config_payload = {
        "grid_output_base": str(args.grid_output_base),
        "candidate_rank": int(candidate_rank),
        "repeat_index": int(repeat_index),
        "task_index": int(args.task_index),
        "runs_per_candidate": int(args.runs_per_candidate),
        "selected_grid_row": json_ready(selected),
        "selected_params": params,
        "narrowed_bounds": {key: list(value) for key, value in narrowed.items()},
        "initial_raw_mean": raw_mean,
        "initial_raw_std": raw_std,
    }
    write_json(args.output_dir / "warm_start_config.json", config_payload)

    checkpoint_path = args.output_dir / "checkpoint_latest.json"
    best_path = args.output_dir / "best_params.json"
    if best_path.exists():
        return
    if checkpoint_path.exists() and not args.force_warm_start:
        return
    write_json(
        checkpoint_path,
        {
            "completed_generations": 0,
            "final_raw_mean": raw_mean,
            "final_raw_std": raw_std,
            "generation_records": [],
            "finalized": False,
            "warm_start_config": config_payload,
        },
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_args(args: argparse.Namespace) -> None:
    if args.rank_count <= 0:
        raise ValueError("--rank-count must be positive.")
    if args.runs_per_candidate <= 0:
        raise ValueError("--runs-per-candidate must be positive.")
    if args.task_index < 0:
        raise ValueError("--task-index must be non-negative.")
    if args.warm_start_raw_std <= 0.0 or not math.isfinite(args.warm_start_raw_std):
        raise ValueError("--warm-start-raw-std must be positive and finite.")
    for name in [
        "fresh_negative_radius",
        "fresh_positive_radius",
        "fresh_probability_radius",
        "previous_negative_radius",
        "previous_positive_radius",
        "previous_probability_radius",
    ]:
        value = float(getattr(args, name))
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")


if __name__ == "__main__":
    main()
