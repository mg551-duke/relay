#!/usr/bin/env python3
"""Analyze partial or complete message-mix grid-search outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

PARAMETER_NAMES = [
    "fresh_negative",
    "fresh_positive",
    "fresh_p_positive",
    "previous_negative",
    "previous_positive",
    "previous_p_positive",
]

SHORT_NAMES = {
    "fresh_negative": "fresh -",
    "fresh_positive": "fresh +",
    "fresh_p_positive": "fresh p+",
    "previous_negative": "prev -",
    "previous_positive": "prev +",
    "previous_p_positive": "prev p+",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-output-base", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to GRID_OUTPUT_BASE/analysis.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--max-neighbor-jumps",
        type=int,
        default=200,
        help="Rows to keep in largest_neighbor_jumps.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, source_files = load_grid_rows(args.grid_output_base)
    if rows.empty:
        raise SystemExit(f"No grid result rows found under {args.grid_output_base}")
    rows = deduplicate_rows(rows)
    rows = rows.sort_values(
        ["logical_failure_rate", "mean_iterations", "convergence_rate"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    output_dir = args.output_dir or (args.grid_output_base / "analysis")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = load_configs(args.grid_output_base)
    summary = coverage_summary(rows, configs=configs, source_files=source_files)
    write_json(output_dir / "grid_coverage_summary.json", summary)
    rows.to_csv(output_dir / "grid_results_merged.csv", index=False)
    rows.head(args.top_k).to_csv(output_dir / "top_candidates.csv", index=False)

    jumps = neighbor_jumps(rows)
    jumps.to_csv(output_dir / "neighbor_jumps.csv", index=False)
    jumps.head(args.max_neighbor_jumps).to_csv(
        output_dir / "largest_neighbor_jumps.csv",
        index=False,
    )

    plot_pairwise_projection(
        rows,
        value="logical_failure_rate",
        reducer="min",
        output_path=plots_dir / "pairwise_best_logical_error_rate.png",
        title="Best logical error rate seen in each 2D projection",
        cmap="viridis_r",
        log_scale=True,
    )
    plot_pairwise_projection(
        rows,
        value="convergence_rate",
        reducer="mean",
        output_path=plots_dir / "pairwise_mean_convergence_rate.png",
        title="Mean convergence rate in each 2D projection",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plot_pairwise_projection(
        rows,
        value="candidate_index",
        reducer="count",
        output_path=plots_dir / "pairwise_evaluated_counts.png",
        title="Evaluated candidate count in each 2D projection",
        cmap="magma",
    )
    plot_best_point_slices(
        rows,
        output_path=plots_dir / "best_point_logical_error_slices.png",
    )
    plot_neighbor_jump_bars(
        jumps,
        output_path=plots_dir / "largest_neighbor_jumps.png",
        max_rows=min(30, args.max_neighbor_jumps),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_dir}")


def load_grid_rows(base: Path) -> tuple[pd.DataFrame, list[Path]]:
    base = base.resolve()
    if not base.exists():
        raise FileNotFoundError(missing_base_message(base))
    files = grid_result_files(base)
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frame["source_shard"] = shard_from_path(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), []
    rows = pd.concat(frames, ignore_index=True)
    for name in PARAMETER_NAMES:
        rows[name] = rows[name].astype(float)
    for name in [
        "logical_failure_rate",
        "mean_iterations",
        "convergence_rate",
        "trials",
        "logical_failures",
    ]:
        if name in rows.columns:
            rows[name] = rows[name].astype(float)
    if "candidate_index" in rows.columns:
        rows["candidate_index"] = rows["candidate_index"].astype(int)
    return rows, files


def missing_base_message(base: Path) -> str:
    message = [f"Grid output base does not exist: {base}"]
    parent = base.parent
    if parent.exists():
        matches = sorted(
            path
            for pattern in ("*message_mix*grid*", "*msgmix*grid*", "*grid*")
            for path in parent.glob(pattern)
            if path.is_dir()
        )
        seen = []
        for path in matches:
            if path not in seen:
                seen.append(path)
        if seen:
            message.append("Nearby candidate directories:")
            message.extend(f"  {path}" for path in seen[:20])
        else:
            message.append(
                f"No nearby grid-like directories found under existing parent: {parent}"
            )
    else:
        message.append(f"Parent directory also does not exist: {parent}")
    message.append(
        "Find the actual grid output with: "
        "find examples/notebook_data -path '*/seed_*/grid_results.csv' -print"
    )
    return "\n".join(message)


def grid_result_files(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    files = []
    for directory in [
        base,
        *sorted(path for path in base.glob("seed_*") if path.is_dir()),
    ]:
        grid_csv = directory / "grid_results.csv"
        sorted_csv = directory / "grid_results_sorted.csv"
        if grid_csv.exists():
            files.append(grid_csv)
        elif sorted_csv.exists():
            files.append(sorted_csv)
    return files


def shard_from_path(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", str(path))
    return int(match.group(1)) if match else None


def load_configs(base: Path) -> list[dict[str, Any]]:
    if base.is_file():
        return []
    configs = []
    for path in sorted(
        [base / "grid_search_config.json", *base.glob("seed_*/grid_search_config.json")]
    ):
        if path.exists():
            with path.open("r", encoding="utf-8-sig") as f:
                payload = json.load(f)
            payload["_path"] = str(path)
            configs.append(payload)
    return configs


def deduplicate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        "candidate_index",
        "logical_failure_rate",
        "mean_iterations",
        "convergence_rate",
    ]
    present = [column for column in sort_columns if column in rows.columns]
    rows = rows.sort_values(
        present,
        ascending=[True, True, True, False][: len(present)],
    )
    if "candidate_index" in rows.columns:
        return rows.drop_duplicates(["candidate_index"], keep="first").reset_index(
            drop=True
        )
    return rows.drop_duplicates(PARAMETER_NAMES, keep="first").reset_index(drop=True)


def coverage_summary(
    rows: pd.DataFrame,
    *,
    configs: list[dict[str, Any]],
    source_files: list[Path],
) -> dict[str, Any]:
    full_grid_sizes = [
        int(config["full_grid_size"])
        for config in configs
        if config.get("full_grid_size") is not None
    ]
    full_grid_size = (
        max(full_grid_sizes) if full_grid_sizes else inferred_grid_size(rows)
    )
    evaluated = (
        int(rows["candidate_index"].nunique())
        if "candidate_index" in rows
        else len(rows)
    )
    per_shard = (
        rows.groupby("source_shard", dropna=False)
        .size()
        .reset_index(name="evaluated_candidates")
        .to_dict(orient="records")
    )
    top = rows.sort_values(
        ["logical_failure_rate", "mean_iterations", "convergence_rate"],
        ascending=[True, True, False],
    ).head(10)
    return {
        "grid_output_files": [str(path) for path in source_files],
        "config_files": [config.get("_path") for config in configs],
        "evaluated_candidates": evaluated,
        "full_grid_size": int(full_grid_size),
        "coverage_fraction": (
            float(evaluated / full_grid_size) if full_grid_size else None
        ),
        "parameter_values_seen": {
            name: sorted(float(value) for value in rows[name].dropna().unique())
            for name in PARAMETER_NAMES
        },
        "per_shard": per_shard,
        "best_candidates": top.to_dict(orient="records"),
    }


def inferred_grid_size(rows: pd.DataFrame) -> int:
    size = 1
    for name in PARAMETER_NAMES:
        size *= max(1, int(rows[name].nunique()))
    return size


def neighbor_jumps(rows: pd.DataFrame) -> pd.DataFrame:
    key_to_row = {
        tuple(float(row[name]) for name in PARAMETER_NAMES): row
        for row in rows.to_dict(orient="records")
    }
    values_by_param = {
        name: sorted(float(value) for value in rows[name].dropna().unique())
        for name in PARAMETER_NAMES
    }
    jumps = []
    for key, left in key_to_row.items():
        key_list = list(key)
        for dim, name in enumerate(PARAMETER_NAMES):
            values = values_by_param[name]
            index = values.index(key[dim])
            if index + 1 >= len(values):
                continue
            key_list[dim] = values[index + 1]
            right_key = tuple(key_list)
            key_list[dim] = key[dim]
            right = key_to_row.get(right_key)
            if right is None:
                continue
            logical_delta = float(right["logical_failure_rate"]) - float(
                left["logical_failure_rate"]
            )
            convergence_delta = float(right["convergence_rate"]) - float(
                left["convergence_rate"]
            )
            jumps.append(
                {
                    "parameter": name,
                    "from_value": float(left[name]),
                    "to_value": float(right[name]),
                    "candidate_index_left": int(left.get("candidate_index", -1)),
                    "candidate_index_right": int(right.get("candidate_index", -1)),
                    "logical_failure_left": float(left["logical_failure_rate"]),
                    "logical_failure_right": float(right["logical_failure_rate"]),
                    "logical_failure_delta": logical_delta,
                    "abs_logical_failure_delta": abs(logical_delta),
                    "convergence_left": float(left["convergence_rate"]),
                    "convergence_right": float(right["convergence_rate"]),
                    "convergence_delta": convergence_delta,
                    "abs_convergence_delta": abs(convergence_delta),
                    "mean_iterations_left": float(left["mean_iterations"]),
                    "mean_iterations_right": float(right["mean_iterations"]),
                }
                | {
                    f"fixed_{param}": float(left[param])
                    for param in PARAMETER_NAMES
                    if param != name
                }
            )
    if not jumps:
        return pd.DataFrame()
    return pd.DataFrame(jumps).sort_values(
        ["abs_logical_failure_delta", "abs_convergence_delta"],
        ascending=[False, False],
    )


def plot_pairwise_projection(
    rows: pd.DataFrame,
    *,
    value: str,
    reducer: str,
    output_path: Path,
    title: str,
    cmap: str,
    log_scale: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    pairs = parameter_pairs()
    fig, axes = plt.subplots(5, 3, figsize=(15, 20), squeeze=False)
    for ax, (x_name, y_name) in zip(axes.ravel(), pairs, strict=False):
        x_values, y_values, matrix = aggregate_pairwise(
            rows,
            x_name=x_name,
            y_name=y_name,
            value=value,
            reducer=reducer,
        )
        plot_matrix(
            ax,
            x_values=x_values,
            y_values=y_values,
            matrix=matrix,
            x_label=SHORT_NAMES[x_name],
            y_label=SHORT_NAMES[y_name],
            cmap=cmap,
            log_scale=log_scale,
            vmin=vmin,
            vmax=vmax,
        )
    for ax in axes.ravel()[len(pairs) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, y=1.002)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_point_slices(rows: pd.DataFrame, *, output_path: Path) -> None:
    best = rows.sort_values(
        ["logical_failure_rate", "mean_iterations", "convergence_rate"],
        ascending=[True, True, False],
    ).iloc[0]
    pairs = parameter_pairs()
    fig, axes = plt.subplots(5, 3, figsize=(15, 20), squeeze=False)
    for ax, (x_name, y_name) in zip(axes.ravel(), pairs, strict=False):
        fixed = [name for name in PARAMETER_NAMES if name not in {x_name, y_name}]
        mask = np.ones(len(rows), dtype=bool)
        for name in fixed:
            mask &= np.isclose(rows[name].to_numpy(dtype=float), float(best[name]))
        slice_df = rows.loc[mask].copy()
        if slice_df.empty:
            ax.axis("off")
            continue
        x_values, y_values, matrix = aggregate_pairwise(
            slice_df,
            x_name=x_name,
            y_name=y_name,
            value="logical_failure_rate",
            reducer="min",
        )
        plot_matrix(
            ax,
            x_values=x_values,
            y_values=y_values,
            matrix=matrix,
            x_label=SHORT_NAMES[x_name],
            y_label=SHORT_NAMES[y_name],
            cmap="viridis_r",
            log_scale=True,
        )
        ax.set_title(f"slice at best, n={len(slice_df)}", fontsize=9)
    for ax in axes.ravel()[len(pairs) :]:
        ax.axis("off")
    fig.suptitle(
        "Logical error slices through the current best grid point", fontsize=14, y=1.002
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_neighbor_jump_bars(
    jumps: pd.DataFrame, *, output_path: Path, max_rows: int
) -> None:
    if jumps.empty:
        return
    top = jumps.head(max_rows).copy()
    labels = [
        f"{SHORT_NAMES[row.parameter]} {row.from_value:g}->{row.to_value:g}"
        for row in top.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.28 * len(top))))
    y = np.arange(len(top))
    ax.barh(y, top["abs_logical_failure_delta"].to_numpy(dtype=float), color="#3268a8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("|neighbor logical-error-rate jump|")
    ax.set_title("Largest local jumps between adjacent grid values")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_pairwise(
    rows: pd.DataFrame,
    *,
    x_name: str,
    y_name: str,
    value: str,
    reducer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.asarray(sorted(rows[x_name].dropna().unique()), dtype=float)
    y_values = np.asarray(sorted(rows[y_name].dropna().unique()), dtype=float)
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    grouped = rows.groupby([y_name, x_name], dropna=False)
    for (y_value, x_value), group in grouped:
        y_idx = int(np.where(np.isclose(y_values, float(y_value)))[0][0])
        x_idx = int(np.where(np.isclose(x_values, float(x_value)))[0][0])
        if reducer == "min":
            matrix[y_idx, x_idx] = float(group[value].min())
        elif reducer == "mean":
            matrix[y_idx, x_idx] = float(group[value].mean())
        elif reducer == "count":
            matrix[y_idx, x_idx] = float(len(group))
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
    return x_values, y_values, matrix


def plot_matrix(
    ax: plt.Axes,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    matrix: np.ndarray,
    x_label: str,
    y_label: str,
    cmap: str,
    log_scale: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if x_values.size == 0 or y_values.size == 0 or np.all(~np.isfinite(matrix)):
        ax.axis("off")
        return
    data = matrix.copy()
    norm = None
    if log_scale:
        finite_positive = data[np.isfinite(data) & (data > 0)]
        if finite_positive.size:
            floor = min(float(np.nanmin(finite_positive)), 1.0e-6)
            upper = float(np.nanmax(finite_positive))
        else:
            floor = 1.0e-6
            upper = 1.0e-5
        data[np.isfinite(data) & (data <= 0)] = floor
        if upper <= floor:
            upper = floor * 10.0
        norm = LogNorm(vmin=floor, vmax=upper)
    mesh = ax.pcolormesh(
        heatmap_edges(x_values),
        heatmap_edges(y_values),
        data,
        shading="auto",
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)


def heatmap_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        delta = max(abs(float(values[0])) * 0.05, 0.05)
        return np.asarray([values[0] - delta, values[0] + delta], dtype=float)
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        [
            [values[0] - (midpoints[0] - values[0])],
            midpoints,
            [values[-1] + (values[-1] - midpoints[-1])],
        ]
    )


def parameter_pairs() -> list[tuple[str, str]]:
    pairs = []
    for y_idx, y_name in enumerate(PARAMETER_NAMES):
        for x_name in PARAMETER_NAMES[:y_idx]:
            pairs.append((x_name, y_name))
    return pairs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
