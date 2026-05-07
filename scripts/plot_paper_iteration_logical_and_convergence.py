#!/usr/bin/env python3
"""Plot logical error and convergence heatmaps for the paper iteration run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

DEFAULT_OUTPUT_SUBDIR = (
    "examples/notebook_data/"
    "bicycle_bivariate_144_12_12_paper_iteration_heatmap_cluster_Z_basis"
)


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "tests" / "testdata" / "bicycle_bivariate").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root containing tests/testdata/bicycle_bivariate "
        f"starting from {start}."
    )


def infer_single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {label} matching {pattern!r} found in {directory}."
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple {label} files found in {directory}: {matches}")
    return matches[0]


def load_setup(output_dir: Path) -> dict[str, Any]:
    setup_paths = sorted(output_dir.glob("*_setup.json"))
    if not setup_paths:
        return {}
    if len(setup_paths) > 1:
        raise ValueError(
            f"Multiple setup JSON files found in {output_dir}: {setup_paths}"
        )
    with setup_paths[0].open("r", encoding="utf-8") as f:
        return json.load(f)


def heatmap_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Heatmap values must be a non-empty one-dimensional array.")
    if values.size == 1:
        delta = 0.5 if values[0] == 0 else abs(values[0]) * 0.5
        return np.asarray([values[0] - delta, values[0] + delta], dtype=np.float64)
    midpoints = (values[:-1] + values[1:]) / 2.0
    first_edge = values[0] - (midpoints[0] - values[0])
    last_edge = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first_edge], midpoints, [last_edge]))


def logical_error_lognorm(
    logical_error_rate: np.ndarray,
    *,
    max_shots_per_point: int,
) -> tuple[np.ndarray, LogNorm]:
    display_values = np.asarray(logical_error_rate, dtype=np.float64).copy()
    finite_mask = np.isfinite(display_values)
    positive_mask = finite_mask & (display_values > 0.0)
    floor = 0.5 / max(max_shots_per_point, 1)
    display_values[finite_mask & (display_values <= 0.0)] = floor
    if np.any(positive_mask):
        vmin = min(float(np.nanmin(display_values[positive_mask])), floor)
        vmax = float(np.nanmax(display_values[finite_mask]))
        if not math.isfinite(vmax) or vmax <= 0.0:
            vmax = max(vmin * 10.0, floor * 10.0)
    else:
        vmin = floor
        vmax = floor * 10.0
    if vmax <= vmin:
        vmax = vmin * 1.01
    return display_values, LogNorm(vmin=vmin, vmax=vmax)


def interval_to_center_width(low: float, high: float) -> tuple[float, float]:
    return (float(low + high) / 2.0, float(high - low))


def overlay_paper_guides(
    ax: plt.Axes,
    *,
    center_values: np.ndarray,
    width_values: np.ndarray,
    setup: dict[str, Any],
) -> None:
    if center_values.size == 0 or width_values.size == 0:
        return
    guide_centers = np.linspace(
        float(center_values[0]),
        float(center_values[-1]),
        256,
        dtype=np.float64,
    )
    guide_widths = 2.0 * guide_centers
    mask = guide_widths <= float(width_values[-1])
    ax.plot(
        guide_centers[mask],
        guide_widths[mask],
        linestyle="--",
        linewidth=1.4,
        color="white",
        alpha=0.9,
        label="negative gamma threshold",
    )

    low = setup.get("paper_interval_low")
    high = setup.get("paper_interval_high")
    if low is None or high is None:
        return
    paper_center, paper_width = interval_to_center_width(float(low), float(high))
    ax.plot(
        [paper_center],
        [paper_width],
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor="white",
        markeredgewidth=2.0,
        linestyle="none",
        label="paper gross interval",
    )


def convergence_from_jsonl(details_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mismatch_count = 0
    for detail_path in sorted(details_dir.glob("*.jsonl")):
        with detail_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {detail_path} at line {line_number}."
                    ) from exc
                decoder = str(record.get("decoder", ""))
                if not decoder:
                    continue
                converged = record.get("converged")
                physical_success = record.get("physical_success")
                if converged is None:
                    converged = physical_success
                if converged is None:
                    continue
                if physical_success is not None and bool(converged) != bool(
                    physical_success
                ):
                    mismatch_count += 1
                rows.append({"decoder": decoder, "converged": bool(converged)})

    if mismatch_count:
        print(
            "warning: found "
            f"{mismatch_count} records where converged != physical_success"
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "decoder",
                "detail_shots",
                "converged_count",
                "non_converged_count",
                "convergence_rate",
                "non_convergence_rate",
            ]
        )

    grouped = (
        pd.DataFrame(rows)
        .groupby("decoder", as_index=False)
        .agg(
            detail_shots=("converged", "size"),
            converged_count=("converged", "sum"),
        )
    )
    grouped["non_converged_count"] = (
        grouped["detail_shots"] - grouped["converged_count"]
    )
    grouped["convergence_rate"] = grouped["converged_count"] / grouped["detail_shots"]
    grouped["non_convergence_rate"] = (
        grouped["non_converged_count"] / grouped["detail_shots"]
    )
    return grouped


def grid_values(
    grid_df: pd.DataFrame, column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_values = (
        grid_df[["center_idx", "center"]]
        .drop_duplicates()
        .sort_values("center_idx")["center"]
        .to_numpy(dtype=np.float64)
    )
    width_values = (
        grid_df[["width_idx", "width"]]
        .drop_duplicates()
        .sort_values("width_idx")["width"]
        .to_numpy(dtype=np.float64)
    )
    values = np.full(
        (len(width_values), len(center_values)),
        np.nan,
        dtype=np.float64,
    )
    for row in grid_df.itertuples(index=False):
        value = getattr(row, column)
        if pd.isna(value):
            continue
        values[int(row.width_idx), int(row.center_idx)] = float(value)
    return center_values, width_values, values


def plot_logical_error_and_convergence(
    *,
    grid_df: pd.DataFrame,
    setup: dict[str, Any],
    output_path: Path,
) -> None:
    center_values, width_values, logical_error_rate = grid_values(
        grid_df,
        "logical_error_rate",
    )
    _, _, shots = grid_values(grid_df, "shots")
    _, _, convergence_rate = grid_values(grid_df, "convergence_rate")
    center_edges = heatmap_edges(center_values)
    width_edges = heatmap_edges(width_values)
    max_shots_per_point = int(
        setup.get(
            "max_shots_per_point",
            np.nanmax(shots) if np.isfinite(shots).any() else 1,
        )
    )
    logical_display, logical_norm = logical_error_lognorm(
        logical_error_rate,
        max_shots_per_point=max_shots_per_point,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), squeeze=False)
    logical_ax, convergence_ax = axes.ravel()

    logical_mesh = logical_ax.pcolormesh(
        center_edges,
        width_edges,
        logical_display,
        shading="auto",
        cmap="viridis_r",
        norm=logical_norm,
    )
    overlay_paper_guides(
        logical_ax,
        center_values=center_values,
        width_values=width_values,
        setup=setup,
    )
    logical_ax.set_title("logical error rate")
    logical_ax.set_xlabel("gamma center")
    logical_ax.set_ylabel("gamma width")
    fig.colorbar(logical_mesh, ax=logical_ax, fraction=0.046, pad=0.04)

    convergence_mesh = convergence_ax.pcolormesh(
        center_edges,
        width_edges,
        convergence_rate,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    overlay_paper_guides(
        convergence_ax,
        center_values=center_values,
        width_values=width_values,
        setup=setup,
    )
    convergence_ax.set_title("convergence rate")
    convergence_ax.set_xlabel("gamma center")
    convergence_ax.set_ylabel("gamma width")
    fig.colorbar(convergence_mesh, ax=convergence_ax, fraction=0.046, pad=0.04)

    title_parts = ["Relay-BP-1 logical error and convergence"]
    if setup.get("circuit"):
        title_parts.append(str(setup["circuit"]))
    if setup.get("selected_p") is not None:
        title_parts.append(f"p = {float(setup['selected_p']):.3g}")
    if setup.get("basis_filter") is not None:
        title_parts.append(f"basis filter = {setup['basis_filter']}")
    fig.suptitle(" | ".join(title_parts), fontsize=13, y=1.02)

    handles, labels = logical_ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Paper iteration heatmap output directory. Defaults to the standard "
            "examples/notebook_data location under the repo root."
        ),
    )
    parser.add_argument(
        "--grid-csv",
        type=Path,
        default=None,
        help="Optional explicit *_grid.csv path.",
    )
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=None,
        help="Optional explicit details JSONL directory.",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="logical_error_and_convergence_heatmaps.png",
        help="Filename to write under output_dir/plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd().resolve())
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / DEFAULT_OUTPUT_SUBDIR
    )
    grid_csv = (
        args.grid_csv.resolve()
        if args.grid_csv is not None
        else infer_single_file(output_dir, "*_grid.csv", "grid CSV")
    )
    details_dir = (
        args.details_dir.resolve()
        if args.details_dir is not None
        else output_dir / "details"
    )
    if not details_dir.exists():
        raise FileNotFoundError(f"Details directory does not exist: {details_dir}")

    setup = load_setup(output_dir)
    grid_df = pd.read_csv(grid_csv)
    convergence_df = convergence_from_jsonl(details_dir)
    merged = grid_df.merge(convergence_df, on="decoder", how="left")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = output_dir / "logical_error_and_convergence_grid.csv"
    output_path = plots_dir / args.plot_name

    merged.to_csv(merged_csv, index=False)
    plot_logical_error_and_convergence(
        grid_df=merged,
        setup=setup,
        output_path=output_path,
    )

    print(f"wrote {merged_csv}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
