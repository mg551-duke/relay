#!/usr/bin/env python3
"""Detailed cluster runner for paper-style Relay-BP heatmaps on gross-code memory-Z.

This is the cluster-native version of the Fig. 2 / Fig. 3 gross-code heatmap
task. Unlike the earlier notebook-style draft, this runner uses `sinter.collect`
for each grid square so the job prints normal sinter progress lines on cluster.

Paper-aligned defaults:
- circuit: `bicycle_bivariate_144_12_12_memory_Z`
- XZ-style decoding via `Z`-basis detector filtering
- p = 3e-3
- first stage: gamma0 = 0.125, pre_iter = 80
- relay continuation: 300 additional legs, set_max_iter = 60
- stop after first converged solution (Relay-BP-1)
- paper gross-code interval marker: [-0.24, 0.66]

Each grid square is run as one custom sinter decoder with:
- `max_errors = 10`
- `max_shots = 500_000`

Detailed per-shot decode records are also persisted so the output tables and
heatmaps include the mean decoder iteration count per run.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sinter
import stim
from matplotlib.colors import LogNorm

from relay_bp.stim import CheckMatrices, SinterDecoder_RelayBP
from relay_bp.stim.sinter.runner import _filter_detectors_by_basis
from relay_bp.stim.sinter.utils import write_stats


PAPER_URL = "https://arxiv.org/abs/2506.01779"
DEFAULT_SELECTED_P = 0.003
DEFAULT_MEMORY_BASIS = "Z"
DEFAULT_BASIS_FILTER = "Z"

PAPER_ALPHA = 1.0
PAPER_GAMMA0 = 0.125
PAPER_PRE_ITER = 80
PAPER_NUM_SETS = 300
PAPER_SET_MAX_ITER = 60
PAPER_STOP_NCONV = 1
PAPER_RELAY_POSTERIORS = True
PAPER_INTERVAL_LOW = -0.24
PAPER_INTERVAL_HIGH = 0.66

DEFAULT_CENTER_MIN = 0.1
DEFAULT_CENTER_MAX = 0.6
DEFAULT_CENTER_COUNT = 11
DEFAULT_WIDTH_MIN = 0.5
DEFAULT_WIDTH_MAX = 1.5
DEFAULT_WIDTH_COUNT = 11

DEFAULT_TARGET_LOGICAL_ERRORS = 10
DEFAULT_MAX_SHOTS_PER_POINT = 500_000
DEFAULT_BASE_RELAY_SEED = 0


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "tests" / "testdata" / "bicycle_bivariate").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root containing tests/testdata/bicycle_bivariate "
        f"starting from {start}."
    )


def resolve_num_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return max(1, multiprocessing.cpu_count())


def parse_name_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"stim_path": str(path)}
    for part in path.stem.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        metadata[key] = value
    if "error_rate" in metadata:
        metadata["p"] = float(metadata["error_rate"])
    return metadata


def find_circuit_path(circuits_dir: Path, selected_p: float, memory_basis: str) -> Path:
    wanted = str(selected_p)
    pattern = f"circuit=bicycle_bivariate_144_12_12_memory_{memory_basis},*.stim"
    matches: list[Path] = []
    for path in sorted(circuits_dir.glob(pattern)):
        metadata = parse_name_metadata(path)
        if metadata.get("error_rate") == wanted:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            "Missing bicycle_bivariate_144_12_12 memory circuit for "
            f"basis={memory_basis} and p={selected_p} in {circuits_dir}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Expected exactly one memory_{memory_basis} circuit for p={selected_p}, "
            f"got {len(matches)}."
        )
    return matches[0]


def parse_float_csv(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return np.asarray(values, dtype=np.float64)


def resolve_grid_values(
    *,
    explicit_values: np.ndarray | None,
    lower: float,
    upper: float,
    count: int,
    name: str,
) -> np.ndarray:
    if explicit_values is not None:
        values = np.asarray(explicit_values, dtype=np.float64)
    else:
        values = np.linspace(lower, upper, count, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if np.any(np.diff(values) < 0):
        raise ValueError(f"{name} must be sorted in non-decreasing order.")
    return values


def interval_from_center_width(center: float, width: float) -> tuple[float, float]:
    if width < 0:
        raise ValueError("width must be non-negative")
    low = float(center) - float(width) / 2.0
    high = float(center) + float(width) / 2.0
    return low, high


def interval_to_center_width(low: float, high: float) -> tuple[float, float]:
    return (float(low + high) / 2.0, float(high - low))


def load_filtered_problem(
    *,
    stim_path: Path,
    basis_filter: str,
) -> dict[str, Any]:
    full_circuit = stim.Circuit.from_file(stim_path)
    filtered_circuit = _filter_detectors_by_basis(full_circuit, basis_filter)
    dem = filtered_circuit.detector_error_model()
    check_matrices = CheckMatrices.from_dem(dem)
    metadata = parse_name_metadata(stim_path)
    metadata["basis_filter_applied"] = basis_filter
    return {
        "stim_path": stim_path,
        "full_circuit": full_circuit,
        "filtered_circuit": filtered_circuit,
        "dem": dem,
        "check_matrices": check_matrices,
        "metadata": metadata,
    }


def build_task(
    *,
    filtered_circuit: stim.Circuit,
    dem: stim.DetectorErrorModel,
    metadata: dict[str, Any],
    target_logical_errors: int,
    max_shots_per_point: int,
) -> list[sinter.Task]:
    return [
        sinter.Task(
            circuit=filtered_circuit,
            detector_error_model=dem,
            json_metadata=metadata,
            collection_options=sinter.CollectionOptions(
                max_errors=target_logical_errors,
                max_shots=max_shots_per_point,
            ),
        )
    ]


def format_float_token(value: float) -> str:
    sign = "m" if value < 0 else ""
    magnitude = f"{abs(value):.3f}".replace(".", "p")
    return f"{sign}{magnitude}"


def make_decoder_name(center: float, width: float) -> str:
    return (
        "relay-bp-paper-"
        f"c{format_float_token(center)}-"
        f"w{format_float_token(width)}"
    )


def build_heatmap_points(
    *,
    center_values: np.ndarray,
    width_values: np.ndarray,
    base_relay_seed: int,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for width_idx, width in enumerate(width_values):
        for center_idx, center in enumerate(center_values):
            point_id = width_idx * len(center_values) + center_idx
            interval_low, interval_high = interval_from_center_width(float(center), float(width))
            points.append(
                {
                    "point_id": int(point_id),
                    "width_idx": int(width_idx),
                    "center_idx": int(center_idx),
                    "center": float(center),
                    "width": float(width),
                    "interval_low": float(interval_low),
                    "interval_high": float(interval_high),
                    "decoder": make_decoder_name(float(center), float(width)),
                    "relay_seed": int(base_relay_seed + 1_000_003 * point_id),
                }
            )
    return points


def build_point_decoder(
    *,
    point: dict[str, Any],
    num_fault_mechanisms: int,
    alpha: float,
    gamma0: float,
    pre_iter: int,
    num_sets: int,
    set_max_iter: int,
    relay_posteriors: bool,
    stop_nconv: int,
    details_dir: Path | None = None,
) -> SinterDecoder_RelayBP:
    explicit_gammas = None
    gamma_dist_interval = (float(point["interval_low"]), float(point["interval_high"]))
    if math.isclose(float(point["width"]), 0.0, abs_tol=0.0):
        explicit_gammas = np.full(
            (num_sets, num_fault_mechanisms),
            float(point["center"]),
            dtype=np.float64,
        )
        gamma_dist_interval = (float(point["center"]), float(point["center"]) + 1e-12)

    return SinterDecoder_RelayBP(
        alpha=alpha,
        gamma0=gamma0,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        gamma_dist_interval=gamma_dist_interval,
        explicit_gammas=explicit_gammas,
        relay_posteriors=relay_posteriors,
        stop_nconv=stop_nconv,
        stopping_criterion="nconv",
        logging=False,
        seed=int(point["relay_seed"]),
        details_dir=str(details_dir) if details_dir is not None else None,
        decoder_label=str(point["decoder"]),
    )


def build_setup_dict(
    *,
    repo_root: Path,
    output_dir: Path,
    plots_dir: Path,
    selected_p: float,
    memory_basis: str,
    basis_filter: str,
    alpha: float,
    gamma0: float,
    pre_iter: int,
    num_sets: int,
    set_max_iter: int,
    stop_nconv: int,
    relay_posteriors: bool,
    center_values: np.ndarray,
    width_values: np.ndarray,
    target_logical_errors: int,
    max_shots_per_point: int,
    base_relay_seed: int,
    metadata: dict[str, Any],
    dem: stim.DetectorErrorModel,
    check_matrices: CheckMatrices,
    num_workers: int,
    paper_interval_low: float,
    paper_interval_high: float,
) -> dict[str, Any]:
    paper_center, paper_width = interval_to_center_width(paper_interval_low, paper_interval_high)
    return {
        "paper_reference_url": PAPER_URL,
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "stim_path": str(metadata.get("stim_path", "")),
        "circuit": metadata.get("circuit", "unknown"),
        "selected_p": float(selected_p),
        "memory_basis": memory_basis,
        "basis_filter": basis_filter,
        "alpha": float(alpha),
        "gamma0": float(gamma0),
        "pre_iter": int(pre_iter),
        "num_sets": int(num_sets),
        "total_legs": int(num_sets + 1),
        "set_max_iter": int(set_max_iter),
        "stop_nconv": int(stop_nconv),
        "relay_posteriors": bool(relay_posteriors),
        "center_values": center_values.tolist(),
        "width_values": width_values.tolist(),
        "target_logical_errors": int(target_logical_errors),
        "max_shots_per_point": int(max_shots_per_point),
        "base_relay_seed": int(base_relay_seed),
        "paper_interval_low": float(paper_interval_low),
        "paper_interval_high": float(paper_interval_high),
        "paper_interval_center": float(paper_center),
        "paper_interval_width": float(paper_width),
        "num_detectors": int(dem.num_detectors),
        "num_observables": int(dem.num_observables),
        "num_fault_mechanisms": int(check_matrices.check_matrix.shape[1]),
        "num_workers": int(num_workers),
        "has_syndrome_bias": bool(check_matrices.syndrome_bias is not None),
        "has_observables_bias": bool(check_matrices.observables_bias is not None),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
    }


def validate_resume_setup(setup_json_path: Path, current_setup: dict[str, Any]) -> None:
    if not setup_json_path.exists():
        return
    existing_setup = json.loads(setup_json_path.read_text(encoding="utf-8"))
    fields_to_compare = [
        "selected_p",
        "memory_basis",
        "basis_filter",
        "alpha",
        "gamma0",
        "pre_iter",
        "num_sets",
        "set_max_iter",
        "stop_nconv",
        "relay_posteriors",
        "center_values",
        "width_values",
        "target_logical_errors",
        "max_shots_per_point",
        "base_relay_seed",
        "paper_interval_low",
        "paper_interval_high",
    ]
    mismatches = [
        field
        for field in fields_to_compare
        if existing_setup.get(field) != current_setup.get(field)
    ]
    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise ValueError(
            "Existing heatmap outputs were created with different settings: "
            f"{mismatch_text}. Use --reset-data or choose a different output directory."
        )


def load_samples_from_resume(resume_csv: Path) -> list[sinter.TaskStats]:
    if not resume_csv.exists():
        return []
    return list(sinter.stats_from_csv_files(resume_csv))


def summarize_iteration_details(
    details_dir: Path,
    *,
    points_by_decoder: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not details_dir.exists():
        return pd.DataFrame(
            columns=[
                "decoder",
                "detail_shots",
                "detail_iteration_sum",
                "mean_iterations",
            ]
        )

    aggregates: dict[str, dict[str, float]] = {}
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
                        f"Invalid JSON in detail file {detail_path} at line {line_number}."
                    ) from exc
                decoder = str(record.get("decoder", ""))
                if decoder not in points_by_decoder:
                    continue
                iterations = record.get("iterations")
                if iterations is None:
                    continue
                aggregate = aggregates.setdefault(
                    decoder,
                    {
                        "detail_shots": 0.0,
                        "detail_iteration_sum": 0.0,
                    },
                )
                aggregate["detail_shots"] += 1.0
                aggregate["detail_iteration_sum"] += float(iterations)

    for decoder, aggregate in sorted(aggregates.items()):
        detail_shots = float(aggregate["detail_shots"])
        detail_iteration_sum = float(aggregate["detail_iteration_sum"])
        rows.append(
            {
                "decoder": decoder,
                "detail_shots": int(detail_shots),
                "detail_iteration_sum": detail_iteration_sum,
                "mean_iterations": (
                    detail_iteration_sum / detail_shots if detail_shots > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def attach_iteration_metrics(
    summary_df: pd.DataFrame,
    *,
    details_dir: Path,
    points_by_decoder: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    detail_df = summarize_iteration_details(
        details_dir,
        points_by_decoder=points_by_decoder,
    )
    if summary_df.empty:
        result = summary_df.copy()
        result["detail_shots"] = pd.Series(dtype=np.int64)
        result["detail_iteration_sum"] = pd.Series(dtype=np.float64)
        result["mean_iterations"] = pd.Series(dtype=np.float64)
        return result

    result = summary_df.merge(detail_df, on="decoder", how="left")
    if "detail_shots" not in result.columns:
        result["detail_shots"] = np.nan
    if "detail_iteration_sum" not in result.columns:
        result["detail_iteration_sum"] = np.nan
    if "mean_iterations" not in result.columns:
        result["mean_iterations"] = np.nan
    return result


def samples_to_df(
    *,
    samples: list[sinter.TaskStats],
    points_by_decoder: dict[str, dict[str, Any]],
    target_logical_errors: int,
    max_shots_per_point: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stat in samples:
        point = points_by_decoder.get(stat.decoder)
        if point is None:
            continue
        circuit_name = stat.json_metadata.get("circuit", "unknown")
        rows.append(
            {
                "decoder": stat.decoder,
                "point_id": int(point["point_id"]),
                "width_idx": int(point["width_idx"]),
                "center_idx": int(point["center_idx"]),
                "center": float(point["center"]),
                "width": float(point["width"]),
                "interval_low": float(point["interval_low"]),
                "interval_high": float(point["interval_high"]),
                "relay_seed": int(point["relay_seed"]),
                "circuit": circuit_name,
                "memory": circuit_name.rsplit("_", 1)[-1],
                "p": float(stat.json_metadata["p"]),
                "basis_filter_applied": stat.json_metadata.get("basis_filter_applied", "none"),
                "shots": int(stat.shots),
                "logical_error_count": int(stat.errors),
                "seconds": float(stat.seconds),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "decoder",
                "point_id",
                "width_idx",
                "center_idx",
                "center",
                "width",
                "interval_low",
                "interval_high",
                "relay_seed",
                "circuit",
                "memory",
                "p",
                "basis_filter_applied",
                "shots",
                "logical_error_count",
                "seconds",
                "logical_error_rate",
                "logical_success_rate",
                "stderr",
                "relative_error",
                "hit_target",
                "shot_cap_reached",
            ]
        )

    df = pd.DataFrame(rows)
    summary_df = (
        df.groupby(
            [
                "decoder",
                "point_id",
                "width_idx",
                "center_idx",
                "center",
                "width",
                "interval_low",
                "interval_high",
                "relay_seed",
                "circuit",
                "memory",
                "p",
                "basis_filter_applied",
            ],
            as_index=False,
        )
        .agg(
            shots=("shots", "sum"),
            logical_error_count=("logical_error_count", "sum"),
            seconds=("seconds", "sum"),
        )
        .sort_values(["width_idx", "center_idx"])
        .reset_index(drop=True)
    )
    summary_df["logical_error_rate"] = summary_df["logical_error_count"] / summary_df["shots"]
    summary_df["logical_success_rate"] = 1.0 - summary_df["logical_error_rate"]
    summary_df["stderr"] = (
        (
            summary_df["logical_error_rate"]
            * (1.0 - summary_df["logical_error_rate"])
            / summary_df["shots"]
        ).clip(lower=0.0)
    ) ** 0.5
    summary_df["relative_error"] = np.where(
        summary_df["logical_error_rate"] > 0.0,
        summary_df["stderr"] / summary_df["logical_error_rate"],
        np.inf,
    )
    summary_df["hit_target"] = summary_df["logical_error_count"] >= int(target_logical_errors)
    summary_df["shot_cap_reached"] = (
        (summary_df["shots"] >= int(max_shots_per_point)) & (~summary_df["hit_target"])
    )
    return summary_df


def heatmap_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
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
    positive_mask = np.isfinite(display_values) & (display_values > 0.0)
    floor = 0.5 / max_shots_per_point
    display_values[np.isfinite(display_values) & (display_values <= 0.0)] = floor
    if np.any(positive_mask):
        vmin = min(float(np.nanmin(display_values[positive_mask])), floor)
        vmax = float(np.nanmax(display_values[np.isfinite(display_values)]))
        if not math.isfinite(vmax) or vmax <= 0.0:
            vmax = max(vmin * 10.0, floor * 10.0)
    else:
        vmin = floor
        vmax = floor * 10.0
    if vmax <= vmin:
        vmax = vmin * 1.01
    return display_values, LogNorm(vmin=vmin, vmax=vmax)


def positive_lognorm(values: np.ndarray, *, floor: float = 1.0) -> tuple[np.ndarray, LogNorm]:
    display_values = np.asarray(values, dtype=np.float64).copy()
    finite_mask = np.isfinite(display_values)
    positive_mask = finite_mask & (display_values > 0.0)
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


def overlay_paper_guides(
    ax: plt.Axes,
    *,
    center_values: np.ndarray,
    width_values: np.ndarray,
    paper_center: float,
    paper_width: float,
) -> None:
    guide_centers = np.linspace(
        float(center_values[0]),
        float(center_values[-1]),
        256,
        dtype=np.float64,
    )
    guide_widths = 2.0 * guide_centers
    width_limit = float(width_values[-1])
    mask = guide_widths <= width_limit
    ax.plot(
        guide_centers[mask],
        guide_widths[mask],
        linestyle="--",
        linewidth=1.4,
        color="white",
        alpha=0.9,
        label="negative gamma threshold",
    )
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


def grid_to_artifact(
    grid_df: pd.DataFrame,
    *,
    center_values: np.ndarray,
    width_values: np.ndarray,
) -> dict[str, np.ndarray]:
    shape = (len(width_values), len(center_values))
    artifact = {
        "logical_error_rate": np.full(shape, np.nan, dtype=np.float64),
        "logical_success_rate": np.full(shape, np.nan, dtype=np.float64),
        "shots": np.full(shape, np.nan, dtype=np.float64),
        "logical_error_count": np.full(shape, np.nan, dtype=np.float64),
        "seconds": np.full(shape, np.nan, dtype=np.float64),
        "detail_shots": np.full(shape, np.nan, dtype=np.float64),
        "mean_iterations": np.full(shape, np.nan, dtype=np.float64),
    }
    for row in grid_df.itertuples(index=False):
        artifact["logical_error_rate"][row.width_idx, row.center_idx] = float(row.logical_error_rate)
        artifact["logical_success_rate"][row.width_idx, row.center_idx] = float(row.logical_success_rate)
        artifact["shots"][row.width_idx, row.center_idx] = float(row.shots)
        artifact["logical_error_count"][row.width_idx, row.center_idx] = float(row.logical_error_count)
        artifact["seconds"][row.width_idx, row.center_idx] = float(row.seconds)
        if hasattr(row, "detail_shots") and pd.notna(row.detail_shots):
            artifact["detail_shots"][row.width_idx, row.center_idx] = float(row.detail_shots)
        if hasattr(row, "mean_iterations") and pd.notna(row.mean_iterations):
            artifact["mean_iterations"][row.width_idx, row.center_idx] = float(
                row.mean_iterations
            )
    return artifact


def plot_paper_style_heatmap(
    *,
    artifact: dict[str, np.ndarray],
    center_values: np.ndarray,
    width_values: np.ndarray,
    metadata: dict[str, Any],
    selected_p: float,
    basis_filter: str,
    output_path: Path,
    max_shots_per_point: int,
    gamma0: float,
    pre_iter: int,
    num_sets: int,
    set_max_iter: int,
    stop_nconv: int,
    paper_interval_low: float,
    paper_interval_high: float,
) -> None:
    center_edges = heatmap_edges(center_values)
    width_edges = heatmap_edges(width_values)
    paper_center, paper_width = interval_to_center_width(paper_interval_low, paper_interval_high)
    display_values, norm = logical_error_lognorm(
        artifact["logical_error_rate"],
        max_shots_per_point=max_shots_per_point,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(
        center_edges,
        width_edges,
        display_values,
        shading="auto",
        cmap="viridis_r",
        norm=norm,
    )
    overlay_paper_guides(
        ax,
        center_values=center_values,
        width_values=width_values,
        paper_center=paper_center,
        paper_width=paper_width,
    )
    ax.set_xlabel("gamma center")
    ax.set_ylabel("gamma width")
    ax.set_title(
        "Relay-BP-1 logical error rate | "
        f"{metadata.get('circuit', 'unknown')} | p = {selected_p:.3g} | basis filter = {basis_filter}"
    )
    subtitle = (
        f"gamma0 = {gamma0:.3f}, pre_iter = {pre_iter}, total_legs = {num_sets + 1}, "
        f"set_max_iter = {set_max_iter}, stop_nconv = {stop_nconv}"
    )
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("logical error rate")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_collection_heatmaps(
    *,
    artifact: dict[str, np.ndarray],
    center_values: np.ndarray,
    width_values: np.ndarray,
    metadata: dict[str, Any],
    selected_p: float,
    basis_filter: str,
    output_path: Path,
    max_shots_per_point: int,
    paper_interval_low: float,
    paper_interval_high: float,
) -> None:
    center_edges = heatmap_edges(center_values)
    width_edges = heatmap_edges(width_values)
    paper_center, paper_width = interval_to_center_width(paper_interval_low, paper_interval_high)
    ler_display, ler_norm = logical_error_lognorm(
        artifact["logical_error_rate"],
        max_shots_per_point=max_shots_per_point,
    )
    iterations_display, iterations_norm = positive_lognorm(
        artifact["mean_iterations"],
        floor=1.0,
    )
    shots_display, shots_norm = positive_lognorm(artifact["shots"], floor=1.0)
    seconds_display, seconds_norm = positive_lognorm(artifact["seconds"], floor=1e-3)

    fig, axes = plt.subplots(2, 3, figsize=(17, 11), squeeze=False)
    flat_axes = axes.ravel()

    specs = [
        ("logical error rate", ler_display, "viridis_r", ler_norm),
        ("mean iterations", iterations_display, "viridis", iterations_norm),
        ("shots", shots_display, "magma", shots_norm),
        ("logical errors observed", artifact["logical_error_count"], "viridis", None),
        ("wall-clock seconds", seconds_display, "magma", seconds_norm),
    ]

    for ax, (title, data, cmap, norm) in zip(flat_axes, specs, strict=False):
        mesh = ax.pcolormesh(
            center_edges,
            width_edges,
            np.asarray(data, dtype=np.float64),
            shading="auto",
            cmap=cmap,
            norm=norm,
            vmin=None if norm is not None else 0.0,
            vmax=None if norm is not None else float(max(1.0, np.nanmax(np.asarray(data, dtype=np.float64)))),
        )
        overlay_paper_guides(
            ax,
            center_values=center_values,
            width_values=width_values,
            paper_center=paper_center,
            paper_width=paper_width,
        )
        ax.set_title(title)
        ax.set_xlabel("gamma center")
        ax.set_ylabel("gamma width")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    for ax in flat_axes[len(specs) :]:
        ax.axis("off")

    fig.suptitle(
        "Collection summary heatmaps | "
        f"{metadata.get('circuit', 'unknown')} | p = {selected_p:.3g} | basis filter = {basis_filter}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def best_points_table(grid_df: pd.DataFrame) -> pd.DataFrame:
    if grid_df.empty:
        return pd.DataFrame()

    best_ler = grid_df.sort_values(
        ["logical_error_rate", "logical_error_count", "shots", "seconds"],
        ascending=[True, False, True, True],
    ).head(8).copy()
    best_ler.insert(0, "ranking", ["lowest logical error"] * len(best_ler))

    most_sampled = grid_df.sort_values(
        ["shots", "logical_error_rate", "seconds"],
        ascending=[False, True, True],
    ).head(8).copy()
    most_sampled.insert(0, "ranking", ["most sampled"] * len(most_sampled))

    fastest = grid_df.sort_values(
        ["seconds", "logical_error_rate", "shots"],
        ascending=[True, True, True],
    ).head(8).copy()
    fastest.insert(0, "ranking", ["fastest"] * len(fastest))

    return pd.concat([best_ler, most_sampled, fastest], ignore_index=True)


def save_outputs(
    *,
    summary_df: pd.DataFrame,
    center_values: np.ndarray,
    width_values: np.ndarray,
    metadata: dict[str, Any],
    selected_p: float,
    basis_filter: str,
    output_dir: Path,
    save_stem: str,
    paper_interval_low: float,
    paper_interval_high: float,
    gamma0: float,
    pre_iter: int,
    num_sets: int,
    set_max_iter: int,
    stop_nconv: int,
    max_shots_per_point: int,
) -> None:
    grid_csv = output_dir / f"{save_stem}_grid.csv"
    best_csv = output_dir / f"{save_stem}_best_points.csv"
    paper_point_csv = output_dir / f"{save_stem}_paper_interval_nearest_point.csv"
    artifact_npz = output_dir / f"{save_stem}_artifact.npz"
    plots_dir = output_dir / "plots"
    paper_plot_png = plots_dir / "paper_style_logical_error_heatmap.png"
    collection_plot_png = plots_dir / "collection_heatmaps.png"

    summary_df.sort_values(["width_idx", "center_idx"]).to_csv(grid_csv, index=False)
    best_df = best_points_table(summary_df)
    best_df.to_csv(best_csv, index=False)

    if summary_df.empty:
        return

    paper_center, paper_width = interval_to_center_width(paper_interval_low, paper_interval_high)
    paper_dist2 = (
        (summary_df["center"] - paper_center) ** 2
        + (summary_df["width"] - paper_width) ** 2
    )
    nearest_paper_point = summary_df.loc[[int(paper_dist2.idxmin())]].copy()
    nearest_paper_point.insert(0, "paper_interval_center", paper_center)
    nearest_paper_point.insert(1, "paper_interval_width", paper_width)
    nearest_paper_point.to_csv(paper_point_csv, index=False)

    artifact = grid_to_artifact(
        summary_df,
        center_values=center_values,
        width_values=width_values,
    )
    np.savez_compressed(
        artifact_npz,
        center_values=center_values,
        width_values=width_values,
        **artifact,
    )

    plot_paper_style_heatmap(
        artifact=artifact,
        center_values=center_values,
        width_values=width_values,
        metadata=metadata,
        selected_p=selected_p,
        basis_filter=basis_filter,
        output_path=paper_plot_png,
        max_shots_per_point=max_shots_per_point,
        gamma0=gamma0,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        stop_nconv=stop_nconv,
        paper_interval_low=paper_interval_low,
        paper_interval_high=paper_interval_high,
    )
    plot_collection_heatmaps(
        artifact=artifact,
        center_values=center_values,
        width_values=width_values,
        metadata=metadata,
        selected_p=selected_p,
        basis_filter=basis_filter,
        output_path=collection_plot_png,
        max_shots_per_point=max_shots_per_point,
        paper_interval_low=paper_interval_low,
        paper_interval_high=paper_interval_high,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing tests/testdata/bicycle_bivariate. Auto-detected by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for CSV and plot outputs. Defaults to "
            "examples/notebook_data/bicycle_bivariate_144_12_12_paper_iteration_heatmap_cluster_Z_basis "
            "under the repo root."
        ),
    )
    parser.add_argument(
        "--reset-data",
        action="store_true",
        help="Delete prior CSV/JSON/NPZ/PNG outputs before running.",
    )
    parser.add_argument(
        "--print-progress",
        action="store_true",
        default=False,
        help="Enable sinter progress output.",
    )
    parser.add_argument("--p", type=float, default=DEFAULT_SELECTED_P)
    parser.add_argument("--memory-basis", choices=["X", "Z"], default=DEFAULT_MEMORY_BASIS)
    parser.add_argument("--basis-filter", choices=["X", "Z"], default=DEFAULT_BASIS_FILTER)
    parser.add_argument("--alpha", type=float, default=PAPER_ALPHA)
    parser.add_argument("--gamma0", type=float, default=PAPER_GAMMA0)
    parser.add_argument("--pre-iter", type=int, default=PAPER_PRE_ITER)
    parser.add_argument("--num-sets", type=int, default=PAPER_NUM_SETS)
    parser.add_argument("--set-max-iter", type=int, default=PAPER_SET_MAX_ITER)
    parser.add_argument("--stop-nconv", type=int, default=PAPER_STOP_NCONV)
    parser.add_argument(
        "--no-relay-posteriors",
        action="store_true",
        help="Disable relayed posterior reuse between legs.",
    )
    parser.add_argument("--center-values", type=str, default=None)
    parser.add_argument("--width-values", type=str, default=None)
    parser.add_argument("--center-min", type=float, default=DEFAULT_CENTER_MIN)
    parser.add_argument("--center-max", type=float, default=DEFAULT_CENTER_MAX)
    parser.add_argument("--center-count", type=int, default=DEFAULT_CENTER_COUNT)
    parser.add_argument("--width-min", type=float, default=DEFAULT_WIDTH_MIN)
    parser.add_argument("--width-max", type=float, default=DEFAULT_WIDTH_MAX)
    parser.add_argument("--width-count", type=int, default=DEFAULT_WIDTH_COUNT)
    parser.add_argument("--target-logical-errors", type=int, default=DEFAULT_TARGET_LOGICAL_ERRORS)
    parser.add_argument("--max-shots-per-point", type=int, default=DEFAULT_MAX_SHOTS_PER_POINT)
    parser.add_argument("--base-relay-seed", type=int, default=DEFAULT_BASE_RELAY_SEED)
    parser.add_argument("--paper-interval-low", type=float, default=PAPER_INTERVAL_LOW)
    parser.add_argument("--paper-interval-high", type=float, default=PAPER_INTERVAL_HIGH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path.cwd().resolve())
    circuits_dir = repo_root / "tests" / "testdata" / "bicycle_bivariate"
    save_stem = "bicycle_bivariate_144_12_12_paper_iteration_heatmap_cluster_Z_basis"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "examples" / "notebook_data" / save_stem
    )
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    resume_csv = output_dir / f"{save_stem}_resume.csv"
    summary_csv = output_dir / f"{save_stem}_summary.csv"
    grid_csv = output_dir / f"{save_stem}_grid.csv"
    best_csv = output_dir / f"{save_stem}_best_points.csv"
    paper_point_csv = output_dir / f"{save_stem}_paper_interval_nearest_point.csv"
    artifact_npz = output_dir / f"{save_stem}_artifact.npz"
    setup_json = output_dir / f"{save_stem}_setup.json"
    selected_circuit_csv = output_dir / "selected_circuit.csv"
    point_manifest_csv = output_dir / f"{save_stem}_point_manifest.csv"
    details_dir = output_dir / "details"
    paper_plot_png = plots_dir / "paper_style_logical_error_heatmap.png"
    collection_plot_png = plots_dir / "collection_heatmaps.png"

    if args.reset_data:
        for path in [
            resume_csv,
            summary_csv,
            grid_csv,
            best_csv,
            paper_point_csv,
            artifact_npz,
            setup_json,
            selected_circuit_csv,
            point_manifest_csv,
            paper_plot_png,
            collection_plot_png,
        ]:
            if path.exists():
                path.unlink()
        if details_dir.exists():
            for path in details_dir.glob("*.jsonl"):
                path.unlink()
    details_dir.mkdir(parents=True, exist_ok=True)

    center_values = resolve_grid_values(
        explicit_values=parse_float_csv(args.center_values),
        lower=args.center_min,
        upper=args.center_max,
        count=args.center_count,
        name="center_values",
    )
    width_values = resolve_grid_values(
        explicit_values=parse_float_csv(args.width_values),
        lower=args.width_min,
        upper=args.width_max,
        count=args.width_count,
        name="width_values",
    )
    if np.any(width_values < 0.0):
        raise ValueError("All width values must be non-negative.")
    if args.target_logical_errors <= 0:
        raise ValueError("--target-logical-errors must be positive.")
    if args.max_shots_per_point <= 0:
        raise ValueError("--max-shots-per-point must be positive.")
    if args.paper_interval_high < args.paper_interval_low:
        raise ValueError("--paper-interval-high must be >= --paper-interval-low.")

    stim_path = find_circuit_path(circuits_dir, args.p, args.memory_basis)
    problem = load_filtered_problem(stim_path=stim_path, basis_filter=args.basis_filter)
    metadata = problem["metadata"]
    metadata["stim_path"] = str(stim_path)
    pd.DataFrame([metadata]).to_csv(selected_circuit_csv, index=False)

    num_workers = resolve_num_workers()
    relay_posteriors = not args.no_relay_posteriors
    tasks = build_task(
        filtered_circuit=problem["filtered_circuit"],
        dem=problem["dem"],
        metadata=metadata,
        target_logical_errors=args.target_logical_errors,
        max_shots_per_point=args.max_shots_per_point,
    )

    points = build_heatmap_points(
        center_values=center_values,
        width_values=width_values,
        base_relay_seed=args.base_relay_seed,
    )
    points_by_decoder = {str(point["decoder"]): point for point in points}
    pd.DataFrame(points).to_csv(point_manifest_csv, index=False)

    current_setup = build_setup_dict(
        repo_root=repo_root,
        output_dir=output_dir,
        plots_dir=plots_dir,
        selected_p=args.p,
        memory_basis=args.memory_basis,
        basis_filter=args.basis_filter,
        alpha=args.alpha,
        gamma0=args.gamma0,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        stop_nconv=args.stop_nconv,
        relay_posteriors=relay_posteriors,
        center_values=center_values,
        width_values=width_values,
        target_logical_errors=args.target_logical_errors,
        max_shots_per_point=args.max_shots_per_point,
        base_relay_seed=args.base_relay_seed,
        metadata=metadata,
        dem=problem["dem"],
        check_matrices=problem["check_matrices"],
        num_workers=num_workers,
        paper_interval_low=args.paper_interval_low,
        paper_interval_high=args.paper_interval_high,
    )
    validate_resume_setup(setup_json, current_setup)
    setup_json.write_text(json.dumps(current_setup, indent=2, sort_keys=True), encoding="utf-8")

    initial_samples = load_samples_from_resume(resume_csv)
    initial_summary_df = samples_to_df(
        samples=initial_samples,
        points_by_decoder=points_by_decoder,
        target_logical_errors=args.target_logical_errors,
        max_shots_per_point=args.max_shots_per_point,
    )
    initial_summary_df = attach_iteration_metrics(
        initial_summary_df,
        details_dir=details_dir,
        points_by_decoder=points_by_decoder,
    )
    if initial_samples:
        write_stats(initial_samples, summary_csv)
    save_outputs(
        summary_df=initial_summary_df,
        center_values=center_values,
        width_values=width_values,
        metadata=metadata,
        selected_p=args.p,
        basis_filter=args.basis_filter,
        output_dir=output_dir,
        save_stem=save_stem,
        paper_interval_low=args.paper_interval_low,
        paper_interval_high=args.paper_interval_high,
        gamma0=args.gamma0,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        stop_nconv=args.stop_nconv,
        max_shots_per_point=args.max_shots_per_point,
    )

    completed_decoders = set(
        initial_summary_df.loc[
            initial_summary_df["hit_target"] | initial_summary_df["shot_cap_reached"],
            "decoder",
        ].tolist()
    )

    print(f"repo_root   = {repo_root}")
    print(f"circuits_dir= {circuits_dir}")
    print(f"stim_path   = {stim_path}")
    print(f"output_dir  = {output_dir}")
    print(f"plots_dir   = {plots_dir}")
    print(f"details_dir = {details_dir}")
    print(f"num_workers = {num_workers}")
    print(f"paper_url   = {PAPER_URL}")
    print(f"selected_p  = {args.p}")
    print(f"memory_basis= {args.memory_basis}")
    print(f"basis_filter= {args.basis_filter}")
    print(f"gamma0      = {args.gamma0}")
    print(f"pre_iter    = {args.pre_iter}")
    print(f"num_sets    = {args.num_sets} (total legs = {args.num_sets + 1})")
    print(f"set_max_iter= {args.set_max_iter}")
    print(f"stop_nconv  = {args.stop_nconv}")
    print(f"center_values = {center_values.tolist()}")
    print(f"width_values  = {width_values.tolist()}")
    print(f"target_logical_errors = {args.target_logical_errors}")
    print(f"max_shots_per_point   = {args.max_shots_per_point}")
    print(
        "paper_interval = "
        f"[{args.paper_interval_low:.3f}, {args.paper_interval_high:.3f}] "
        f"(center={interval_to_center_width(args.paper_interval_low, args.paper_interval_high)[0]:.3f}, "
        f"width={interval_to_center_width(args.paper_interval_low, args.paper_interval_high)[1]:.3f})"
    )
    print(f"completed_points = {len(completed_decoders)}/{len(points)}")

    for step_index, point in enumerate(points, start=1):
        decoder_name = str(point["decoder"])
        if decoder_name in completed_decoders:
            print(
                f"\n=== Skipping heatmap point {step_index}/{len(points)}: "
                f"{decoder_name} (already complete) ==="
            )
            continue

        print(
            f"\n=== Running heatmap point {step_index}/{len(points)}: {decoder_name} "
            f"| center={point['center']:.3f} width={point['width']:.3f} ==="
        )
        decoder = build_point_decoder(
            point=point,
            num_fault_mechanisms=int(problem["check_matrices"].check_matrix.shape[1]),
            alpha=args.alpha,
            gamma0=args.gamma0,
            pre_iter=args.pre_iter,
            num_sets=args.num_sets,
            set_max_iter=args.set_max_iter,
            relay_posteriors=relay_posteriors,
            stop_nconv=args.stop_nconv,
            details_dir=details_dir,
        )
        sinter.collect(
            tasks=tasks,
            decoders=[decoder_name],
            custom_decoders={decoder_name: decoder},
            num_workers=num_workers,
            print_progress=args.print_progress,
            save_resume_filepath=resume_csv,
        )

        samples = load_samples_from_resume(resume_csv)
        write_stats(samples, summary_csv)
        summary_df = samples_to_df(
            samples=samples,
            points_by_decoder=points_by_decoder,
            target_logical_errors=args.target_logical_errors,
            max_shots_per_point=args.max_shots_per_point,
        )
        summary_df = attach_iteration_metrics(
            summary_df,
            details_dir=details_dir,
            points_by_decoder=points_by_decoder,
        )
        save_outputs(
            summary_df=summary_df,
            center_values=center_values,
            width_values=width_values,
            metadata=metadata,
            selected_p=args.p,
            basis_filter=args.basis_filter,
            output_dir=output_dir,
            save_stem=save_stem,
            paper_interval_low=args.paper_interval_low,
            paper_interval_high=args.paper_interval_high,
            gamma0=args.gamma0,
            pre_iter=args.pre_iter,
            num_sets=args.num_sets,
            set_max_iter=args.set_max_iter,
            stop_nconv=args.stop_nconv,
            max_shots_per_point=args.max_shots_per_point,
        )
        completed_decoders.add(decoder_name)

    final_samples = load_samples_from_resume(resume_csv)
    final_summary_df = samples_to_df(
        samples=final_samples,
        points_by_decoder=points_by_decoder,
        target_logical_errors=args.target_logical_errors,
        max_shots_per_point=args.max_shots_per_point,
    )
    final_summary_df = attach_iteration_metrics(
        final_summary_df,
        details_dir=details_dir,
        points_by_decoder=points_by_decoder,
    )
    if final_samples:
        write_stats(final_samples, summary_csv)
    save_outputs(
        summary_df=final_summary_df,
        center_values=center_values,
        width_values=width_values,
        metadata=metadata,
        selected_p=args.p,
        basis_filter=args.basis_filter,
        output_dir=output_dir,
        save_stem=save_stem,
        paper_interval_low=args.paper_interval_low,
        paper_interval_high=args.paper_interval_high,
        gamma0=args.gamma0,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        stop_nconv=args.stop_nconv,
        max_shots_per_point=args.max_shots_per_point,
    )

    print("\nDone.")
    print(f"setup_json       = {setup_json}")
    print(f"selected_circuit = {selected_circuit_csv}")
    print(f"point_manifest   = {point_manifest_csv}")
    print(f"details_dir      = {details_dir}")
    print(f"resume_csv       = {resume_csv}")
    print(f"summary_csv      = {summary_csv}")
    print(f"grid_csv         = {grid_csv}")
    print(f"best_csv         = {best_csv}")
    print(f"paper_point_csv  = {paper_point_csv}")
    print(f"artifact_npz     = {artifact_npz}")
    print(f"paper_plot_png   = {paper_plot_png}")
    print(f"collection_png   = {collection_plot_png}")


if __name__ == "__main__":
    main()
