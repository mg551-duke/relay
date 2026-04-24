#!/usr/bin/env python3
"""Cluster runner for damping-only BP heatmaps on gross-code memory-Z.

This mirrors the gross-code heatmap workflow, but disables the memory/gamma
effect entirely and uses the Relay wrapper only to run 301 BP legs with
per-leg edge damping intervals swept across a center/width grid.

Defaults:
- circuit: `bicycle_bivariate_144_12_12_memory_Z`
- XZ-style decoding via `Z`-basis detector filtering
- p = 3e-3
- gamma0 = 0.0 and explicit relay gammas = 0 everywhere
- relay continuation: 300 additional legs, set_max_iter = 60
- relayed posteriors reused between legs by default
- stop after first converged solution
- damping reference interval from the repo's disordered-damping specs: [0.5, 1.0]

Each grid square is run as one custom sinter decoder with:
- `max_errors = 1`
- `max_shots = 500_000`

Requested damping intervals are parameterized by center and width exactly like
the gamma heatmap runner. Because damping values must lie in [0, 1], requested
intervals are clipped to that range before decoding, and the effective interval
is recorded in the CSV outputs.
"""

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
import sinter
from matplotlib.colors import LogNorm

from relay_bp.stim import SinterDecoder_RelayBP
from relay_bp.stim.sinter.utils import write_stats
from run_bivariate_bicycle144_12_12_paper_heatmap_cluster_Z_basis import (
    best_points_table,
    build_task,
    find_circuit_path,
    find_repo_root,
    format_float_token,
    grid_to_artifact,
    heatmap_edges,
    load_filtered_problem,
    load_samples_from_resume,
    logical_error_lognorm,
    parse_float_csv,
    positive_lognorm,
    resolve_grid_values,
    resolve_num_workers,
    samples_to_df,
)


PAPER_URL = "https://arxiv.org/abs/2506.01779"
DEFAULT_SELECTED_P = 0.003
DEFAULT_MEMORY_BASIS = "Z"
DEFAULT_BASIS_FILTER = "Z"

PAPER_ALPHA = 1.0
PAPER_GAMMA0 = 0.0
PAPER_PRE_ITER = 80
PAPER_NUM_SETS = 300
PAPER_SET_MAX_ITER = 60
PAPER_STOP_NCONV = 1
PAPER_RELAY_POSTERIORS = True

REFERENCE_DAMP_INTERVAL_LOW = 0.5
REFERENCE_DAMP_INTERVAL_HIGH = 1.0

DEFAULT_CENTER_MIN = 0.0
DEFAULT_CENTER_MAX = 1.0
DEFAULT_CENTER_COUNT = 11
DEFAULT_WIDTH_MIN = 0.0
DEFAULT_WIDTH_MAX = 2.0
DEFAULT_WIDTH_COUNT = 11

DEFAULT_TARGET_LOGICAL_ERRORS = 1
DEFAULT_MAX_SHOTS_PER_POINT = 500_000
DEFAULT_BASE_RELAY_SEED = 0


def interval_from_center_width(center: float, width: float) -> tuple[float, float]:
    if width < 0:
        raise ValueError("width must be non-negative")
    low = float(center) - float(width) / 2.0
    high = float(center) + float(width) / 2.0
    return low, high


def interval_to_center_width(low: float, high: float) -> tuple[float, float]:
    return (float(low + high) / 2.0, float(high - low))


def clip_unit_interval(low: float, high: float) -> tuple[float, float, bool]:
    clipped_low = max(0.0, float(low))
    clipped_high = min(1.0, float(high))
    clipped = (not math.isclose(clipped_low, float(low))) or (
        not math.isclose(clipped_high, float(high))
    )
    return clipped_low, clipped_high, clipped


def make_decoder_name(center: float, width: float) -> str:
    return (
        "relay-bp-paper-damping-"
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
            raw_low, raw_high = interval_from_center_width(float(center), float(width))
            interval_low, interval_high, interval_clipped = clip_unit_interval(
                raw_low, raw_high
            )
            effective_center, effective_width = interval_to_center_width(
                interval_low, interval_high
            )
            points.append(
                {
                    "point_id": int(point_id),
                    "width_idx": int(width_idx),
                    "center_idx": int(center_idx),
                    "center": float(center),
                    "width": float(width),
                    "raw_interval_low": float(raw_low),
                    "raw_interval_high": float(raw_high),
                    "interval_low": float(interval_low),
                    "interval_high": float(interval_high),
                    "effective_center": float(effective_center),
                    "effective_width": float(effective_width),
                    "interval_clipped": bool(interval_clipped),
                    "decoder": make_decoder_name(float(center), float(width)),
                    "relay_seed": int(base_relay_seed + 1_000_003 * point_id),
                }
            )
    return points


def build_point_decoder(
    *,
    point: dict[str, Any],
    num_sets: int,
    num_fault_mechanisms: int,
    num_edges: int,
    alpha: float,
    gamma0: float,
    pre_iter: int,
    set_max_iter: int,
    relay_posteriors: bool,
    stop_nconv: int,
) -> SinterDecoder_RelayBP:
    explicit_c_damp_messages = None
    c_damp_dist_interval: tuple[float, float] | None = (
        float(point["interval_low"]),
        float(point["interval_high"]),
    )
    if math.isclose(float(point["effective_width"]), 0.0, abs_tol=1e-12):
        explicit_c_damp_messages = np.full(
            (num_sets + 1, num_edges),
            float(point["effective_center"]),
            dtype=np.float64,
        )
        c_damp_dist_interval = None

    return SinterDecoder_RelayBP(
        alpha=alpha,
        gamma0=gamma0,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        gamma_dist_interval=(0.0, 1e-12),
        explicit_gammas=np.zeros(
            (num_sets, num_fault_mechanisms),
            dtype=np.float64,
        ),
        explicit_c_damp_messages=explicit_c_damp_messages,
        c_damp_dist_interval=c_damp_dist_interval,
        relay_posteriors=relay_posteriors,
        stop_nconv=stop_nconv,
        stopping_criterion="nconv",
        logging=False,
        seed=int(point["relay_seed"]),
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
    num_detectors: int,
    num_observables: int,
    num_fault_mechanisms: int,
    num_edges: int,
    num_workers: int,
    reference_interval_low: float,
    reference_interval_high: float,
) -> dict[str, Any]:
    reference_center, reference_width = interval_to_center_width(
        reference_interval_low,
        reference_interval_high,
    )
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
        "memory_disabled": True,
        "explicit_gamma_mode": "all_zero",
        "reference_interval_low": float(reference_interval_low),
        "reference_interval_high": float(reference_interval_high),
        "reference_interval_center": float(reference_center),
        "reference_interval_width": float(reference_width),
        "damping_interval_clipping": "clip_to_[0,1]",
        "num_detectors": int(num_detectors),
        "num_observables": int(num_observables),
        "num_fault_mechanisms": int(num_fault_mechanisms),
        "num_edges": int(num_edges),
        "num_workers": int(num_workers),
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
        "memory_disabled",
        "explicit_gamma_mode",
        "reference_interval_low",
        "reference_interval_high",
        "damping_interval_clipping",
    ]
    mismatches = [
        field
        for field in fields_to_compare
        if existing_setup.get(field) != current_setup.get(field)
    ]
    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise ValueError(
            "Existing damping heatmap outputs were created with different settings: "
            f"{mismatch_text}. Use --reset-data or choose a different output directory."
        )


def samples_to_df_with_point_metadata(
    *,
    samples: list[sinter.TaskStats],
    points: list[dict[str, Any]],
    points_by_decoder: dict[str, dict[str, Any]],
    target_logical_errors: int,
    max_shots_per_point: int,
) -> pd.DataFrame:
    summary_df = samples_to_df(
        samples=samples,
        points_by_decoder=points_by_decoder,
        target_logical_errors=target_logical_errors,
        max_shots_per_point=max_shots_per_point,
    )
    metadata_df = pd.DataFrame(
        [
            {
                "decoder": point["decoder"],
                "raw_interval_low": point["raw_interval_low"],
                "raw_interval_high": point["raw_interval_high"],
                "effective_center": point["effective_center"],
                "effective_width": point["effective_width"],
                "interval_clipped": point["interval_clipped"],
            }
            for point in points
        ]
    )
    if summary_df.empty:
        for column in metadata_df.columns:
            if column not in summary_df.columns:
                summary_df[column] = pd.Series(dtype=metadata_df[column].dtype)
        return summary_df
    return summary_df.merge(metadata_df, on="decoder", how="left")


def damping_grid_to_artifact(
    grid_df: pd.DataFrame,
    *,
    center_values: np.ndarray,
    width_values: np.ndarray,
) -> dict[str, np.ndarray]:
    artifact = grid_to_artifact(
        grid_df,
        center_values=center_values,
        width_values=width_values,
    )
    shape = (len(width_values), len(center_values))
    artifact["interval_clipped"] = np.full(shape, np.nan, dtype=np.float64)
    artifact["effective_interval_low"] = np.full(shape, np.nan, dtype=np.float64)
    artifact["effective_interval_high"] = np.full(shape, np.nan, dtype=np.float64)
    artifact["effective_center"] = np.full(shape, np.nan, dtype=np.float64)
    artifact["effective_width"] = np.full(shape, np.nan, dtype=np.float64)
    for row in grid_df.itertuples(index=False):
        artifact["interval_clipped"][row.width_idx, row.center_idx] = (
            1.0 if bool(row.interval_clipped) else 0.0
        )
        artifact["effective_interval_low"][row.width_idx, row.center_idx] = float(
            row.interval_low
        )
        artifact["effective_interval_high"][row.width_idx, row.center_idx] = float(
            row.interval_high
        )
        artifact["effective_center"][row.width_idx, row.center_idx] = float(
            row.effective_center
        )
        artifact["effective_width"][row.width_idx, row.center_idx] = float(
            row.effective_width
        )
    return artifact


def overlay_damping_guides(
    ax: plt.Axes,
    *,
    center_values: np.ndarray,
    width_values: np.ndarray,
    reference_center: float,
    reference_width: float,
) -> None:
    domain_low = max(0.0, float(np.min(center_values)))
    domain_high = min(1.0, float(np.max(center_values)))
    if domain_high > domain_low:
        guide_centers = np.linspace(domain_low, domain_high, 512, dtype=np.float64)
        guide_widths = 2.0 * np.minimum(guide_centers, 1.0 - guide_centers)
        guide_widths = np.minimum(guide_widths, float(np.max(width_values)))
        ax.plot(
            guide_centers,
            guide_widths,
            linestyle="--",
            linewidth=1.4,
            color="white",
            alpha=0.9,
            label="no-clipping boundary",
        )
    ax.plot(
        [reference_center],
        [reference_width],
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor="white",
        markeredgewidth=2.0,
        linestyle="none",
        label="reference damping interval",
    )


def plot_damping_heatmap(
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
    relay_posteriors: bool,
    reference_interval_low: float,
    reference_interval_high: float,
) -> None:
    center_edges = heatmap_edges(center_values)
    width_edges = heatmap_edges(width_values)
    reference_center, reference_width = interval_to_center_width(
        reference_interval_low,
        reference_interval_high,
    )
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
    overlay_damping_guides(
        ax,
        center_values=center_values,
        width_values=width_values,
        reference_center=reference_center,
        reference_width=reference_width,
    )
    ax.set_xlabel("damping center")
    ax.set_ylabel("damping width")
    ax.set_title(
        "BP+damping logical error rate | "
        f"{metadata.get('circuit', 'unknown')} | p = {selected_p:.3g} | basis filter = {basis_filter}"
    )
    subtitle = (
        f"gamma0 = {gamma0:.3f}, pre_iter = {pre_iter}, total_legs = {num_sets + 1}, "
        f"set_max_iter = {set_max_iter}, stop_nconv = {stop_nconv}\n"
        f"explicit relay gammas = 0, relay_posteriors = {relay_posteriors}, "
        "damping intervals clipped to [0, 1]"
    )
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("logical error rate")
    ax.legend(loc="upper right")
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
    reference_interval_low: float,
    reference_interval_high: float,
) -> None:
    center_edges = heatmap_edges(center_values)
    width_edges = heatmap_edges(width_values)
    reference_center, reference_width = interval_to_center_width(
        reference_interval_low,
        reference_interval_high,
    )
    ler_display, ler_norm = logical_error_lognorm(
        artifact["logical_error_rate"],
        max_shots_per_point=max_shots_per_point,
    )
    shots_display, shots_norm = positive_lognorm(artifact["shots"], floor=1.0)
    seconds_display, seconds_norm = positive_lognorm(artifact["seconds"], floor=1e-3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), squeeze=False)
    flat_axes = axes.ravel()

    specs = [
        ("logical error rate", ler_display, "viridis_r", ler_norm),
        ("shots", shots_display, "magma", shots_norm),
        ("logical errors observed", artifact["logical_error_count"], "viridis", None),
        ("wall-clock seconds", seconds_display, "magma", seconds_norm),
    ]

    for index, (ax, (title, data, cmap, norm)) in enumerate(
        zip(flat_axes, specs, strict=True)
    ):
        numeric_data = np.asarray(data, dtype=np.float64)
        if norm is None:
            finite_max = np.nanmax(numeric_data)
            vmax = float(max(1.0, finite_max)) if math.isfinite(finite_max) else 1.0
        else:
            vmax = None
        mesh = ax.pcolormesh(
            center_edges,
            width_edges,
            numeric_data,
            shading="auto",
            cmap=cmap,
            norm=norm,
            vmin=None if norm is not None else 0.0,
            vmax=vmax,
        )
        overlay_damping_guides(
            ax,
            center_values=center_values,
            width_values=width_values,
            reference_center=reference_center,
            reference_width=reference_width,
        )
        ax.set_title(title)
        ax.set_xlabel("damping center")
        ax.set_ylabel("damping width")
        if index == 0:
            ax.legend(loc="upper right")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "BP+damping collection summary heatmaps | "
        f"{metadata.get('circuit', 'unknown')} | p = {selected_p:.3g} | basis filter = {basis_filter}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    gamma0: float,
    pre_iter: int,
    num_sets: int,
    set_max_iter: int,
    stop_nconv: int,
    max_shots_per_point: int,
    relay_posteriors: bool,
    reference_interval_low: float,
    reference_interval_high: float,
) -> None:
    grid_csv = output_dir / f"{save_stem}_grid.csv"
    best_csv = output_dir / f"{save_stem}_best_points.csv"
    reference_point_csv = output_dir / f"{save_stem}_reference_interval_nearest_point.csv"
    artifact_npz = output_dir / f"{save_stem}_artifact.npz"
    plots_dir = output_dir / "plots"
    main_plot_png = plots_dir / "paper_style_logical_error_heatmap.png"
    collection_plot_png = plots_dir / "collection_heatmaps.png"

    summary_df.sort_values(["width_idx", "center_idx"]).to_csv(grid_csv, index=False)
    best_df = best_points_table(summary_df)
    best_df.to_csv(best_csv, index=False)

    if summary_df.empty:
        return

    reference_center, reference_width = interval_to_center_width(
        reference_interval_low,
        reference_interval_high,
    )
    reference_dist2 = (
        (summary_df["center"] - reference_center) ** 2
        + (summary_df["width"] - reference_width) ** 2
    )
    nearest_reference_point = summary_df.loc[[int(reference_dist2.idxmin())]].copy()
    nearest_reference_point.insert(0, "reference_interval_center", reference_center)
    nearest_reference_point.insert(1, "reference_interval_width", reference_width)
    nearest_reference_point.to_csv(reference_point_csv, index=False)

    artifact = damping_grid_to_artifact(
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

    plot_damping_heatmap(
        artifact=artifact,
        center_values=center_values,
        width_values=width_values,
        metadata=metadata,
        selected_p=selected_p,
        basis_filter=basis_filter,
        output_path=main_plot_png,
        max_shots_per_point=max_shots_per_point,
        gamma0=gamma0,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        stop_nconv=stop_nconv,
        relay_posteriors=relay_posteriors,
        reference_interval_low=reference_interval_low,
        reference_interval_high=reference_interval_high,
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
        reference_interval_low=reference_interval_low,
        reference_interval_high=reference_interval_high,
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
            "examples/notebook_data/bicycle_bivariate_144_12_12_paper_damping_heatmap_cluster_Z_basis "
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
    parser.add_argument(
        "--reference-interval-low",
        type=float,
        default=REFERENCE_DAMP_INTERVAL_LOW,
        help="Reference damping interval low endpoint used for the plot marker and nearest-point CSV.",
    )
    parser.add_argument(
        "--reference-interval-high",
        type=float,
        default=REFERENCE_DAMP_INTERVAL_HIGH,
        help="Reference damping interval high endpoint used for the plot marker and nearest-point CSV.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path.cwd().resolve())
    circuits_dir = repo_root / "tests" / "testdata" / "bicycle_bivariate"
    save_stem = "bicycle_bivariate_144_12_12_paper_damping_heatmap_cluster_Z_basis"
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
    reference_point_csv = output_dir / f"{save_stem}_reference_interval_nearest_point.csv"
    artifact_npz = output_dir / f"{save_stem}_artifact.npz"
    setup_json = output_dir / f"{save_stem}_setup.json"
    selected_circuit_csv = output_dir / "selected_circuit.csv"
    point_manifest_csv = output_dir / f"{save_stem}_point_manifest.csv"
    main_plot_png = plots_dir / "paper_style_logical_error_heatmap.png"
    collection_plot_png = plots_dir / "collection_heatmaps.png"

    if args.reset_data:
        for path in [
            resume_csv,
            summary_csv,
            grid_csv,
            best_csv,
            reference_point_csv,
            artifact_npz,
            setup_json,
            selected_circuit_csv,
            point_manifest_csv,
            main_plot_png,
            collection_plot_png,
        ]:
            if path.exists():
                path.unlink()

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
    if args.reference_interval_high < args.reference_interval_low:
        raise ValueError("--reference-interval-high must be >= --reference-interval-low.")

    stim_path = find_circuit_path(circuits_dir, args.p, args.memory_basis)
    problem = load_filtered_problem(stim_path=stim_path, basis_filter=args.basis_filter)
    metadata = dict(problem["metadata"])
    metadata["stim_path"] = str(stim_path)
    metadata["memory_disabled"] = True
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
    point_manifest_df = pd.DataFrame(points)
    point_manifest_df.to_csv(point_manifest_csv, index=False)

    num_fault_mechanisms = int(problem["check_matrices"].check_matrix.shape[1])
    num_edges = int(problem["check_matrices"].check_matrix.nnz)
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
        num_detectors=int(problem["dem"].num_detectors),
        num_observables=int(problem["dem"].num_observables),
        num_fault_mechanisms=num_fault_mechanisms,
        num_edges=num_edges,
        num_workers=num_workers,
        reference_interval_low=args.reference_interval_low,
        reference_interval_high=args.reference_interval_high,
    )
    validate_resume_setup(setup_json, current_setup)
    setup_json.write_text(json.dumps(current_setup, indent=2, sort_keys=True), encoding="utf-8")

    initial_samples = load_samples_from_resume(resume_csv)
    initial_summary_df = samples_to_df_with_point_metadata(
        samples=initial_samples,
        points=points,
        points_by_decoder=points_by_decoder,
        target_logical_errors=args.target_logical_errors,
        max_shots_per_point=args.max_shots_per_point,
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
        gamma0=args.gamma0,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        stop_nconv=args.stop_nconv,
        max_shots_per_point=args.max_shots_per_point,
        relay_posteriors=relay_posteriors,
        reference_interval_low=args.reference_interval_low,
        reference_interval_high=args.reference_interval_high,
    )

    completed_decoders = set(
        initial_summary_df.loc[
            initial_summary_df["hit_target"] | initial_summary_df["shot_cap_reached"],
            "decoder",
        ].tolist()
    )
    num_clipped_points = int(point_manifest_df["interval_clipped"].sum())

    print(f"repo_root   = {repo_root}")
    print(f"circuits_dir= {circuits_dir}")
    print(f"stim_path   = {stim_path}")
    print(f"output_dir  = {output_dir}")
    print(f"plots_dir   = {plots_dir}")
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
    print("explicit_gammas = 0 everywhere (memory disabled)")
    print(f"relay_posteriors = {relay_posteriors}")
    print(
        "reference_damping_interval = "
        f"[{args.reference_interval_low:.3f}, {args.reference_interval_high:.3f}] "
        f"(center={interval_to_center_width(args.reference_interval_low, args.reference_interval_high)[0]:.3f}, "
        f"width={interval_to_center_width(args.reference_interval_low, args.reference_interval_high)[1]:.3f})"
    )
    print(f"center_values = {center_values.tolist()}")
    print(f"width_values  = {width_values.tolist()}")
    print(f"target_logical_errors = {args.target_logical_errors}")
    print(f"max_shots_per_point   = {args.max_shots_per_point}")
    print(f"clipped_points = {num_clipped_points}/{len(points)}")
    print(f"completed_points = {len(completed_decoders)}/{len(points)}")

    for step_index, point in enumerate(points, start=1):
        decoder_name = str(point["decoder"])
        if decoder_name in completed_decoders:
            print(
                f"\n=== Skipping damping heatmap point {step_index}/{len(points)}: "
                f"{decoder_name} (already complete) ==="
            )
            continue

        clipping_note = " clipped" if bool(point["interval_clipped"]) else ""
        print(
            f"\n=== Running damping heatmap point {step_index}/{len(points)}: {decoder_name} "
            f"| center={point['center']:.3f} width={point['width']:.3f} "
            f"| effective=[{point['interval_low']:.3f}, {point['interval_high']:.3f}]{clipping_note} ==="
        )
        decoder = build_point_decoder(
            point=point,
            num_sets=args.num_sets,
            num_fault_mechanisms=num_fault_mechanisms,
            num_edges=num_edges,
            alpha=args.alpha,
            gamma0=args.gamma0,
            pre_iter=args.pre_iter,
            set_max_iter=args.set_max_iter,
            relay_posteriors=relay_posteriors,
            stop_nconv=args.stop_nconv,
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
        summary_df = samples_to_df_with_point_metadata(
            samples=samples,
            points=points,
            points_by_decoder=points_by_decoder,
            target_logical_errors=args.target_logical_errors,
            max_shots_per_point=args.max_shots_per_point,
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
            gamma0=args.gamma0,
            pre_iter=args.pre_iter,
            num_sets=args.num_sets,
            set_max_iter=args.set_max_iter,
            stop_nconv=args.stop_nconv,
            max_shots_per_point=args.max_shots_per_point,
            relay_posteriors=relay_posteriors,
            reference_interval_low=args.reference_interval_low,
            reference_interval_high=args.reference_interval_high,
        )
        completed_decoders.add(decoder_name)

    final_samples = load_samples_from_resume(resume_csv)
    final_summary_df = samples_to_df_with_point_metadata(
        samples=final_samples,
        points=points,
        points_by_decoder=points_by_decoder,
        target_logical_errors=args.target_logical_errors,
        max_shots_per_point=args.max_shots_per_point,
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
        gamma0=args.gamma0,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        stop_nconv=args.stop_nconv,
        max_shots_per_point=args.max_shots_per_point,
        relay_posteriors=relay_posteriors,
        reference_interval_low=args.reference_interval_low,
        reference_interval_high=args.reference_interval_high,
    )

    print("\nDone.")
    print(f"setup_json       = {setup_json}")
    print(f"selected_circuit = {selected_circuit_csv}")
    print(f"point_manifest   = {point_manifest_csv}")
    print(f"resume_csv       = {resume_csv}")
    print(f"summary_csv      = {summary_csv}")
    print(f"grid_csv         = {grid_csv}")
    print(f"best_csv         = {best_csv}")
    print(f"reference_csv    = {reference_point_csv}")
    print(f"artifact_npz     = {artifact_npz}")
    print(f"main_plot_png    = {main_plot_png}")
    print(f"collection_png   = {collection_plot_png}")


if __name__ == "__main__":
    main()
