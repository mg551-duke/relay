from __future__ import annotations

import json
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _lines(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.split("\n")]


def _markdown(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text: str, *, tags: list[str] | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if tags:
        metadata["tags"] = tags
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": _lines(text),
    }


def _notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(_notebook(cells), indent=2) + "\n", encoding="utf-8")


SETUP_CELL = r"""
import importlib
from math import comb
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm
"""


CONFIG_CELL = r"""
def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
            return candidate
    raise FileNotFoundError("Could not locate the relay repo root from the current working directory.")


REPO_ROOT = find_repo_root(Path.cwd())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from relay_bp.analysis.code_capacity_common import default_export_code_paths

CODE_PATHS = {
    "hgp_625": Path(r"C:\Users\User\Documents\projects-git\giulio\hgp_code_625_25_6_peg.npz"),
    "hgp_225": Path(r"C:\Users\User\Documents\cluster_decoder\cluster_decoder\codes\hgp_code_225.npz"),
    "surface5": Path(r"C:\Users\User\Documents\Code from BPGD\surface5_HxHzLxLz.npz"),
    "surface13": Path(r"C:\Users\User\Documents\Code from BPGD\surface13_HxHzLxLz.npz"),
    "B1": Path(r"C:\Users\User\Documents\projects-git\bpgd_low_llr_and_hybrid\B1_HxHzLxLz.npz"),
    "gross": Path(r"C:\Users\User\Documents\projects-git\bpgd_low_llr_and_hybrid\gross_HxHzLxLz.npz"),
    "two_gross": Path(r"C:\Users\User\Documents\projects-git\bpgd_low_llr_and_hybrid\two_gross_HxHzLxLz.npz"),
}
for fallback_key, fallback_path in default_export_code_paths(REPO_ROOT).items():
    if fallback_key not in CODE_PATHS or not CODE_PATHS[fallback_key].exists():
        CODE_PATHS[fallback_key] = fallback_path

available_code_keys = [key for key, path in CODE_PATHS.items() if Path(path).exists()]
if not available_code_keys:
    raise FileNotFoundError("No code paths are available. Update CODE_PATHS for this machine.")

selected_code = "gross"  # change here
if selected_code not in CODE_PATHS or not Path(CODE_PATHS[selected_code]).exists():
    selected_code = available_code_keys[0]
npz_path = CODE_PATHS[selected_code]

prior_error_rate = 0.10
max_iter = 40
alpha = 1.0
center_values = np.linspace(0.00, 1.00, 11)
width_values = np.linspace(0.00, 2.00, 11)
random_draws_per_point = 2
n_error_samples = 300
base_seed = 0

show_progress = True
show_inner_case_progress = False
selected_row_weights = None
case_limit = None
sample_limit = None

weak_memory_interval = (-0.10, 0.10)
weak_damping_interval = (0.85, 0.95)

assert np.all(np.diff(center_values) >= 0), "center_values must be sorted."
assert np.all(np.diff(width_values) >= 0), "width_values must be sorted."
assert np.all(width_values >= 0.0), "width_values must be non-negative."
"""


COMMON_HELPERS_CELL = r"""
def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
            return candidate
    raise FileNotFoundError("Could not locate the relay repo root from the current working directory.")


REPO_ROOT = globals().get("REPO_ROOT", find_repo_root(Path.cwd()))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import relay_bp
import relay_bp.analysis.code_capacity_common as code_capacity_common_module
import relay_bp.analysis.code_capacity_edge_weights as code_capacity_edge_weights_module

importlib.reload(code_capacity_common_module)
importlib.reload(code_capacity_edge_weights_module)

from relay_bp.analysis.code_capacity_common import (
    default_export_code_paths,
    decode_with_edge_message_weights,
    decode_with_memory_strengths,
    enumerate_half_stabilizer_cases,
    evaluate_decode_result,
    load_code_capacity_problem,
    make_min_sum_tracer,
    sample_random_error_cases,
)
from relay_bp.analysis.code_capacity_edge_weights import (
    best_edge_weight_rows,
    build_memory_strengths,
    edge_weight_heatmap_rows,
    evaluate_edge_weight_heatmap,
    interval_from_center_width,
)

CODE_PATHS = globals().get("CODE_PATHS", {})
for fallback_key, fallback_path in default_export_code_paths(REPO_ROOT).items():
    if fallback_key not in CODE_PATHS or not Path(CODE_PATHS[fallback_key]).exists():
        CODE_PATHS[fallback_key] = fallback_path
selected_code = globals().get("selected_code", "gross")
if selected_code not in CODE_PATHS or not Path(CODE_PATHS[selected_code]).exists():
    selected_code = next(key for key, path in CODE_PATHS.items() if Path(path).exists())
npz_path = Path(globals().get("npz_path", CODE_PATHS[selected_code]))

prior_error_rate = float(globals().get("prior_error_rate", 0.10))
max_iter = int(globals().get("max_iter", 40))
alpha = float(globals().get("alpha", 1.0))
center_values = np.asarray(globals().get("center_values", np.linspace(0.00, 1.00, 11)), dtype=np.float64)
width_values = np.asarray(globals().get("width_values", np.linspace(0.00, 2.00, 11)), dtype=np.float64)
random_draws_per_point = int(globals().get("random_draws_per_point", 2))
n_error_samples = int(globals().get("n_error_samples", 300))
base_seed = int(globals().get("base_seed", 0))
show_progress = bool(globals().get("show_progress", False))
show_inner_case_progress = bool(globals().get("show_inner_case_progress", False))
selected_row_weights = globals().get("selected_row_weights", None)
case_limit = globals().get("case_limit", None)
sample_limit = globals().get("sample_limit", None)
weak_memory_interval = tuple(globals().get("weak_memory_interval", (-0.10, 0.10)))
weak_damping_interval = tuple(globals().get("weak_damping_interval", (0.85, 0.95)))


def load_css_code(path: Path) -> dict[str, object]:
    problem = load_code_capacity_problem(path, default_prior_error_rate=prior_error_rate)
    return {"problem": problem, "hx": problem.hx, "hz": problem.hz, "lz": problem.lz, "metadata": problem.metadata}


def dynamic_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        eps = 1e-6 if np.isclose(vmin, 0.0) else max(1e-6, abs(vmin) * 0.05)
        return (vmin - eps, vmax + eps)
    return (vmin, vmax)


def plot_heatmaps(artifact: dict[str, object], *, title: str):
    metrics = [
        ("logical_success_rate", "Logical success rate"),
        ("convergence_rate", "Convergence rate"),
        ("exact_recovery_rate", "Exact recovery rate"),
        ("mean_iterations", "Mean iterations"),
    ]
    centers = np.asarray(artifact["centers"], dtype=np.float64)
    widths = np.asarray(artifact["widths"], dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric_key, label) in zip(axes.flat, metrics):
        values = np.asarray(artifact[metric_key], dtype=np.float64)
        vmin, vmax = dynamic_limits(values)
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=[float(centers.min()), float(centers.max()), float(widths.min()), float(widths.max())],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(label)
        axis.set_xlabel("interval center")
        axis.set_ylabel("interval width")
        fig.colorbar(image, ax=axis, shrink=0.85)
    fig.suptitle(title, fontsize=14)
    return fig


def edge_grid_table(artifact: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(edge_weight_heatmap_rows(artifact)).sort_values(
        ["logical_success_rate", "convergence_rate", "exact_recovery_rate", "mean_iterations", "width", "center"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)


def edge_best_points_table(artifact: dict[str, object], top_k: int = 8) -> pd.DataFrame:
    return pd.DataFrame(best_edge_weight_rows(edge_weight_heatmap_rows(artifact), topk=top_k)).reset_index(drop=True)


code_data = load_css_code(npz_path)
hz_check_matrix = code_data["hz"]
lz_matrix = code_data["lz"]
"""


HALF_CASES_CELL = r"""
def stabilizer_statistics(hx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = hx.sum(axis=1).astype(int)
    unique_weights, counts = np.unique(weights, return_counts=True)
    rows = []
    total_even_weight_stabilizers = 0
    total_half_flip_patterns = 0
    for weight, count in zip(unique_weights, counts):
        weight = int(weight)
        count = int(count)
        is_even = weight > 0 and weight % 2 == 0
        half_subsets_per_row = comb(weight, weight // 2) if is_even else 0
        total_half = count * half_subsets_per_row
        if is_even:
            total_even_weight_stabilizers += count
            total_half_flip_patterns += total_half
        rows.append(
            {
                "row_weight": weight,
                "n_x_stabilizers": count,
                "is_even_weight": is_even,
                "half_flip_patterns_per_row": half_subsets_per_row,
                "total_half_flip_patterns": total_half,
            }
        )
    summary = pd.DataFrame(
        [
            {
                "code": selected_code,
                "Hx_rows": int(hx.shape[0]),
                "n_qubits": int(hx.shape[1]),
                "even_weight_x_stabilizers": total_even_weight_stabilizers,
                "total_half_flip_patterns": total_half_flip_patterns,
                "random_draws_per_point": random_draws_per_point,
                "n_center_values": len(center_values),
                "n_width_values": len(width_values),
            }
        ]
    )
    return pd.DataFrame(rows), summary


def case_manifest(cases: list[dict[str, object]], head: int | None = 20) -> pd.DataFrame:
    manifest = pd.DataFrame(
        [
            {
                "case_id": int(case["case_id"]),
                "row_idx": int(case["row_idx"]),
                "row_weight": int(case["row_weight"]),
                "half_subset_idx": int(case["half_subset_idx"]),
                "syndrome_weight": int(case["syndrome_weight"]),
                "support_bits": list(case["support_bits"]),
                "flipped_bits": list(case["flipped_bits"]),
            }
            for case in cases
        ]
    )
    return manifest.head(head) if head is not None else manifest


stabilizer_rows, stabilizer_summary = stabilizer_statistics(code_data["hx"])
all_cases = enumerate_half_stabilizer_cases(
    code_data["problem"],
    selected_row_weights=None if selected_row_weights is None else set(int(weight) for weight in selected_row_weights),
    max_cases=case_limit,
)
case_manifest_df = case_manifest(all_cases)
"""


EDGE_BINDING_SMOKE_CELL = r"""
_smoke_problem = load_code_capacity_problem(default_export_code_paths(REPO_ROOT)["surface13"])
_smoke_tracer = relay_bp.MinSumBPDecoderTraceF64(
    _smoke_problem.hz,
    error_priors=_smoke_problem.error_priors,
    max_iter=6,
    alpha=1.0,
)
if hasattr(_smoke_tracer, "set_explicit_edge_message_weights"):
    _smoke_cases = enumerate_half_stabilizer_cases(_smoke_problem, max_cases=4)
    _smoke_artifact = evaluate_edge_weight_heatmap(
        problem=_smoke_problem,
        cases=_smoke_cases,
        max_iter=6,
        alpha=1.0,
        centers=[0.2],
        widths=[0.4],
        base_seed=3,
        random_draws_per_point=1,
        support_only=True,
    )
    assert _smoke_artifact["logical_success_rate"].shape == (1, 1)
    assert int(_smoke_artifact["trial_count"][0, 0]) == len(_smoke_cases)
else:
    print("Skipping edge-weight smoke decode because the local relay_bp extension is stale.")
"""


def build_memory_half_heatmap_notebook() -> list[dict[str, object]]:
    return [
        _markdown(
            """
            # Code-Capacity Half-Stabilizer Memory Heat Maps

            This restores the original half-stabilizer memory-weight heat-map workflow. The swept interval is applied as per-bit memory strength, with support-local or all-bit application controlled by `memory_application_scope`.
            """
        ),
        _code(SETUP_CELL),
        _code(CONFIG_CELL + '\nmemory_application_scope = "stabilizer_support"  # options: "stabilizer_support", "all_bits"\n'),
        _code(
            COMMON_HELPERS_CELL
            + HALF_CASES_CELL
            + r"""
memory_application_scope = globals().get("memory_application_scope", "stabilizer_support")


def evaluate_memory_heatmap(cases: list[dict[str, object]]) -> dict[str, object]:
    tracer = make_min_sum_tracer(code_data["problem"], max_iter=max_iter, alpha=alpha, gamma0=0.0)
    shape = (len(width_values), len(center_values))
    logical_success_rate = np.full(shape, np.nan)
    convergence_rate = np.full(shape, np.nan)
    exact_recovery_rate = np.full(shape, np.nan)
    mean_iterations = np.full(shape, np.nan)
    trial_count = np.zeros(shape, dtype=np.int64)
    interval_lows = np.full(shape, np.nan)
    interval_highs = np.full(shape, np.nan)
    parameter_points = [
        (width_idx, float(width), center_idx, float(center))
        for width_idx, width in enumerate(width_values)
        for center_idx, center in enumerate(center_values)
    ]
    parameter_points_iter = tqdm(
        parameter_points,
        total=len(parameter_points),
        desc="memory-weight grid",
        disable=not show_progress,
    )
    for width_idx, width, center_idx, center in parameter_points_iter:
        interval = interval_from_center_width(float(center), float(width))
        interval_lows[width_idx, center_idx], interval_highs[width_idx, center_idx] = interval
        logical_sum = conv_sum = exact_sum = iter_sum = 0.0
        trials = 0
        cases_iter = tqdm(
            cases,
            total=len(cases),
            desc=f"w={width:.3f}, c={center:.3f}",
            leave=False,
            disable=(not show_progress) or (not show_inner_case_progress),
        )
        for case in cases_iter:
            support_bits = case["support_bits"] if memory_application_scope == "stabilizer_support" else None
            for draw_idx in range(random_draws_per_point):
                coeff_seed = base_seed + 1_000_003 * int(case["case_id"]) + 10_007 * width_idx + 1_009 * center_idx + draw_idx
                memory_strengths = build_memory_strengths(
                    code_data["problem"].n_bits,
                    interval=interval,
                    coeff_seed=coeff_seed,
                    support_bits=support_bits,
                )
                result = decode_with_memory_strengths(tracer, case["syndrome"], memory_strengths)
                evaluation = evaluate_decode_result(result=result, error=case["error"], lz=code_data["problem"].lz)
                logical_sum += float(evaluation.logical_success)
                conv_sum += float(evaluation.converged)
                exact_sum += float(evaluation.exact_recovery)
                iter_sum += float(evaluation.iterations)
                trials += 1
        logical_success_rate[width_idx, center_idx] = logical_sum / trials
        convergence_rate[width_idx, center_idx] = conv_sum / trials
        exact_recovery_rate[width_idx, center_idx] = exact_sum / trials
        mean_iterations[width_idx, center_idx] = iter_sum / trials
        trial_count[width_idx, center_idx] = trials
        parameter_points_iter.set_postfix(
            width=f"{width:.3f}",
            center=f"{center:.3f}",
            logical=f"{logical_success_rate[width_idx, center_idx]:.3f}",
            conv=f"{convergence_rate[width_idx, center_idx]:.3f}",
        )
    return {
        "centers": center_values,
        "widths": width_values,
        "interval_lows": interval_lows,
        "interval_highs": interval_highs,
        "logical_success_rate": logical_success_rate,
        "convergence_rate": convergence_rate,
        "exact_recovery_rate": exact_recovery_rate,
        "mean_iterations": mean_iterations,
        "trial_count": trial_count,
    }


def memory_rows(artifact: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for width_idx, width in enumerate(artifact["widths"]):
        for center_idx, center in enumerate(artifact["centers"]):
            rows.append(
                {
                    "center": float(center),
                    "width": float(width),
                    "low": float(artifact["interval_lows"][width_idx, center_idx]),
                    "high": float(artifact["interval_highs"][width_idx, center_idx]),
                    "logical_success_rate": float(artifact["logical_success_rate"][width_idx, center_idx]),
                    "convergence_rate": float(artifact["convergence_rate"][width_idx, center_idx]),
                    "exact_recovery_rate": float(artifact["exact_recovery_rate"][width_idx, center_idx]),
                    "mean_iterations": float(artifact["mean_iterations"][width_idx, center_idx]),
                    "trial_count": int(artifact["trial_count"][width_idx, center_idx]),
                }
            )
    return rows


def memory_grid_table(artifact: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(memory_rows(artifact)).sort_values(
        ["logical_success_rate", "convergence_rate", "exact_recovery_rate", "mean_iterations"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


_smoke_problem = load_code_capacity_problem(default_export_code_paths(REPO_ROOT)["surface13"])
_smoke_cases = enumerate_half_stabilizer_cases(_smoke_problem, max_cases=2)
_old_code_data = code_data
code_data = {"problem": _smoke_problem}
_smoke_artifact = evaluate_memory_heatmap(_smoke_cases)
assert _smoke_artifact["logical_success_rate"].shape == (len(width_values), len(center_values))
code_data = _old_code_data
""",
            tags=["smoke-test"],
        ),
        _markdown("## Half-Stabilizer Catalogue"),
        _code("display(stabilizer_summary)\ndisplay(stabilizer_rows)\ndisplay(case_manifest_df)"),
        _markdown("## Memory Heat Maps"),
        _code(
            """
artifact = evaluate_memory_heatmap(all_cases)
heatmap_df = memory_grid_table(artifact)
best_points_df = heatmap_df.head(8)

display(best_points_df)
display(heatmap_df.head(16))
fig = plot_heatmaps(artifact, title=f"{selected_code}: memory heat maps ({memory_application_scope})")
fig
"""
        ),
    ]


def build_edge_half_heatmap_notebook() -> list[dict[str, object]]:
    return [
        _markdown(
            """
            # Code-Capacity Edge-Weight Heat Maps For Half-Stabilizer Errors

            This notebook sweeps explicit edge-message weights on deterministic half-stabilizer `X` errors. The usual cells use no memory effect and no damping; the final cell adds weak memory and damping with all other parameters unchanged.
            """
        ),
        _code(SETUP_CELL),
        _code(CONFIG_CELL + '\nedge_weight_application_scope = "stabilizer_support"  # options: "stabilizer_support", "all_edges"\n'),
        _code(COMMON_HELPERS_CELL + HALF_CASES_CELL + EDGE_BINDING_SMOKE_CELL, tags=["smoke-test"]),
        _markdown("## Half-Stabilizer Catalogue"),
        _code("display(stabilizer_summary)\ndisplay(stabilizer_rows)\ndisplay(case_manifest_df)"),
        _markdown("## Edge-Weight Heat Maps"),
        _code(
            """
artifact = evaluate_edge_weight_heatmap(
    problem=code_data["problem"],
    cases=all_cases,
    max_iter=max_iter,
    alpha=alpha,
    centers=center_values.tolist(),
    widths=width_values.tolist(),
    base_seed=base_seed,
    random_draws_per_point=random_draws_per_point,
    support_only=edge_weight_application_scope == "stabilizer_support",
    show_progress=show_progress,
    show_inner_case_progress=show_inner_case_progress,
)
heatmap_df = edge_grid_table(artifact)
best_points_df = edge_best_points_table(artifact)

display(best_points_df)
display(heatmap_df.head(16))
fig = plot_heatmaps(artifact, title=f"{selected_code}: edge-weight heat maps ({edge_weight_application_scope})")
fig
"""
        ),
        _markdown("## Weak Memory + Damping Extension"),
        _code(
            """
weak_joint_artifact = evaluate_edge_weight_heatmap(
    problem=code_data["problem"],
    cases=all_cases,
    max_iter=max_iter,
    alpha=alpha,
    centers=center_values.tolist(),
    widths=width_values.tolist(),
    base_seed=base_seed,
    random_draws_per_point=random_draws_per_point,
    support_only=edge_weight_application_scope == "stabilizer_support",
    memory_interval=weak_memory_interval,
    damping_interval=weak_damping_interval,
    show_progress=show_progress,
    show_inner_case_progress=show_inner_case_progress,
)
weak_joint_heatmap_df = edge_grid_table(weak_joint_artifact)
weak_joint_best_points_df = edge_best_points_table(weak_joint_artifact)

display(pd.DataFrame([{
    "weak_memory_low": float(weak_memory_interval[0]),
    "weak_memory_high": float(weak_memory_interval[1]),
    "weak_damping_low": float(weak_damping_interval[0]),
    "weak_damping_high": float(weak_damping_interval[1]),
}]))
display(weak_joint_best_points_df)
display(weak_joint_heatmap_df.head(16))
weak_joint_fig = plot_heatmaps(weak_joint_artifact, title=f"{selected_code}: edge weights + weak memory+damping")
weak_joint_fig
"""
        ),
    ]


def build_edge_performance_heatmap_notebook() -> list[dict[str, object]]:
    return [
        _markdown(
            """
            # Code-Capacity Edge-Weight Performance Heat Maps

            This notebook uses random code-capacity `X` errors instead of half-stabilizer cases. The sweep is edge-message weights only, with a final weak-memory-plus-damping extension.
            """
        ),
        _code(SETUP_CELL),
        _code(CONFIG_CELL),
        _code(
            COMMON_HELPERS_CELL
            + r"""
random_error_cases = sample_random_error_cases(
    code_data["problem"],
    num_samples=n_error_samples,
    error_rate=prior_error_rate,
    base_seed=base_seed,
)
if sample_limit is not None:
    random_error_cases = random_error_cases[: int(sample_limit)]

sample_stats_df = pd.DataFrame([{
    "code": selected_code,
    "n_samples": len(random_error_cases),
    "prior_error_rate": prior_error_rate,
    "mean_error_weight": float(np.mean([case["error_weight"] for case in random_error_cases])),
    "mean_syndrome_weight": float(np.mean([case["syndrome_weight"] for case in random_error_cases])),
}])
"""
            + EDGE_BINDING_SMOKE_CELL,
            tags=["smoke-test"],
        ),
        _markdown("## Random Error Sample Summary"),
        _code("display(sample_stats_df)"),
        _markdown("## Edge-Weight Performance Heat Maps"),
        _code(
            """
artifact = evaluate_edge_weight_heatmap(
    problem=code_data["problem"],
    cases=random_error_cases,
    max_iter=max_iter,
    alpha=alpha,
    centers=center_values.tolist(),
    widths=width_values.tolist(),
    base_seed=base_seed,
    random_draws_per_point=random_draws_per_point,
    support_only=False,
    show_progress=show_progress,
    show_inner_case_progress=show_inner_case_progress,
)
heatmap_df = edge_grid_table(artifact)
best_points_df = edge_best_points_table(artifact)

display(best_points_df)
display(heatmap_df.head(16))
fig = plot_heatmaps(artifact, title=f"{selected_code}: random-error edge-weight heat maps")
fig
"""
        ),
        _markdown("## Weak Memory + Damping Extension"),
        _code(
            """
weak_joint_artifact = evaluate_edge_weight_heatmap(
    problem=code_data["problem"],
    cases=random_error_cases,
    max_iter=max_iter,
    alpha=alpha,
    centers=center_values.tolist(),
    widths=width_values.tolist(),
    base_seed=base_seed,
    random_draws_per_point=random_draws_per_point,
    support_only=False,
    memory_interval=weak_memory_interval,
    damping_interval=weak_damping_interval,
    show_progress=show_progress,
    show_inner_case_progress=show_inner_case_progress,
)
display(edge_best_points_table(weak_joint_artifact))
weak_joint_fig = plot_heatmaps(weak_joint_artifact, title=f"{selected_code}: random-error edge weights + weak memory+damping")
weak_joint_fig
"""
        ),
    ]


def build_edge_trace_notebook() -> list[dict[str, object]]:
    return [
        _markdown(
            """
            # Code-Capacity Edge-Weight Trace

            This notebook compares plain BP, scalar edge weighting, and random edge weighting on individual half-stabilizer cases.
            """
        ),
        _code(SETUP_CELL),
        _code(CONFIG_CELL + "\ntrace_center = 0.0\ntrace_width = 1.0\ntrace_seed_offsets = [0, 1, 2]\ncore_seed = 0\n"),
        _code(
            COMMON_HELPERS_CELL
            + r"""
case = enumerate_half_stabilizer_cases(code_data["problem"], max_cases=1)[0]
trace_interval = interval_from_center_width(trace_center, trace_width)
_supports_edge_weights = hasattr(
    relay_bp.MinSumBPDecoderTraceF64(
        code_data["problem"].hz,
        error_priors=code_data["problem"].error_priors,
        max_iter=6,
        alpha=1.0,
    ),
    "set_explicit_edge_message_weights",
)


def run_trace(case: dict[str, object], *, label: str, edge_weights: np.ndarray | None):
    tracer = make_min_sum_tracer(code_data["problem"], max_iter=max_iter, alpha=alpha, gamma0=None)
    snapshots = []
    tracer.reset()
    if edge_weights is not None:
        if not hasattr(tracer, "set_explicit_edge_message_weights"):
            raise RuntimeError("Rebuild relay_bp to expose explicit edge-message-weight bindings.")
        tracer.set_explicit_edge_message_weights(edge_weights)
    result = tracer.snapshot(case["syndrome"])
    snapshots.append(result)
    while tracer.current_iteration < result.max_iter and not result.success:
        result = tracer.run_iteration(case["syndrome"])
        snapshots.append(result)
    evaluation = evaluate_decode_result(result=result, error=case["error"], lz=code_data["problem"].lz)
    posterior_trace = np.vstack([np.asarray(snapshot.posterior_ratios, dtype=np.float64) for snapshot in snapshots])
    return {
        "label": label,
        "iterations": np.asarray([int(snapshot.iterations) for snapshot in snapshots]),
        "posterior_trace": posterior_trace,
        "final_success": bool(result.success),
        "logical_success": bool(evaluation.logical_success),
        "exact_recovery": bool(evaluation.exact_recovery),
        "final_iterations": int(result.iterations),
    }


trace_runs = [run_trace(case, label="plain BP", edge_weights=None)]
if _supports_edge_weights:
    same_weights = np.full(code_data["problem"].hz.nnz, trace_center, dtype=np.float64)
    trace_runs.append(run_trace(case, label=f"same edge weight {trace_center:.3f}", edge_weights=same_weights))
    for seed_offset in trace_seed_offsets:
        weights = code_capacity_common_module.build_edge_message_weights(
            code_data["problem"].hz,
            interval=trace_interval,
            coeff_seed=base_seed + int(seed_offset),
            support_bits=case["support_bits"],
        )
        trace_runs.append(run_trace(case, label=f"random edge weight seed {seed_offset}", edge_weights=weights))
else:
    print("Edge-weight trace needs a rebuilt relay_bp extension.")

trace_summary_df = pd.DataFrame([
    {
        "label": run["label"],
        "final_success": run["final_success"],
        "logical_success": run["logical_success"],
        "exact_recovery": run["exact_recovery"],
        "final_iterations": run["final_iterations"],
    }
    for run in trace_runs
])

assert len(trace_runs) >= 1
""",
            tags=["smoke-test"],
        ),
        _markdown("## Trace Summary"),
        _code("display(trace_summary_df)"),
        _markdown("## Posterior Traces"),
        _code(
            """
support_bits = list(case["support_bits"])
fig, axes = plt.subplots(len(trace_runs), 1, figsize=(10, 2.6 * len(trace_runs)), sharex=True, constrained_layout=True)
if len(trace_runs) == 1:
    axes = [axes]
for axis, run in zip(axes, trace_runs):
    trace = run["posterior_trace"][:, support_bits]
    for col_idx, bit in enumerate(support_bits):
        axis.plot(run["iterations"], trace[:, col_idx], marker="o", label=f"q{bit}")
    axis.set_title(run["label"])
    axis.set_ylabel("posterior ratio")
    axis.grid(alpha=0.25)
axes[-1].set_xlabel("iteration")
axes[0].legend(ncol=min(4, len(support_bits)), fontsize=8)
fig
"""
        ),
    ]


def main() -> None:
    _write_notebook(EXAMPLES_DIR / "CodeCapacityHalfStabilizerHeatmaps.ipynb", build_memory_half_heatmap_notebook())
    _write_notebook(
        EXAMPLES_DIR / "CodeCapacityEdgeWeightHalfStabilizerHeatmaps.ipynb",
        build_edge_half_heatmap_notebook(),
    )
    _write_notebook(
        EXAMPLES_DIR / "CodeCapacityEdgeWeightPerformanceHeatmaps.ipynb",
        build_edge_performance_heatmap_notebook(),
    )
    _write_notebook(EXAMPLES_DIR / "CodeCapacityEdgeWeightTrace.ipynb", build_edge_trace_notebook())


if __name__ == "__main__":
    main()
