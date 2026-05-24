#!/usr/bin/env python3
"""Render surface-code Tanner graph visuals for trained deterministic assignment banks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from relay_bp.analysis import load_code_capacity_problem

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".tmp" / "assignment_bank_visualizations_2026_04_23"

DEFAULT_RUN_SPECS = {
    "surface5_memory": {
        "code_path": Path(
            r"C:\Users\User\Documents\Code from BPGD\surface5_HxHzLxLz.npz"
        ),
        "run_dir": REPO_ROOT
        / ".tmp"
        / "assignment_bank_decision_2026_04_23"
        / "surface5_raw"
        / "p_0p11"
        / "memory"
        / "bank_4"
        / "restart_0",
        "family": "memory",
        "label": "surface5",
        "p_value": 0.11,
    },
    "surface5_damping": {
        "code_path": Path(
            r"C:\Users\User\Documents\Code from BPGD\surface5_HxHzLxLz.npz"
        ),
        "run_dir": REPO_ROOT
        / ".tmp"
        / "assignment_bank_decision_2026_04_23"
        / "surface5_raw"
        / "p_0p11"
        / "damping"
        / "bank_4"
        / "restart_0",
        "family": "damping",
        "label": "surface5",
        "p_value": 0.11,
    },
    "surface5_joint": {
        "code_path": Path(
            r"C:\Users\User\Documents\Code from BPGD\surface5_HxHzLxLz.npz"
        ),
        "run_dir": REPO_ROOT
        / ".tmp"
        / "assignment_bank_decision_2026_04_23"
        / "surface5_raw"
        / "p_0p11"
        / "joint"
        / "bank_4"
        / "restart_0",
        "family": "joint",
        "label": "surface5",
        "p_value": 0.11,
    },
    "surface13_memory": {
        "code_path": Path(
            r"C:\Users\User\Documents\Code from BPGD\surface13_HxHzLxLz.npz"
        ),
        "run_dir": REPO_ROOT
        / ".tmp"
        / "assignment_bank_decision_2026_04_23"
        / "surface13_raw"
        / "p_0p03"
        / "memory"
        / "bank_8"
        / "restart_0",
        "family": "memory",
        "label": "surface13",
        "p_value": 0.03,
    },
    "surface13_damping": {
        "code_path": Path(
            r"C:\Users\User\Documents\Code from BPGD\surface13_HxHzLxLz.npz"
        ),
        "run_dir": REPO_ROOT
        / ".tmp"
        / "assignment_bank_decision_2026_04_23"
        / "surface13_raw"
        / "p_0p03"
        / "damping"
        / "bank_4"
        / "restart_0",
        "family": "damping",
        "label": "surface13",
        "p_value": 0.03,
    },
    "surface13_joint": {
        "code_path": Path(
            r"C:\Users\User\Documents\Code from BPGD\surface13_HxHzLxLz.npz"
        ),
        "run_dir": REPO_ROOT
        / ".tmp"
        / "assignment_bank_decision_2026_04_23"
        / "surface13_raw"
        / "p_0p03"
        / "joint"
        / "bank_4"
        / "restart_0",
        "family": "joint",
        "label": "surface13",
        "p_value": 0.03,
    },
}


@dataclass(frozen=True)
class VisualRunSpec:
    key: str
    code_path: Path
    run_dir: Path
    family: str
    label: str
    p_value: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--which",
        default=",".join(DEFAULT_RUN_SPECS),
        help="Comma-separated subset of default plot keys.",
    )
    return parser.parse_args()


def _load_run_spec(key: str) -> VisualRunSpec:
    payload = DEFAULT_RUN_SPECS[key]
    return VisualRunSpec(
        key=key,
        code_path=Path(payload["code_path"]),
        run_dir=Path(payload["run_dir"]),
        family=str(payload["family"]),
        label=str(payload["label"]),
        p_value=float(payload["p_value"]),
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_best_member_index(run_dir: Path) -> tuple[int, dict[str, Any]]:
    diagnostics = _load_json(run_dir / "bank_member_diagnostics.json")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError(f"Expected a non-empty diagnostics list in {run_dir}.")

    def key_fn(entry: dict[str, Any]) -> tuple[Any, ...]:
        random_metrics = entry["random_test"]
        half_metrics = entry["half"]
        return (
            -float(random_metrics["logical_success_rate"]),
            -float(half_metrics["logical_success_rate"]),
            -float(random_metrics["convergence_rate"]),
            -float(half_metrics["convergence_rate"]),
            float(random_metrics["mean_iterations"]),
            float(half_metrics["mean_iterations"]),
            int(entry["bank_idx"]),
        )

    best = min(diagnostics, key=key_fn)
    return int(best["bank_idx"]), best


def _build_tanner_graph_positions(problem) -> tuple[nx.Graph, dict[str, np.ndarray]]:
    graph = nx.Graph()
    for check_idx in range(problem.hz.shape[0]):
        graph.add_node(f"c{check_idx}", kind="check", index=int(check_idx))
    hz = problem.hz.tocsc()
    for bit_idx in range(problem.n_bits):
        graph.add_node(f"v{bit_idx}", kind="variable", index=int(bit_idx))
        start = int(hz.indptr[bit_idx])
        end = int(hz.indptr[bit_idx + 1])
        for edge_idx in range(start, end):
            check_idx = int(hz.indices[edge_idx])
            graph.add_edge(
                f"c{check_idx}",
                f"v{bit_idx}",
                edge_idx=int(edge_idx),
                check_idx=int(check_idx),
                bit_idx=int(bit_idx),
            )
    try:
        raw_pos = nx.kamada_kawai_layout(graph)
    except Exception:
        raw_pos = nx.spring_layout(graph, seed=0, iterations=500)
    xs = np.asarray(
        [float(raw_pos[node][0]) for node in graph.nodes()], dtype=np.float64
    )
    ys = np.asarray(
        [float(raw_pos[node][1]) for node in graph.nodes()], dtype=np.float64
    )
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    positions = {
        node: np.asarray(
            [
                (float(raw_pos[node][0]) - min_x) / span_x,
                1.0 - (float(raw_pos[node][1]) - min_y) / span_y,
            ],
            dtype=np.float64,
        )
        for node in graph.nodes()
    }
    return graph, positions


def _value_norm(values: np.ndarray) -> tuple[Normalize, tuple[float, float]]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        pad = 0.5 if np.isclose(vmin, 0.0) else max(abs(vmin) * 0.05, 1e-6)
        vmin -= pad
        vmax += pad
    return Normalize(vmin=vmin, vmax=vmax), (vmin, vmax)


def _load_bank_arrays(
    spec: VisualRunSpec, member_idx: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    training_dir = spec.run_dir / "training"
    memory = None
    damping = None
    if spec.family in {"memory", "joint"}:
        bank = np.load(training_dir / "best_memory_bank.npy")
        memory = np.asarray(bank[int(member_idx)], dtype=np.float64)
    if spec.family in {"damping", "joint"}:
        bank = np.load(training_dir / "best_damping_bank.npy")
        damping = np.asarray(bank[int(member_idx)], dtype=np.float64)
    return memory, damping


def _draw_assignment_panel(
    *,
    fig: plt.Figure,
    ax: plt.Axes,
    graph: nx.Graph,
    positions: dict[str, np.ndarray],
    title: str,
    memory_vector: np.ndarray | None,
    damping_vector: np.ndarray | None,
) -> dict[str, Any]:
    neutral_edge = "#9ba1a6"
    neutral_node = "#8f959a"
    cmap = plt.cm.Greys

    edge_segments = []
    edge_colors = []
    edge_widths = []
    damping_range = None
    damping_norm = None
    if damping_vector is not None:
        damping_norm, damping_range = _value_norm(
            np.asarray(damping_vector, dtype=np.float64)
        )
    for left, right, data in graph.edges(data=True):
        edge_segments.append([positions[left], positions[right]])
        if damping_vector is None:
            edge_colors.append(neutral_edge)
            edge_widths.append(1.6)
        else:
            value = float(damping_vector[int(data["edge_idx"])])
            edge_colors.append(cmap(damping_norm(value)))
            edge_widths.append(2.0)
    line_collection = LineCollection(
        edge_segments, colors=edge_colors, linewidths=edge_widths, zorder=1
    )
    ax.add_collection(line_collection)

    check_nodes = [
        node for node, data in graph.nodes(data=True) if data["kind"] == "check"
    ]
    variable_nodes = [
        node for node, data in graph.nodes(data=True) if data["kind"] == "variable"
    ]
    check_xy = np.asarray([positions[node] for node in check_nodes], dtype=np.float64)
    variable_xy = np.asarray(
        [positions[node] for node in variable_nodes], dtype=np.float64
    )
    scale = max(1.0, np.sqrt(len(variable_nodes)) / 6.0)
    check_size = 34.0 / scale
    variable_size = 28.0 / scale
    ax.scatter(
        check_xy[:, 0],
        check_xy[:, 1],
        s=check_size,
        c=neutral_node,
        edgecolors="#666b70",
        linewidths=0.5,
        zorder=3,
    )

    memory_range = None
    memory_norm = None
    if memory_vector is None:
        variable_colors = neutral_node
    else:
        memory_norm, memory_range = _value_norm(
            np.asarray(memory_vector, dtype=np.float64)
        )
        variable_colors = [
            cmap(memory_norm(float(memory_vector[int(graph.nodes[node]["index"])])))
            for node in variable_nodes
        ]
    ax.scatter(
        variable_xy[:, 0],
        variable_xy[:, 1],
        s=variable_size,
        c=variable_colors,
        edgecolors="#666b70",
        linewidths=0.45,
        zorder=4,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    colorbar_payload: dict[str, Any] = {}
    if memory_norm is not None:
        memory_sm = ScalarMappable(norm=memory_norm, cmap=cmap)
        cbar = fig.colorbar(memory_sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("memory", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        colorbar_payload["memory_range"] = [
            float(memory_range[0]),
            float(memory_range[1]),
        ]
    if damping_norm is not None:
        damping_sm = ScalarMappable(norm=damping_norm, cmap=cmap)
        cbar = fig.colorbar(
            damping_sm,
            ax=ax,
            fraction=0.046,
            pad=0.08 if memory_norm is not None else 0.02,
        )
        cbar.set_label("damping", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        colorbar_payload["damping_range"] = [
            float(damping_range[0]),
            float(damping_range[1]),
        ]
    return colorbar_payload


def _render_single_plot(
    *,
    spec: VisualRunSpec,
    graph: nx.Graph,
    positions: dict[str, np.ndarray],
    memory_vector: np.ndarray | None,
    damping_vector: np.ndarray | None,
    member_idx: int,
    metrics: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    title = (
        f"{spec.label} | {spec.family} | p={spec.p_value:g} | member {member_idx}\n"
        f"logical={metrics['random_test']['logical_success_rate']:.3f}, "
        f"conv={metrics['random_test']['convergence_rate']:.3f}, "
        f"iter={metrics['random_test']['mean_iterations']:.3f}"
    )
    payload = _draw_assignment_panel(
        fig=fig,
        ax=ax,
        graph=graph,
        positions=positions,
        title=title,
        memory_vector=memory_vector,
        damping_vector=damping_vector,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return payload


def main() -> None:
    args = _parse_args()
    selected_keys = [
        item.strip() for item in str(args.which).split(",") if item.strip()
    ]
    specs = [_load_run_spec(key) for key in selected_keys]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layout_cache: dict[Path, tuple[nx.Graph, dict[str, np.ndarray]]] = {}
    summary: dict[str, Any] = {"plots": {}}
    for spec in specs:
        if spec.code_path not in layout_cache:
            problem = load_code_capacity_problem(spec.code_path)
            layout_cache[spec.code_path] = _build_tanner_graph_positions(problem)
        graph, positions = layout_cache[spec.code_path]
        member_idx, metrics = _select_best_member_index(spec.run_dir)
        memory_vector, damping_vector = _load_bank_arrays(spec, member_idx)
        output_path = args.output_dir / f"{spec.key}.png"
        colorbar_payload = _render_single_plot(
            spec=spec,
            graph=graph,
            positions=positions,
            memory_vector=memory_vector,
            damping_vector=damping_vector,
            member_idx=member_idx,
            metrics=metrics,
            output_path=output_path,
        )
        summary["plots"][spec.key] = {
            "family": spec.family,
            "code_label": spec.label,
            "p_value": spec.p_value,
            "member_idx": member_idx,
            "output_path": str(output_path.resolve()),
            "run_dir": str(spec.run_dir.resolve()),
            "random_test": metrics["random_test"],
            "half": metrics["half"],
            **colorbar_payload,
        }

    summary_path = args.output_dir / "visual_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
