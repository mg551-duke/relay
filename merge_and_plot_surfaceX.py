#!/usr/bin/env python3
"""
merge_and_plot_surfaceX.py

- Reads results_surface_x/surfaceX_seed*.csv
- Merges by strong_id (sums shots/errors/seconds/discards)
- Aggregates across runs to (d, p, decoder, xyz)
- Plots one figure per distance:
    mem-bp(XYZ=False/True),
    msl-bp(XYZ=False/True),
    relay-bp(XYZ=False/True),
    pymatching(XYZ=False)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import sinter


def parse_meta(s: str) -> dict:
    if s in ("null", "None", "", None):
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def read_and_merge(indir: Path) -> pd.DataFrame:
    files = sorted(indir.glob("surfaceX_seed*.csv"))
    if not files:
        raise FileNotFoundError(f"No surfaceX_seed*.csv in {indir}")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    # merge by strong_id to combine identical tasks across seeds
    numeric = [c for c in ["shots", "errors", "discards", "seconds"] if c in df.columns]
    keep_first = [c for c in df.columns if c not in numeric]
    agg = {c: "sum" for c in numeric}
    for c in keep_first:
        agg[c] = "first"
    merged = df.groupby("strong_id", as_index=False).agg(agg)
    return merged


def plot_per_distance(merged: pd.DataFrame, plotdir: Path):
    plotdir.mkdir(parents=True, exist_ok=True)

    metas = merged["json_metadata"].apply(parse_meta)
    merged = merged.copy()
    merged["d"] = metas.apply(lambda m: m.get("d"))
    merged["r"] = metas.apply(lambda m: m.get("r", m.get("d")))
    merged["p"] = metas.apply(lambda m: m.get("p"))
    merged["xyz"] = metas.apply(lambda m: m.get("xyz"))

    merged = merged.dropna(subset=["d", "r", "p", "xyz"])

    # Aggregate across any residual duplicates: (d,p,decoder,xyz,r)
    agg = merged.groupby(["d", "p", "decoder", "xyz", "r"], as_index=False).agg(
        shots=("shots", "sum"),
        errors=("errors", "sum"),
        discards=("discards", "sum") if "discards" in merged.columns else ("shots", "count"),
        seconds=("seconds", "sum") if "seconds" in merged.columns else ("shots", "count"),
    )

    # Make a TaskStats list so sinter.plot_error_rate can do error bars (if available)
    TaskStats = getattr(sinter, "TaskStats", None)
    use_sinter_plot = TaskStats is not None and hasattr(sinter, "plot_error_rate")

    stats = []
    if use_sinter_plot:
        for _, row in agg.iterrows():
            stats.append(TaskStats(
                shots=int(row["shots"]),
                errors=int(row["errors"]),
                discards=int(row.get("discards", 0)),
                seconds=float(row.get("seconds", 0.0)),
                decoder=str(row["decoder"]),
                strong_id=f'd={int(row["d"])}|p={float(row["p"])}|dec={row["decoder"]}|xyz={row["xyz"]}',
                json_metadata={"d": int(row["d"]), "p": float(row["p"]), "r": int(row["r"]), "xyz": bool(row["xyz"])},
            ))

    for d in sorted(agg["d"].unique()):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Probability (per round/qubit)")
        ax.set_title(f"rotated_surface_code_memory_X — LER vs PER (d={int(d)})")

        if use_sinter_plot:
            stats_d = [s for s in stats if s.json_metadata["d"] == int(d)]

            def group_func(s):
                if s.decoder == "pymatching":
                    return "pymatching(XYZ=False)"
                return f'{s.decoder}(XYZ={s.json_metadata["xyz"]})'

            def x_func(s):
                return float(s.json_metadata["p"])

            def failure_units_per_shot_func(s):
                # divide errors/shots by rounds to get per-round
                return float(s.json_metadata.get("r", s.json_metadata["d"]))

            sinter.plot_error_rate(
                ax=ax,
                stats=stats_d,
                group_func=group_func,
                x_func=x_func,
                failure_units_per_shot_func=failure_units_per_shot_func,
            )
        else:
            sub = agg[agg["d"] == d].copy()
            sub["label"] = sub.apply(lambda r: "pymatching(XYZ=False)" if r["decoder"] == "pymatching"
                                     else f'{r["decoder"]}(XYZ={r["xyz"]})', axis=1)
            for label, g in sub.groupby("label"):
                g = g.sort_values("p")
                y = (g["errors"] / g["shots"]) / g["r"]
                ax.plot(g["p"], y, marker="o", label=label)

        ax.legend(loc="best")
        ax.grid(True, which="both", alpha=0.3)

        png = plotdir / f"rotated_surface_code_memory_X_d{int(d)}.png"
        pdf = plotdir / f"rotated_surface_code_memory_X_d{int(d)}.pdf"
        fig.tight_layout()
        fig.savefig(png, dpi=200)
        fig.savefig(pdf)
        plt.close(fig)
        print("Wrote:", png, pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, type=str)
    ap.add_argument("--out-merged", type=str, default=None)
    ap.add_argument("--plotdir", type=str, default=None)
    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    out_merged = Path(args.out_merged).resolve() if args.out_merged else indir / "merged_surfaceX.csv"
    plotdir = Path(args.plotdir).resolve() if args.plotdir else indir / "plots"

    merged = read_and_merge(indir)
    merged.to_csv(out_merged, index=False)
    print("Wrote merged CSV:", out_merged)

    plot_per_distance(merged, plotdir)


if __name__ == "__main__":
    main()
