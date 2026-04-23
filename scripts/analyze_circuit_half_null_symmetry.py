#!/usr/bin/env python3
"""Audit complementary half-null symmetry and posterior bias in projected circuit fault exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relay_bp.analysis import (
    DEFAULT_GAP_THRESHOLD_EXACT,
    DEFAULT_GAP_THRESHOLD_NEAR,
    PAIR_CLASS_ORDER,
    audit_half_null_posteriors,
    default_export_code_paths,
    enumerate_half_null_cases,
    load_code_capacity_problem,
    pair_half_null_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORTS = default_export_code_paths(REPO_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code",
        choices=["gross", "two_gross", "surface13", "all"],
        default="all",
        help="Projected circuit export to analyze.",
    )
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--pair-class",
        nargs="+",
        choices=list(PAIR_CLASS_ORDER),
        default=None,
        help="Optional posterior-gap classes to include in the decode audit.",
    )
    parser.add_argument(
        "--max-pairs-per-class",
        type=int,
        default=None,
        help="Optional cap on decoded pairs per posterior class.",
    )
    parser.add_argument(
        "--gap-threshold-exact",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_EXACT,
        help="Threshold for classifying exact-equal posterior gaps.",
    )
    parser.add_argument(
        "--gap-threshold-near",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_NEAR,
        help="Threshold for classifying near-equal posterior gaps.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected_codes = (
        ["gross", "two_gross", "surface13"] if args.code == "all" else [args.code]
    )
    payload: dict[str, object] = {}
    for code in selected_codes:
        problem = load_code_capacity_problem(DEFAULT_EXPORTS[code])
        cases = enumerate_half_null_cases(problem)
        pairs = pair_half_null_cases(cases)
        audit = audit_half_null_posteriors(
            problem=problem,
            pairs=pairs,
            max_iter=args.max_iter,
            alpha=args.alpha,
            pair_classes=args.pair_class,
            max_pairs_per_class=args.max_pairs_per_class,
            gap_threshold_exact=args.gap_threshold_exact,
            gap_threshold_near=args.gap_threshold_near,
        )
        payload[code] = {
            "summary": audit["summary"],
            "pair_class_counts": audit["pair_class_counts"],
            "pair_class_summary": audit["pair_class_summary"],
            "decoded_matches_counts": audit["decoded_matches_counts"],
            "exact_equal_examples": audit["representative_examples"]["exact_equal"][:5],
            "near_equal_examples": audit["representative_examples"]["near_equal"][:5],
            "strong_bias_examples": audit["representative_examples"]["strong_bias"][:5],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
