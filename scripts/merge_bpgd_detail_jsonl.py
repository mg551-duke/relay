#!/usr/bin/env python3
"""Merge native BPGD JSONL detail sidecars into a flat CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_dir", type=Path)
    parser.add_argument("out_csv", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.detail_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["source_file"] = path.name
                rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
