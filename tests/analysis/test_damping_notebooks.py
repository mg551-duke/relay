# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = [
    REPO_ROOT / "examples" / "CodeCapacityDampingPerformanceHeatMap.ipynb",
    REPO_ROOT / "examples" / "CodeCapacityDampingTrace.ipynb",
    REPO_ROOT / "examples" / "CodeCapacityHalfStabilizerDampingPerformance.ipynb",
    REPO_ROOT / "examples" / "CodeCapacityHalfStabilizerDampingHeatmaps.ipynb",
]


def test_damping_notebook_smoke_cells_execute():
    for notebook_path in NOTEBOOKS:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        smoke_cell_count = 0
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            tags = cell.get("metadata", {}).get("tags", [])
            if "smoke-test" not in tags:
                continue
            smoke_cell_count += 1
            exec(
                compile(
                    "".join(cell["source"]), f"{notebook_path}:{cell_index}", "exec"
                ),
                namespace,
            )
        assert (
            smoke_cell_count >= 1
        ), f"{notebook_path.name} is missing a smoke-test cell."
