"""Perform lightweight structural and per-cell syntax checks on the research notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Custom_DL_Optimizer_Research_Colab.ipynb"

data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert data["nbformat"] == 4
assert isinstance(data.get("cells"), list) and data["cells"]

errors = []
for index, cell in enumerate(data["cells"]):
    assert cell["cell_type"] in {"code", "markdown", "raw"}
    assert isinstance(cell.get("source"), list)
    if cell["cell_type"] != "code":
        continue
    cell_source = "".join(
        line for line in cell["source"] if not line.lstrip().startswith(("!", "%"))
    )
    try:
        compile(cell_source, f"cell_{index}", "exec")
    except SyntaxError as exc:
        errors.append({"cell": index, "line": exc.lineno, "message": exc.msg})

if errors:
    raise SystemExit(f"Notebook syntax errors: {errors}")

print(
    f"Validated {NOTEBOOK.name}: {len(data['cells'])} cells, "
    f"{sum(cell['cell_type'] == 'code' for cell in data['cells'])} code cells, "
    f"{NOTEBOOK.stat().st_size} bytes"
)
