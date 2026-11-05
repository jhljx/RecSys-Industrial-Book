#!/usr/bin/env python3
"""Check syntax of chapter Python sources without running expensive examples."""
from __future__ import annotations

import ast
from pathlib import Path

from recsys_code import CI_SMOKE_SOURCES, catalog


def check(root: Path) -> tuple[int, int]:
    sources = catalog(root)
    missing_labels = [row["path"] for row in sources if not row["description"]]
    missing_smoke = CI_SMOKE_SOURCES - {str(row["path"]) for row in sources}
    errors = []
    python_count = 0
    for row in sources:
        if row["language"] != "py":
            continue
        python_count += 1
        try:
            ast.parse((root / str(row["path"])).read_text(encoding="utf-8"), filename=str(row["path"]))
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"{row['path']}: {error}")
    if missing_labels or missing_smoke or errors:
        raise ValueError("\n".join([*(f"missing README label: {path}" for path in missing_labels),
                                    *(f"missing CI smoke source: {path}" for path in sorted(missing_smoke)), *errors]))
    return len(sources), python_count


if __name__ == "__main__":
    total, parsed = check(Path(__file__).resolve().parents[1])
    print(f"chapter code: {total} catalogued, {parsed} Python syntax parsed; three CI smoke sources are checked separately")
