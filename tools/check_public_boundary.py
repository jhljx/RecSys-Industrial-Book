#!/usr/bin/env python3
"""Fail when the public repository contains private authoring or extracted prose."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

DENIED_SUFFIXES = {".tex", ".aux", ".xdv", ".synctex", ".synctex.gz"}
DENIED_PARTS = {".recsys-agent", ".recsys-workspace", "__pycache__", ".pytest_cache", "target"}
DENIED_NAMES = {"book.md"}
TEXT_SUFFIXES = {".md", ".py", ".html", ".js", ".json", ".jsonl", ".xml", ".txt", ".yml", ".yaml", ".css"}
# Code and documentation copied from Draft must not expose authoring-source coordinates.
PRIVATE_SOURCE_REF = re.compile(r"(?<![\w/])(?:chapters/)?ch\d+\.tex\b", re.IGNORECASE)
ALLOWED_MARKDOWN = {
    "README.md",
    "README.en.md",
    "AGENTS.md",
    "CLAUDE.md",
    "CHANGELOG.md",
    "docs/agent-guide.md",
    "docs/ai-native-validation-report.md",
    "docs/discoverability.md",
    "docs/local-preview.md",
    "CONTRIBUTING.md",
    "docs/learning-agent-guide.md",
}


def markdown_allowed(relative: Path) -> bool:
    path = relative.as_posix()
    return (
        path in ALLOWED_MARKDOWN
        or (path.startswith(".claude/skills/") and path.endswith(".md"))
        or (path.startswith(".agents/skills/") and path.endswith(".md"))
        or (path.startswith("knowledge/") and path.endswith(".md"))
        or (path.startswith("learning-paths/") and path.endswith(".md"))
        or (path.startswith("cases/") and path.endswith(".md"))
        or (path.startswith("evals/") and path.endswith(".md"))
        or (path.startswith("templates/") and path.endswith("/README.md"))
        or ("/code/" in path and path.endswith("/README.md"))
    )


def tracked_files(root: Path) -> list[Path]:
    system_git = Path("/usr/bin/git")
    git = str(system_git) if system_git.is_file() else shutil.which("git")
    if git is None:
        raise SystemExit("privacy check failed: git executable not found")
    result = subprocess.run(
        [git, "-C", str(root), "ls-files", "-co", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [root / line for line in result.stdout.splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    violations: list[str] = []
    for path in tracked_files(root):
        relative = path.relative_to(root)
        suffix = "".join(path.suffixes[-2:]) if "".join(path.suffixes[-2:]) in DENIED_SUFFIXES else path.suffix
        denied_markdown = path.suffix.lower() == ".md" and not markdown_allowed(relative)
        if suffix in DENIED_SUFFIXES or path.name in DENIED_NAMES or DENIED_PARTS.intersection(relative.parts) or denied_markdown:
            violations.append(relative.as_posix())
            continue
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
            # Exclude this checker itself; the detection regex is intentionally shown here.
            if relative.as_posix() != "tools/check_public_boundary.py" and PRIVATE_SOURCE_REF.search(path.read_text(encoding="utf-8")):
                violations.append(f"{relative.as_posix()}: private source reference")
    if violations:
        print("privacy check failed:")
        for violation in sorted(violations):
            print(f"  {violation}")
        raise SystemExit(1)
    print("privacy check passed: public files match the source and Markdown boundary")


if __name__ == "__main__":
    main()
