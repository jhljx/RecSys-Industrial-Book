#!/usr/bin/env python3
"""Validate local Agent Skills without external dependencies."""
from __future__ import annotations
import re
from pathlib import Path

def main() -> None:
    root=Path(__file__).resolve().parents[1]; failures=[]; count=0
    for skill in sorted((root/".agents/skills").glob("*/SKILL.md")):
        count+=1; text=skill.read_text(encoding="utf-8")
        match=re.match(r"---\n(.*?)\n---\n",text,re.DOTALL)
        if not match: failures.append(f"{skill}: missing frontmatter"); continue
        fields={line.split(":",1)[0]:line.split(":",1)[1].strip() for line in match.group(1).splitlines() if ":" in line}
        if fields.get("name") != skill.parent.name: failures.append(f"{skill}: name does not match directory")
        if not fields.get("description"): failures.append(f"{skill}: missing description")
    if failures: print("\n".join(failures)); raise SystemExit(1)
    print(f"skill lint passed: {count} canonical skills")
if __name__=="__main__": main()
