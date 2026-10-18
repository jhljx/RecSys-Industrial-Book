#!/usr/bin/env python3
"""Install canonical Agent Skills into supported local client directories."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--client",choices=("claude","agents","all"),default="all"); parser.add_argument("--target",type=Path,default=Path.cwd()); args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]/".agents"/"skills"
    destinations=[]
    if args.client in ("agents","all"): destinations.append(args.target/".agents"/"skills")
    if args.client in ("claude","all"): destinations.append(args.target/".claude"/"skills")
    installed=[]
    for destination in destinations:
        destination.mkdir(parents=True,exist_ok=True)
        for skill in source.iterdir():
            if not skill.is_dir(): continue
            target=destination/skill.name
            if target.resolve()==skill.resolve(): continue
            shutil.copytree(skill,target,dirs_exist_ok=True); installed.append(str(target))
    print("\n".join(installed) if installed else "canonical skills already available")
if __name__=="__main__": main()
