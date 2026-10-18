#!/usr/bin/env python3
"""Prepare isolated old/new usability runs and summarize blinded reviews."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import shutil
import tempfile
import statistics
from pathlib import Path

TASKS = ("Q1", "Q2", "L1", "L2", "D1", "G1", "R1", "P1")
SYSTEMS = ("old", "new")
LABELS = ("A", "B")
SCORE_FIELDS = ("correctness", "evidence", "completeness", "actionability", "boundary")


def git(root: Path, *args: str) -> str:
    return subprocess.run(["/usr/bin/git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def revision(root: Path, value: str) -> str:
    return git(root, "rev-parse", "--verify", f"{value}^{{commit}}")


def prepare(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{output}")
    old_revision = revision(root, args.old_revision)
    new_revision = revision(root, args.new_revision)
    output.mkdir(parents=True)
    prompt = args.system_prompt.read_bytes()
    manifest = {
        "old_revision": old_revision,
        "new_revision": new_revision,
        "model_runtime": args.model_runtime,
        "system_prompt": args.system_prompt.name,
        "system_prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "tasks": list(TASKS),
        "controls": ["same model/runtime", "same system prompt", "fresh session per task", "no answer sharing", "reviewers see labels A/B only"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="recsys-old-") as temporary:
        worktree = Path(temporary) / "repo"
        subprocess.run(["/usr/bin/git", "-C", str(root), "worktree", "add", "--detach", str(worktree), old_revision], check=True, capture_output=True, text=True)
        try:
            subprocess.run(["python3", "tools/recsys_agent.py", "index"], cwd=worktree, check=True, capture_output=True, text=True)
            old_index = worktree / ".recsys-agent/index.jsonl"
            old_index_sha256 = hashlib.sha256(old_index.read_bytes()).hexdigest()
        finally:
            subprocess.run(["/usr/bin/git", "-C", str(root), "worktree", "remove", "--force", str(worktree)], check=True, capture_output=True, text=True)
    manifest["old_index_sha256"] = old_index_sha256
    assignment: dict[str, dict[str, str]] = {}
    for task in TASKS:
        swap = int(hashlib.sha256(f"{task}:{old_revision}:{new_revision}".encode()).hexdigest(), 16) & 1
        assignment[task] = dict(zip(LABELS, reversed(SYSTEMS) if swap else SYSTEMS))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    private = output / "private"
    private.mkdir()
    (private / "assignment.json").write_text(json.dumps(assignment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(args.system_prompt, private / "system-prompt.txt")
    for task in TASKS:
        for label, system in assignment[task].items():
            record = {
                "task_id": task, "label": label, "system": system,
                "revision": old_revision if system == "old" else new_revision,
                "started_at": None, "first_success_at": None, "ended_at": None,
                "commands_run": [], "actual_output_evidence": [], "answer": "",
            }
            path = private / "runs" / task
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{label}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "reviews.jsonl").write_text("", encoding="utf-8")
    print(json.dumps({"status": "prepared", "output": str(output), "private_assignment": str(private / "assignment.json"), **manifest}, ensure_ascii=False, indent=2))

def load_reviews(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        identity = (str(row.get("task_id")), str(row.get("label")), str(row.get("reviewer")))
        if row.get("task_id") not in TASKS or row.get("label") not in LABELS or not row.get("reviewer"):
            raise SystemExit(f"无效评审标识：{row}")
        if identity in seen:
            raise SystemExit(f"重复评审：{'/'.join(identity)}")
        seen.add(identity)
        for field in SCORE_FIELDS:
            value = row.get(field)
            if not isinstance(value, int) or not 0 <= value <= 2:
                raise SystemExit(f"{row['task_id']}/{row['label']} 的 {field} 必须是 0..2 整数")
        elapsed = row.get("first_success_seconds")
        if not isinstance(elapsed, (int, float)) or elapsed < 0:
            raise SystemExit(f"{row['task_id']}/{row['label']} 缺少非负 first_success_seconds")
    return rows


def summarize(args: argparse.Namespace) -> None:
    rows = load_reviews(args.reviews)
    assignment = json.loads(args.assignment.read_text(encoding="utf-8"))
    if set(assignment) != set(TASKS) or any(set(labels) != set(LABELS) or set(labels.values()) != set(SYSTEMS) for labels in assignment.values()):
        raise SystemExit("assignment 必须为每个任务提供 A/B 到 old/new 的一一映射")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task_id"]), str(row["label"])), []).append(row)
    missing: list[str] = []
    adjudication: list[str] = []
    task_rows: list[dict[str, object]] = []
    agreements = comparisons = 0
    for task in TASKS:
        report: dict[str, object] = {"task_id": task}
        for label in LABELS:
            reviews = grouped.get((task, label), [])
            distinct = {str(row["reviewer"]) for row in reviews}
            system = assignment[task][label]
            if len(distinct) < 2:
                missing.append(f"{task}/{label}: 至少需要两名独立评审")
                continue
            first_two = reviews[:2]
            for field in SCORE_FIELDS:
                delta = abs(int(first_two[0][field]) - int(first_two[1][field]))
                comparisons += 1
                agreements += delta <= 1
                if delta > 1 and len(distinct) < 3:
                    adjudication.append(f"{task}/{label}/{field}")
            report[system] = {
                "blind_label": label,
                "median_score": statistics.median(sum(int(row[field]) for field in SCORE_FIELDS) for row in reviews),
                "median_first_success_seconds": statistics.median(float(row["first_success_seconds"]) for row in reviews),
                "reviewers": sorted(distinct),
            }
        task_rows.append(report)
    complete = not missing and not adjudication
    result = {
        "status": "complete" if complete else "incomplete",
        "tasks": task_rows,
        "reviewer_agreement_within_one": agreements / comparisons if comparisons else None,
        "missing": missing,
        "requires_third_reviewer": adjudication,
        "publication_decision": "May compare reader efficiency and quality." if complete else "Do not claim old/new improvement.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not complete:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="冻结版本、提示和 16 个隔离任务记录")
    prepare_parser.add_argument("--root", type=Path, default=Path.cwd())
    prepare_parser.add_argument("--old-revision", default="1af34d036fa22f21e69b254bb96949ed28b28156")
    prepare_parser.add_argument("--new-revision", required=True, help="必须是已提交的待评 revision")
    prepare_parser.add_argument("--model-runtime", required=True)
    prepare_parser.add_argument("--system-prompt", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    summary_parser = sub.add_parser("summarize", help="校验双评审/裁决并汇总盲评")
    summary_parser.add_argument("--reviews", type=Path, required=True)
    summary_parser.add_argument("--assignment", type=Path, required=True, help="prepare 生成的私有 A/B 映射")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else summarize(args)


if __name__ == "__main__":
    main()
