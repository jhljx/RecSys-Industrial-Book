"""Bounded, read-only engineering context and user-supplied contract evidence.

Evidence is checked against explicit domain invariants. It is never executed and
never treated as live telemetry merely because it was supplied in a JSON file.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

DOMAINS = ("feature-time", "serving", "release-version")
MAX_EVIDENCE_BYTES = 1_000_000
MAX_RECORDS = 1000
PLAYBOOKS = {
    "feature-time": {
        "options": ["按预测时刻做 point-in-time join 并拒绝未来特征", "为迟到数据配置水位线并按快照版本回填"],
        "next_check": "追踪选中特征的事件时间、可用时间与标签时间；以边界时间样本复现。",
        "acceptance": "未来特征未入样、等时刻样本仍可用、迟到与回填有明确版本。",
    },
    "serving": {
        "options": ["对子路由设置短于请求总预算的超时，并合并已返回候选", "依赖超时后走非空且版本兼容的兜底候选"],
        "next_check": "对照请求 trace 中的子路由耗时、总耗时、候选数及兜底原因。",
        "acceptance": "故障注入后总耗时不超预算且候选非空，撤销故障后恢复正常路由。",
    },
    "release-version": {
        "options": ["用同一发布清单绑定模型、向量和索引", "拒绝错配并回退至已知兼容的整套版本"],
        "next_check": "核对同一请求的模型、向量、索引与回退版本，以及错配是否被拒绝。",
        "acceptance": "正常版本被接受，错配版本被拒绝，回滚后可恢复正常请求。",
    },
}


def project_context(project: Path) -> dict[str, object]:
    root = project.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--project 必须是目录。")
    paths: list[str] = []
    visited = 0
    for directory, subdirs, files in os.walk(root, followlinks=False):
        visited += 1
        if visited > 500 or len(paths) >= 30:
            break
        depth = len(Path(directory).relative_to(root).parts)
        subdirs[:] = sorted(name for name in subdirs if depth < 4 and not name.startswith(".")
                            and name not in {"node_modules", "venv", "__pycache__"})
        for name in sorted(files):
            if name in {"pyproject.toml", "requirements.txt", "README.md", "AGENTS.md", "app.py"} or name.startswith("test_") and name.endswith(".py"):
                path = Path(directory) / name
                if not path.is_symlink():
                    paths.append(path.relative_to(root).as_posix())
                if len(paths) >= 30:
                    break
    return {"root": str(root), "entry_candidates": paths[:30],
            "note": "仅列出候选文件；需阅读目标实现、调用方与测试后才能判断行为。"}


def _time(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("时间不能是布尔值")
    if isinstance(value, (int, float)) and -1e15 < value < 1e15:
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("时间缺少时区")
            return parsed.timestamp()
        except ValueError as error:
            raise ValueError("时间应为数值或带时区的 ISO 8601") from error
    raise ValueError("时间应为数值或带时区的 ISO 8601")


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1e12:
        raise ValueError(f"{name} 必须是正数")
    return float(value)


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1e12:
        raise ValueError(f"{name} 必须是非负数")
    return float(value)


def check_evidence(path: Path, domain: str) -> dict[str, object]:
    if domain not in DOMAINS:
        raise ValueError("未知工程领域。")
    if not path.is_file() or path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError("证据文件不存在或超过 1MB。")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"证据文件不是有效 UTF-8 JSON：{error}") from error
    if not isinstance(document, dict) or document.get("domain") != domain:
        raise ValueError("证据必须是对象，domain 必须与 --domain 一致。")
    return check_records(domain, document.get("records"), source=str(path.resolve()))


def check_records(domain: str, records: object, source: str = "local_command_output") -> dict[str, object]:
    if domain not in DOMAINS:
        raise ValueError("未知工程领域。")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError("records 必须包含 1–1000 条记录。")
    checks: list[dict[str, object]] = []
    phases: set[str] = set()
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict) or record.get("phase") not in {"baseline", "fault", "recovery"}:
            raise ValueError(f"记录 {number} 缺少有效 phase。")
        phase = record["phase"]
        phases.add(phase)
        if domain == "feature-time":
            label, feature = _time(record.get("label_time")), _time(record.get("feature_time"))
            selected = record.get("selected")
            if not isinstance(selected, bool):
                raise ValueError(f"记录 {number} 的 selected 必须是布尔值。")
            passed = not (selected and feature > label)
            invariant = "selected_feature_time <= label_time"
            observation = "未来特征进入样本" if not passed else "当前记录未见未来特征进入样本"
        elif domain == "serving":
            budget = _positive(record.get("budget_ms"), "budget_ms")
            elapsed = _nonnegative(record.get("elapsed_ms"), "elapsed_ms")
            candidates = record.get("candidates")
            if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 0:
                raise ValueError(f"记录 {number} 的 candidates 必须是非负整数。")
            passed = elapsed <= budget and candidates > 0
            invariant = "elapsed_ms <= budget_ms and candidates > 0"
            observation = "超时或空结果" if not passed else "当前请求在预算内且非空"
        else:
            versions = [record.get(key) for key in ("model_release", "embedding_release", "index_release")]
            accepted = record.get("accepted")
            if any(not isinstance(value, str) or not value.strip() for value in versions) or not isinstance(accepted, bool):
                raise ValueError(f"记录 {number} 需要三个非空版本和布尔 accepted。")
            compatible = len(set(versions)) == 1
            passed = not accepted if not compatible else accepted if phase in {"baseline", "recovery"} else True
            invariant = "incompatible_releases_rejected_and_compatible_baseline_accepted"
            observation = ("当前记录满足版本契约" if passed else
                           "不兼容版本被接受" if not compatible else "兼容版本未被接受")
        checks.append({"record": number, "phase": phase, "passed": passed, "invariant": invariant,
                       "observation": observation})
    if domain == "feature-time" and {"baseline", "recovery"}.issubset(phases) and not any(
        record.get("selected") for record in records if record["phase"] in {"baseline", "recovery"}
    ):
        raise ValueError("基线或恢复阶段必须包含一条实际选中的特征。")
    failed = [item["record"] for item in checks if not item["passed"]]
    return {"domain": domain, "source": source, "provenance": "user_supplied_local_file" if source != "local_command_output" else "local_command_output",
            "status": "violation_observed" if failed else "no_violation_in_samples", "checked_records": len(records),
            "failed_records": failed, "checks": checks,
            "coverage": {phase: phase in phases for phase in ("baseline", "fault", "recovery")},
            "limits": "仅检查提供的记录；不能证明目标代码执行了这些路径、数据真实或线上指标达标。"}


def project_report(project: Path | None, evidence: Path | None, domain: str | None) -> dict[str, object]:
    if bool(evidence) != bool(domain):
        raise ValueError("--evidence 与 --domain 必须同时提供。")
    report = {}
    if project is not None:
        report["project_context"] = project_context(project)
    if evidence is not None:
        report["evidence_check"] = check_evidence(evidence, domain)
    return report


def verify_commands(project: Path, domain: str, cases: list[list[str]], timeout: int = 30) -> dict[str, object]:
    """Run only explicit commands inside the named target project."""
    context = project_context(project)
    if domain not in DOMAINS or not 0 < timeout <= 300:
        raise ValueError("领域无效或超时不在 1–300 秒内。")
    if len(cases) != 3 or {phase for phase, _ in cases} != {"baseline", "fault", "recovery"}:
        raise ValueError("必须各提供一条 baseline、fault、recovery 命令。")
    runs = []
    records = []
    rejected_fault = False
    rejection_domain_supported = domain in {"feature-time", "release-version"}
    project_root = Path(context["root"])
    for phase, command in cases:
        arguments = shlex.split(command)
        if not arguments:
            raise ValueError(f"{phase} 命令为空。")
        executable = Path(arguments[0])
        if executable.name not in {"python", "python3"} or executable.parent not in {Path("."), Path("/usr/bin"), Path("/usr/local/bin")}:
            raise ValueError("verify 只接受 python 或 python3 脚本命令。")
        if executable.name.startswith("python"):
            if len(arguments) < 2 or arguments[1].startswith("-"):
                raise ValueError("verify 需要项目内的 Python 脚本路径，不接受 -c/-m。")
            script = (project_root / arguments[1]).resolve()
            if not script.is_relative_to(project_root) or not script.is_file() or script.suffix != ".py":
                raise ValueError("脚本必须是 --project 目录内的 .py 文件。")
            arguments[1] = str(script)
        try:
            result = subprocess.run(arguments, cwd=context["root"], capture_output=True, text=True,
                                    timeout=timeout, encoding="utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError(f"{phase} 执行失败：{error}") from error
        output = result.stdout[-100_000:]
        runs.append({"phase": phase, "command": arguments, "exit_code": result.returncode,
                     "stdout_tail": output[-600:], "stderr_tail": result.stderr[-600:]})
        try:
            observed = json.loads(output)
        except json.JSONDecodeError as error:
            if result.returncode:
                continue
            raise ValueError(f"{phase} 标准输出必须是单个 JSON 记录：{error}") from error
        rejection_reason = observed.get("error", "") if isinstance(observed, dict) else ""
        expected_reason = {"feature-time": "future feature", "release-version": "version mismatch"}
        if (phase == "fault" and result.returncode == 2 and rejection_domain_supported
                and isinstance(observed, dict) and observed.get("status") == "rejected"
                and isinstance(rejection_reason, str)
                and expected_reason[domain] in rejection_reason.casefold()):
            rejected_fault = True
            continue
        if result.returncode:
            continue
        if not isinstance(observed, dict) or "phase" in observed or "domain" in observed:
            raise ValueError(f"{phase} 输出必须是不含 phase/domain 的记录对象。")
        if domain == "feature-time" and isinstance(observed.get("rows"), list):
            records.extend({"phase": phase, "label_time": row.get("label_time"),
                            "feature_time": row.get("feature_time"), "selected": True}
                           for row in observed["rows"] if isinstance(row, dict))
        elif domain == "release-version" and observed.get("status") == "ok" and isinstance(observed.get("version"), str):
            records.append({"phase": phase, "model_release": observed["version"],
                            "embedding_release": observed["version"], "index_release": observed["version"],
                            "accepted": True})
        else:
            records.append({**observed, "phase": phase})
    if rejected_fault and records and {record["phase"] for record in records} == {"baseline", "recovery"}:
        checked = check_records(domain, records)
        return {"status": "fault_rejected_and_recovered" if not checked["failed_records"] else "baseline_or_recovery_failed",
                "project_context": context, "runs": runs, "evidence_check": checked,
                "verification": "local fault returned exit 2 with a domain-specific rejection message; baseline/recovery observations checked; cause and output mapping require code inspection"}
    if not records or {record["phase"] for record in records} != {"baseline", "fault", "recovery"}:
        return {"status": "command_failed", "project_context": context, "runs": runs,
                "verification": "incomplete; only exit-0 commands supply contract observations"}
    checked = check_records(domain, records)
    failures = {item["phase"] for item in checked["checks"] if not item["passed"]}
    outcome = ("fault_reproduced_and_recovered" if failures == {"fault"} else
               "baseline_or_recovery_failed" if failures & {"baseline", "recovery"} else
               "fault_not_reproduced")
    return {"status": outcome, "project_context": context, "runs": runs,
            "evidence_check": checked,
            "verification": "three user-specified local commands executed; output provenance is local run, not production telemetry"}
