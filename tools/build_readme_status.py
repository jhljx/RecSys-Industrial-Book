#!/usr/bin/env python3
"""Refresh the public README release summary from chapter maps and PDFs."""
from __future__ import annotations

import re
from pathlib import Path

from build_knowledge_map import build
from check_knowledge import pdf_page_count
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- released-status:start -->"
END = "<!-- released-status:end -->"


def compact(numbers: list[int]) -> str:
    groups = []
    start = end = numbers[0]
    for number in numbers[1:]:
        if number == end + 1:
            end = number
        else:
            groups.append(str(start) if start == end else f"{start}–{end}")
            start = end = number
    groups.append(str(start) if start == end else f"{start}–{end}")
    return "、".join(groups)


def render(root: Path = ROOT) -> str:
    chapters = build(root)["chapters"]
    numbers = sorted(chapter["number"] for chapter in chapters)
    pages = sum(pdf_page_count(root / unquote(chapter["pdf"].removeprefix("../"))) for chapter in chapters)
    return (f"{START}\n**公开发布快照**：{len(chapters)} 份章节 PDF（含前言），共 {pages} 个 PDF 页。"
            f"已发布章节编号：{compact(numbers)}。这里仅统计仓库公开 PDF，不推断未发布章节的写作进度。\n{END}")


def update(root: Path = ROOT) -> None:
    path = root / "README.md"
    text = path.read_text(encoding="utf-8")
    match = re.search(re.escape(START) + r".*?" + re.escape(END), text, flags=re.S)
    if not match or text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("README release status markers missing or duplicated")
    expected = {chapter["number"] for chapter in build(root)["chapters"]}
    table = text.split("## 📚 已发布章节 PDF 阅读入口", 1)[1].split("## ✍️ 当前写作进度", 1)[0]
    listed = {int(number) for number in re.findall(r"reader\.html#ch(\d+)&page=1", table)}
    if listed != expected:
        raise ValueError(f"README published chapter table mismatch: missing={sorted(expected-listed)} extra={sorted(listed-expected)}")
    path.write_text(text[:match.start()] + render(root) + text[match.end():], encoding="utf-8")


if __name__ == "__main__":
    update()
    print("README released-PDF summary matches chapter maps")
