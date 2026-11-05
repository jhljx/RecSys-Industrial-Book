#!/usr/bin/env python3
"""Generate a static HTML page for every released chapter source file.

Pages contain existing public source, README labels and structural anchors only.
No PDF extraction, model-generated claims or private authoring files are included.
"""
from __future__ import annotations

import ast
import html
import json
import re
from pathlib import Path
from urllib.parse import quote

from recsys_code import catalog, page_name
from recsys_navigation import NavigationIndex

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "knowledge-map/code"
INDEX = ROOT / "knowledge-map/code-data.js"


def overview(path: Path, text: str) -> tuple[str, list[dict[str, object]]]:
    """Return a source-authored docstring and navigable top-level declarations."""
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return "", []
        summary = (ast.get_docstring(tree) or "").strip()[:400]
        symbols = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = (ast.get_docstring(node) or "").strip().splitlines()
                symbols.append({"name": node.name, "line": node.lineno, "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                                "description": doc[0][:160] if doc else ""})
        return summary, symbols[:80]
    # Other languages remain readable, without guessing method semantics.
    return "", []


def build(root: Path = ROOT) -> list[dict[str, object]]:
    output = root / "knowledge-map/code"
    entries = []
    for row in catalog(root):
        path = root / str(row["path"])
        source = path.read_text(encoding="utf-8", errors="replace")
        summary, symbols = overview(path, source)
        url = page_name(str(row["path"]))
        entry = {**row, "url": "./code/" + url, "symbols": len(symbols)}
        entries.append(entry)
        target = output / url
        target.parent.mkdir(parents=True, exist_ok=True)
        # Store code as escaped text (not executable markup), one anchor per source line.
        lines = source.splitlines()
        code = "".join(f'<span id="L{i}" class="source-line"><a class="line-number" href="#L{i}" aria-label="第 {i} 行">{i}</a><span>{html.escape(line)}</span></span>'
                         for i, line in enumerate(lines, 1))
        description = html.escape(str(row["description"]))
        if not description:
            description = "本文件在章节 README 中暂无逐文件说明；请对照源码与原书 PDF 核查。"
        overview_text = f"<p>{html.escape(summary)}</p>" if summary else ""
        symbol_items = "".join(f'<li><a href="#L{item["line"]}"><code>{html.escape(str(item["name"]))}</code> · 第 {item["line"]} 行</a>'
                               + (f'<small>{html.escape(str(item["description"]))}</small>' if item["description"] else "") + "</li>" for item in symbols)
        structure = f'<ol class="symbol-list">{symbol_items}</ol>' if symbols else '<p>此文件没有可提取的 Python 顶层类/函数目录；可直接按行阅读完整源码。</p>'
        chapter = int(row["chapter"])
        relative = "../"
        pdf_link = f'{relative}reader.html#ch{chapter}'
        readme_link = "../../" + quote(str(row["readme"]), safe="/")
        source_link = "../../" + quote(str(row["path"]), safe="/")
        target.write_text(f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(path.name)} · 第 {chapter} 章代码讲解</title>
<meta name="description" content="第 {chapter} 章教学源码 {html.escape(path.name, quote=True)}：{html.escape(str(row['description'])[:115], quote=True)}">
<link rel="canonical" href="https://jhljx.github.io/RecSys-Industrial-Book/knowledge-map/code/{url}">
<link rel="stylesheet" href="{relative}style.css"><link rel="stylesheet" href="{relative}code.css">
<script src="{relative}code.js" defer></script></head><body>
<a class="skip" href="#main">跳到正文</a><header class="site-header"><a class="brand" href="{relative}index.html">RECSYS <span>／ 代码讲解</span></a>
<nav aria-label="页面导航"><a href="{relative}index.html">知识地图</a><a href="{relative}code.html#ch{chapter}">代码目录</a><a href="{pdf_link}">阅读 PDF</a></nav></header>
<main id="main" class="code-page" data-source="{html.escape(str(row['path']), quote=True)}">
<div class="overline">CH {chapter:02d} / SOURCE EXAMPLE</div><h1>{html.escape(path.name)}</h1>
<p class="source-path">{html.escape(str(row['path']))}</p>
<div class="detail-links"><a href="{readme_link}">章节代码说明 ↗</a><a href="{source_link}">原始源码 ↗</a><a href="{pdf_link}">阅读第 {chapter} 章 PDF ↗</a></div>
<section class="code-intro" aria-labelledby="overview"><h2 id="overview">这个示例讲什么</h2><p>{description}</p>{overview_text}
<p class="boundary-note">上面来自章节 README 或源码自身的说明，不是自动验证的运行结果，也不能代替 PDF 的书中论断。讲解时请结合实际代码、输入输出与原书页码。</p></section>
<section class="code-intro" aria-labelledby="structure"><h2 id="structure">源码结构与阅读入口</h2>{structure}</section>
<section class="code-source" aria-labelledby="source-title"><div class="source-toolbar"><h2 id="source-title">完整源码 · {len(lines)} 行</h2><button type="button" id="copy-code">复制源码</button></div><pre><code id="source-code" class="language-{html.escape(str(row['language']))}">{code}</code></pre></section>
<section class="code-intro"><h2>如何用于学习或编写新代码</h2><ol><li>先看章节 README 的依赖和运行成本，定位这个例子的输入、输出及关键函数。</li><li>在原书 PDF 中找到对应概念的具体页；代码演示与书中论断分开标注。</li><li>只在依赖和耗时可控时运行单个示例，修改一个参数并记录真实输出与失败情形。</li><li>参考实现新代码时说明复用、改动、测试与尚未完成的生产依赖；教学示例并非生产系统。</li></ol></section>
</main><footer><span>公开代码示例 · 与全书知识地图一致的阅读体验</span><a href="{relative}code.html#ch{chapter}">返回本章代码 ↑</a></footer></body></html>
''', encoding="utf-8")
    # Reuse the Agent's candidate mapping so a concept leads to the same public
    # source pages in both interfaces. Templates are intentionally not code pages.
    if (root / "knowledge/graph.json").is_file():
        navigation = NavigationIndex(root)
        by_path = {entry["path"]: entry for entry in entries}
        for ident in navigation.concepts:
            papers = [paper for paper in navigation.papers if ident in paper["concepts"]]
            for path in navigation._code(ident, papers):
                if path in by_path:
                    by_path[path].setdefault("concept_ids", []).append(ident)
    for entry in entries:
        entry.setdefault("concept_ids", [])
    (root / "knowledge-map/code-data.js").write_text(
        "/* Generated by tools/build_code_pages.py; do not edit manually. */\nwindow.CHAPTER_CODE = "
        + json.dumps(entries, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    return entries


if __name__ == "__main__":
    rows = build()
    print(f"code pages: {len(rows)} sources across {len({row['chapter'] for row in rows})} chapters")
