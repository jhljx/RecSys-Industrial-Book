# 贡献与勘误

感谢提交可复核的错误报告、概念导航改进和教学代码修复。请先核对[已发布章节目录](knowledge-map/catalog.html)和原始 PDF；未发布章节、作者私稿和本地学习状态不属于公开贡献范围。

报告书中勘误时请提供：发布 PDF 文件名、PDF 页码（不是印刷页码）、原问题的简要描述、建议改法，以及是否影响概念卡或课程。不要粘贴长段 PDF 正文或私有写作素材。

修改概念卡时保留「本书依据」「扩展说明」分节，只引用公开 PDF 的文件名与 PDF 页码。教学代码应说明输入、核心变换、输出、依赖、一个参数实验与一个失败路径；`[MOCK]` 仍需明确标出。代码演示不作为书籍论断的证据。

提交前在仓库根目录运行：

```bash
python3 tools/check_knowledge.py
python3 tools/check_public_boundary.py
python3 tools/check_chapter_code.py
python3 tools/build_knowledge_map.py
python3 tools/build_code_pages.py
python3 tools/build_search_pages.py
python3 tools/build_readme_status.py
python3 -m unittest discover -s tools/tests -v
```

若修改章节代码，先看其 `code/README.md` 的依赖和运行成本，再在合适的隔离环境中运行目标示例。生成的站点页须随源码一起更新；不要提交 `.recsys-agent/`、`.recsys-workspace/`、PDF 提取文本或私有 `.tex` 路径。提交与部署是不同步骤，本地测试不能证明线上链接已经更新。
