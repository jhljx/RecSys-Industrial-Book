# 可引用的公开入口与搜索收录

本仓库发布的是中文书稿《推荐系统：工业架构与核心算法》。PDF 是正文权威；概念卡、代码页和 Agent 工具用于导航、练习和工程参考，不是英文版正文或生产级实现。

## 外部文章可引用的稳定链接

- 项目总览（中文）：https://jhljx.github.io/RecSys-Industrial-Book/knowledge-map/
- 项目总览（英文，书稿仍为中文）：https://jhljx.github.io/RecSys-Industrial-Book/README.en.html
- 不依赖 JavaScript 的章节／概念目录：https://jhljx.github.io/RecSys-Industrial-Book/knowledge-map/catalog.html
- PDF 阅读器：https://jhljx.github.io/RecSys-Industrial-Book/knowledge-map/reader.html
- 章节代码目录：https://jhljx.github.io/RecSys-Industrial-Book/knowledge-map/code.html
- 例：双塔召回概念卡：https://jhljx.github.io/RecSys-Industrial-Book/knowledge/concepts/two-tower.html；对应书中出处和代码候选见卡片与静态索引。引用具体书籍论断时还需核对第 10 章 PDF 原页，不能只引代码。

站外技术文章、课程主页或个人主页应按实际内容链接到相关概念卡或章节，而不是批量贴首页；保持准确、披露中文语言与 `[MOCK]` 限制，不购买链接、不制造虚假背书。这里提供可复用入口，**不自动在外部平台发布**。

## 收录与监测（站点发布后由有权限的维护者执行）

1. 仓库当前在 GitHub Pages 的项目路径 `/RecSys-Industrial-Book/` 下。生成器 `python3 tools/build_search_pages.py` 更新静态目录与 `sitemap.xml`；GitHub Pages 正式部署后访问 https://jhljx.github.io/RecSys-Industrial-Book/sitemap.xml ，核对 XML 和示例 URL 状态。
2. 在 Google Search Console 验证 `https://jhljx.github.io/RecSys-Industrial-Book/` URL 前缀资源并提交上述完整 sitemap 地址；在 Bing Webmaster Tools 验证站点并提交同一 sitemap。`google24bd46984e7fcbf2.html` 是仓库中已有的验证文件，但不代表当前账号已完成验证；不主动操作账号。
3. 因 GitHub Pages 是 `jhljx.github.io` 共享域名，**项目内 `robots.txt` 不能充当域名根路径 `/robots.txt`**；没有该域名根站点的控制权时不要写无效的项目级规则。sitemap 仍可直接提交及外部引用。
4. 定期检查索引覆盖、英文/中文搜索词、点击和展示量；抽样用中英文问题核查 AI 回答是否找到**具体的原始页面与 PDF 页码**。这些是可观察指标，不保证排序或 AI 引用。
5. 新发布 PDF/章节/概念或代码后重新生成站点数据、代码页和静态目录与 sitemap，跑测试与公开边界检查，再手动审阅提交；本地生成不等于线上已更新。

## Bing / Copilot 与 IndexNow

- Bing Webmaster Tools 中验证本项目站点后提交完整地址 `https://jhljx.github.io/RecSys-Industrial-Book/sitemap.xml`，用 URL Inspection 抽查知识地图、静态目录和代表性概念页；AI Performance（若账户可用）用于观察引用趋势，而非保证 Copilot 引用。可用中英文问题分别核查 AI 引用是否指向概念卡、章节地图和准确 PDF 页码。
- IndexNow 的公开密钥文件位于项目根目录 `9d4390e1830e48f0b4d52fb03d1a4b2c.txt`；因本站在 GitHub Pages 项目子路径下，通知必须携带 `keyLocation=https://jhljx.github.io/RecSys-Industrial-Book/9d4390e1830e48f0b4d52fb03d1a4b2c.txt`，只用于本项目路径内的 URL。密钥用于公开所有权验证，不是保密 API 凭据；不要借它通知同域名下别的项目。
- `.github/workflows/indexnow.yml` 在成功的 legacy Pages 构建事件后读取相邻的成功构建提交，`tools/notify_indexnow.py` 只通知 sitemap 中新增、更新或删除的公开 URL；先核对线上密钥和完整 sitemap 与构建提交一致，再向 Bing 的 IndexNow 接口发送。不会在本地生成、PR 或仅推送未部署的文件时通知。异常与非 200/202 状态要查看 Actions 日志；202 表示密钥验证待完成，200/202 都不是收录承诺。
- 手工预览本次相邻构建差异：`python3 tools/notify_indexnow.py --before <上一成功构建SHA> --after <当前成功构建SHA>`。初次启用没有前一次成功构建记录时不会批量推送旧 URL；继续依赖 sitemap 与内链。手动重跑 workflow 只用于构建已成功、自动事件漏报的情况，不要对未更新内容反复发送。
