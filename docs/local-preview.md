# 本地站点预览

直接打开 `knowledge-map/index.html` 可离线使用知识地图、章节阅读器与代码目录；页面中的文档链接会指向原始 Markdown。若想检查和 GitHub Pages 相近的文档排版与目录 URL，可以在已安装 Ruby/Jekyll 的本机，从仓库根目录启动 Jekyll：

```bash
preview_dir=$(mktemp -d /tmp/recsys-site.XXXXXX)
jekyll serve --baseurl /RecSys-Industrial-Book --destination "$preview_dir"
```

在浏览器打开 `http://127.0.0.1:4000/RecSys-Industrial-Book/knowledge-map/`，检查概念页、`/templates/`、子项目目录和章节 PDF。预览输出写在临时目录，不进入仓库。Jekyll 是可选的本地预览依赖；持续集成仍检查源码链接，Pages 构建成功后还会检查已部署关键入口。若没有 Jekyll，继续用 `file://` 查看静态知识地图，文档显示 Markdown 原文。
