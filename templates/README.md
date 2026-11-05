# 可运行教学项目

四个项目使用小规模合成数据与 `[MOCK]` 依赖，供本地练习；不包含生产集成或真实性能承诺。

| 项目 | 练习重点 | 起步命令 |
|---|---|---|
| [推荐漏斗](recommendation-pipeline/README.md) | 多路召回、去重、超时与过载兜底 | `python3 templates/recommendation-pipeline/app.py --user u1 --limit 5` |
| [向量检索版本契约](two-tower-ann/README.md) | 固定示例向量的精确排序、版本错配拒绝 | `python3 templates/two-tower-ann/app.py --user u1 --limit 3` |
| [特征时间一致性](feature-consistency/README.md) | 时点拼接、未来特征拒绝 | `python3 templates/feature-consistency/app.py` |
| [服务稳定性](serving-reliability/README.md) | 容量、依赖超时、降级 | `python3 templates/serving-reliability/app.py --requests 100 --capacity 80` |

具体输入、故障注入和生产替换边界见各项目说明。仓库根目录运行 `python3 -m unittest discover -s tools/tests -p test_template_cli.py -v` 验证四个 CLI 的主路径和失败路径。
