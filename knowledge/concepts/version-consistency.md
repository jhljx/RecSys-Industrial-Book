---
id: version-consistency
title: 模型与数据版本一致性
chapter: 17
prerequisites: [embedding-server]
---

## 核心判断
模型、特征、Embedding 和索引必须来自兼容发布集合；单独更新任一组件可能产生静默错误。

## 工程检查
维护 release manifest，加载前校验兼容性，灰度和回滚以 release 为单位。

## 本书依据
《第十七章-推荐系统在线服务.pdf》，PDF 第17页起；《第二十二章-链路一致性建模.pdf》，PDF 第1页起。

## 扩展说明
跨地域发布需处理最终一致窗口。
