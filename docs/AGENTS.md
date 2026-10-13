# AGENTS.md — docs/

本目录的协作须知。

## 导航

- 三档权威性（规范 / 架构 / 一次性报告）与逐文件分类见 [`README.md`](README.md)。
- 协作规范的唯一权威是 [`dev-convention.md`](dev-convention.md)；[`../CONTRIBUTING.md`](../CONTRIBUTING.md) 是简版入口。

## 风格

- 每文件一个 H1，且与索引名一致；标题不跳级；相对链接；代码块标语言。
- 文件名可用中文；不改名既有文档。
- 编码 UTF-8。

## 坑

- 本目录里既有规范、也有设计历史与一次性报告——**不要从「名字更像规范」去猜**，先查 [`README.md`](README.md) 的分类。
- `architecture/beta_ver.md` 是设计历史，`architecture/测试数据格式.md` 描述的是上游生产者格式，都不是当前状态。

## 流程

- 新增、重命名、删除本目录文档，必须**在同一提交**更新 [`README.md`](README.md) 索引。
- 规范被取代时，在同一提交标注取代关系；不并列两套互相矛盾的规范。
- 一次性报告必须注明测量时间点（或 commit）、数据集、环境，并声明其非当前状态。
- `doc/book` 分支的书稿由维护者手工撰写，禁止批量改写。
