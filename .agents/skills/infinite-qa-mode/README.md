# 无限问答模式

让 Cursor Agent 进入持续问答循环，适合需要反复讨论、逐步确认方案后再执行的场景。

## 解决的痛点

默认情况下 Agent 回复后对话就结束了，用户需要手动输入才能继续。无限问答模式让 Agent 每次回复后自动提供上下文相关的快捷选项，形成持续的交互循环。Agent 可以正常执行所有操作（修改代码、搜索、分析等），只是每一步都有选项引导，直到用户主动退出。

## 安装

从 Skill Market 下载并解压到 `~/.cursor/skills/infinite-qa-mode/`

## 使用

在 Cursor 中对 Agent 说：

```
开始无限问答模式
```

Agent 会进入问答循环，每次回复后自动提供选项供你选择。选择"退出问答"即可结束循环并开始执行。

## 项目结构

```
infinite-qa-mode/
├── .skill/
│   └── skill-manifest.json
├── SKILL.md
└── README.md
```

---

作者：@yunshanpeng
