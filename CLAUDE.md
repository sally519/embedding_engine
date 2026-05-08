# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本地 PyTorch 运行器，运行 `Qwen/Qwen3-Embedding-0.6B` 向量模型和 `BAAI/bge-reranker-v2-m3` 重排序模型，封装为可供其他 Python 工程直接 import 的 SDK，同时提供 CLI 调试工具。

## 常用命令

```powershell
# 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# 快速语法校验（不需要下载模型）
python -m compileall src

# 向量模型 CLI 冒烟验证
qwen-embed --device cpu demo
qwen-embed --device cpu embed --text "你好" --show-shape
qwen-embed --device cpu similarity --query "北京首都" --document "中国首都是北京"

# 重排序模型 CLI 冒烟验证
qwen-embed --device cpu rerank --query "中国首都是哪里" --document "北京是中国的首都" --document "巴黎是法国的首都"
```

注意：CLI 全局参数（`--model-id`、`--device`、`--max-length`、`--cache-dir`、`--batch-size`）必须放在子命令前面。`--model-id` 不传时，各子命令自动使用对应默认模型（embedding 用 Qwen3，rerank 用 bge-reranker）。

## 架构

两套平行子系统（embedding + reranking），共享分层调用模式：

```
config.py              EngineConfig / RerankerConfig 数据类与默认常量
schemas.py             Pydantic 协议对象（Request / Response / UsageInfo / Result）
engine/
  model.py             EmbeddingEngine — 向量模型加载、分词、池化、归一化、维度截断、相似度矩阵
  pooling.py           last_token_pool — 左侧 padding + 取最后一个有效 token
  service.py           EmbeddingService — 协议映射、引擎缓存、preload
  reranker.py          RerankerEngine — 重排序模型加载、query-doc 对推理、分数提取
  reranker_service.py  RerankerService — 协议映射、引擎缓存、preload
sdk.py                 EmbeddingSDK + RerankerSDK + 工厂函数
cli.py                 argparse CLI，四个子命令：demo / embed / similarity / rerank
```

关键设计点：
- **向量模型**：query 自动拼接 `Instruct: {task}\nQuery:{text}` 前缀，document 原样传入；池化使用左侧 padding + 最后一个 token
- **重排序模型**：使用 `AutoModelForSequenceClassification` 加载，输入 `(queries, documents)` 对，输出 `logits.squeeze(-1)` 作为相关性分数
- 两套 Service 均按 model_id 惰性初始化并缓存引擎实例，均支持 `preload()` 预加载
- SDK 同时接受协议对象和普通 dict
- `batch_size` 从 config 到 engine 全链路透传，未设置时一次性全量推理

## 编码约定

- 4 空格缩进，UTF-8 编码，行尾 CRLF（见 `.editorconfig`）
- `snake_case` 函数/变量，`PascalCase` 类名，`UPPER_SNAKE_CASE` 常量
- 新增代码注释、文档统一使用中文；仅在引用外部库参数名、模型名、命令名时保留英文原文
- 每个方法尽量加上中文注释
- 不要在源码中硬编码令牌、账号信息或机器私有路径

## 测试

当前无测试目录。后续新增测试放在 `tests/` 下，命名为 `test_*.py`。提交前至少执行 `python -m compileall src` 并做一次 CLI 冒烟验证。
