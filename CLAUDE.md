# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本地 PyTorch 运行器，运行 `Qwen/Qwen3-Embedding-0.6B` 向量模型，封装为可供其他 Python 工程直接 import 的 SDK，同时提供 CLI 调试工具。

## 常用命令

```powershell
# 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# 快速语法校验（不需要下载模型）
python -m compileall src

# CLI 冒烟验证
qwen-embed --device cpu demo
qwen-embed --device cpu embed --text "你好" --show-shape
qwen-embed --device cpu similarity --query "北京首都" --document "中国首都是北京"
```

注意：CLI 全局参数（`--model-id`、`--device`、`--max-length`、`--cache-dir`）必须放在子命令前面。

## 架构

分层调用链（从底层到顶层）：

```
config.py          默认模型 ID、任务描述、EngineConfig 数据类
contracts/         Pydantic 协议对象：EmbeddingRequest / EmbeddingResponse / UsageInfo
core/pooling.py    last_token_pool — 取最后一个有效 token 向量作为整句表示
core/engine.py     EmbeddingEngine — 模型加载、分词、推理、归一化、维度截断、相似度矩阵
core/service.py    EmbeddingService — 协议对象到模型层的映射，按 model_id 缓存引擎实例
sdk.py             EmbeddingSDK — 面向外部工程的稳定 API（embed / embed_texts）
factory.py         create_embedding_sdk() 工厂函数
cli.py             argparse CLI，三个子命令：demo / embed / similarity
```

关键设计点：
- query 类型文本会自动拼接 `Instruct: {task}\nQuery:{text}` 前缀（`build_instruction`），document 类型原样传入
- 池化使用左侧 padding + 取最后一个 token（Qwen3 Embedding 官方推荐方式）
- `EmbeddingService` 按 model_id 惰性初始化并缓存 `EmbeddingEngine` 实例
- `EmbeddingSDK.embed()` 同时接受 `EmbeddingRequest` 对象和普通 dict

## 编码约定

- 4 空格缩进，UTF-8 编码，行尾 CRLF（见 `.editorconfig`）
- `snake_case` 函数/变量，`PascalCase` 类名，`UPPER_SNAKE_CASE` 常量
- 新增代码注释、文档统一使用中文；仅在引用外部库参数名、模型名、命令名时保留英文原文
- 每个方法尽量加上中文注释
- 不要在源码中硬编码令牌、账号信息或机器私有路径

## 测试

当前无测试目录。后续新增测试放在 `tests/` 下，命名为 `test_*.py`。优先补充池化逻辑、指令拼接、相似度计算等轻量单元测试。提交前至少执行 `python -m compileall src` 并做一次 CLI 冒烟验证。
