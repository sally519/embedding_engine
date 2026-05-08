# Repository Guidelines

## 项目结构与模块组织

本仓库用于在本地基于 PyTorch 和 Hugging Face Transformers 运行 `Qwen/Qwen3-Embedding-0.6B` 向量模型。

- `src/embedding_engine/model.py`：模型加载、分词、池化与相似度计算逻辑。
- `src/embedding_engine/cli.py`：命令行入口，对外暴露 `qwen-embed` 命令。
- `src/embedding_engine/__init__.py`：包导出入口。
- `pyproject.toml`：项目元数据与依赖声明。
- `README.md`：本地安装、运行方式与示例说明。

不要提交生成目录或本地环境文件，例如 `.venv/`、`__pycache__/`、`.idea/`、`src/*.egg-info/`。

## 构建、测试与开发命令

- `python -m venv .venv`：创建虚拟环境。
- `.\.venv\Scripts\Activate.ps1`：在 PowerShell 中激活虚拟环境。
- `pip install -e .`：以可编辑模式安装项目。
- `qwen-embed --device cpu demo`：运行内置示例，验证模型推理链路。
- `qwen-embed --device cpu embed --text "你好" --show-shape`：生成文本向量并输出形状。
- `python -m compileall src`：做一次快速语法校验，不依赖模型下载。

注意：全局参数必须放在子命令前，例如 `qwen-embed --device cpu demo`，不要写成 `qwen-embed demo --device cpu`。

## 编码风格与命名约定

- 使用 4 空格缩进，源码文件统一为 UTF-8 编码。
- 函数、变量使用 `snake_case`，类名使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。
- 模块职责保持单一：模型相关逻辑放在 `model.py`，命令行参数处理放在 `cli.py`。
- 优先写清晰、直接的小函数，避免无必要的抽象。

当前仓库未配置格式化或静态检查工具，提交前请保持与现有风格一致。

## 测试约定

当前还没有正式测试目录。后续新增测试时请遵循：

- 测试文件放在 `tests/` 目录下，命名为 `test_*.py`。
- 优先补充池化逻辑、指令拼接、相似度计算等轻量单元测试。
- 提交前至少执行一次 `python -m compileall src`，并补一次真实 CLI 冒烟验证。

示例：
`qwen-embed --device cpu similarity --query "北京首都" --document "中国首都是北京"`

## 提交与合并请求约定

当前目录还不是 Git 仓库，没有可参考的提交历史。默认采用简洁的祈使句提交信息，例如：

- `新增批量向量生成命令`
- `修复命令行参数顺序问题`

如果后续接入 Git 或代码评审，合并请求至少应包含：

- 变更目的与范围；
- 本地验证方式；
- 是否影响依赖安装或模型下载；
- 若命令行行为变化，附一段实际输出示例。

## 仓库长期约定

本仓库从现在开始采用以下中文规范，视为仓库级长期记忆：

- 新增代码注释统一使用中文。
- 新增说明文档、使用手册、排障文档统一使用中文。
- 面向仓库协作的说明文字优先使用中文；仅在引用外部库参数名、模型名、命令名时保留英文原文。
- 所有的代码中每个方法都尽量加上中文注释，方便阅读。

如果需要引入第三方英文内容，请在不改变原意的前提下补充中文说明。

## 安全与配置提示

模型首次运行会从 Hugging Face 下载。若遇到限流，可配置 `HF_TOKEN`。不要在源码中硬编码令牌、账号信息或机器私有路径。
