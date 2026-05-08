# Qwen3 Embedding 本地运行器

这个项目用于在本地基于 PyTorch 运行 `Qwen/Qwen3-Embedding-0.6B` 向量模型，并封装成可被其他 Python 工程直接调用的 SDK。

## 环境要求

- Python 3.10 到 3.12
- Windows PowerShell
- 首次运行时可访问 Hugging Face

## 安装方式

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

如果你使用 NVIDIA GPU，建议先安装与你本机 CUDA 版本匹配的 PyTorch，再执行：

```powershell
pip install -e .
```

## 快速开始

运行内置示例：

```powershell
qwen-embed --device cpu demo
```

生成文本向量：

```powershell
qwen-embed --device cpu embed `
  --text "OpenAI develops AI systems." `
  --text "Beijing is the capital of China." `
  --show-shape
```

计算查询与文档相似度：

```powershell
qwen-embed --device cpu similarity `
  --query "What is the capital of China?" `
  --query "Explain gravity" `
  --document "The capital of China is Beijing." `
  --document "Gravity is a force that attracts two bodies towards each other."
```

## 常用参数

- `--model-id`：模型名称，默认是 `Qwen/Qwen3-Embedding-0.6B`
- `--device`：运行设备，可选 `auto`、`cpu`、`cuda`
- `--max-length`：最大分词长度，默认 `2048`
- `--task-description`：检索任务下 query 使用的指令描述
- `--output-dimension`：向量截断维度，例如 `256`
- `--cache-dir`：自定义 Hugging Face 模型缓存目录

## SDK 模块结构

- `config.py`：默认模型与运行配置
- `contracts/`：输入输出协议对象
- `core/engine.py`：模型加载、分词、池化、相似度计算
- `core/service.py`：统一协议到模型层的服务编排
- `sdk.py`：提供给外部工程直接调用的 SDK 入口
- `factory.py`：SDK 工厂函数
- `cli.py`：本地调试命令行工具

## 统一协议

本工程对外暴露统一输入输出协议，供另一个工程直接 import 调用。

输入字段：

- `texts`：必传，待编码文本列表
- `type`：可选，`query` 或 `document`
- `model`：非必传；为空时默认使用本地 `Qwen/Qwen3-Embedding-0.6B`
- `output_dimension`：可选，输出维度截断
- `normalized`：是否归一化

输出字段：

- `embeddings`
- `dimension`
- `model_name`
- `normalized`
- `usage`

## 在其他工程中调用

```python
from embedding_engine import create_embedding_sdk

sdk = create_embedding_sdk(device="cpu")

result = sdk.embed_texts(
    texts=["中国首都是北京", "北京是中国首都"],
    type="document",
    output_dimension=128,
)

print(result.model_name)
print(result.dimension)
print(result.usage)
print(len(result.embeddings))
```

如果你想直接传统一协议对象：

```python
from embedding_engine import EmbeddingRequest, create_embedding_sdk

sdk = create_embedding_sdk()
request = EmbeddingRequest(
    texts=["中国首都是北京"],
    type="query",
    model=None,
    output_dimension=256,
)
result = sdk.embed(request)
```

## 缓存位置

如果未显式指定 `--cache-dir`、`HF_HOME` 或 `HF_HUB_CACHE`，模型会默认下载到：

`C:\Users\你的用户名\.cache\huggingface\hub`

本机当前默认缓存根目录为：

`C:\Users\zsq51\.cache\huggingface\hub`

## 编码说明

本仓库统一使用 UTF-8 编码保存源码和文档。若 PowerShell 控制台显示中文乱码，不要直接把终端乱码内容复制回文件，优先重新打开源文件并按 UTF-8 方式编辑。
