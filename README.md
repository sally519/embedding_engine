# Embedding Engine

本地 PyTorch 运行器，集成向量模型和重排序模型，封装为可供其他 Python 工程直接 import 的 SDK，同时提供 CLI 调试工具。

**向量模型**：`Qwen/Qwen3-Embedding-0.6B` — 文本向量化，支持 query/document 双模式编码、维度截断、相似度计算。

**重排序模型**：`BAAI/bge-reranker-v2-m3` — Cross-Encoder 重排序，输入 query + 文档列表，输出按相关性降序排列的结果。

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

### 向量模型

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

### 重排序模型

对文档按查询相关性重排序：

```powershell
qwen-embed --device cpu rerank `
  --query "中国首都是哪里" `
  --document "北京是中国的首都" `
  --document "巴黎是法国的首都" `
  --document "东京是日本的首都"
```

输出按相关性降序排列的 JSON 结果，每条包含 `index`（原始位置）、`document`（文档文本）、`relevance_score`（相关性分数，越高越相关）。

## 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-id` | Hugging Face 模型 ID | embedding 子命令默认 `Qwen/Qwen3-Embedding-0.6B`，rerank 子命令默认 `BAAI/bge-reranker-v2-m3` |
| `--device` | 运行设备，可选 `auto`、`cpu`、`cuda` | `auto` |
| `--max-length` | 最大分词长度 | embedding 默认 `2048`，rerank 默认 `1024` |
| `--batch-size` | 每批推理数量 | 不分批 |
| `--cache-dir` | 自定义模型缓存目录 | 系统默认 |
| `--task-description` | 检索任务下 query 的指令描述 | 内置默认描述 |
| `--output-dimension` | 向量截断维度，如 `256` | 不截断 |

注意：全局参数必须放在子命令前面，例如 `qwen-embed --device cpu embed --text "你好"`。

## 在其他工程中调用

### 向量编码

```python
from embedding_engine import create_embedding_sdk

sdk = create_embedding_sdk(device="cpu")

# 便捷方法
result = sdk.embed_texts(
    texts=["中国首都是北京", "北京是中国首都"],
    type="document",
    output_dimension=128,
)
print(result.dimension)       # 128
print(len(result.embeddings)) # 2

# 或传协议对象 / dict
result = sdk.embed({"texts": ["你好"], "type": "query"})
```

### 重排序

```python
from embedding_engine import create_reranker_sdk

sdk = create_reranker_sdk(device="cpu")

# 便捷方法
result = sdk.rerank_documents(
    query="什么是机器学习？",
    documents=[
        "机器学习是人工智能的一个分支。",
        "巴黎是法国的首都。",
        "ML models learn patterns from data.",
    ],
    top_n=2,
)
for r in result.results:
    print(f"score={r.relevance_score:.2f} | {r.document}")

# 或传协议对象 / dict
result = sdk.rerank({"query": "你好", "documents": ["世界"]})
```

### 模型预加载

默认惰性加载（首次调用时才加载模型），可通过 `preload` 提前加载：

```python
sdk = create_embedding_sdk(device="cpu", preload=True)
sdk = create_reranker_sdk(device="cpu", preload=True)

# 或创建后手动调用
sdk.preload()
```

### 分批推理

文本量大时设置 `batch_size` 避免内存或显存溢出：

```python
sdk = create_embedding_sdk(device="cuda", batch_size=16)
sdk = create_reranker_sdk(device="cuda", batch_size=32)
```

## SDK 模块结构

```
src/embedding_engine/
├── config.py              EngineConfig / RerankerConfig 数据类与默认常量
├── schemas.py             Pydantic 协议对象（Request / Response / Result / UsageInfo）
├── engine/
│   ├── model.py           EmbeddingEngine — 向量模型加载、分词、池化、归一化、相似度
│   ├── pooling.py         last_token_pool — 左侧 padding + 最后一个有效 token
│   ├── service.py         EmbeddingService — 协议映射、引擎缓存
│   ├── reranker.py        RerankerEngine — 重排序模型加载、query-doc 对推理
│   └── reranker_service.py RerankerService — 协议映射、引擎缓存
├── sdk.py                 EmbeddingSDK + RerankerSDK + 工厂函数
└── cli.py                 argparse CLI（demo / embed / similarity / rerank）
```

## 缓存位置

如果未显式指定 `--cache-dir`、`HF_HOME` 或 `HF_HUB_CACHE`，模型会默认下载到：

`C:\Users\你的用户名\.cache\huggingface\hub`

## 编码说明

本仓库统一使用 UTF-8 编码保存源码和文档。若 PowerShell 控制台显示中文乱码，不要直接把终端乱码内容复制回文件，优先重新打开源文件并按 UTF-8 方式编辑。
