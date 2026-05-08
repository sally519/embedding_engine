from __future__ import annotations

from dataclasses import dataclass

# 默认使用的向量模型 ID，对应 Hugging Face 上的 Qwen3-Embedding-0.6B。
DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

# 默认检索任务描述，用于 query 类型文本的 instruction 拼接。
# 当用户未显式指定 task_description 时，会使用这段英文描述。
DEFAULT_TASK_DESCRIPTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


@dataclass
class EngineConfig:
    """引擎全局配置。

    这是整个 SDK 的基础配置项集合。外部工程通常只需要修改 device 或 cache_dir，
    其余字段保持默认值即可正常工作。所有字段均有默认值，可以直接无参构造。

    Attributes:
        default_model_id: 默认模型 ID，对应 Hugging Face 上的模型仓库路径。
            默认值为 ``Qwen/Qwen3-Embedding-0.6B``。
        device: 推理设备。可选 ``"auto"``（自动检测）、``"cpu"``、``"cuda"``。
            ``"auto"`` 模式下优先使用 CUDA，不可用时回退到 CPU。
        max_length: 分词器最大序列长度。超过此长度的输入文本会被截断。
            默认 2048，与 Qwen3-Embedding-0.6B 模型的推荐值一致。
        cache_dir: Hugging Face 模型缓存目录。为 None 时使用系统默认路径
            （通常为 ``~/.cache/huggingface/hub``）。
        batch_size: 每批推理的文本条数。为 None 时不分批，一次性全部处理；
            设置后文本量大于此值时自动分批推理，避免内存或显存溢出。
    """

    default_model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    max_length: int = 2048
    cache_dir: str | None = None
    batch_size: int | None = None


# 默认使用的重排序模型 ID，对应 BAAI/bge-reranker-v2-m3。
DEFAULT_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

# bge-reranker-v2-m3 模型微调时使用的最大序列长度。
DEFAULT_RERANKER_MAX_LENGTH = 1024


@dataclass
class RerankerConfig:
    """重排序引擎全局配置。

    与 :class:`EngineConfig` 对称设计，但使用独立的默认模型和序列长度。
    bge-reranker-v2-m3 基于 XLMRoberta 架构，微调时 max_length 为 1024。

    Attributes:
        default_model_id: 默认重排序模型 ID。
            默认为 ``BAAI/bge-reranker-v2-m3``。
        device: 推理设备。可选 ``"auto"``（自动检测）、``"cpu"``、``"cuda"``。
        max_length: 分词器最大序列长度，默认 1024。
        cache_dir: Hugging Face 模型缓存目录。为 None 时使用系统默认路径。
        batch_size: 每批推理的 query-doc 对数。为 None 时不分批。
    """

    default_model_id: str = DEFAULT_RERANKER_MODEL_ID
    device: str = "auto"
    max_length: int = DEFAULT_RERANKER_MAX_LENGTH
    cache_dir: str | None = None
    batch_size: int | None = None
