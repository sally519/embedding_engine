from __future__ import annotations

from .config import DEFAULT_MODEL_ID, EngineConfig
from .schemas import EmbeddingRequest, EmbeddingResponse
from .engine.service import EmbeddingService


class EmbeddingSDK:
    """面向外部工程的向量编码 SDK 主入口。

    本类是本仓库对外暴露的核心 API。外部工程只需 ``import`` 后创建实例，
    即可调用 ``embed`` 或 ``embed_texts`` 方法进行文本向量编码。

    内部会根据配置创建 :class:`EmbeddingService`，由服务层管理模型加载和推理。
    模型在首次调用编码方法时才会实际加载（惰性初始化），创建 SDK 实例本身不会
    触发模型下载或 GPU 内存占用。

    典型用法::

        from embedding_engine import create_embedding_sdk

        sdk = create_embedding_sdk(device="cpu")
        result = sdk.embed_texts(
            texts=["中国首都是北京", "北京是中国首都"],
            type="document",
            output_dimension=128,
        )
        print(result.dimension)       # 128
        print(len(result.embeddings)) # 2

    Args:
        model: 默认模型 ID，默认 ``Qwen/Qwen3-Embedding-0.6B``。
        device: 推理设备，``"auto"``（自动检测）、``"cpu"`` 或 ``"cuda"``。
        max_length: 分词器最大序列长度，默认 2048。
        cache_dir: Hugging Face 模型缓存目录，为 None 时使用系统默认路径。

    Attributes:
        config: 引擎全局配置实例。
        service: 底层服务实例，负责协议转换和模型编排。
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        max_length: int = 2048,
        cache_dir: str | None = None,
    ) -> None:
        self.config = EngineConfig(
            default_model_id=model,
            device=device,
            max_length=max_length,
            cache_dir=cache_dir,
        )
        self.service = EmbeddingService(self.config)

    def embed(self, request: EmbeddingRequest | dict) -> EmbeddingResponse:
        """通过统一协议对象进行向量编码。

        接受 :class:`EmbeddingRequest` 协议对象或等价的 ``dict`` 字典，
        执行编码后返回 :class:`EmbeddingResponse` 协议对象。
        传 ``dict`` 时会自动通过 Pydantic 校验并转换为 ``EmbeddingRequest``。

        适合已有一套标准协议的外部工程使用，可以直接传递结构化数据。

        Args:
            request: 编码请求，支持两种形式：
                - :class:`EmbeddingRequest` 实例：完整协议对象
                - ``dict``：等价字典，如 ``{"texts": ["你好"], "type": "document"}``

        Returns:
            :class:`EmbeddingResponse` 协议对象，包含编码结果和元信息。

        Example::

            sdk = create_embedding_sdk()
            # 方式一：传协议对象
            result = sdk.embed(EmbeddingRequest(texts=["你好"], type="document"))
            # 方式二：传字典
            result = sdk.embed({"texts": ["你好"], "type": "document"})
        """
        if isinstance(request, dict):
            request = EmbeddingRequest(**request)
        return self.service.embed(request)

    def embed_texts(
        self,
        texts: list[str],
        *,
        type: str = "document",
        model: str | None = None,
        output_dimension: int | None = None,
        normalized: bool = True,
        task_description: str | None = None,
    ) -> EmbeddingResponse:
        """便捷方法：通过关键字参数直接编码文本列表。

        无需手动构造 :class:`EmbeddingRequest` 对象，直接传入文本列表和
        可选参数即可完成编码。内部会自动构建请求并调用服务层。

        这是最常用的编码方法，适合大多数场景。

        Args:
            texts: 待编码文本列表，至少包含一条文本。
            type: 编码类型，``"query"``（检索查询，自动拼接 instruction）或
                ``"document"``（文档，原样编码）。默认 ``"document"``。
            model: 模型 ID，为 None 时使用默认模型。
            output_dimension: 输出维度截断值。为 None 时不截断，输出模型原始维度。
            normalized: 是否对输出向量做 L2 归一化。默认为 True。
            task_description: 检索任务描述，仅在 ``type="query"`` 时生效。
                为 None 时使用默认描述。

        Returns:
            :class:`EmbeddingResponse` 协议对象，包含编码结果和元信息。

        Example::

            sdk = create_embedding_sdk()

            # 编码文档
            doc_result = sdk.embed_texts(
                texts=["中国首都是北京"],
                type="document",
                output_dimension=256,
            )

            # 编码查询
            query_result = sdk.embed_texts(
                texts=["中国首都是哪里？"],
                type="query",
                output_dimension=256,
            )
        """
        request = EmbeddingRequest(
            texts=texts,
            type=type,
            model=model,
            output_dimension=output_dimension,
            normalized=normalized,
            task_description=task_description,
        )
        return self.service.embed(request)


def create_embedding_sdk(
    *,
    model: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    max_length: int = 2048,
    cache_dir: str | None = None,
) -> EmbeddingSDK:
    """创建向量编码 SDK 实例的工厂函数。

    这是本仓库推荐的外部接入方式。使用工厂函数而非直接构造
    :class:`EmbeddingSDK` 的好处是：

    - 外部工程不需要关心内部类名，只导入这一个函数即可
    - 未来如果内部实现变更（如切换到异步引擎），只需修改此函数，
      外部调用代码无需改动
    - 所有参数都有合理默认值，最简调用 ``create_embedding_sdk()`` 即可工作

    模型在首次调用编码方法时才会实际加载，创建 SDK 实例本身不会
    触发模型下载或 GPU 内存占用。

    Args:
        model: 默认模型 ID，对应 Hugging Face 上的模型仓库路径。
            默认为 ``Qwen/Qwen3-Embedding-0.6B``。
        device: 推理设备，可选值：
            - ``"auto"``：自动检测，优先使用 CUDA，不可用时回退到 CPU
            - ``"cpu"``：强制使用 CPU
            - ``"cuda"``：强制使用 GPU
        max_length: 分词器最大序列长度，超过此长度的文本会被截断。
            默认 2048，与模型推荐值一致。
        cache_dir: Hugging Face 模型缓存目录。为 None 时使用系统默认路径
            （Windows 通常为 ``C:\\Users\\<用户名>\\.cache\\huggingface\\hub``）。
            可通过此参数将模型缓存到自定义目录，如网络磁盘或本地大容量磁盘。

    Returns:
        已初始化的 :class:`EmbeddingSDK` 实例，可直接调用 ``embed`` 或
        ``embed_texts`` 方法进行文本向量编码。

    Example::

        from embedding_engine import create_embedding_sdk

        # 最简用法：全部使用默认值
        sdk = create_embedding_sdk()

        # 指定设备和缓存目录
        sdk = create_embedding_sdk(device="cuda", cache_dir="D:/models")

        # 指定输出维度并编码
        result = sdk.embed_texts(
            texts=["你好世界"],
            output_dimension=256,
        )
    """
    return EmbeddingSDK(
        model=model,
        device=device,
        max_length=max_length,
        cache_dir=cache_dir,
    )
