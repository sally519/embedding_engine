from __future__ import annotations

from ..config import DEFAULT_TASK_DESCRIPTION, EngineConfig
from ..schemas import EmbeddingRequest, EmbeddingResponse, UsageInfo
from .model import EmbeddingEngine, build_instruction


class EmbeddingService:
    """向量编码服务层，负责统一协议到模型层的映射与编排。

    本类是协议层（schemas）与模型层（model）之间的桥梁：
    - 接收 :class:`EmbeddingRequest` 协议对象
    - 根据 ``type`` 字段决定编码方式（query 拼接 instruction / document 原样编码）
    - 调用底层 :class:`EmbeddingEngine` 执行推理
    - 统计 token 用量，组装 :class:`EmbeddingResponse` 返回

    本类不关心 HTTP 传输或存储实现，可在 SDK、CLI、HTTP 服务等多种场景中复用。

    内部按 ``model_id`` 惰性初始化并缓存 :class:`EmbeddingEngine` 实例，
    同一个 ``model_id`` 只会加载一次模型，避免重复占用内存。

    Args:
        config: 引擎全局配置。为 None 时使用默认配置（所有字段取默认值）。
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._engines: dict[str, EmbeddingEngine] = {}

    def get_engine(self, model_id: str | None) -> EmbeddingEngine:
        """获取或创建指定模型的编码引擎实例。

        内部维护一个按 ``model_id`` 索引的引擎缓存字典。
        首次请求某个 ``model_id`` 时会加载模型并缓存；
        后续请求直接返回缓存实例，避免重复加载。

        当 ``model_id`` 为 None 时，使用配置中的 ``default_model_id``。

        Args:
            model_id: 模型 ID，如 ``"Qwen/Qwen3-Embedding-0.6B"``。
                为 None 时使用默认模型。

        Returns:
            对应模型 ID 的 :class:`EmbeddingEngine` 实例（已加载模型，处于 eval 模式）。
        """
        resolved_model_id = model_id or self.config.default_model_id
        if resolved_model_id not in self._engines:
            self._engines[resolved_model_id] = EmbeddingEngine(
                model_id=resolved_model_id,
                device=self.config.device,
                max_length=self.config.max_length,
                cache_dir=self.config.cache_dir,
            )
        return self._engines[resolved_model_id]

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """执行向量编码，将请求协议对象转换为响应协议对象。

        这是服务层的核心方法，处理流程如下：

        1. 根据请求中的 ``model`` 字段获取（或创建）对应的引擎实例
        2. 根据 ``type`` 字段选择编码路径：
           - ``"query"``：调用 :meth:`EmbeddingEngine.embed_queries`，自动拼接 instruction
           - ``"document"``：调用 :meth:`EmbeddingEngine.embed`，原样编码
        3. 统计本次请求的 token 用量
        4. 组装并返回 :class:`EmbeddingResponse`

        Args:
            request: 统一编码请求协议对象，包含待编码文本、类型、维度等参数。

        Returns:
            :class:`EmbeddingResponse` 协议对象，包含编码结果向量、维度、模型名称和用量统计。

        Example::

            service = EmbeddingService()
            request = EmbeddingRequest(texts=["你好"], type="document")
            response = service.embed(request)
            print(response.dimension)    # 1024
            print(len(response.embeddings))  # 1
        """
        engine = self.get_engine(request.model)

        if request.type == "query":
            task_description = request.task_description or DEFAULT_TASK_DESCRIPTION
            result = engine.embed_queries(
                request.texts,
                task_description=task_description,
                normalize=request.normalized,
                output_dimension=request.output_dimension,
            )
            # query 类型统计 token 时，需要使用拼接 instruction 后的文本，
            # 因为实际送入分词器的是拼接后的文本。
            texts_for_token_count = [
                build_instruction(task_description, text) for text in request.texts
            ]
        else:
            result = engine.embed(
                request.texts,
                normalize=request.normalized,
                output_dimension=request.output_dimension,
            )
            texts_for_token_count = request.texts

        prompt_tokens = self._count_tokens(engine, texts_for_token_count)
        return EmbeddingResponse(
            embeddings=result.embeddings.tolist(),
            dimension=int(result.embeddings.shape[1]),
            model_name=engine.model_id,
            normalized=request.normalized,
            usage=UsageInfo(
                input_count=len(request.texts),
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )

    @staticmethod
    def _count_tokens(engine: EmbeddingEngine, texts: list[str]) -> int:
        """统计文本列表经过分词后的总 token 数。

        使用引擎的分词器对文本进行分词（含 padding 和截断），
        然后通过 ``attention_mask`` 的求和统计有效 token 总数。
        这确保了统计结果与实际送入模型的 token 数一致。

        Args:
            engine: 已加载模型的引擎实例，使用其分词器进行分词。
            texts: 待统计的文本列表（query 类型应为拼接 instruction 后的文本）。

        Returns:
            所有文本的有效 token 总数（int）。
        """
        batch_dict = engine.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=engine.max_length,
            return_tensors="pt",
        )
        return int(batch_dict["attention_mask"].sum().item())
