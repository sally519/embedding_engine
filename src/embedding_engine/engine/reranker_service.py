from __future__ import annotations

from ..config import RerankerConfig
from ..schemas import RerankRequest, RerankResponse, RerankResult, RerankUsageInfo
from .reranker import RerankerEngine


class RerankerService:
    """重排序服务层，负责统一协议到模型层的映射与编排。

    与 :class:`EmbeddingService` 对称设计：
    - 接收 :class:`RerankRequest` 协议对象
    - 将 query × documents 展开为 ``(query, doc)`` 对列表
    - 调用底层 :class:`RerankerEngine` 执行推理
    - 组装 :class:`RerankResponse` 返回，包含按分数降序排列的结果

    内部按 ``model_id`` 惰性初始化并缓存 :class:`RerankerEngine` 实例。

    Args:
        config: 重排序引擎全局配置。为 None 时使用默认配置。
    """

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig()
        self._engines: dict[str, RerankerEngine] = {}

    def get_engine(self, model_id: str | None) -> RerankerEngine:
        """获取或创建指定模型的重排序引擎实例（带缓存）。

        首次请求某个 ``model_id`` 时会加载模型并缓存；
        后续请求直接返回缓存实例。当 ``model_id`` 为 None 时
        使用配置中的 ``default_model_id``。

        Args:
            model_id: 模型 ID，为 None 时使用默认模型。

        Returns:
            对应模型 ID 的 :class:`RerankerEngine` 实例。
        """
        resolved_model_id = model_id or self.config.default_model_id
        if resolved_model_id not in self._engines:
            self._engines[resolved_model_id] = RerankerEngine(
                model_id=resolved_model_id,
                device=self.config.device,
                max_length=self.config.max_length,
                cache_dir=self.config.cache_dir,
                batch_size=self.config.batch_size,
            )
        return self._engines[resolved_model_id]

    def preload(self) -> None:
        """提前加载默认模型到内存/显存。

        正常情况下模型在首次调用 :meth:`rerank` 时才会加载（惰性初始化）。
        调用此方法可以在服务启动阶段就完成模型加载，避免首次请求延迟。

        如果模型已经加载过，此方法不做任何操作。
        """
        self.get_engine(None)

    def rerank(self, request: RerankRequest) -> RerankResponse:
        """执行重排序，将请求协议对象转换为响应协议对象。

        处理流程：

        1. 根据 ``model`` 字段获取（或创建）引擎
        2. 将 query × documents 展开为 ``(query, doc)`` 对列表
        3. 调用引擎推理得到分数
        4. 组装 ``RerankResult`` 列表（按分数降序排列）
        5. 如果请求指定了 ``top_n``，截断结果
        6. 统计 token 用量
        7. 返回 ``RerankResponse``

        Args:
            request: 统一重排序请求协议对象。

        Returns:
            :class:`RerankResponse` 协议对象。

        Example::

            service = RerankerService()
            request = RerankRequest(
                query="中国首都是哪里",
                documents=["北京是中国的首都", "巴黎是法国的首都"],
            )
            response = service.rerank(request)
            for r in response.results:
                print(f"{r.relevance_score:.2f} | {r.document}")
        """
        engine = self.get_engine(request.model)

        # 展开 query × documents 为对列表。
        pairs = [(request.query, doc) for doc in request.documents]

        result = engine.rerank(pairs)

        # 组装结果列表，每个元素包含原始文档索引和分数。
        scores_list = result.scores.tolist()
        raw_results = []
        for idx, (score, (_, doc)) in enumerate(zip(scores_list, pairs)):
            raw_results.append(
                RerankResult(
                    index=idx,
                    document=doc,
                    relevance_score=score,
                )
            )

        # 按分数降序排列。
        sorted_results = sorted(
            raw_results, key=lambda r: r.relevance_score, reverse=True
        )

        # 如果指定了 top_n，截断结果列表。
        if request.top_n is not None and len(sorted_results) > request.top_n:
            sorted_results = sorted_results[: request.top_n]

        # 统计 token 用量。
        prompt_tokens = self._count_tokens(engine, pairs)

        return RerankResponse(
            model_name=engine.model_id,
            results=sorted_results,
            usage=RerankUsageInfo(
                input_count=len(pairs),
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )

    @staticmethod
    def _count_tokens(
        engine: RerankerEngine, pairs: list[tuple[str, str]]
    ) -> int:
        """统计 query-doc 对经过分词后的总 token 数。

        Args:
            engine: 已加载模型的重排序引擎实例。
            pairs: 待统计的 ``(query, doc)`` 对列表。

        Returns:
            所有对的有效 token 总数（int）。
        """
        queries = [q for q, _ in pairs]
        documents = [d for _, d in pairs]
        batch_dict = engine.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=engine.max_length,
            return_tensors="pt",
        )
        return int(batch_dict["attention_mask"].sum().item())
