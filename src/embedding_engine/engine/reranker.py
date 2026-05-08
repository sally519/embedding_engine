from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import DEFAULT_RERANKER_MODEL_ID


@dataclass
class RerankEngineResult:
    """重排序推理的原始结果（内部使用）。

    与 schemas.RerankResult 不同，本类保存 PyTorch 张量形式的分数，
    仅在引擎内部和服务层之间传递。

    Attributes:
        scores: 相关性分数张量，形状 ``(batch_size,)``。
            每个元素是一个标量 logit，值越大表示 query-doc 对越相关。
        pairs: 原始输入的 ``(query, doc)`` 对列表，保留便于调试对照。
    """

    scores: Tensor
    pairs: list[tuple[str, str]]


class RerankerEngine:
    """重排序推理引擎，负责 bge-reranker-v2-m3 的模型加载与推理。

    封装了模型加载、分词（query-doc pair 拼接）、前向推理、分数提取的完整流程。
    使用 ``AutoModelForSequenceClassification`` 加载 XLMRoberta 模型，
    输出 ``logits.squeeze(-1)`` 作为相关性分数。

    与 :class:`EmbeddingEngine` 的主要区别：
    - 使用 ``AutoModelForSequenceClassification`` 而非 ``AutoModel``
    - 输入是 ``(query, document)`` 对而非独立文本
    - 输出是标量相关性分数而非向量
    - 不需要池化、归一化或维度截断

    典型用法::

        engine = RerankerEngine(device="cpu")
        pairs = [("中国首都是哪里", "北京是中国的首都"), ("中国首都是哪里", "巴黎是法国的首都")]
        result = engine.rerank(pairs)
        print(result.scores)  # tensor([ 5.2, -8.7])

    Args:
        model_id: Hugging Face 模型 ID，默认 ``BAAI/bge-reranker-v2-m3``。
        device: 推理设备，``"auto"``（自动检测）、``"cpu"`` 或 ``"cuda"``。
        max_length: 分词器最大序列长度，默认 1024（模型微调时使用的长度）。
        cache_dir: Hugging Face 模型缓存目录。为 None 时使用系统默认路径。
        batch_size: 每批推理的 query-doc 对数。为 None 时不分批。

    Attributes:
        model_id: 实际使用的模型 ID。
        device: 实际使用的 ``torch.device`` 对象。
        max_length: 分词器最大序列长度。
        batch_size: 每批推理的对数。
        tokenizer: AutoTokenizer 实例（XLMRobertaTokenizer）。
        model: AutoModelForSequenceClassification 实例，已加载到设备并处于 eval 模式。
    """

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER_MODEL_ID,
        device: str = "auto",
        max_length: int = 1024,
        cache_dir: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size

        tokenizer_kwargs = {}
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        model_kwargs = {}
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, **model_kwargs
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """将设备字符串解析为 ``torch.device`` 对象。

        Args:
            device: 设备字符串，支持 ``"auto"``、``"cpu"``、``"cuda"`` 等。

        Returns:
            对应的 ``torch.device`` 对象。
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    def rerank(
        self,
        pairs: Iterable[tuple[str, str]],
        *,
        batch_size: int | None = None,
    ) -> RerankEngineResult:
        """对 query-doc 对列表进行重排序打分。

        每对 ``(query, document)`` 送入 Cross-Encoder 模型，输出一个相关性分数。
        分数为原始 logit，值越大表示越相关，典型范围约 -11 到 +7。

        当对数较多时，按 ``batch_size`` 分批推理，避免占满内存或显存。

        Args:
            pairs: ``(query, document)`` 元组的可迭代对象。不能为空。
            batch_size: 每批推理的对数。为 None 时使用引擎初始化时的
                ``self.batch_size``；如果仍为 None 则一次性全部处理。

        Returns:
            :class:`RerankEngineResult` 对象，包含相关性分数张量和原始输入对。

        Raises:
            ValueError: 传入的 pairs 为空列表时抛出。

        Example::

            engine = RerankerEngine(device="cpu", batch_size=8)
            pairs = [("什么是AI?", "人工智能是..."), ("什么是AI?", "机器学习是...")]
            result = engine.rerank(pairs)
            print(result.scores.tolist())  # [5.2, 3.1]
        """
        pair_list = list(pairs)
        if not pair_list:
            raise ValueError("pairs must not be empty")

        # 确定实际使用的批量大小：参数 > 引擎默认 > 不分批。
        effective_bs = batch_size or self.batch_size

        # 不分批：一次性处理。
        if effective_bs is None or len(pair_list) <= effective_bs:
            return self._rerank_batch(pair_list)

        # 分批处理：按 batch_size 切分，逐批推理后拼接。
        all_scores: list[Tensor] = []
        for start in range(0, len(pair_list), effective_bs):
            batch_pairs = pair_list[start : start + effective_bs]
            batch_result = self._rerank_batch(batch_pairs)
            all_scores.append(batch_result.scores)

        return RerankEngineResult(
            scores=torch.cat(all_scores, dim=0),
            pairs=pair_list,
        )

    def _rerank_batch(
        self,
        pair_list: list[tuple[str, str]],
    ) -> RerankEngineResult:
        """对单批 query-doc 对执行推理（内部方法）。

        分词时 tokenizer 接受 ``(queries, documents)`` 双参数调用，
        自动在 query 和 doc 之间插入 ``</s></s>`` 分隔符（XLMRoBERTa 格式）。
        前向推理后 ``logits.squeeze(-1)`` 得到每对的一个标量分数。

        Args:
            pair_list: 当前批次的 ``(query, doc)`` 对列表。

        Returns:
            当前批次的推理结果。
        """
        queries = [q for q, _ in pair_list]
        documents = [d for _, d in pair_list]

        # tokenizer 接受 (text_pairs_first, text_pairs_second) 格式，
        # 自动拼接为 <s> query </s></s> document </s>
        batch_dict = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}

        with torch.inference_mode():
            outputs = self.model(**batch_dict)
            # logits 形状: (batch_size, 1)，squeeze(-1) 得到 (batch_size,)。
            scores = outputs.logits.squeeze(-1)

        return RerankEngineResult(
            scores=scores.detach().cpu(),
            pairs=pair_list,
        )
