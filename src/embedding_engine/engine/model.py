from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from ..config import DEFAULT_MODEL_ID, DEFAULT_TASK_DESCRIPTION
from .pooling import last_token_pool


def build_instruction(task_description: str, query: str) -> str:
    """为检索查询拼接 instruction 前缀。

    在检索（retrieval）场景下，query 端需要拼接一段任务描述（instruction），
    让模型区分查询和文档的语义角色。生成的格式为::

        Instruct: {task_description}
        Query:{query}

    注意：``Query:`` 前没有换行符，``{query}`` 后也没有，这是 Qwen3-Embedding
    官方示例中的格式。文档侧不需要拼接 instruction，原样传入即可。

    Args:
        task_description: 检索任务描述，例如
            ``"Given a web search query, retrieve relevant passages that answer the query"``。
        query: 原始查询文本，例如 ``"中国首都是哪里？"``。

    Returns:
        拼接后的完整字符串，例如::

            Instruct: Given a web search query, retrieve relevant passages that answer the query
            Query:中国首都是哪里？

    Example::

        >>> build_instruction("检索相关文档", "什么是引力？")
        'Instruct: 检索相关文档\\nQuery:什么是引力？'
    """
    return f"Instruct: {task_description}\nQuery:{query}"


@dataclass
class EmbeddingResult:
    """模型编码的原始结果。

    封装了 PyTorch 张量形式的向量结果和原始输入文本，
    便于在内部各层之间传递，也方便调试时对照输入与输出。

    Attributes:
        embeddings: 编码结果向量张量，形状 ``(batch_size, hidden_dim)``。
            如果指定了 ``output_dimension`` 则最后一维为截断后的维度。
            默认已做 L2 归一化（可通过参数关闭）。
        texts: 原始输入文本列表（query 类型时为拼接 instruction 后的文本），
            保留此字段方便调试和结果对照。
    """

    embeddings: Tensor
    texts: list[str]


class EmbeddingEngine:
    """向量编码引擎，负责模型加载与推理的核心逻辑。

    封装了从模型加载、分词、前向推理、池化到归一化的完整编码流程。
    本类是底层组件，直接操作 PyTorch 模型和张量，不涉及协议转换。

    初始化时会从 Hugging Face 加载模型和分词器到指定设备。
    首次运行会自动下载模型文件（约 1.2GB），后续使用本地缓存。

    典型用法::

        engine = EmbeddingEngine(device="cpu", batch_size=8)
        result = engine.embed(["你好世界"])
        print(result.embeddings.shape)  # torch.Size([1, 1024])

    Args:
        model_id: Hugging Face 模型 ID，默认 ``Qwen/Qwen3-Embedding-0.6B``。
        device: 推理设备，``"auto"``（自动检测）、``"cpu"`` 或 ``"cuda"``。
        max_length: 分词器最大序列长度，超出部分会被截断。默认 2048。
        cache_dir: Hugging Face 模型缓存目录。为 None 时使用系统默认路径。
        batch_size: 每批送入模型的文本条数。文本量大时自动按此值分批推理，
            避免一次性占满内存或显存。为 None 时不分批，一次性全部处理。

    Attributes:
        model_id: 实际使用的模型 ID。
        device: 实际使用的 ``torch.device`` 对象。
        max_length: 分词器最大序列长度。
        batch_size: 每批推理的文本条数。
        tokenizer: Hugging Face 分词器实例，配置为左侧 padding。
        model: Hugging Face 模型实例，已加载到指定设备并处于 eval 模式。
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        max_length: int = 2048,
        cache_dir: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size

        # Qwen3 Embedding 推荐使用左侧 padding，便于最后一个有效 token 池化。
        # 左侧 padding 意味着 padding token 放在序列左侧，有效 token 靠右排列，
        # 这样序列最后一个位置就是最后一个有效 token，池化时可以直接取。
        tokenizer_kwargs = {"padding_side": "left"}
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        model_kwargs = {}
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir

        self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """将设备字符串解析为 ``torch.device`` 对象。

        ``"auto"`` 模式下优先检测 CUDA 是否可用，可用则返回 ``cuda``，
        否则回退到 ``cpu``。其他字符串（如 ``"cpu"``、``"cuda"``、
        ``"cuda:0"``）直接透传给 ``torch.device`` 构造。

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

    def embed(
        self,
        texts: Iterable[str],
        *,
        normalize: bool = True,
        output_dimension: int | None = None,
        batch_size: int | None = None,
    ) -> EmbeddingResult:
        """对文本列表进行向量编码。

        完整流程：分词 -> 移入设备 -> 模型前向推理 -> last-token 池化 ->
        可选维度截断 -> 可选 L2 归一化。

        当文本数量较多时，按 ``batch_size`` 分批推理，避免一次性占满
        内存或显存。每批独立完成分词、推理、池化和归一化后，
        将各批向量拼接为一个完整张量返回。

        推理过程使用 ``torch.inference_mode()`` 上下文管理器，
        禁用梯度计算以节省显存和加速推理。

        Args:
            texts: 待编码文本的可迭代对象。不能为空。
                注意：如果是 query 类型的文本，调用方应先通过
                :func:`build_instruction` 拼接 instruction，或直接使用
                :meth:`embed_queries` 方法。
            normalize: 是否对输出向量做 L2 归一化。归一化后向量点积即为余弦相似度。
                默认为 True。
            output_dimension: 输出维度截断值。为 None 时输出模型原始维度；
                指定后只取前 N 维，适用于需要降维以节省存储空间的场景。
            batch_size: 每批推理的文本条数。为 None 时使用引擎初始化时的
                ``self.batch_size``；如果仍为 None 则一次性全部处理。
                显式传入可覆盖引擎默认值，适合某次调用需要特殊批量的场景。

        Returns:
            :class:`EmbeddingResult` 对象，包含编码后的向量张量和原始输入文本。

        Raises:
            ValueError: 传入的 texts 为空列表时抛出。

        Example::

            engine = EmbeddingEngine(device="cpu", batch_size=8)
            # 100 条文本会自动分成 13 批（12×8 + 1×4）推理
            result = engine.embed(["文本"] * 100)
            print(result.embeddings.shape)  # torch.Size([100, 1024])

            # 或在单次调用时覆盖 batch_size
            result = engine.embed(["文本"] * 100, batch_size=32)
        """
        text_list = list(texts)
        if not text_list:
            raise ValueError("texts must not be empty")

        # 确定实际使用的批量大小：参数 > 引擎默认 > 不分批。
        effective_bs = batch_size or self.batch_size

        # 不分批：一次性处理（文本量少或未设置 batch_size 时走这条路径）。
        if effective_bs is None or len(text_list) <= effective_bs:
            return self._embed_batch(
                text_list, normalize=normalize, output_dimension=output_dimension
            )

        # 分批处理：按 batch_size 切分文本，逐批推理后拼接。
        all_embeddings: list[Tensor] = []
        for start in range(0, len(text_list), effective_bs):
            batch_texts = text_list[start : start + effective_bs]
            batch_result = self._embed_batch(
                batch_texts, normalize=normalize, output_dimension=output_dimension
            )
            all_embeddings.append(batch_result.embeddings)

        return EmbeddingResult(
            embeddings=torch.cat(all_embeddings, dim=0),
            texts=text_list,
        )

    def _embed_batch(
        self,
        text_list: list[str],
        *,
        normalize: bool,
        output_dimension: int | None,
    ) -> EmbeddingResult:
        """对单批文本执行编码（内部方法）。

        执行分词、前向推理、池化、维度截断和归一化的完整流程。
        由 :meth:`embed` 调用，每次只处理一个批次。

        Args:
            text_list: 当前批次的文本列表。
            normalize: 是否做 L2 归一化。
            output_dimension: 维度截断值，为 None 时不截断。

        Returns:
            当前批次的编码结果。
        """
        batch_dict = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}

        with torch.inference_mode():
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            # 维度截断：只取前 output_dimension 个维度，相当于 embeddings[:, :N]。
            if output_dimension is not None:
                embeddings = embeddings[:, :output_dimension]
            # L2 归一化：将每个向量缩放到单位长度，归一化后点积等于余弦相似度。
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        return EmbeddingResult(
            embeddings=embeddings.detach().cpu(),
            texts=text_list,
        )

    def embed_queries(
        self,
        queries: Iterable[str],
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        *,
        normalize: bool = True,
        output_dimension: int | None = None,
        batch_size: int | None = None,
    ) -> EmbeddingResult:
        """对检索查询文本进行向量编码（自动拼接 instruction）。

        与 :meth:`embed` 的区别在于，本方法会自动为每条查询文本拼接
        instruction 前缀（通过 :func:`build_instruction`），
        以适配 Qwen3-Embedding 的检索模式。

        拼接后的文本格式为::

            Instruct: {task_description}
            Query:{原始查询}

        Args:
            queries: 原始查询文本的可迭代对象，不需要预先拼接 instruction。
            task_description: 检索任务描述，默认使用 ``DEFAULT_TASK_DESCRIPTION``。
                可自定义以适配不同的检索场景（如代码搜索、问答检索等）。
            normalize: 是否对输出向量做 L2 归一化。默认为 True。
            output_dimension: 输出维度截断值。为 None 时不截断。
            batch_size: 每批推理的文本条数。为 None 时使用引擎默认值。

        Returns:
            :class:`EmbeddingResult` 对象，包含编码后的向量张量。
            注意 ``texts`` 字段保存的是拼接 instruction 后的完整文本。

        Example::

            engine = EmbeddingEngine(device="cpu", batch_size=8)
            result = engine.embed_queries(["中国首都是哪里？"])
            # result.texts[0] == "Instruct: ...\\nQuery:中国首都是哪里？"
        """
        instructed_queries = [
            build_instruction(task_description, query) for query in queries
        ]
        return self.embed(
            instructed_queries,
            normalize=normalize,
            output_dimension=output_dimension,
            batch_size=batch_size,
        )

    @staticmethod
    def similarity_matrix(left: Tensor, right: Tensor) -> Tensor:
        """计算两组向量之间的两两相似度矩阵。

        使用矩阵乘法计算点积。在向量已做 L2 归一化的前提下，
        点积等价于余弦相似度，结果范围 ``[-1, 1]``。

        输出矩阵中 ``result[i][j]`` 表示 ``left[i]`` 与 ``right[j]`` 的相似度。

        Args:
            left: 左侧向量矩阵，形状 ``(m, dim)``。
            right: 右侧向量矩阵，形状 ``(n, dim)``。
                ``left`` 和 ``right`` 的最后一维（dim）必须一致。

        Returns:
            相似度矩阵，形状 ``(m, n)``。其中 ``result[i][j]`` 是
            ``left[i]`` 与 ``right[j]`` 的余弦相似度（归一化前提下）。

        Example::

            query_emb = engine.embed_queries(["北京"])    # shape: (1, 1024)
            doc_emb = engine.embed(["中国首都是北京"])     # shape: (1, 1024)
            scores = engine.similarity_matrix(query_emb.embeddings, doc_emb.embeddings)
            # scores.shape == (1, 1)，值越大越相似
        """
        # 在默认已归一化的前提下，点积可以直接视为余弦相似度。
        return left @ right.T
