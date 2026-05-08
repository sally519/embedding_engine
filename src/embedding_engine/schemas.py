from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# 向量编码类型。query 会自动拼接检索 instruction，document 则原样编码。
EmbeddingType = Literal["query", "document"]


class EmbeddingRequest(BaseModel):
    """统一向量编码请求协议。

    这是本 SDK 对外暴露的标准输入结构，所有调用方式（SDK 方法、CLI、未来 HTTP 接口）
    最终都会转换为该对象。使用 Pydantic BaseModel 进行数据校验。

    ``texts`` 是唯一必传字段，其余字段均有合理默认值。外部工程可以按需设置
    ``type``、``output_dimension`` 等参数来控制编码行为。

    Example::

        request = EmbeddingRequest(
            texts=["中国首都是北京", "北京是中国首都"],
            type="document",
            output_dimension=128,
        )

    Attributes:
        texts: 待编码文本列表，至少包含一条文本。
            会自动去除每条文本首尾空白，不允许传入空字符串。
        type: 向量编码类型。``"query"`` 表示检索查询，会自动在文本前拼接
            instruction 前缀；``"document"`` 表示待检索文档，原样编码。
            默认为 ``"document"``。
        model: 模型名称，非必传。为 None 时使用本地默认模型
            ``Qwen/Qwen3-Embedding-0.6B``。
        output_dimension: 输出向量维度截断值。为 None 时输出模型原始维度（如 1024）；
            指定后只取前 N 维，适用于需要降维的场景。必须 >= 1。
        normalized: 是否对输出向量做 L2 归一化。归一化后可直接用点积计算余弦相似度。
            默认为 True。
        task_description: 检索任务描述，仅在 ``type="query"`` 时生效。
            用于覆盖默认的 instruction 文本。为 None 时使用
            ``DEFAULT_TASK_DESCRIPTION``。
    """

    texts: list[str] = Field(..., min_length=1, description="待编码文本列表。")
    type: EmbeddingType = Field(
        default="document",
        description="向量类型。query 会自动拼接检索 instruction。",
    )
    model: str | None = Field(
        default=None,
        description="模型名称，非必传；为空时默认使用本地 Qwen3 向量模型。",
    )
    output_dimension: int | None = Field(
        default=None,
        ge=1,
        description="可选的输出维度截断值。",
    )
    normalized: bool = Field(
        default=True,
        description="是否对输出向量做 L2 归一化。",
    )
    task_description: str | None = Field(
        default=None,
        description="仅在 query 类型下生效，用于覆盖默认检索 instruction。",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        """校验并清洗 texts 字段。

        对每条文本执行 strip() 去除首尾空白后，检查是否包含空字符串。
        如果存在空字符串则抛出 ValueError，防止空文本进入模型推理流程。

        Args:
            texts: 用户传入的原始文本列表。

        Returns:
            清洗后的文本列表（每条文本已去除首尾空白）。

        Raises:
            ValueError: 清洗后存在空字符串。
        """
        cleaned_texts = [text.strip() for text in texts]
        if any(not text for text in cleaned_texts):
            raise ValueError("texts 中不能包含空字符串")
        return cleaned_texts


class UsageInfo(BaseModel):
    """调用用量统计信息。

    用于让外部工程记录本次编码请求的批量大小与 token 消耗，
    便于成本统计和日志记录。

    Attributes:
        input_count: 输入文本条数，即请求中 ``texts`` 的长度。
        prompt_tokens: 输入 token 总数（含 instruction 拼接后的 token）。
            query 类型会统计 instruction 拼接后的 token 数。
        total_tokens: 总 token 数。当前与 prompt_tokens 一致，
            因为模型侧没有额外生成 token。
    """

    input_count: int
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """统一向量编码响应协议。

    这是本 SDK 对外暴露的标准输出结构，包含编码结果向量和相关元信息。

    Attributes:
        embeddings: 编码结果向量列表，外层长度等于输入文本条数，
            内层每个列表是一条文本对应的浮点向量。
        dimension: 输出向量的实际维度。如果请求中指定了 ``output_dimension``，
            此处为截断后的维度；否则为模型原始输出维度。
        model_name: 实际使用的模型 ID（如 ``Qwen/Qwen3-Embedding-0.6B``），
            用于外部工程确认当前使用的模型版本。
        normalized: 向量是否已做 L2 归一化，与请求中的 ``normalized`` 一致。
        usage: 本次调用的用量统计信息。
    """

    embeddings: list[list[float]]
    dimension: int
    model_name: str
    normalized: bool
    usage: UsageInfo
