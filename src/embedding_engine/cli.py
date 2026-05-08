from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import DEFAULT_MODEL_ID, DEFAULT_RERANKER_MODEL_ID, DEFAULT_TASK_DESCRIPTION
from .engine.model import EmbeddingEngine
from .engine.reranker_service import RerankerService
from .schemas import RerankRequest


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    定义了 ``qwen-embed`` 命令的完整参数结构：

    - **全局参数**（必须放在子命令前面）：
        - ``--model-id``：Hugging Face 模型 ID
        - ``--device``：推理设备，可选 ``auto``、``cpu``、``cuda``
        - ``--max-length``：分词器最大序列长度
        - ``--cache-dir``：模型缓存目录
        - ``--batch-size``：每批推理数量

    - **子命令**：
        - ``demo``：运行内置相似度示例，验证向量模型推理链路
        - ``embed``：生成文本向量
        - ``similarity``：计算查询与文档的相似度矩阵
        - ``rerank``：对文档按查询相关性重排序

    Returns:
        配置好的 ``ArgumentParser`` 实例。
    """
    parser = argparse.ArgumentParser(
        description="Run Qwen3-Embedding-0.6B locally with PyTorch."
    )
    # 全局 --model-id 默认为 None，各子命令按需 fallback 到自己的默认模型。
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model id. Defaults depend on the subcommand.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Tokenizer max length. Defaults depend on the subcommand.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of texts per inference batch. None means no batching.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a built-in similarity demo.")
    demo.add_argument(
        "--task-description",
        default=DEFAULT_TASK_DESCRIPTION,
        help="Instruction used for query embedding.",
    )

    embed = subparsers.add_parser("embed", help="Generate embeddings for text.")
    embed.add_argument(
        "--text",
        action="append",
        required=True,
        help="Text to embed. Repeat for multiple inputs.",
    )
    embed.add_argument(
        "--output-dimension",
        type=int,
        default=None,
        help="Optional output dimension slice, e.g. 256.",
    )
    embed.add_argument(
        "--show-shape",
        action="store_true",
        help="Print embedding matrix shape.",
    )

    similarity = subparsers.add_parser(
        "similarity", help="Compute similarity between queries and documents."
    )
    similarity.add_argument(
        "--query",
        action="append",
        required=True,
        help="Query text. Repeat for multiple queries.",
    )
    similarity.add_argument(
        "--document",
        action="append",
        required=True,
        help="Document text. Repeat for multiple documents.",
    )
    similarity.add_argument(
        "--task-description",
        default=DEFAULT_TASK_DESCRIPTION,
        help="Instruction used for query embedding.",
    )
    similarity.add_argument(
        "--output-dimension",
        type=int,
        default=None,
        help="Optional output dimension slice, e.g. 256.",
    )

    rerank = subparsers.add_parser(
        "rerank", help="Rerank documents by relevance to a query."
    )
    rerank.add_argument(
        "--query",
        required=True,
        help="Query text for reranking.",
    )
    rerank.add_argument(
        "--document",
        action="append",
        required=True,
        help="Document text. Repeat for multiple documents.",
    )
    rerank.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Return only top N results.",
    )

    return parser


def run_demo(engine: EmbeddingEngine, task_description: str) -> int:
    """运行内置相似度示例。

    使用一组预定义的查询和文档，分别编码后计算相似度矩阵。
    主要用于快速验证模型加载、推理、池化、相似度计算的完整链路是否正常。

    Args:
        engine: 已加载模型的编码引擎实例。
        task_description: 检索任务描述，用于 query 侧的 instruction 拼接。

    Returns:
        退出码，0 表示成功。
    """
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
    ]

    query_embeddings = engine.embed_queries(
        queries,
        task_description=task_description,
    )
    document_embeddings = engine.embed(documents)
    scores = engine.similarity_matrix(
        query_embeddings.embeddings,
        document_embeddings.embeddings,
    )

    print("Queries:")
    print(json.dumps(queries, ensure_ascii=False, indent=2))
    print("Documents:")
    print(json.dumps(documents, ensure_ascii=False, indent=2))
    print("Similarity:")
    print(json.dumps(scores.tolist(), ensure_ascii=False, indent=2))
    return 0


def run_embed(
    engine: EmbeddingEngine,
    texts: Sequence[str],
    output_dimension: int | None,
    show_shape: bool,
) -> int:
    """生成文本向量并输出到标准输出。

    Args:
        engine: 已加载模型的编码引擎实例。
        texts: 待编码文本序列。
        output_dimension: 输出维度截断值。为 None 时不截断。
        show_shape: 是否打印矩阵形状。

    Returns:
        退出码，0 表示成功。
    """
    result = engine.embed(texts, output_dimension=output_dimension)
    if show_shape:
        print(list(result.embeddings.shape))
    print(json.dumps(result.embeddings.tolist(), ensure_ascii=False))
    return 0


def run_similarity(
    engine: EmbeddingEngine,
    queries: Sequence[str],
    documents: Sequence[str],
    task_description: str,
    output_dimension: int | None,
) -> int:
    """计算查询与文档之间的相似度矩阵并输出。

    Args:
        engine: 已加载模型的编码引擎实例。
        queries: 查询文本序列。
        documents: 文档文本序列。
        task_description: 检索任务描述。
        output_dimension: 输出维度截断值。

    Returns:
        退出码，0 表示成功。
    """
    query_embeddings = engine.embed_queries(
        queries,
        task_description=task_description,
        output_dimension=output_dimension,
    )
    document_embeddings = engine.embed(
        documents,
        output_dimension=output_dimension,
    )
    scores = engine.similarity_matrix(
        query_embeddings.embeddings,
        document_embeddings.embeddings,
    )
    print(json.dumps(scores.tolist(), ensure_ascii=False, indent=2))
    return 0


def run_rerank(
    service: RerankerService,
    query: str,
    documents: Sequence[str],
    top_n: int | None,
) -> int:
    """对文档按查询相关性重排序并输出。

    将查询和文档送入 Cross-Encoder 模型，输出按相关性降序排列的结果。
    输出为 JSON 格式，包含模型名称和排序结果列表。

    Args:
        service: 重排序服务实例。
        query: 查询文本。
        documents: 文档文本序列。
        top_n: 仅返回前 N 个结果。

    Returns:
        退出码，0 表示成功。
    """
    request = RerankRequest(
        query=query,
        documents=list(documents),
        top_n=top_n,
    )
    response = service.rerank(request)
    output = {
        "model": response.model_name,
        "results": [
            {
                "index": r.index,
                "document": r.document,
                "relevance_score": r.relevance_score,
            }
            for r in response.results
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    """CLI 主入口函数。

    解析命令行参数，根据子命令分发到对应的处理函数。
    rerank 子命令使用独立的 RerankerService 和默认重排序模型；
    其余子命令共用 EmbeddingEngine。

    Returns:
        退出码：0 表示成功，2 表示参数错误。
    """
    parser = build_parser()
    args = parser.parse_args()

    # rerank 子命令使用独立的重排序引擎和配置。
    if args.command == "rerank":
        from .config import RerankerConfig

        config = RerankerConfig(
            default_model_id=args.model_id or DEFAULT_RERANKER_MODEL_ID,
            device=args.device,
            max_length=args.max_length or 1024,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
        )
        service = RerankerService(config)
        return run_rerank(service, args.query, args.document, args.top_n)

    # embedding 子命令使用向量引擎。
    engine = EmbeddingEngine(
        model_id=args.model_id or DEFAULT_MODEL_ID,
        device=args.device,
        max_length=args.max_length or 2048,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )

    if args.command == "demo":
        return run_demo(engine, args.task_description)
    if args.command == "embed":
        return run_embed(
            engine,
            args.text,
            args.output_dimension,
            args.show_shape,
        )
    if args.command == "similarity":
        return run_similarity(
            engine,
            args.query,
            args.document,
            args.task_description,
            args.output_dimension,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
