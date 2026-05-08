from __future__ import annotations

import torch
from torch import Tensor


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """最后有效 token 池化（last-token pooling）。

    从 Transformer 最后一层隐藏状态中，提取每个样本最后一个 **有效** token
    的向量作为整条文本的语义表示。这是 Qwen3-Embedding 官方推荐的池化方式。

    池化过程示意::

        输入文本 -> token 序列 -> 每个 token 一个向量 -> 取最后一个有效 token 向量
        例如：
          [中国] [的] [首都] [是] [北京]
            |      |      |     |     |
           [v1]   [v2]   [v3]  [v4]  [v5]
                                         |
                                         v
                                    选择 [v5] 作为整句向量

    该函数同时兼容左侧补齐（left padding）和右侧补齐（right padding）两种
    ``attention_mask`` 形态：
    - **左侧补齐**（本项目的默认模式）：所有有效 token 靠右排列，
      每个样本的最后一个位置就是最后一个有效 token，直接取 ``[:, -1]`` 即可。
    - **右侧补齐**：有效 token 靠左排列，需要根据 ``attention_mask`` 的和
      计算每个样本的实际序列长度，再逐行索引到对应位置。

    Args:
        last_hidden_states: 模型最后一层隐藏状态，形状 ``(batch_size, seq_len, hidden_dim)``。
        attention_mask: 注意力掩码，形状 ``(batch_size, seq_len)``。
            1 表示有效 token，0 表示 padding。

    Returns:
        池化后的向量张量，形状 ``(batch_size, hidden_dim)``。
        每行是对应文本的语义向量。
    """
    # 判断是否为左侧补齐：如果所有样本最后一个位置都是有效 token，
    # 说明做了左侧补齐，最后一个位置就是每个样本的最后一个有效 token。
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    # 右侧补齐的情况：计算每个样本的实际序列长度（有效 token 数 - 1），
    # 然后按行索引到每个样本的最后一个有效 token 位置。
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]
