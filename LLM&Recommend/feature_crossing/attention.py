"""
特征交叉方式 2/5：Attention。
对多组嵌入做 Attention 加权聚合或交叉，用于 DIN 等序列兴趣建模。
"""

from typing import Optional
import torch
import torch.nn as nn


class AttentionCross(nn.Module):
    """Attention 特征交叉：query/key/value 形式加权聚合或交叉。"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        query: [B, len_q, dim], key/value: [B, len_k, dim]
        输出: [B, len_q, dim] 或 [B, dim]（聚合后），由调用方约定。
        """
        pass
