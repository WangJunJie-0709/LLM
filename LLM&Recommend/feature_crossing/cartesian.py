"""
特征交叉方式 1/5：笛卡尔积。
对离散特征做 one-hot 后的笛卡尔积（或等价嵌入空间笛卡尔积），得到显式交叉特征。
"""

from typing import Optional
import torch
import torch.nn as nn


class CartesianProductCross(nn.Module):
    """笛卡尔积特征交叉：将多域嵌入或 one-hot 展开为笛卡尔积再映射。"""

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        out_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        embeddings: [B, num_fields, embedding_dim]
        输出: [B, out_dim] 或 [B, num_fields*(num_fields-1)//2, embedding_dim] 等，由实现约定。
        """
        pass
