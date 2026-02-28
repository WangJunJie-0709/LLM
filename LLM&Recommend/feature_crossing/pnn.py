"""
特征交叉方式 3/5：PNN (Product-based Neural Network)。
内积 / 外积得到交叉矩阵，再经 MLP 或与原始嵌入 concat 后向下游传递。
"""

from typing import Optional, Literal
import torch
import torch.nn as nn


class PNNCross(nn.Module):
    """PNN 特征交叉：inner product / outer product 层 + 可选 MLP。"""

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        product_type: Literal["inner", "outer", "both"] = "inner",
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
        输出: [B, out_dim] 或交叉向量，供 DeepFM/DIN 等使用。
        """
        pass
