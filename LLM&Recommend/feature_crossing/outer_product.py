"""
特征交叉方式：外积。
对字段两两嵌入做外积，得到矩阵后展平或池化，再进 DNN。
"""

from typing import Optional
import torch
import torch.nn as nn


class OuterProductCross(nn.Module):
    """外积特征交叉：pairwise 外积，展平或池化后输出向量。"""

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
        输出: [B, out_dim] 外积展平/池化后的向量。
        """
        pass
