"""
特征交叉方式 4/5：内积。
对字段两两嵌入做内积，得到标量交叉（FM 二阶形式），可 sum 或 concat 后进 DNN。
"""

from typing import Optional
import torch
import torch.nn as nn


class InnerProductCross(nn.Module):
    """内积特征交叉：pairwise 内积，输出标量和或向量。"""

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        output_style: Optional[str] = None,
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
        输出: [B, 1]（sum of pairs）或 [B, num_pairs] 等。
        """
        pass
