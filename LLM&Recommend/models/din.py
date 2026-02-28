"""
DIN (Deep Interest Network)：基于 Attention 的兴趣建模。
用户行为序列与候选 item 做 Attention 特征交叉，再与 DNN 结合。
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from .base import BaseCTRModel


class DIN(BaseCTRModel):
    """DIN：行为序列 + 候选 item，Attention 加权聚合后进 DNN。"""

    def __init__(
        self,
        feature_spec: Dict[str, Any],
        embedding_dim: int = 16,
        attention_units: Optional[tuple] = None,
        hidden_dims: Optional[tuple] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_spec, **kwargs)

    def forward(
        self,
        sparse_ids: Optional[torch.Tensor] = None,
        dense: Optional[torch.Tensor] = None,
        behavior_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """DIN 前向：behavior_ids 为序列，与候选做 Attention 后输出 logits。"""
        pass

    def get_user_embedding(self, **kwargs: Any) -> Optional[torch.Tensor]:
        """返回 Attention 聚合后的用户兴趣向量，供 T-SNE。"""
        pass
