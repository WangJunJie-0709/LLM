"""
Matrix Factorization：双塔用户/物品嵌入，内积得到 logits。
可与 feature_crossing 的内积方式配合或独立使用。
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from .base import BaseCTRModel


class MF(BaseCTRModel):
    """MF 模型：用户嵌入与物品/特征嵌入内积做 CTR 预测。"""

    def __init__(self, feature_spec: Dict[str, Any], embedding_dim: int = 16, **kwargs: Any) -> None:
        super().__init__(feature_spec, **kwargs)

    def forward(
        self,
        sparse_ids: Optional[torch.Tensor] = None,
        dense: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """MF 前向：根据 sparse_ids 取嵌入并做内积，输出 logits。"""
        pass

    def get_user_embedding(self, **kwargs: Any) -> Optional[torch.Tensor]:
        """返回当前 batch 的用户嵌入，供 T-SNE 使用。"""
        pass
