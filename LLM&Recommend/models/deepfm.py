"""
DeepFM：FM 一阶/二阶 + DNN，二阶部分即特征交叉。
可选用 feature_crossing 中的实现（如内积/外积）或内置 FM 层。
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from .base import BaseCTRModel


class DeepFM(BaseCTRModel):
    """DeepFM：线性 + FM 二阶交叉 + 深度网络。"""

    def __init__(
        self,
        feature_spec: Dict[str, Any],
        embedding_dim: int = 16,
        hidden_dims: Optional[tuple] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_spec, **kwargs)

    def forward(
        self,
        sparse_ids: Optional[torch.Tensor] = None,
        dense: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """DeepFM 前向：一阶 + FM 二阶 + DNN，输出 logits。"""
        pass

    def get_user_embedding(self, **kwargs: Any) -> Optional[torch.Tensor]:
        """可选：对用户侧特征做 pooling 作为用户表示。"""
        pass
