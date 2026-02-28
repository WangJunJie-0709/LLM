"""
CTR 模型基类：统一 forward 接口与 feature_spec 约定。
MF/DeepFM/DIN 继承此类，便于 pipeline 解耦调用。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class BaseCTRModel(nn.Module, ABC):
    """点击率预估模型基类。子类需实现 forward，输入与 DataLoader 产出格式一致。"""

    def __init__(self, feature_spec: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__()
        self.feature_spec = feature_spec

    @abstractmethod
    def forward(
        self,
        sparse_ids: Optional[torch.Tensor] = None,
        dense: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        前向传播，输出 logits（未 sigmoid）。
        具体键名由 data.criteo 的 feature_spec 与 DataLoader 约定。
        """
        pass

    def get_user_embedding(self, **kwargs: Any) -> Optional[torch.Tensor]:
        """
        若模型有用户侧嵌入（如 MF/DIN），返回用于 T-SNE 可视化的用户嵌入矩阵；
        否则返回 None，由 visualization 模块处理。
        """
        return None
