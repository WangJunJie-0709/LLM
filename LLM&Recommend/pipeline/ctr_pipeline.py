"""
点击率预估 Pipeline：在 Criteo 上训练 / 评估 / 预测。
可切换 MF、DeepFM、DIN，并复用 5 种特征交叉（由模型内部或外部注入）。
"""

from typing import Any, Dict, Optional, Union
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# 本模块依赖 data / models，通过参数传入或从 config 读取，保持解耦


def run_ctr_pipeline(
    model_name: str,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    feature_spec: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    完整 pipeline：根据 model_name 构建 MF/DeepFM/DIN，
    训练若干 epoch，在 test 上评估，返回 metrics 与训练好的 model（或路径）。
    """
    pass


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: Union[str, torch.device],
    **kwargs: Any,
) -> float:
    """训练一个 epoch，返回平均 loss。"""
    pass


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: Union[str, torch.device],
    **kwargs: Any,
) -> Dict[str, float]:
    """评估：返回 AUC、LogLoss 等指标。"""
    pass
