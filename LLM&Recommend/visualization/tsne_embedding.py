"""
T-SNE 可视化用户嵌入空间。
从已训练模型 + DataLoader 收集用户嵌入，降维后绘图并保存。
"""

from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
import numpy as np


def collect_user_embeddings(
    model: Any,
    data_loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    遍历 data_loader，调用 model.get_user_embedding（或等价接口）收集用户嵌入。
    返回 [N, embedding_dim] 的 numpy 数组；若模型无用户嵌入则抛错或返回空。
    """
    pass


def plot_tsne_user_embedding(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    **kwargs: Any,
) -> None:
    """
    对 embeddings 做 T-SNE 降维到 2D，绘图（可选 labels 着色），保存到 output_path。
    """
    pass
