"""
数据模块：Criteo 数据集加载与预处理。
供 pipeline 与 models 调用，与具体模型解耦。
"""

from .criteo import load_criteo, preprocess_criteo, get_criteo_dataloaders

__all__ = ["load_criteo", "preprocess_criteo", "get_criteo_dataloaders"]
