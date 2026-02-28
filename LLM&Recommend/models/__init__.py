"""
模型模块：MF、DeepFM、DIN 等 CTR 模型。
统一通过 base 接口与 pipeline / feature_crossing 解耦调用。
"""

from .base import BaseCTRModel
from .mf import MF
from .deepfm import DeepFM
from .din import DIN

__all__ = ["BaseCTRModel", "MF", "DeepFM", "DIN"]
