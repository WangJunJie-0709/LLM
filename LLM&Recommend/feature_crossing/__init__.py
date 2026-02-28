"""
特征交叉模块：5 种工程实现（笛卡尔积 / Attention / PNN / 内积 / 外积）。
各模块接收嵌入或特征张量，输出交叉后的表示，供 MF/DeepFM/DIN 等模型调用。
"""

from .cartesian import CartesianProductCross
from .attention import AttentionCross
from .pnn import PNNCross
from .inner_product import InnerProductCross
from .outer_product import OuterProductCross

__all__ = [
    "CartesianProductCross",
    "AttentionCross",
    "PNNCross",
    "InnerProductCross",
    "OuterProductCross",
]
