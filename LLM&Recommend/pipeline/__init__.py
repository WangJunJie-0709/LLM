"""
Pipeline 模块：Criteo 上的点击率预估全流程。
串联 data -> model -> train/eval，与具体模型和特征交叉解耦。
"""

from .ctr_pipeline import run_ctr_pipeline, train_one_epoch, evaluate

__all__ = ["run_ctr_pipeline", "train_one_epoch", "evaluate"]
