"""
第1-2周实战项目入口：Criteo 点击率预估 + T-SNE 用户嵌入可视化。
串联 data -> models (MF/DeepFM/DIN) -> pipeline -> visualization，各模块解耦调用。
"""

import argparse
from typing import Optional

# 以下仅做流程编排，不写具体实现

def parse_args():
    """解析命令行：数据路径、模型名、是否跑 T-SNE 等。"""
    p = argparse.ArgumentParser(description="Criteo CTR Pipeline (MF/DeepFM/DIN) + T-SNE")
    p.add_argument("--model_name", type=str, default="deepfm", choices=["mf", "deepfm", "din"])
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--test_path", type=str, default=None)
    p.add_argument("--run_tsne", action="store_true", default=True)
    return p.parse_args()


def main(
    model_name: str = "deepfm",
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    run_tsne: bool = True,
    **kwargs,
) -> None:
    """
    1. 使用 data 模块加载并预处理 Criteo，得到 train_loader / test_loader、feature_spec
    2. 使用 pipeline 模块根据 model_name 构建 MF/DeepFM/DIN，训练并评估
    3. 若 run_tsne 为 True，用 visualization 模块收集用户嵌入并画 T-SNE 图
    """
    pass


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
