"""
可视化模块：用户嵌入 T-SNE 等。
依赖 models 的 get_user_embedding 或等价接口，与 pipeline 解耦。
"""

from .tsne_embedding import plot_tsne_user_embedding, collect_user_embeddings

__all__ = ["plot_tsne_user_embedding", "collect_user_embeddings"]
