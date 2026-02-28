"""
Matrix Factorization：在 Criteo 这类多域场景下做一个「广义 MF」基线。

设计约定（与 data.criteo 中的 feature_spec 对齐）：
- 每个稀疏字段各自一个 Embedding，维度相同；
- 一条样本的「用户表示」= 所有稀疏字段嵌入的平均向量；
- 使用一个线性层把用户表示映射到标量 logits，必要时再叠加 dense 特征的线性部分。

这样既保留了 MF「低秩嵌入 + 线性打分」的思想，又能与 DeepFM / DIN 共用同一份 feature_spec。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseCTRModel


class MF(BaseCTRModel):
    """
    MF 模型：基于多域稀疏特征的广义 Matrix Factorization。

    关键实现点（配合详细注释便于理解与改造）：
    - 使用 feature_spec["sparse_vocab_sizes"] 构建「字段级」Embedding 列表；
    - forward 时把 sparse_ids 中的每一列喂入对应 Embedding；
    - 对所有字段嵌入做 mean pooling 得到「用户嵌入」；
    - 通过一层线性变换映射到 logits，同时可选叠加 dense 特征的线性打分。
    """

    def __init__(self, feature_spec: Dict[str, Any], embedding_dim: int = 16, **kwargs: Any) -> None:
        super().__init__(feature_spec, **kwargs)

        n_sparse: int = int(feature_spec.get("n_sparse", 0))
        n_dense: int = int(feature_spec.get("n_dense", 0))
        vocab_sizes = feature_spec.get("sparse_vocab_sizes")
        if vocab_sizes is None or len(vocab_sizes) != n_sparse:
            raise ValueError(
                "MF 需要在 feature_spec 中提供 'n_sparse' 与 'sparse_vocab_sizes'，且长度一致。"
            )

        # 为每个稀疏字段单独建一个 Embedding。
        # 这样在需要做 field-aware 操作（例如仅挑选其中若干字段构成用户行为）时更灵活。
        self.sparse_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=int(v), embedding_dim=embedding_dim)
                for v in vocab_sizes
            ]
        )

        # 用于把「用户嵌入」映射到 logits 的线性层。
        self.user_proj = nn.Linear(embedding_dim, 1)

        # 可选：对 dense 特征做一个线性打分再加到 logits 上。
        self.use_dense = n_dense > 0
        self.dense_linear = nn.Linear(n_dense, 1) if self.use_dense else None

        # 额外的偏置项，便于统一调节整体 CTR 水平。
        self.bias = nn.Parameter(torch.zeros(1))

        self._last_user_embedding: Optional[torch.Tensor] = None

    def _embed_sparse_fields(self, sparse_ids: torch.Tensor) -> torch.Tensor:
        """
        把 shape = [B, F] 的 sparse_ids 映射到字段级嵌入。

        返回：
        - embeddings: [B, F, D]
        """
        if sparse_ids.dim() != 2:
            raise ValueError(f"sparse_ids 期望为 [batch, num_fields]，实际为 {tuple(sparse_ids.shape)}")

        batch_size, num_fields = sparse_ids.shape
        if num_fields != len(self.sparse_embeddings):
            raise ValueError(
                f"sparse_ids 的字段数 {num_fields} 与 Embedding 数量 {len(self.sparse_embeddings)} 不一致"
            )

        # 对每个字段 i，取出对应的一列 id，送入第 i 个 Embedding，再在字段维度上 stack。
        field_embs = []
        for i, emb in enumerate(self.sparse_embeddings):
            # 每列形状：[B] -> [B, D]
            field_embs.append(emb(sparse_ids[:, i]))
        # [F, B, D] -> [B, F, D]
        embeddings = torch.stack(field_embs, dim=1)
        return embeddings

    def forward(
        self,
        sparse_ids: Optional[torch.Tensor] = None,
        dense: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        MF 前向：把 batch 内每个样本的多字段稀疏特征压缩成一个低维「用户向量」，并输出 logits。

        参数：
        - sparse_ids: [B, F]，来自 CriteoCTRDataset 中的 "sparse_ids"
        - dense: [B, D_dense]，可选，来自 "dense"

        返回：
        - logits: [B]，尚未过 sigmoid 的 CTR 预测值。
        """
        if sparse_ids is None:
            raise ValueError("MF 需要 sparse_ids 作为输入（shape = [batch, num_fields]）。")

        # 1) 字段级 Embedding：得到 [B, F, D]
        field_embeddings = self._embed_sparse_fields(sparse_ids)

        # 2) mean pooling 得到「用户嵌入」：[B, D]
        # 这里选择简单平均，工程上你可以换成加权平均或更复杂的聚合策略。
        user_embedding = field_embeddings.mean(dim=1)

        # 保留一份用户嵌入用于 T-SNE 可视化。
        self._last_user_embedding = user_embedding.detach()

        # 3) 用户嵌入 -> logits（MF 主干）
        logits = self.user_proj(user_embedding).squeeze(-1)  # [B]

        # 4) 可选叠加 dense 线性部分：相当于给 MF 加一个简单的 wide 分支。
        if self.use_dense and dense is not None:
            logits = logits + self.dense_linear(dense).squeeze(-1)

        logits = logits + self.bias
        return logits

    def get_user_embedding(self, **kwargs: Any) -> Optional[torch.Tensor]:
        """
        返回最近一次 forward 中的用户嵌入，供 T-SNE 使用。

        说明：
        - visualization.tsne_embedding.collect_user_embeddings 会在 eval 模式下调用模型，
          因此这里只需返回一个 [B, D] 的 tensor 即可；
        - 这里返回的是 detach 后的张量，以避免构建无用的梯度图。
        """
        return self._last_user_embedding
