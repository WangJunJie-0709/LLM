"""
DeepFM：FM 一阶/二阶 + DNN，二阶部分即特征交叉。

这里实现一个经典形态的 DeepFM，并与 data.criteo 的 feature_spec 对齐：
- 一阶部分：dense 使用线性层，sparse 使用「维度为 1 的 Embedding」；
- 二阶部分（FM）：对稀疏字段 Embedding 做 pairwise 交叉（sum-of-squares 公式）；
- Deep 部分：把所有 sparse Embedding 展开后与 dense 拼接，送入 MLP。

最终 logits = linear_part + fm_second_order + deep_part。
"""

from typing import Any, Dict, Optional, Sequence, List

import torch
import torch.nn as nn

from .base import BaseCTRModel


class DeepFM(BaseCTRModel):
    """DeepFM：线性 + FM 二阶交叉 + 深度网络。"""

    def __init__(
        self,
        feature_spec: Dict[str, Any],
        embedding_dim: int = 16,
        hidden_dims: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_spec, **kwargs)

        n_sparse: int = int(feature_spec.get("n_sparse", 0))
        n_dense: int = int(feature_spec.get("n_dense", 0))
        vocab_sizes = feature_spec.get("sparse_vocab_sizes")
        if vocab_sizes is None or len(vocab_sizes) != n_sparse:
            raise ValueError(
                "DeepFM 需要在 feature_spec 中提供 'n_sparse' 与 'sparse_vocab_sizes'，且长度一致。"
            )

        # ---------- 一阶部分 ----------
        # dense：一个线性层即可表达所有数值特征的线性关系。
        self.linear_dense = nn.Linear(n_dense, 1) if n_dense > 0 else None

        # sparse：每个字段一个「标量 Embedding」，等价于对 one-hot 做线性层，
        # 但存储更省且能共享哈希空间。
        self.linear_sparse = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=int(v), embedding_dim=1)
                for v in vocab_sizes
            ]
        )

        # ---------- 二阶 FM 部分 ----------
        # 与 MF 一样，为每个稀疏字段建一个 D 维 Embedding，用于做 pairwise 内积。
        self.sparse_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=int(v), embedding_dim=embedding_dim)
                for v in vocab_sizes
            ]
        )

        # ---------- Deep 部分 ----------
        # Deep 输入由两部分组成：
        # - 所有字段的 Embedding 展开：F * D
        # - 原始 dense 特征：n_dense
        deep_input_dim = n_sparse * embedding_dim + n_dense
        if hidden_dims is None:
            hidden_dims = (256, 128, 64)

        mlp_layers: List[nn.Module] = []
        in_dim = deep_input_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, h))
            mlp_layers.append(nn.ReLU())
            # 工程实践中常见 dropout，这里可以按需调整
            mlp_layers.append(nn.Dropout(p=kwargs.get("dropout", 0.0)))
            in_dim = h
        # 最后一层输出标量，用于 deep_part logits
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.deep_mlp = nn.Sequential(*mlp_layers)

        self._last_user_embedding: Optional[torch.Tensor] = None

    def _embed_sparse_fields(self, sparse_ids: torch.Tensor) -> torch.Tensor:
        """
        与 MF 中的实现类似：把 [B, F] 的 id 张量映射为 [B, F, D] 的 Embedding。
        该表示既用于 FM 二阶交叉，也用于 Deep 部分。
        """
        if sparse_ids.dim() != 2:
            raise ValueError(f"sparse_ids 期望为 [batch, num_fields]，实际为 {tuple(sparse_ids.shape)}")

        batch_size, num_fields = sparse_ids.shape
        if num_fields != len(self.sparse_embeddings):
            raise ValueError(
                f"sparse_ids 的字段数 {num_fields} 与 Embedding 数量 {len(self.sparse_embeddings)} 不一致"
            )

        field_embs = []
        for i, emb in enumerate(self.sparse_embeddings):
            field_embs.append(emb(sparse_ids[:, i]))
        embeddings = torch.stack(field_embs, dim=1)  # [B, F, D]
        return embeddings

    def _first_order_term(
        self,
        sparse_ids: torch.Tensor,
        dense: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算一阶线性部分：
        - dense: linear_dense(dense)
        - sparse: 对每个字段用标量 Embedding 映射后按字段相加。
        """
        batch_size, num_fields = sparse_ids.shape
        if num_fields != len(self.linear_sparse):
            raise ValueError(
                f"sparse_ids 的字段数 {num_fields} 与 linear_sparse 数量 {len(self.linear_sparse)} 不一致"
            )

        # 稀疏一阶：逐字段 lookup 后相加，形状 [B, 1]
        sparse_terms = []
        for i, emb in enumerate(self.linear_sparse):
            sparse_terms.append(emb(sparse_ids[:, i]))  # [B, 1]
        sparse_linear = torch.stack(sparse_terms, dim=1).sum(dim=1)  # [B, 1]

        # dense 一阶：若存在 dense，则加上线性变换
        if self.linear_dense is not None and dense is not None:
            dense_linear = self.linear_dense(dense)  # [B, 1]
            first_order = sparse_linear + dense_linear
        else:
            first_order = sparse_linear

        # 展平成 [B]
        return first_order.squeeze(-1)

    def _second_order_fm(self, field_embeddings: torch.Tensor) -> torch.Tensor:
        """
        FM 二阶交叉项（经典公式）：
        0.5 * [ (sum(v_i))^2 - sum(v_i^2) ] 在 embedding 维度上再求和。

        参数：
        - field_embeddings: [B, F, D]

        返回：
        - fm_second_order: [B]
        """
        # [B, D]，先在字段维度上求和，再平方
        summed = field_embeddings.sum(dim=1)
        summed_square = summed * summed  # (sum v_i)^2

        # [B, D]，先逐字段平方，再在字段维度上求和
        squared = field_embeddings * field_embeddings
        squared_sum = squared.sum(dim=1)  # sum(v_i^2)

        # [B, D] -> [B]
        fm_term = 0.5 * (summed_square - squared_sum).sum(dim=1)
        return fm_term

    def forward(
        self,
        sparse_ids: Optional[torch.Tensor] = None,
        dense: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        DeepFM 前向：一阶 + FM 二阶 + Deep，输出 logits。

        参数：
        - sparse_ids: [B, F]，来自 CriteoCTRDataset 的 "sparse_ids"
        - dense: [B, D_dense]，来自 "dense"
        """
        if sparse_ids is None:
            raise ValueError("DeepFM 需要 sparse_ids 作为输入（shape = [batch, num_fields]）。")

        # ---------- Embedding ----------
        field_embeddings = self._embed_sparse_fields(sparse_ids)  # [B, F, D]

        # ---------- 一阶部分 ----------
        first_order = self._first_order_term(sparse_ids, dense)  # [B]

        # ---------- FM 二阶部分 ----------
        fm_second = self._second_order_fm(field_embeddings)  # [B]

        # ---------- Deep 部分 ----------
        # 把所有字段的 Embedding 展开，与 dense 特征拼接。
        B, F, D = field_embeddings.shape
        deep_input = field_embeddings.reshape(B, F * D)  # [B, F*D]
        if dense is not None:
            deep_input = torch.cat([deep_input, dense], dim=-1)  # [B, F*D + D_dense]

        deep_out = self.deep_mlp(deep_input).squeeze(-1)  # [B]

        logits = first_order + fm_second + deep_out

        # 为 T-SNE 准备一个「用户表示」，这里采用字段 Embedding 的平均。
        self._last_user_embedding = field_embeddings.mean(dim=1).detach()  # [B, D]

        return logits

    def get_user_embedding(self, **kwargs: Any) -> Optional[torch.Tensor]:
        """
        返回 DeepFM 中用于可视化的「用户嵌入」。

        这里简单取所有字段 Embedding 的平均，工程上可以根据业务定义更精细的 user-side 字段集合。
        """
        return self._last_user_embedding
