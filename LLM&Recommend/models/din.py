"""
DIN (Deep Interest Network)：基于 Attention 的兴趣建模。

在本实现中，我们与 data.criteo 的约定如下：
- CriteoCTRDataset 提供：
  - sparse_ids: [B, F]，一条样本的所有稀疏字段；
  - behavior_ids: [B, L]，由若干字段（feature_spec["behavior_field_indices"]）拼出的伪行为序列；
  - dense: [B, D_dense]，数值特征。
- DIN 使用：
  - behavior_ids 作为用户「历史行为」序列；
  - sparse_ids 中的第一个字段作为「候选 item id」（可以按需调整）。

核心思想：用候选 item 对历史行为做 Attention，加权聚合得到兴趣向量，再与候选 + dense 一起送入 DNN。
"""

from typing import Any, Dict, Optional, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseCTRModel


class DIN(BaseCTRModel):
    """DIN：行为序列 + 候选 item，Attention 加权聚合后进 DNN。"""

    def __init__(
        self,
        feature_spec: Dict[str, Any],
        embedding_dim: int = 16,
        attention_units: Optional[Sequence[int]] = None,
        hidden_dims: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_spec, **kwargs)

        n_sparse: int = int(feature_spec.get("n_sparse", 0))
        n_dense: int = int(feature_spec.get("n_dense", 0))
        vocab_sizes = feature_spec.get("sparse_vocab_sizes")
        if vocab_sizes is None or len(vocab_sizes) != n_sparse:
            raise ValueError(
                "DIN 需要在 feature_spec 中提供 'n_sparse' 与 'sparse_vocab_sizes'，且长度一致。"
            )

        # 为行为序列与候选 item 构建一个共享的「item 级」Embedding 表。
        # 为简单起见，这里使用最大 vocab_size 作为 Embedding 大小，
        # 假定所有 id 都已经哈希到 [0, max_vocab)。
        max_vocab = max(int(v) for v in vocab_sizes)
        self.item_embedding = nn.Embedding(num_embeddings=max_vocab, embedding_dim=embedding_dim)

        # 可选：对剩余稀疏字段（非行为字段）做额外建模，这里先简化为不显式使用。

        # dense 特征将在最终 DNN 中与兴趣向量 / 候选向量拼接。
        self.n_dense = n_dense

        # ---------- Attention 网络 ----------
        # 典型 DIN Attention 会用 [beh, target, beh-target, beh*target] 作为输入，
        # 这里实现一个轻量版：使用点积再加一层可学习的 MLP 做打分。
        if attention_units is None:
            attention_units = (64, 32)

        att_layers: List[nn.Module] = []
        att_input_dim = embedding_dim * 4  # beh, target, beh-target, beh*target
        in_dim = att_input_dim
        for h in attention_units:
            att_layers.append(nn.Linear(in_dim, h))
            att_layers.append(nn.ReLU())
            in_dim = h
        att_layers.append(nn.Linear(in_dim, 1))  # 输出标量打分
        self.att_mlp = nn.Sequential(*att_layers)

        # ---------- 主干 DNN ----------
        if hidden_dims is None:
            hidden_dims = (256, 128, 64)

        # DIN 的 DNN 输入 = [兴趣向量, 候选 item 向量, dense 特征]
        dnn_input_dim = embedding_dim * 2 + n_dense
        dnn_layers: List[nn.Module] = []
        in_dim = dnn_input_dim
        for h in hidden_dims:
            dnn_layers.append(nn.Linear(in_dim, h))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(p=kwargs.get("dropout", 0.0)))
            in_dim = h
        dnn_layers.append(nn.Linear(in_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)

        self._last_user_embedding: Optional[torch.Tensor] = None

    def _compute_attention(
        self,
        behavior_emb: torch.Tensor,
        target_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算行为序列对候选 item 的 Attention 权重，并返回聚合后的兴趣向量。

        参数：
        - behavior_emb: [B, L, D]
        - target_emb: [B, D]

        返回：
        - user_interest: [B, D]
        """
        B, L, D = behavior_emb.shape

        # 将 target_emb broadcast 为 [B, L, D]
        target_expanded = target_emb.unsqueeze(1).expand(B, L, D)

        # 构造 DIN 风格的注意力输入：[beh, target, beh-target, beh*target]
        att_input = torch.cat(
            [
                behavior_emb,
                target_expanded,
                behavior_emb - target_expanded,
                behavior_emb * target_expanded,
            ],
            dim=-1,
        )  # [B, L, 4D]

        # 经过 MLP 得到每个行为的打分，再 softmax 得到权重
        scores = self.att_mlp(att_input).squeeze(-1)  # [B, L]
        att_weights = F.softmax(scores, dim=-1)  # [B, L]

        # 加权求和得到兴趣向量 [B, D]
        user_interest = torch.bmm(att_weights.unsqueeze(1), behavior_emb).squeeze(1)
        return user_interest

    def forward(
        self,
        sparse_ids: Optional[torch.Tensor] = None,
        dense: Optional[torch.Tensor] = None,
        behavior_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        DIN 前向：behavior_ids 为序列，与候选 item 做 Attention 后输出 logits。

        参数：
        - sparse_ids: [B, F]，其中第 0 列视为候选 item id（可根据业务调整）；
        - behavior_ids: [B, L]，用户行为序列；
        - dense: [B, D_dense]，数值特征。
        """
        if sparse_ids is None:
            raise ValueError("DIN 需要 sparse_ids（至少包含一个候选 item 字段）。")
        if behavior_ids is None:
            raise ValueError("DIN 需要 behavior_ids 作为用户行为序列（shape = [batch, seq_len]）。")

        # 候选 item id：简单取 sparse_ids 的第一个字段，
        # 工程上可以在 feature_spec 中额外显式标记。
        target_ids = sparse_ids[:, 0]  # [B]

        # 行为 & 候选 Embedding
        behavior_emb = self.item_embedding(behavior_ids)  # [B, L, D]
        target_emb = self.item_embedding(target_ids)  # [B, D]

        # 基于候选 item 的注意力加权聚合行为序列 -> 用户兴趣向量
        user_interest = self._compute_attention(behavior_emb, target_emb)  # [B, D]

        # 保留一份兴趣向量用于 T-SNE。
        self._last_user_embedding = user_interest.detach()

        # DNN 输入：兴趣向量 + 候选向量 + dense
        if dense is not None:
            dnn_input = torch.cat([user_interest, target_emb, dense], dim=-1)
        else:
            dnn_input = torch.cat([user_interest, target_emb], dim=-1)

        logits = self.dnn(dnn_input).squeeze(-1)  # [B]
        return logits

    def get_user_embedding(self, **kwargs: Any) -> Optional[torch.Tensor]:
        """
        返回 Attention 聚合后的用户兴趣向量，供 T-SNE 可视化使用。

        注意：该向量基于「行为序列 + 候选 item」的注意力权重，能更好反映当前推荐场景下的兴趣状态。
        """
        return self._last_user_embedding
