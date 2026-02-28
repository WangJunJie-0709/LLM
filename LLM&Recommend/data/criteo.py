"""
Criteo 数据集：加载、预处理、构建 DataLoader。
用于点击率预估 pipeline，输出与 MF/DeepFM/DIN 等模型接口一致。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Sequence, Union, Iterable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

"""
Criteo 数据集的特征数量
"""
CRITEO_NUM_DENSE = 13  # 稠密特征数量
CRITEO_NUM_SPARSE = 26  # 稀疏特征数量


@dataclass(frozen=True)
class CriteoMeta:
    """Criteo 标准列信息（Kaggle Display Ads 版本）。"""

    label_col: str = "label"
    dense_cols: Tuple[str, ...] = tuple(f"I{i}" for i in range(1, CRITEO_NUM_DENSE + 1))
    sparse_cols: Tuple[str, ...] = tuple(f"C{i}" for i in range(1, CRITEO_NUM_SPARSE + 1))
    delimiter: str = "\t"


class CriteoCTRDataset(Dataset):
    """
    CTR 任务 Dataset。
    每条样本返回 dict，键名固定，便于模型解耦消费：
    - label: float32 标量
    - dense: float32 [13]
    - sparse_ids: int64 [26]
    - behavior_ids: int64 [seq_len]（为 DIN 预留；在 Criteo 上用部分 sparse 字段拼成伪序列）
    """

    def __init__(
        self,
        label: torch.Tensor,
        dense: torch.Tensor,
        sparse_ids: torch.Tensor,
        behavior_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self.label = label
        self.dense = dense
        self.sparse_ids = sparse_ids
        self.behavior_ids = behavior_ids

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "label": self.label[idx],
            "dense": self.dense[idx],
            "sparse_ids": self.sparse_ids[idx],
        }
        if self.behavior_ids is not None:
            item["behavior_ids"] = self.behavior_ids[idx]
        return item


def _stable_hash_to_bucket(text: str, num_buckets: int) -> int:
    """
    稳定哈希到 [1, num_buckets)。
    0 预留给缺失/unknown。
    """
    if not text:
        return 0
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return (int(digest, 16) % (num_buckets - 1)) + 1


def _cache_key(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _read_criteo_tsv(
    path: str,
    *,
    delimiter: str = "\t",
    has_label: bool = True,
    max_rows: Optional[int] = None,
) -> List[Tuple[int, List[str], List[str]]]:
    """
    读取 Criteo TSV（逐行解析，避免强依赖 pandas）。
    返回 list[(label, dense_strs[13], sparse_strs[26])]。
    """
    rows: List[Tuple[int, List[str], List[str]]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(delimiter)
            if has_label:
                if len(parts) < 1 + CRITEO_NUM_DENSE + CRITEO_NUM_SPARSE:
                    continue
                label = 0 if parts[0] in ("", "NA", "null", "None") else int(parts[0])
                dense = parts[1 : 1 + CRITEO_NUM_DENSE]
                sparse = parts[1 + CRITEO_NUM_DENSE : 1 + CRITEO_NUM_DENSE + CRITEO_NUM_SPARSE]
            else:
                # 无 label 的测试集：用 0 占位
                if len(parts) < CRITEO_NUM_DENSE + CRITEO_NUM_SPARSE:
                    continue
                label = 0
                dense = parts[0:CRITEO_NUM_DENSE]
                sparse = parts[CRITEO_NUM_DENSE : CRITEO_NUM_DENSE + CRITEO_NUM_SPARSE]
            rows.append((label, dense, sparse))
    return rows


def load_criteo(
    train_path: str,
    test_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """
    从路径加载 Criteo 原始数据（或从缓存加载）。
    返回 (train_data, test_data, meta_info)。
    meta_info 包含：特征数、字段名、稀疏/稠密划分等，供后续预处理与模型使用。
    """
    meta = CriteoMeta()
    max_rows: Optional[int] = kwargs.get("max_rows")

    cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)

    def load_one(path: str, *, has_label: bool, tag: str) -> Any:
        if not cache_root:
            return _read_criteo_tsv(path, delimiter=meta.delimiter, has_label=has_label, max_rows=max_rows)

        p = Path(path).expanduser().resolve()
        key = _cache_key(tag, str(p), str(p.stat().st_mtime_ns), str(max_rows))
        cache_file = cache_root / f"raw_{tag}_{key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        data = _read_criteo_tsv(str(p), delimiter=meta.delimiter, has_label=has_label, max_rows=max_rows)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return data

    train_data = load_one(train_path, has_label=True, tag="train")
    test_data = load_one(test_path, has_label=True, tag="test") if test_path else None

    meta_info: Dict[str, Any] = {
        "label_col": meta.label_col,
        "dense_cols": list(meta.dense_cols),
        "sparse_cols": list(meta.sparse_cols),
        "n_dense": CRITEO_NUM_DENSE,
        "n_sparse": CRITEO_NUM_SPARSE,
        "delimiter": meta.delimiter,
    }
    return train_data, test_data, meta_info


def preprocess_criteo(
    train_data: Any,
    test_data: Optional[Any] = None,
    meta_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any]]:
    """
    对 Criteo 做数值化、归一化、缺失填充等，得到 PyTorch Dataset。
    返回 (train_dataset, val_dataset, test_dataset, feature_spec)。
    feature_spec 供模型构建 embedding 与特征交叉使用。
    """
    seed: int = int(kwargs.get("seed", 42))
    split_ratio: Tuple[float, float, float] = tuple(kwargs.get("split_ratio", (0.8, 0.1, 0.1)))  # type: ignore[assignment]
    if not math.isclose(sum(split_ratio), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"split_ratio 必须和为 1.0，当前为 {split_ratio}")

    cache_dir: Optional[str] = kwargs.get("cache_dir")
    force_reprocess: bool = bool(kwargs.get("force_reprocess", False))
    max_rows: Optional[int] = kwargs.get("max_rows")  # 参与缓存 key，保持一致

    # categorical hashing 配置：每个 field 一个 bucket（推荐工程常用做法）
    num_buckets_per_field: Union[int, Sequence[int]] = kwargs.get("num_buckets_per_field", 2**18)
    if isinstance(num_buckets_per_field, int):
        sparse_vocab_sizes = [int(num_buckets_per_field)] * CRITEO_NUM_SPARSE
    else:
        sparse_vocab_sizes = [int(x) for x in num_buckets_per_field]
        if len(sparse_vocab_sizes) != CRITEO_NUM_SPARSE:
            raise ValueError("num_buckets_per_field 长度必须为 26（对应 C1..C26）")

    # 为 DIN 生成伪行为序列：默认取最后 5 个 sparse 字段（每条样本固定长度 K）
    behavior_field_indices: Sequence[int] = kwargs.get("behavior_field_indices", (21, 22, 23, 24, 25))
    seq_len = len(behavior_field_indices)

    # dense 归一化：先 log1p，再标准化（可关闭）
    normalize_dense: Literal["log1p+zscore", "log1p", "none"] = kwargs.get("normalize_dense", "log1p+zscore")

    cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)

        key = _cache_key(
            "processed",
            str(len(train_data)),
            str(len(test_data) if test_data is not None else 0),
            json.dumps(split_ratio),
            str(seed),
            str(max_rows),
            json.dumps(sparse_vocab_sizes),
            json.dumps(list(behavior_field_indices)),
            normalize_dense,
        )
        cache_file = cache_root / f"processed_{key}.pt"
        if cache_file.exists() and not force_reprocess:
            obj = torch.load(str(cache_file), map_location="cpu")
            feature_spec = obj["feature_spec"]
            splits = obj["splits"]
            train_ds = CriteoCTRDataset(**splits["train"])
            val_ds = CriteoCTRDataset(**splits["val"])
            test_ds = CriteoCTRDataset(**splits["test"])
            return train_ds, val_ds, test_ds, feature_spec

    # ---- 1) 原始 -> numpy ----
    def to_arrays(rows: List[Tuple[int, List[str], List[str]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(rows)
        y = np.zeros((n,), dtype=np.float32)
        dense = np.zeros((n, CRITEO_NUM_DENSE), dtype=np.float32)
        sparse = np.zeros((n, CRITEO_NUM_SPARSE), dtype=np.int64)

        for i, (label, dense_strs, sparse_strs) in enumerate(rows):
            y[i] = float(label)

            # dense：缺失 -> 0；脏值 -> 0
            for j, s in enumerate(dense_strs):
                if not s or s in ("NA", "null", "None"):
                    v = 0.0
                else:
                    try:
                        v = float(s)
                        if not math.isfinite(v):
                            v = 0.0
                    except Exception:
                        v = 0.0
                dense[i, j] = v

            # sparse：缺失 -> 0；其余用稳定哈希到 bucket
            for j, s in enumerate(sparse_strs):
                if not s or s in ("NA", "null", "None"):
                    sparse[i, j] = 0
                else:
                    # field-aware hashing，避免同值跨字段冲突
                    sparse[i, j] = _stable_hash_to_bucket(f"F{j}:{s}", sparse_vocab_sizes[j])

        return y, dense, sparse

    y, dense_x, sparse_x = to_arrays(train_data)

    # ---- 2) dense 规范化 ----
    if normalize_dense in ("log1p+zscore", "log1p"):
        dense_x = np.log1p(np.clip(dense_x, a_min=0.0, a_max=None))

    dense_mean = dense_x.mean(axis=0, keepdims=True).astype(np.float32)
    dense_std = dense_x.std(axis=0, keepdims=True).astype(np.float32)
    dense_std[dense_std < 1e-6] = 1.0

    if normalize_dense == "log1p+zscore":
        dense_x = (dense_x - dense_mean) / dense_std

    # ---- 3) 构造伪序列特征（DIN 输入） ----
    behavior_x = sparse_x[:, list(behavior_field_indices)].astype(np.int64) if seq_len > 0 else None

    # ---- 4) 8:1:1 切分 train/val/test（始终基于 train_data 做切分，满足你的要求） ----
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_train = int(len(y) * split_ratio[0])
    n_val = int(len(y) * split_ratio[1])
    n_test = len(y) - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    assert len(test_idx) == n_test

    def slice_to_tensors(sel: np.ndarray) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            "label": torch.from_numpy(y[sel]).float(),
            "dense": torch.from_numpy(dense_x[sel]).float(),
            "sparse_ids": torch.from_numpy(sparse_x[sel]).long(),
        }
        if behavior_x is not None:
            out["behavior_ids"] = torch.from_numpy(behavior_x[sel]).long()
        return out

    train_t = slice_to_tensors(train_idx)
    val_t = slice_to_tensors(val_idx)
    test_t = slice_to_tensors(test_idx)

    train_ds = CriteoCTRDataset(**train_t)
    val_ds = CriteoCTRDataset(**val_t)
    test_ds = CriteoCTRDataset(**test_t)

    feature_spec: Dict[str, Any] = {
        "n_dense": CRITEO_NUM_DENSE,
        "n_sparse": CRITEO_NUM_SPARSE,
        "dense_cols": meta_info.get("dense_cols") if meta_info else [f"I{i}" for i in range(1, CRITEO_NUM_DENSE + 1)],
        "sparse_cols": meta_info.get("sparse_cols") if meta_info else [f"C{i}" for i in range(1, CRITEO_NUM_SPARSE + 1)],
        "sparse_vocab_sizes": sparse_vocab_sizes,
        "behavior_field_indices": list(behavior_field_indices),
        "behavior_seq_len": seq_len,
        "dense_normalize": normalize_dense,
        "dense_mean": dense_mean.squeeze(0).tolist(),
        "dense_std": dense_std.squeeze(0).tolist(),
        "split_ratio": list(split_ratio),
        "seed": seed,
    }

    if cache_root:
        torch.save(
            {
                "feature_spec": feature_spec,
                "splits": {"train": train_t, "val": val_t, "test": test_t},
            },
            str(cache_file),
        )

    return train_ds, val_ds, test_ds, feature_spec


def get_criteo_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 4096,
    num_workers: int = 0,
    **kwargs: Any,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    根据 train/val/test Dataset 构建 DataLoader。
    返回 (train_loader, val_loader, test_loader)。
    """
    pin_memory: bool = bool(kwargs.get("pin_memory", False))
    drop_last: bool = bool(kwargs.get("drop_last", False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        if test_dataset is not None
        else None
    )
    return train_loader, val_loader, test_loader
