# train_gate_fusion.py
# -*- coding: utf-8 -*-
"""
门控融合训练脚本（Mixture-of-Experts Gate Fusion, TRAIN ONLY）

你这次报错核心原因：
- 你的 ML bundle 里使用了 XGBoost 的 sklearn wrapper，来自旧版本序列化；
- 新版本 xgboost/sklearn 在 get_params / clone 时会访问 feature_weights，
  旧对象没有这个属性 -> AttributeError

本脚本做了两件事保证能跑：
1) 自动给 XGBModel 补齐 feature_weights=None（避免 clone/get_params 报错）
2) 生成 estimator 时：优先 sklearn.clone，失败则 deepcopy，再失败就直接复用对象 fit（fit 会覆盖 booster）

同时，为了避免你当前 dataset.py 的推理 bug，本脚本 **不再依赖 dataset.py**，
而是内置一个 YarnDLInferDataset 来做 DL OOF 推理（依赖 one_hot_map.pkl + model_i.py）。

产物：
- out_dir/gate_model.pkl
- out_dir/ml_refit_bundle.pkl
- out_dir/train_oof_predictions.csv

运行示例（Windows CMD 一行）：
python train_gate_fusion.py --train_csv .\without\single_train_data_without.csv --ml_bundle .\pkl\yarn_strength_model_gpu.pkl --dl_ckpt_dir .\cv_models --out_dir .\gate_fusion_out --one_hot_map ..\one_hot_map.pkl
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "纱强力"
BATCH_COL = "纱批"
RATIO_COL = "物料名称使用比例"

# ML 聚合用（和 predict_clean.py 对齐）
HVI_COLS_ALL = ["MIC", "MAT%", "LEN(INCH)", "STR(CN/TEX)"]
AFIS_COLS_ALL = ["AFIS-细度(MTEX)", "AFIS-成熟度", "AFIS-平均长度mm", "SFC(N)-%"]


# ----------------- 指标 -----------------
def rel_err(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)


def within5(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(rel_err(y_true, y_pred) <= 0.05))


# ----------------- 兼容 XGBoost 旧序列化 -----------------
def _patch_xgb_model_attrs(est):
    """给旧的 XGBModel 补齐新版本需要的属性（目前主要是 feature_weights）。"""
    try:
        mod = getattr(est.__class__, "__module__", "") or ""
        if "xgboost" in mod.lower():
            if not hasattr(est, "feature_weights"):
                setattr(est, "feature_weights", None)
    except Exception:
        pass
    return est


def _safe_make_estimator(model_template):
    """
    生成一个“未训练”的 estimator 副本来 fit。
    - 先 sklearn.clone
    - 失败则 deepcopy
    - 再失败则直接返回原对象（fit 会覆盖 booster）
    """
    model_template = _patch_xgb_model_attrs(model_template)
    try:
        return clone(model_template)
    except Exception:
        try:
            return copy.deepcopy(model_template)
        except Exception:
            return model_template


# ----------------- ML：清洗+聚合 -----------------
def clean_invalid_rows(df: pd.DataFrame, hvi_cols: list, afis_cols: list) -> pd.DataFrame:
    df = df.copy()

    hvi_cols = [c for c in hvi_cols if c in df.columns]
    afis_cols = [c for c in afis_cols if c in df.columns]
    if not hvi_cols and not afis_cols:
        return df

    hvi_vals = df[hvi_cols].fillna(0) if hvi_cols else pd.DataFrame(index=df.index)
    afis_vals = df[afis_cols].fillna(0) if afis_cols else pd.DataFrame(index=df.index)

    if hvi_cols:
        all_hvi_zero = (hvi_vals == 0).all(axis=1)
        any_hvi_zero = (hvi_vals == 0).any(axis=1)
    else:
        all_hvi_zero = pd.Series(False, index=df.index)
        any_hvi_zero = pd.Series(False, index=df.index)

    if afis_cols:
        all_afis_zero = (afis_vals == 0).all(axis=1)
        any_afis_zero = (afis_vals == 0).any(axis=1)
    else:
        all_afis_zero = pd.Series(False, index=df.index)
        any_afis_zero = pd.Series(False, index=df.index)

    remove_mask = (all_hvi_zero & any_afis_zero) | (all_afis_zero & any_hvi_zero)
    return df.loc[~remove_mask].copy()


def weighted_mean_ignore_zero(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = (values != 0) & ~np.isnan(values)
    if not np.any(mask):
        return 0.0
    v = values[mask]
    w = weights[mask]
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return float(np.sum(v * w))


def aggregate_by_batch_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if RATIO_COL in df.columns and df[RATIO_COL].max() > 1.5:
        df[RATIO_COL] = df[RATIO_COL] / 100.0

    groups = df.groupby(BATCH_COL)
    agg_rows = []

    hvi_cols = [c for c in HVI_COLS_ALL if c in df.columns]
    afis_cols = [c for c in AFIS_COLS_ALL if c in df.columns]
    feature_cols = hvi_cols + afis_cols

    for batch, g in groups:
        g = g.copy()
        total_ratio = g[RATIO_COL].sum() if RATIO_COL in g.columns else 0.0
        if total_ratio <= 0:
            g["norm_p"] = 1.0 / len(g)
        else:
            g["norm_p"] = g[RATIO_COL] / total_ratio

        row = {BATCH_COL: batch}

        for col in feature_cols:
            row[col] = weighted_mean_ignore_zero(g[col].values, g["norm_p"].values)

        if "纺纱纱支" in g.columns:
            row["纺纱纱支"] = g["纺纱纱支"].iloc[0]

        if "实测单纱捻度" in g.columns:
            row["捻度"] = g["实测单纱捻度"].iloc[0]
        elif "捻度" in g.columns:
            row["捻度"] = g["捻度"].iloc[0]

        row["纺纱方式"] = g["纺纱方式"].iloc[0] if "纺纱方式" in g.columns else ""
        row["梳棉工艺名"] = g["梳棉工艺名"].iloc[0] if "梳棉工艺名" in g.columns else ""
        row["精梳工艺名"] = g["精梳工艺名"].iloc[0] if "精梳工艺名" in g.columns else ""

        if TARGET_COL in g.columns:
            row[TARGET_COL] = g[TARGET_COL].iloc[0]
        else:
            row[TARGET_COL] = np.nan

        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def ensure_columns(df: pd.DataFrame, cols: List[str], fill_value) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def build_preprocessor(num_features: List[str], cat_features: List[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if num_features:
        transformers.append(("num", StandardScaler(), num_features))
    if cat_features:
        transformers.append(("cat", ohe, cat_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def ml_oof_predictions(
    batch_df: pd.DataFrame,
    unique_batches_in_order: np.ndarray,
    ml_bundle: dict,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.Series, dict]:
    model_template = _patch_xgb_model_attrs(ml_bundle["model"])
    num_features = list(ml_bundle.get("num_features", []))
    cat_features = list(ml_bundle.get("cat_features", []))

    batch_df = batch_df.copy()
    batch_df = ensure_columns(batch_df, num_features, 0.0)
    batch_df = ensure_columns(batch_df, cat_features, "")
    batch_df = batch_df.set_index(BATCH_COL, drop=False)

    X_all = batch_df[num_features + cat_features]
    y_all = batch_df[TARGET_COL].values.astype(float)

    oof_pred = pd.Series(index=batch_df.index, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(unique_batches_in_order), start=1):
        tr_batches = set(unique_batches_in_order[tr_idx].tolist())
        va_batches = set(unique_batches_in_order[va_idx].tolist())

        tr_mask = batch_df.index.isin(tr_batches)
        va_mask = batch_df.index.isin(va_batches)

        X_tr = X_all.loc[tr_mask]
        y_tr = y_all[tr_mask]
        X_va = X_all.loc[va_mask]

        pre = build_preprocessor(num_features, cat_features)
        X_tr_enc = pre.fit_transform(X_tr)
        X_va_enc = pre.transform(X_va)

        est = _safe_make_estimator(model_template)
        est.fit(X_tr_enc, y_tr)
        pred_va = est.predict(X_va_enc)
        oof_pred.loc[X_va.index] = pred_va

        print(f"[ML OOF] fold {fold}: val batches={va_mask.sum()}, Within5={within5(y_all[va_mask], pred_va):.4f}")

    # refit：用于后续测试集推理
    pre_full = build_preprocessor(num_features, cat_features)
    X_full_enc = pre_full.fit_transform(X_all)

    est_full = _safe_make_estimator(model_template)
    est_full.fit(X_full_enc, y_all)

    refit_bundle = {
        "model": est_full,
        "preprocessor": pre_full,
        "num_features": num_features,
        "cat_features": cat_features,
        "model_name": ml_bundle.get("model_name", type(est_full).__name__),
        "metrics": ml_bundle.get("metrics", {}),
    }
    return oof_pred, refit_bundle


# ----------------- Gate：meta 特征（预测阶段可得） -----------------
def batch_meta_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if RATIO_COL in df.columns and df[RATIO_COL].max() > 1.5:
        df[RATIO_COL] = df[RATIO_COL] / 100.0

    g = df.groupby(BATCH_COL, sort=False)
    patch_count = g.size().rename("patch_count").astype(int)

    def _ratio_stats(x: pd.Series) -> Tuple[float, float, float]:
        p = x.fillna(0).astype(float).values
        s = p.sum()
        if s <= 0:
            p = np.ones_like(p) / len(p)
        else:
            p = p / s
        eps = 1e-12
        ent = float(-np.sum(p * np.log(p + eps)))
        p_sorted = np.sort(p)[::-1]
        pmax = float(p_sorted[0]) if len(p_sorted) else 0.0
        top2 = float(p_sorted[:2].sum()) if len(p_sorted) >= 2 else pmax
        return ent, pmax, top2

    stats = g[RATIO_COL].apply(_ratio_stats).rename("ratio_stats")
    meta = pd.DataFrame({
        "ratio_entropy": stats.apply(lambda t: t[0]),
        "ratio_max": stats.apply(lambda t: t[1]),
        "ratio_top2_sum": stats.apply(lambda t: t[2]),
    })
    meta = pd.concat([patch_count, meta], axis=1)
    meta.index.name = BATCH_COL
    return meta.reset_index()


def build_gate_features(meta_i: pd.DataFrame, pred_ml: pd.Series, pred_dl: pd.Series) -> pd.DataFrame:
    df = meta_i.set_index(BATCH_COL, drop=False).copy()
    df["pred_ml"] = pred_ml.reindex(df.index).values
    df["pred_dl"] = pred_dl.reindex(df.index).values
    df["diff"] = df["pred_dl"] - df["pred_ml"]
    df["abs_diff"] = np.abs(df["diff"])
    df["rel_diff"] = df["abs_diff"] / (np.abs(df["pred_ml"]) + 1e-6)

    feature_cols = [
        "pred_ml", "pred_dl", "diff", "abs_diff", "rel_diff",
        "patch_count", "ratio_entropy", "ratio_max", "ratio_top2_sum",
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_cols]


def compute_w_star(y_true: np.ndarray, y_ml: np.ndarray, y_dl: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    denom = (y_dl - y_ml)
    w = np.empty_like(denom, dtype=float)
    mask = np.abs(denom) > eps
    w[mask] = (y_true[mask] - y_ml[mask]) / denom[mask]
    w[~mask] = 0.5
    return np.clip(w, 0.0, 1.0)


def train_gate_regressor(X: pd.DataFrame, w_star: np.ndarray):
    try:
        from lightgbm import LGBMRegressor  # type: ignore
        model = LGBMRegressor(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X, w_star)
        return model
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=600,
            random_state=42,
        )
        model.fit(X, w_star)
        return model


# ----------------- DL：不依赖 dataset.py 的推理 Dataset -----------------
def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _list_fold_ckpts(ckpt_dir: Path) -> List[Path]:
    ckpts = sorted(ckpt_dir.glob("fold_*_best.pth"))
    if not ckpts:
        ckpts = sorted(ckpt_dir.glob("*.pth"))
    return ckpts


def _maybe_create_afis_alias_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "AFIS-细度(MTEX)": "afis-细度",
        "AFIS-成熟度": "afis-成熟度",
        "AFIS-平均长度mm": "afis-均长",
        "SFC(N)-%": "afis-短绒率",   # 近似映射：如你有真实“AFIS短绒率”列名，请在这里改成正确映射
    }
    for src, dst in mapping.items():
        if dst not in df.columns and src in df.columns:
            df[dst] = df[src]
    return df


class YarnDLInferDataset:
    """
    内置推理 Dataset：最大限度复刻你的 MyDataset 的输出结构，并按 ckpt 维度做 pad/trim
    """
    def __init__(self,
                 df_raw: pd.DataFrame,
                 one_hot_map: dict,
                 max_len: int,
                 is_twist: bool,
                 hvi_num: int,
                 comber_num: int, d_yc: int):
        self.df = _maybe_create_afis_alias_cols(df_raw).copy()
        self.one_hot_map = one_hot_map
        self.max_len = int(max_len)
        self.is_twist = bool(is_twist)
        self.hvi_num = int(hvi_num)
        self.comber_num = int(comber_num)
        self.d_yc = int(d_yc)

        if RATIO_COL in self.df.columns and self.df[RATIO_COL].max() > 1.5:
            self.df[RATIO_COL] = self.df[RATIO_COL] / 100.0

        # label 仅作占位，不参与推理
        if TARGET_COL in self.df.columns and self.df[TARGET_COL].max() > 5:
            self.df["_y_scaled"] = self.df[TARGET_COL] / 100.0
        elif TARGET_COL in self.df.columns:
            self.df["_y_scaled"] = self.df[TARGET_COL]
        else:
            self.df["_y_scaled"] = 0.0

        self.unique_batches = self.df[BATCH_COL].unique().tolist()

        self.base_hvi_cols = ["MIC", "MAT%", "LEN(INCH)", "SFI(%)"]
        self.afis_cols = ['AFIS-细度(MTEX)','AFIS-成熟度','AFIS-平均长度mm','SFC(N)-%']
        self.comber_cols = ["梳棉工艺名", "精梳工艺名"]
        self.single_twist_cols = ["纺纱股数", "单纱捻度"]
        self.ply_twist_cols = ["纺纱股数", "股线捻度"]

    def __len__(self):
        return len(self.unique_batches)

    def _get_onehot(self, key) -> np.ndarray:
        return np.asarray(self.one_hot_map[str(key)], dtype=np.float32)

    @staticmethod
    def _pad_or_trim_last_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
        cur = arr.shape[-1]
        if cur == target_dim:
            return arr
        if cur > target_dim:
            return arr[..., :target_dim]
        pad = np.zeros((*arr.shape[:-1], target_dim - cur), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=-1)

    def _norm_cols_by_sum(self, g: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        g = g.copy()
        present = [c for c in cols if c in g.columns]
        for c in present:
            s = float(np.nansum(g[c].values.astype(float)))
            if s != 0:
                g[c] = g[c].astype(float) / s
            else:
                g[c] = 0.0
        return g

    def __getitem__(self, idx: int):
        import torch
        from torch import nn as tnn

        b = self.unique_batches[idx]
        g = self.df[self.df[BATCH_COL] == b].copy()
        length = len(g)

        if length > self.max_len:
            g = g.iloc[:self.max_len].copy()
            length = self.max_len

        # use_afis 判断：若 HVI 合计为0，则用 AFIS
        use_afis = False
        present_hvi = [c for c in self.base_hvi_cols if c in g.columns]
        if present_hvi:
            use_afis = float(np.nansum(g[present_hvi].values.astype(float))) == 0.0

        g = self._norm_cols_by_sum(g, self.base_hvi_cols)
        g = self._norm_cols_by_sum(g, self.afis_cols)

        feat_cols = self.afis_cols if use_afis else self.base_hvi_cols
        for c in feat_cols:
            if c not in g.columns:
                g[c] = 0.0

        # feature1 = [degree_onehot, 5维特征]，再按 hvi_num pad/trim
        if "棉等级" not in g.columns:
            g["棉等级"] = "UNKNOWN"
        degree_oh = np.stack([self._get_onehot(x) for x in g["棉等级"].values], axis=0)  # [L, d_deg_raw]
        degree_target = max(self.hvi_num - 5, 0)
        degree_oh = self._pad_or_trim_last_dim(degree_oh, degree_target).astype(np.float32)

        feat_vals = g[feat_cols].values.astype(np.float32)  # [L, 5]
        feature1 = np.concatenate([degree_oh, feat_vals], axis=1).astype(np.float32)      # [L, *]
        feature1 = self._pad_or_trim_last_dim(feature1, self.hvi_num)
        feature1 = torch.from_numpy(feature1).float()

        # comber_feature：拼接两个工艺 onehot，按 comber_num pad/trim
        comber_parts = []
        for c in self.comber_cols:
            if c not in g.columns:
                g[c] = "UNKNOWN"
            oh = np.stack([self._get_onehot(x) for x in g[c].values], axis=0)
            comber_parts.append(oh)
        comber_feature = np.concatenate(comber_parts, axis=1).astype(np.float32)
        comber_feature = self._pad_or_trim_last_dim(comber_feature, self.comber_num)
        comber_feature = torch.from_numpy(comber_feature).float()

        # p：比例向量（和为1）
        if RATIO_COL not in g.columns:
            g[RATIO_COL] = 0.0
        p = g[RATIO_COL].values.astype(np.float32)
        s = float(np.sum(p))
        if s > 0:
            p = p / s
        p = torch.from_numpy(p).float()

        # sp_ps：纺纱方式 onehot + ps，再按 d_y pad/trim
        if "纺纱方式" not in g.columns:
            g["纺纱方式"] = "UNKNOWN"
        sp = self._get_onehot(g["纺纱方式"].values[0]).astype(np.float32)

        if "纺纱纱支" not in g.columns:
            g["纺纱纱支"] = 0.0

        if self.is_twist:
            ps = np.array([float(g["纺纱纱支"].values[0])], dtype=np.float32)
        else:
            for c in self.single_twist_cols:
                if c not in g.columns:
                    g[c] = 0.0
            ps = g[["纺纱纱支"] + self.single_twist_cols].values[0].astype(np.float32)

        sp_ps = np.concatenate([sp, ps], axis=0).astype(np.float32)
        sp_ps = self._pad_or_trim_last_dim(sp_ps.reshape(1, -1), self.d_yc).reshape(-1).astype(np.float32)
        sp_ps = torch.from_numpy(sp_ps).float()

        # twist（仅 is_twist=True 时需要）
        for c in self.ply_twist_cols:
            if c not in g.columns:
                g[c] = 0.0
        twist = g[self.ply_twist_cols].values[0].astype(np.float32)
        twist = torch.from_numpy(twist).float()

        y = torch.tensor(float(g["_y_scaled"].values[0])).float()

        # padding 到 max_len
        pad1 = tnn.ZeroPad2d(padding=(0, 0, 0, (self.max_len - length)))
        feature1 = pad1(feature1)
        comber_feature = pad1(comber_feature)

        pad2 = tnn.ZeroPad2d(padding=(0, (self.max_len - length)))
        p = pad2(p)

        if self.is_twist:
            return feature1, p, comber_feature, sp_ps, twist, y
        return feature1, p, comber_feature, sp_ps, y


def _build_mydataset(DS, csv_path: str, max_len: int, is_twist: bool):
    """
    兼容不同 MyDataset 构造函数签名：
    - 常见：MyDataset(csv_path, max_len=..., is_twist=...)
    - 也可能需要 one_hot_map / is_train 等
    这里通过 inspect.signature 自动适配。
    """
    import inspect

    sig = inspect.signature(DS.__init__)
    params = sig.parameters

    # 第一个参数 self 已被包含，csv_path 通常是第二个；我们用位置参数传入
    kwargs = {}
    if "max_len" in params:
        kwargs["max_len"] = int(max_len)
    if "is_twist" in params:
        kwargs["is_twist"] = bool(is_twist)
    # 其他参数保持默认（如 is_train 等）
    return DS(csv_path, **kwargs)


def dl_predict_from_df(
    df_raw: pd.DataFrame,
    ckpt_path: Path,
    one_hot_map_path: Path,
    batch_size: int = 16,
    device: Optional[str] = None,
    output_scale: float = 100.0,
) -> Dict[str, float]:
    """
    用单个 ckpt 对 df_raw 中的所有纱批预测，返回 {纱批 -> 预测强力(原始量纲)}。
    优先使用你项目中的 dataset_use.py 里的 MyDataset（与训练完全一致）；
    若导入失败或运行失败，则退回到脚本内置的 YarnDLInferDataset（兼容兜底）。
    """
    import torch
    from torch.utils.data import DataLoader
    from model_i import Blendmapping

    device = _resolve_device(device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    is_twist = bool(ckpt.get("is_twist", False))

    model = Blendmapping(
        d_model=int(ckpt["d_model"]),
        hvi_num=int(ckpt["hvi_num"]),
        comber_num=int(ckpt["comber_num"]),
        d_yc=int(ckpt["d_yc"]),
        d_y=int(ckpt["d_y"]),
        N=int(ckpt["N"]),
        heads=int(ckpt["heads"]),
        dropout=0.05,
        is_twist=is_twist,
        use_dirichlet=True,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)

    # 需要的 max_len
    max_len_needed = int(df_raw.groupby(BATCH_COL).size().max())
    max_padding = int(ckpt.get("max_padding", max_len_needed))
    max_len = max(max_padding, max_len_needed)

    # ========== 首选：dataset_use.MyDataset ==========
    try:
        from dataset_use import MyDataset as DS  # type: ignore

        # 写临时 csv 给 MyDataset（多数实现都是读 csv_path）
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            tmp_csv = f.name
            df_raw.to_csv(tmp_csv, index=False)

        ds = _build_mydataset(DS, tmp_csv, max_len=max_len, is_twist=is_twist)

        # 需要拿到 unique_name，用于把 batch index 映射回纱批名
        if not hasattr(ds, "unique_name"):
            raise AttributeError("dataset_use.MyDataset 没有 unique_name 属性，无法对齐批次名。")

        # DataLoader: 取索引以便回写纱批名
        class _IdxDS(torch.utils.data.Dataset):
            def __init__(self, base):
                self.base = base
            def __len__(self):
                return len(self.base)
            def __getitem__(self, idx):
                return (idx, *self.base[idx])

        loader = DataLoader(_IdxDS(ds), batch_size=batch_size, shuffle=False, num_workers=0)

        preds: Dict[str, float] = {}
        with torch.no_grad():
            for batch in loader:
                idx = batch[0].cpu().numpy().astype(int)

                if is_twist:
                    x, prop, x1, sp_ps, twist, _y = batch[1:]
                    out = model(x.to(device), prop.to(device), x1.to(device), sp_ps.to(device), twist.to(device))
                else:
                    x, prop, x1, sp_ps, _y = batch[1:]
                    out = model(x.to(device), prop.to(device), x1.to(device), sp_ps.to(device))

                out = out.view(-1).detach().cpu().numpy().astype(float) * float(output_scale)

                for i, p in zip(idx, out):
                    name = str(ds.unique_name[i])
                    preds[name] = float(p)

        # 删除临时文件
        try:
            Path(tmp_csv).unlink(missing_ok=True)
        except Exception:
            pass

        return preds

    except Exception as e:
        print(f"[DL] WARN: dataset_use.MyDataset 推理失败，回退到内置 Dataset。原因: {type(e).__name__}: {e}")

    # ========== 兜底：内置 YarnDLInferDataset ==========
    one_hot_map = pd.read_pickle(one_hot_map_path)

    ds = YarnDLInferDataset(
        df_raw=df_raw,
        one_hot_map=one_hot_map,
        max_len=max_len,
        is_twist=is_twist,
        hvi_num=int(ckpt["hvi_num"]),
        comber_num=int(ckpt["comber_num"]),
        d_yc=int(ckpt["d_yc"]),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds: Dict[str, float] = {}
    with torch.no_grad():
        start = 0
        for batch in loader:
            if is_twist:
                x, prop, x1, sp_ps, twist, _y = batch
                out = model(x.to(device), prop.to(device), x1.to(device), sp_ps.to(device), twist.to(device))
            else:
                x, prop, x1, sp_ps, _y = batch
                out = model(x.to(device), prop.to(device), x1.to(device), sp_ps.to(device))

            out = out.view(-1).detach().cpu().numpy().astype(float) * float(output_scale)

            bs = len(out)
            names = ds.unique_batches[start:start + bs]
            start += bs
            for name, p in zip(names, out):
                preds[str(name)] = float(p)

    return preds


# ----------------- Gate 训练主流程 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--ml_bundle", type=str, required=True)
    ap.add_argument("--dl_ckpt_dir", type=str, default="./cv_models")
    ap.add_argument("--out_dir", type=str, default="./gate_fusion_out")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--dl_batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--one_hot_map", type=str, default="../one_hot_map.pkl")
    ap.add_argument("--dl_output_scale", type=float, default=100.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train_raw = pd.read_csv(args.train_csv, low_memory=False)
    unique_batches = df_train_raw[BATCH_COL].unique()
    print(f"[Train] raw rows={len(df_train_raw)}, unique batches={len(unique_batches)}")

    meta_train = batch_meta_features(df_train_raw).set_index(BATCH_COL, drop=False)

    # ML batch-level
    df_train_clean = clean_invalid_rows(df_train_raw, HVI_COLS_ALL, AFIS_COLS_ALL)
    batch_train = aggregate_by_batch_for_ml(df_train_clean).set_index(BATCH_COL, drop=False)
    if TARGET_COL not in batch_train.columns:
        raise RuntimeError(f"聚合后找不到标签列：{TARGET_COL}")

    # ML OOF + refit
    ml_bundle = joblib.load(args.ml_bundle)
    ml_oof, ml_refit_bundle = ml_oof_predictions(batch_train.reset_index(drop=True), unique_batches, ml_bundle, n_splits=args.n_splits)
    joblib.dump(ml_refit_bundle, out_dir / "ml_refit_bundle.pkl")
    print(f"[ML] saved refit bundle: {out_dir/'ml_refit_bundle.pkl'}")

    # DL OOF：按 fold ckpt 对每折 val 预测
    ckpt_paths = _list_fold_ckpts(Path(args.dl_ckpt_dir))
    fold_ckpts = {int(m.group(1)): p for p in ckpt_paths
                  if (m := re.search(r"fold_(\d+)_best\.pth", p.name))}
    if len(fold_ckpts) < args.n_splits:
        raise FileNotFoundError(f"fold ckpt 不足：需要 >= {args.n_splits} 个 fold_*_best.pth，当前={len(fold_ckpts)}（目录：{args.dl_ckpt_dir}）")

    dl_oof = pd.Series(index=ml_oof.index, dtype=float)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    for fold, (_tr_idx, va_idx) in enumerate(kf.split(unique_batches), start=1):
        val_batches = unique_batches[va_idx]
        df_val = df_train_raw[df_train_raw[BATCH_COL].isin(val_batches)].copy()

        pred_map = dl_predict_from_df(
            df_raw=df_val,
            ckpt_path=fold_ckpts[fold],
            one_hot_map_path=Path(args.one_hot_map),
            batch_size=args.dl_batch_size,
            device=args.device,
            output_scale=args.dl_output_scale,
        )

        for b in val_batches:
            b = str(b)
            if b in pred_map:
                dl_oof.loc[b] = pred_map[b]

        # fold 内评估
        bt = batch_train
        val_present = [str(b) for b in val_batches if str(b) in bt.index]
        y_true_val = bt.loc[val_present, TARGET_COL].values.astype(float)
        y_pred_val = np.array([pred_map.get(str(b), np.nan) for b in val_present], dtype=float)
        ok = ~np.isnan(y_pred_val)
        print(f"[DL OOF] fold {fold}: val batches={len(val_present)}, pred_ok={ok.sum()}, Within5={within5(y_true_val[ok], y_pred_val[ok]):.4f}")

    # Gate 训练对齐
    common = meta_train.index.intersection(ml_oof.index).intersection(dl_oof.index)
    common = common[~dl_oof.loc[common].isna()]
    print(f"[Gate] common batches: {len(common)}")

    y_true = batch_train.loc[common, TARGET_COL].values.astype(float)
    y_ml = ml_oof.loc[common].values.astype(float)
    y_dl = dl_oof.loc[common].values.astype(float)

    w_star = compute_w_star(y_true, y_ml, y_dl)
    X_gate = build_gate_features(meta_train.loc[common].reset_index(drop=True), ml_oof.loc[common], dl_oof.loc[common])

    gate_model = train_gate_regressor(X_gate, w_star)

    w_hat = np.clip(gate_model.predict(X_gate), 0.0, 1.0)
    y_fused = w_hat * y_dl + (1.0 - w_hat) * y_ml
    print(f"[Gate] OOF Within5 (fused on common): {within5(y_true, y_fused):.4f}")

    # 保存 gate
    gate_art = {"gate_model": gate_model, "gate_feature_cols": list(X_gate.columns)}
    joblib.dump(gate_art, out_dir / "gate_model.pkl")
    print(f"[Gate] saved: {out_dir/'gate_model.pkl'}")

    # 保存 OOF 明细
    oof_df = pd.DataFrame({
        BATCH_COL: common,
        "y_true": y_true,
        "pred_ml_oof": y_ml,
        "pred_dl_oof": y_dl,
        "w_star": w_star,
        "w_hat": w_hat,
        "pred_fused_oof": y_fused,
        "rel_err_fused": rel_err(y_true, y_fused),
    })
    oof_df.to_csv(out_dir / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"[Train] saved: {out_dir/'train_oof_predictions.csv'}")


if __name__ == "__main__":
    main()
