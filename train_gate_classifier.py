# train_gate_classifier.py
# -*- coding: utf-8 -*-
"""
训练“选择专家”的 Gate（Classifier Selector）

核心思路：
- 你的评价指标是“相对误差 <= 5% 是否合格”，属于阈值型指标；
- 与其回归一个融合权重 w，不如直接学习：对每个纱批，ML 与 DL 谁更可能更准（更接近真实值）。

训练标签来自训练集 OOF 预测：
label = 1  (DL 的相对误差 < ML 的相对误差)
label = 0  (否则选 ML)
可通过 --margin 抑制边界噪声：只有 DL 至少比 ML 好 margin 才标 1。

避免数据泄漏：
- ML OOF：脚本内部按纱批做 KFold 生成 OOF
- DL OOF：用 fold_*_best.pth 对应折的 val 批次预测（与你训练时划分一致）
  若某 fold 缺 ckpt，会跳过该折（这些批次不参与 gate 训练，保证不泄漏）。

产物：
- out_dir/selector_model.pkl
- out_dir/ml_refit_bundle.pkl
- out_dir/train_selector_oof.csv

用法（Windows CMD）：
python train_gate_classifier.py --train_csv .\\without\\single_train_data_without.csv --ml_bundle .\\pkl\\yarn_strength_model_gpu.pkl --dl_ckpt_dir .\\cv_models --out_dir .\\gate_fusion_out --one_hot_map .\\one_hot_map_without.pkl
"""

from __future__ import annotations

import argparse
import copy
import inspect
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "纱强力"
BATCH_COL = "纱批"
RATIO_COL = "物料名称使用比例"

# ML 聚合用（尽量与 predict_clean.py 对齐）
HVI_COLS_ALL = ["MIC", "MAT%", "LEN(INCH)", "STR(CN/TEX)"]
AFIS_COLS_ALL = ["AFIS-细度(MTEX)", "AFIS-成熟度", "AFIS-平均长度mm", "SFC(N)-%"]

CAT_COLS_CANDIDATES = ["纺纱方式", "梳棉工艺名", "精梳工艺名"]
NUM_COLS_CANDIDATES = ["纺纱纱支", "捻度", "实测单纱捻度", "纺纱股数", "单纱捻度", "股线捻度"]


def rel_err(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)


def within5(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(rel_err(y_true, y_pred) <= 0.05))


# ----------------- 兼容 XGBoost 旧序列化（feature_weights 缺失） -----------------
def _patch_xgb_model_attrs(est):
    try:
        mod = getattr(est.__class__, "__module__", "") or ""
        if "xgboost" in mod.lower():
            if not hasattr(est, "feature_weights"):
                setattr(est, "feature_weights", None)
    except Exception:
        pass
    return est


def _safe_make_estimator(model_template):
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


def _weighted_stats_ignore_zero(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
    """返回 (wmean, wstd, nonzero_frac)，忽略 values==0 与 NaN"""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    m = (values != 0) & ~np.isnan(values)
    if m.sum() == 0:
        return 0.0, 0.0, 0.0

    v = values[m]
    w = weights[m]
    s = float(np.sum(w))
    w = (np.ones_like(w) / len(w)) if s <= 0 else (w / s)

    mean = float(np.sum(v * w))
    var = float(np.sum(((v - mean) ** 2) * w))
    std = float(np.sqrt(max(var, 0.0)))
    frac = float(m.mean())
    return mean, std, frac


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
        g["norm_p"] = (1.0 / len(g)) if total_ratio <= 0 else (g[RATIO_COL] / total_ratio)

        row = {BATCH_COL: batch}
        for col in feature_cols:
            mean, _, _ = _weighted_stats_ignore_zero(g[col].values, g["norm_p"].values)
            row[col] = mean

        # 数值工艺参数（取首行）
        for c in NUM_COLS_CANDIDATES:
            if c in g.columns:
                row[c] = g[c].iloc[0]

        # 类别工艺参数（取首行）
        for c in CAT_COLS_CANDIDATES:
            row[c] = g[c].iloc[0] if c in g.columns else "UNKNOWN"

        row[TARGET_COL] = g[TARGET_COL].iloc[0] if TARGET_COL in g.columns else np.nan
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

        print(f"[ML OOF] fold {fold}: val={va_mask.sum()}, Within5={within5(y_all[va_mask], pred_va):.4f}")

    # refit（供测试集推理）
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


# ----------------- Gate 特征：配比结构 + 缺失模式 + 离散度 -----------------
def _ratio_entropy(p: np.ndarray) -> float:
    eps = 1e-12
    p = np.asarray(p, dtype=float)
    s = float(np.sum(p))
    if s <= 0:
        return 0.0
    p = p / s
    return float(-np.sum(p * np.log(p + eps)))


def _gini(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    s = float(np.sum(p))
    if s <= 0:
        return 0.0
    p = np.sort(p / s)
    n = len(p)
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * p)) / n)


def batch_gate_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if RATIO_COL in df.columns and df[RATIO_COL].max() > 1.5:
        df[RATIO_COL] = df[RATIO_COL] / 100.0

    num_cols = [c for c in (HVI_COLS_ALL + AFIS_COLS_ALL) if c in df.columns]
    proc_num_cols = [c for c in NUM_COLS_CANDIDATES if c in df.columns]
    cat_cols = [c for c in CAT_COLS_CANDIDATES if c in df.columns]

    out_rows = []
    for batch, g in df.groupby(BATCH_COL, sort=False):
        g = g.copy()
        p = g[RATIO_COL].fillna(0).astype(float).values if RATIO_COL in g.columns else np.ones(len(g), dtype=float)
        s = float(np.sum(p))
        w = (np.ones_like(p) / len(p)) if s <= 0 else (p / s)

        p_sorted = np.sort(w)[::-1]
        ratio_max = float(p_sorted[0]) if len(p_sorted) else 0.0
        ratio_top2 = float(p_sorted[:2].sum()) if len(p_sorted) >= 2 else ratio_max
        eff_mat = int(np.sum(w > 1e-6))

        row = {
            BATCH_COL: batch,
            "patch_count": int(len(g)),
            "ratio_entropy": _ratio_entropy(w),
            "ratio_gini": _gini(w),
            "ratio_max": ratio_max,
            "ratio_top2_sum": ratio_top2,
            "ratio_effective_n": eff_mat,
        }

        hvi_present = [c for c in HVI_COLS_ALL if c in g.columns]
        afis_present = [c for c in AFIS_COLS_ALL if c in g.columns]
        row["has_hvi"] = 1 if (hvi_present and float(np.nansum(g[hvi_present].values.astype(float))) != 0.0) else 0
        row["has_afis"] = 1 if (afis_present and float(np.nansum(g[afis_present].values.astype(float))) != 0.0) else 0

        for c in num_cols:
            mean, std, frac = _weighted_stats_ignore_zero(g[c].values, w)
            row[f"{c}__wmean"] = mean
            row[f"{c}__wstd"] = std
            row[f"{c}__nzfrac"] = frac

        for c in proc_num_cols:
            row[c] = float(g[c].iloc[0]) if c in g.columns and pd.notna(g[c].iloc[0]) else 0.0
        for c in cat_cols:
            row[c] = str(g[c].iloc[0]) if c in g.columns and pd.notna(g[c].iloc[0]) else "UNKNOWN"

        out_rows.append(row)

    feat_df = pd.DataFrame(out_rows)
    for c in CAT_COLS_CANDIDATES:
        if c not in feat_df.columns:
            feat_df[c] = "UNKNOWN"
    for c in proc_num_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
    return feat_df


def build_selector_feature_table(gate_feat: pd.DataFrame, pred_ml: pd.Series, pred_dl: pd.Series) -> pd.DataFrame:
    df = gate_feat.set_index(BATCH_COL, drop=False).copy()
    df["pred_ml"] = pred_ml.reindex(df.index).values
    df["pred_dl"] = pred_dl.reindex(df.index).values
    df["diff"] = df["pred_dl"] - df["pred_ml"]
    df["abs_diff"] = np.abs(df["diff"])
    df["rel_diff"] = df["abs_diff"] / (np.abs(df["pred_ml"]) + 1e-6)
    return df


# ----------------- DL OOF：dataset_use.MyDataset -----------------
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


def _build_mydataset(DS, csv_path: str, max_len: int, is_twist: bool, one_hot_map_path: Optional[Path] = None):
    sig = inspect.signature(DS.__init__)
    params = sig.parameters

    kwargs = {}
    if "max_len" in params:
        kwargs["max_len"] = int(max_len)
    if "is_twist" in params:
        kwargs["is_twist"] = bool(is_twist)

    if one_hot_map_path is not None:
        if ("one_hot_map" in params) or ("onehot_map" in params) or ("map_dict" in params) or ("dict_map" in params):
            try:
                oh = pd.read_pickle(one_hot_map_path)
                if "one_hot_map" in params:
                    kwargs["one_hot_map"] = oh
                elif "onehot_map" in params:
                    kwargs["onehot_map"] = oh
                elif "map_dict" in params:
                    kwargs["map_dict"] = oh
                elif "dict_map" in params:
                    kwargs["dict_map"] = oh
            except Exception:
                pass

    if "is_train" in params and "is_train" not in kwargs:
        kwargs["is_train"] = False

    return DS(csv_path, **kwargs)


def dl_predict_fold_val(
    df_val: pd.DataFrame,
    ckpt_path: Path,
    one_hot_map_path: Path,
    batch_size: int = 16,
    device: Optional[str] = None,
    output_scale: float = 100.0,
) -> Dict[str, float]:
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

    max_len_needed = int(df_val.groupby(BATCH_COL).size().max())
    max_padding = int(ckpt.get("max_padding", max_len_needed))
    max_len = max(max_padding, max_len_needed)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        tmp_csv = f.name
        df_val.to_csv(tmp_csv, index=False)

    try:
        from dataset_use import MyDataset as DS  # type: ignore
        ds = _build_mydataset(DS, tmp_csv, max_len=max_len, is_twist=is_twist, one_hot_map_path=one_hot_map_path)

        if not hasattr(ds, "unique_name"):
            raise AttributeError("dataset_use.MyDataset 没有 unique_name 属性。")

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
                    preds[str(ds.unique_name[i])] = float(p)
        return preds
    finally:
        try:
            Path(tmp_csv).unlink(missing_ok=True)
        except Exception:
            pass


# ----------------- 选择器模型 -----------------
def train_selector_model(X: pd.DataFrame, y: np.ndarray):
    # object -> cat, else -> num
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].fillna("UNKNOWN").astype(str)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(float)

    pre = build_preprocessor(num_cols, cat_cols)
    X_enc = pre.fit_transform(X)

    # 优先 lightgbm / catboost，否则 HGBClassifier
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        model = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X_enc, y)
        return model, pre, num_cols, cat_cols
    except Exception:
        try:
            from catboost import CatBoostClassifier  # type: ignore
            model = CatBoostClassifier(
                depth=6, learning_rate=0.05, iterations=800, loss_function="Logloss",
                verbose=False, random_seed=42
            )
            model.fit(X_enc, y)
            return model, pre, num_cols, cat_cols
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.05, max_iter=600, random_state=42
            )
            model.fit(X_enc, y)
            return model, pre, num_cols, cat_cols


def selector_predict_proba(model, pre, X: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> np.ndarray:
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].fillna("UNKNOWN").astype(str)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(float)

    X_enc = pre.transform(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_enc)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X_enc)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(X_enc).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--ml_bundle", type=str, required=True)
    ap.add_argument("--dl_ckpt_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./gate_fusion_out_classifier")
    ap.add_argument("--one_hot_map", type=str, required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--dl_batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dl_output_scale", type=float, default=100.0)
    ap.add_argument("--margin", type=float, default=0.0002)
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.train_csv, low_memory=False)
    unique_batches = df_raw[BATCH_COL].unique()
    print(f"[Train] raw rows={len(df_raw)}, unique batches={len(unique_batches)}")

    # gate features from raw
    gate_feat = batch_gate_features(df_raw).set_index(BATCH_COL, drop=False)

    # ML batch table for ML model
    df_clean = clean_invalid_rows(df_raw, HVI_COLS_ALL, AFIS_COLS_ALL)
    batch_train = aggregate_by_batch_for_ml(df_clean).set_index(BATCH_COL, drop=False)
    if TARGET_COL not in batch_train.columns:
        raise RuntimeError(f"聚合后找不到标签列：{TARGET_COL}")

    # ML OOF
    ml_bundle = joblib.load(args.ml_bundle)
    ml_oof, ml_refit_bundle = ml_oof_predictions(batch_train.reset_index(drop=True), unique_batches, ml_bundle, n_splits=args.n_splits)
    joblib.dump(ml_refit_bundle, out_dir / "ml_refit_bundle.pkl")
    print(f"[ML] saved refit bundle: {out_dir/'ml_refit_bundle.pkl'}")

    # DL OOF by fold
    ckpt_paths = _list_fold_ckpts(Path(args.dl_ckpt_dir))
    fold_ckpts = {int(m.group(1)): p for p in ckpt_paths if (m := re.search(r"fold_(\d+)_best\.pth", p.name))}
    if len(fold_ckpts) < args.n_splits:
        print(f"[DL] WARNING: fold ckpt 不足：需要 >= {args.n_splits} 个 fold_*_best.pth，当前={len(fold_ckpts)}。"
              f" 将跳过缺失 fold 的 OOF（这些批次不参与 selector 训练）。")

    dl_oof = pd.Series(index=ml_oof.index, dtype=float)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    for fold, (_tr_idx, va_idx) in enumerate(kf.split(unique_batches), start=1):
        val_batches = unique_batches[va_idx]
        if fold not in fold_ckpts:
            print(f"[DL OOF] fold {fold}: MISSING checkpoint -> skip")
            continue

        df_val = df_raw[df_raw[BATCH_COL].isin(val_batches)].copy()
        pred_map = dl_predict_fold_val(
            df_val=df_val,
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

        # fold eval
        val_present = [str(b) for b in val_batches if str(b) in batch_train.index]
        y_true_val = batch_train.loc[val_present, TARGET_COL].values.astype(float)
        y_pred_val = np.array([pred_map.get(str(b), np.nan) for b in val_present], dtype=float)
        ok = ~np.isnan(y_pred_val)
        print(f"[DL OOF] fold {fold}: val={len(val_present)}, pred_ok={ok.sum()}, Within5={within5(y_true_val[ok], y_pred_val[ok]):.4f}")

    # align common
    common = gate_feat.index.intersection(batch_train.index).intersection(ml_oof.index).intersection(dl_oof.index)
    common = common[~dl_oof.loc[common].isna()]
    print(f"[Selector] common batches: {len(common)}")

    y_true = batch_train.loc[common, TARGET_COL].values.astype(float)
    y_ml = ml_oof.loc[common].values.astype(float)
    y_dl = dl_oof.loc[common].values.astype(float)

    e_ml = rel_err(y_true, y_ml)
    e_dl = rel_err(y_true, y_dl)
    y_label = (e_dl + float(args.margin) < e_ml).astype(int)

    X_table = build_selector_feature_table(gate_feat.loc[common].reset_index(drop=True), ml_oof.loc[common], dl_oof.loc[common])

    # internal CV to estimate generalization
    kf2 = KFold(n_splits=5, shuffle=True, random_state=7)
    accs, fused_scores = [], []
    common_arr = np.array(common)

    for i, (tr, va) in enumerate(kf2.split(common_arr), start=1):
        tr_batches = common_arr[tr]
        va_batches = common_arr[va]

        X_tr = X_table.set_index(BATCH_COL).loc[tr_batches].reset_index()
        X_va = X_table.set_index(BATCH_COL).loc[va_batches].reset_index()

        y_tr = y_label[np.isin(common_arr, tr_batches)]
        y_va = y_label[np.isin(common_arr, va_batches)]

        model, pre, num_cols, cat_cols = train_selector_model(X_tr.drop(columns=[BATCH_COL], errors="ignore"), y_tr)
        p_va = selector_predict_proba(model, pre, X_va.drop(columns=[BATCH_COL], errors="ignore"), num_cols, cat_cols)
        choose_dl = (p_va >= float(args.threshold)).astype(int)

        acc = accuracy_score(y_va, choose_dl)
        accs.append(acc)

        y_true_va = batch_train.loc[va_batches, TARGET_COL].values.astype(float)
        y_ml_va = ml_oof.loc[va_batches].values.astype(float)
        y_dl_va = dl_oof.loc[va_batches].values.astype(float)
        y_sel = np.where(choose_dl == 1, y_dl_va, y_ml_va)
        fused_scores.append(within5(y_true_va, y_sel))

        print(f"[Selector CV] fold{i}: acc={acc:.4f}, Within5(selected)={fused_scores[-1]:.4f}")

    print(f"[Selector CV] acc mean={np.mean(accs):.4f}, Within5 mean={np.mean(fused_scores):.4f}")

    # train full selector and save
    model, pre, num_cols, cat_cols = train_selector_model(X_table.drop(columns=[BATCH_COL], errors="ignore"), y_label)
    artifact = {
        "selector_model": model,
        "selector_preprocessor": pre,
        "selector_num_cols": num_cols,
        "selector_cat_cols": cat_cols,
        "threshold": float(args.threshold),
        "margin": float(args.margin),
        "dl_output_scale": float(args.dl_output_scale),
        "feature_note": "X = gate_meta + {pred_ml,pred_dl,diff,abs_diff,rel_diff}; label=1 if DL rel_err+margin < ML rel_err",
    }
    joblib.dump(artifact, out_dir / "selector_model.pkl")
    print(f"[Selector] saved: {out_dir/'selector_model.pkl'}")

    oof_df = pd.DataFrame({
        BATCH_COL: common,
        "y_true": y_true,
        "pred_ml_oof": y_ml,
        "pred_dl_oof": y_dl,
        "rel_err_ml": e_ml,
        "rel_err_dl": e_dl,
        "label_dl_better": y_label,
    })
    oof_df.to_csv(out_dir / "train_selector_oof.csv", index=False, encoding="utf-8-sig")
    print(f"[Train] saved: {out_dir/'train_selector_oof.csv'}")


if __name__ == "__main__":
    main()
