# predict_gate_classifier.py
# -*- coding: utf-8 -*-
"""
使用“选择专家”的 Gate 做推理（Classifier Selector Inference）

依赖：
- artifact_dir/selector_model.pkl
- artifact_dir/ml_refit_bundle.pkl

流程：
1) ML：ml_refit_bundle.pkl 对测试集按纱批聚合后的特征预测
2) DL：读取 dl_ckpt_dir 下多个 ckpt（fold_*_best.pth 或 *.pth）逐个预测并取平均
   推理数据特征提取使用 dataset_use.MyDataset（与训练一致）
3) selector：输出 P(DL更准)，按阈值选择
   - mode=select：hard 选择 ML 或 DL
   - mode=soft：soft 融合 y = p*dl + (1-p)*ml

输出：
- artifact_dir/test_selector_predictions.xlsx

用法：
python predict_gate_classifier.py --test_csv .\\without\\single_test_data_without.csv --artifact_dir .\\gate_fusion_out_classifier --dl_ckpt_dir .\\cv_models --one_hot_map .\\one_hot_map_without.pkl
"""

from __future__ import annotations

import argparse
import inspect
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

TARGET_COL = "纱强力"
BATCH_COL = "纱批"
RATIO_COL = "物料名称使用比例"

HVI_COLS_ALL = ["MIC", "MAT%", "LEN(INCH)", "STR(CN/TEX)"]
AFIS_COLS_ALL = ["AFIS-细度(MTEX)", "AFIS-成熟度", "AFIS-平均长度mm", "SFC(N)-%"]
CAT_COLS_CANDIDATES = ["纺纱方式", "梳棉工艺名", "精梳工艺名"]
NUM_COLS_CANDIDATES = ["纺纱纱支", "捻度", "实测单纱捻度", "纺纱股数", "单纱捻度", "股线捻度"]


def rel_err(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)


def within5(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(rel_err(y_true, y_pred) <= 0.05))


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


def _weighted_mean_ignore_zero(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = (values != 0) & ~np.isnan(values)
    if m.sum() == 0:
        return 0.0
    v = values[m]
    w = weights[m]
    s = float(np.sum(w))
    w = (np.ones_like(w) / len(w)) if s <= 0 else (w / s)
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
        g["norm_p"] = (1.0 / len(g)) if total_ratio <= 0 else (g[RATIO_COL] / total_ratio)

        row = {BATCH_COL: batch}
        for col in feature_cols:
            row[col] = _weighted_mean_ignore_zero(g[col].values, g["norm_p"].values)

        for c in NUM_COLS_CANDIDATES:
            if c in g.columns:
                row[c] = g[c].iloc[0]
        for c in CAT_COLS_CANDIDATES:
            row[c] = g[c].iloc[0] if c in g.columns else "UNKNOWN"

        row[TARGET_COL] = g[TARGET_COL].iloc[0] if TARGET_COL in g.columns else np.nan
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


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


def _weighted_stats(values: np.ndarray, weights: np.ndarray):
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
            mean, std, frac = _weighted_stats(g[c].values, w)
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


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _list_ckpts(ckpt_dir: Path) -> List[Path]:
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


def dl_predict_ensemble(
    df_raw: pd.DataFrame,
    ckpt_paths: List[Path],
    one_hot_map_path: Path,
    batch_size: int = 16,
    device: Optional[str] = None,
    output_scale: float = 100.0,
) -> pd.Series:
    import torch
    from torch.utils.data import DataLoader
    from model_i import Blendmapping

    device = _resolve_device(device)
    max_len_needed = int(df_raw.groupby(BATCH_COL).size().max())

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        tmp_csv = f.name
        df_raw.to_csv(tmp_csv, index=False)

    all_preds: Dict[str, List[float]] = {}
    try:
        from dataset_use import MyDataset as DS  # type: ignore

        for ckpt_path in ckpt_paths:
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

            max_padding = int(ckpt.get("max_padding", max_len_needed))
            max_len = max(max_padding, max_len_needed)

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
                        all_preds.setdefault(name, []).append(float(p))

        return pd.Series({k: float(np.mean(v)) for k, v in all_preds.items()}, dtype=float)

    finally:
        try:
            Path(tmp_csv).unlink(missing_ok=True)
        except Exception:
            pass


def selector_predict_proba(selector_art: dict, X: pd.DataFrame) -> np.ndarray:
    model = selector_art["selector_model"]
    pre = selector_art["selector_preprocessor"]
    num_cols = selector_art["selector_num_cols"]
    cat_cols = selector_art["selector_cat_cols"]

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
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--artifact_dir", type=str, required=True)
    ap.add_argument("--dl_ckpt_dir", type=str, required=True)
    ap.add_argument("--one_hot_map", type=str, required=True)
    ap.add_argument("--dl_batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--mode", type=str, default="select", choices=["select", "soft"])
    ap.add_argument("--threshold", type=float, default=None)  # 覆盖保存的阈值
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    selector_art = joblib.load(artifact_dir / "selector_model.pkl")
    ml_bundle = joblib.load(artifact_dir / "ml_refit_bundle.pkl")

    thr = float(selector_art["threshold"]) if args.threshold is None else float(args.threshold)
    dl_scale = float(selector_art.get("dl_output_scale", 100.0))

    df_raw = pd.read_csv(args.test_csv, low_memory=False)
    print(f"[Test] raw rows={len(df_raw)}, unique batches={df_raw[BATCH_COL].nunique()}")

    # ML predictions
    df_clean = clean_invalid_rows(df_raw, HVI_COLS_ALL, AFIS_COLS_ALL)
    batch_test = aggregate_by_batch_for_ml(df_clean).set_index(BATCH_COL, drop=False)

    num_features = list(ml_bundle.get("num_features", []))
    cat_features = list(ml_bundle.get("cat_features", []))

    for c in num_features:
        if c not in batch_test.columns:
            batch_test[c] = 0.0
    for c in cat_features:
        if c not in batch_test.columns:
            batch_test[c] = ""

    X_ml = batch_test[num_features + cat_features].copy()
    pred_ml = pd.Series(
        ml_bundle["model"].predict(ml_bundle["preprocessor"].transform(X_ml)),
        index=batch_test.index,
        dtype=float,
    )

    # DL ensemble
    ckpts = _list_ckpts(Path(args.dl_ckpt_dir))
    pred_dl = dl_predict_ensemble(
        df_raw=df_raw,
        ckpt_paths=ckpts,
        one_hot_map_path=Path(args.one_hot_map),
        batch_size=args.dl_batch_size,
        device=args.device,
        output_scale=dl_scale,
    )

    # selector features
    gate_feat = batch_gate_features(df_raw).set_index(BATCH_COL, drop=False)
    common = gate_feat.index.intersection(pred_ml.index).intersection(pred_dl.index)

    X_table = build_selector_feature_table(gate_feat.loc[common].reset_index(drop=True), pred_ml.loc[common], pred_dl.loc[common])

    p_dl = selector_predict_proba(selector_art, X_table.drop(columns=[BATCH_COL], errors="ignore"))
    choose_dl = (p_dl >= thr).astype(int)

    if args.mode == "select":
        y_pred = np.where(choose_dl == 1, pred_dl.loc[common].values, pred_ml.loc[common].values)
    else:
        y_pred = p_dl * pred_dl.loc[common].values + (1.0 - p_dl) * pred_ml.loc[common].values

    out_df = pd.DataFrame({
        BATCH_COL: common,
        "pred_ml": pred_ml.loc[common].values,
        "pred_dl": pred_dl.loc[common].values,
        "p_dl_better": p_dl,
        "choose_dl": choose_dl,
        "pred_final": y_pred,
    })

    if TARGET_COL in batch_test.columns:
        y_true = batch_test.loc[common, TARGET_COL].values.astype(float)
        out_df["y_true"] = y_true
        out_df["rel_err_final"] = rel_err(y_true, y_pred)
        out_df["rel_err_ml"] = rel_err(y_true, out_df["pred_ml"].values)
        out_df["rel_err_dl"] = rel_err(y_true, out_df["pred_dl"].values)

        print(f"[Test] Within5 ML    : {within5(y_true, out_df['pred_ml'].values):.4f}")
        print(f"[Test] Within5 DL    : {within5(y_true, out_df['pred_dl'].values):.4f}")
        print(f"[Test] Within5 Final : {within5(y_true, y_pred):.4f}")
        print(f"[Test] mode={args.mode}, threshold={thr}")

    excel_path = artifact_dir / "test_selector_predictions.xlsx"
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="predictions")
        if "y_true" in out_df.columns:
            total = len(out_df)
            count_5 = int(np.sum(out_df["rel_err_final"].values <= 0.05))
            summary_df = pd.DataFrame({
                "指标": ["总批次数(有真实值)", "偏差≤5%的批次数", "合格率"],
                "数量": [total, count_5, f"{count_5}/{total}={count_5/total:.4f}"],
            })
            summary_df.to_excel(writer, index=False, sheet_name="summary")

    print(f"[Test] saved: {excel_path}")


if __name__ == "__main__":
    main()
