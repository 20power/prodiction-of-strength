# predict_gate_fusion_v6.py
# -*- coding: utf-8 -*-
"""
门控融合推理脚本（Gate Fusion Inference）

你需要先运行 train_gate_fusion.py（或你已有的训练脚本）产出：
- artifact_dir/gate_model.pkl
- artifact_dir/ml_refit_bundle.pkl

然后本脚本会：
1) 用 ml_refit_bundle.pkl 在测试集上做 ML 预测（按纱批聚合）
2) 用 dl_ckpt_dir 下的多个 fold_*.pth（或 *.pth）对测试集做 DL 预测（按纱批，默认多权重平均）
3) 用 gate_model.pkl 输出权重 w，并做融合：
      y_fused = w * y_dl + (1-w) * y_ml
4) 输出 artifact_dir/test_predictions.xlsx

运行示例：
python predict_gate_fusion.py \
  --test_csv ./without/single_test_data_without.csv \
  --artifact_dir ./gate_fusion_out \
  --dl_ckpt_dir ./cv_models \
  --one_hot_map ../one_hot_map.pkl

注意：
- 本脚本不依赖你项目里的 dataset.py（你当前版本 dataset.py 内部存在 np.concat / base_hvi_cols 等问题会报错）
- DL 预测会把网络输出 *100（因为你训练时把纱强力除以 100）
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# 这些列名尽量和你项目一致
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


# ----------------- ML：清洗+聚合（复制你 predict_clean 的逻辑） -----------------
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
    """
    和 predict_clean.py 对齐：
    - 若比例是 0~100 则除 100
    - 按纱批：按配比归一化后对 HVI/AFIS 做“忽略0的加权均值”
    - 纺纱纱支、捻度等取首行
    """
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

        # 捻度字段兼容
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
    
    # --- v6: 加入工艺参数（纱支/捻度）作为 gate 的 meta 输入 ---
    proc_cols = [c for c in ["纺纱纱支", "捻度"] if c in df.columns]
    if proc_cols:
        proc = g[proc_cols].first().copy()
        for c in proc_cols:
            s = proc[c]
            # 先尝试直接转数值；若失败（如 "32S"），再提取数字部分
            s_num = pd.to_numeric(s, errors="coerce")
            if float(s_num.isna().mean()) > 0.5:
                s_num = pd.to_numeric(
                    s.astype(str).str.replace(r"[^0-9\.\+\-]", "", regex=True),
                    errors="coerce"
                )
            proc[c] = s_num.fillna(0.0).astype(float)
    else:
        proc = pd.DataFrame(index=patch_count.index)

    meta = pd.concat([patch_count, meta, proc], axis=1)
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
        "纺纱纱支", "捻度",
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_cols]


# ----------------- DL：不依赖 dataset.py 的推理 Dataset -----------------
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


def _maybe_create_afis_alias_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容两套 AFIS 命名：
    - 你 ML 脚本里是 "AFIS-细度(MTEX)" 等
    - 你 DL 数据集里常用 'afis-细度' 等
    如果只有大写那套，就复制一份小写别名，便于 DL 处理。
    """
    df = df.copy()
    mapping = {
        "AFIS-细度(MTEX)": "afis-细度",
        "AFIS-成熟度": "afis-成熟度",
        "AFIS-平均长度mm": "afis-均长",
        "SFC(N)-%": "afis-短绒率",   # 近似映射（如果你真实列是 AFIS 短绒率，请替换）
    }
    for src, dst in mapping.items():
        if dst not in df.columns and src in df.columns:
            df[dst] = df[src]
    return df


class YarnDLInferDataset:
    """
    尽量复刻你 MyDataset 的输入格式，但修复 bug 且支持自定义 one_hot_map。
    输出：
      - is_twist=False : feature1, p, comber_feature, sp_ps, y_dummy
      - is_twist=True  : feature1, p, comber_feature, sp_ps, twist, y_dummy
    """
    def __init__(self, df_raw: pd.DataFrame, one_hot_map: dict, max_len: int, is_twist: bool, d_yc: int):
        self.df = _maybe_create_afis_alias_cols(df_raw).copy()
        self.one_hot_map = one_hot_map
        self.max_len = int(max_len)
        self.is_twist = bool(is_twist)

        # 归一化比例到 0~1
        if RATIO_COL in self.df.columns and self.df[RATIO_COL].max() > 1.5:
            self.df[RATIO_COL] = self.df[RATIO_COL] / 100.0

        # 标签也按训练一致缩放（若存在）
        if TARGET_COL in self.df.columns and self.df[TARGET_COL].max() > 5:
            # 只有当看起来是原始量纲时才缩放；否则默认已缩放
            self.df["_y_scaled"] = self.df[TARGET_COL] / 100.0
        elif TARGET_COL in self.df.columns:
            self.df["_y_scaled"] = self.df[TARGET_COL]
        else:
            self.df["_y_scaled"] = 0.0

        self.unique_batches = self.df[BATCH_COL].unique().tolist()

        # 固定列
        self.base_hvi_cols = ["MIC", "MAT%", "LEN(INCH)", "SFI(%)", "STR(CN/TEX)"]
        self.afis_cols = ["afis-细度", "afis-成熟度", "afis-均长", "afis-短绒率", "afis-强度"]
        self.comber_cols = ["梳棉工艺名", "精梳工艺名"]
        self.single_twist_cols = ["纺纱股数", "单纱捻度"]
        self.ply_twist_cols = ["纺纱股数", "股线捻度"]

    def __len__(self):
        return len(self.unique_batches)

    def _get_onehot(self, key) -> np.ndarray:
        # one_hot_map 的 key 可能是字符串
        return np.asarray(self.one_hot_map[str(key)], dtype=np.float32)

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

        # 若某些批次行数超过 max_len，则截断（和训练一致的话通常不会超）
        if length > self.max_len:
            g = g.iloc[:self.max_len].copy()
            length = self.max_len

        # 选择 HVI or AFIS（与你原意一致：若 HVI 合计为0，则用 AFIS）
        use_afis = False
        present_hvi = [c for c in self.base_hvi_cols if c in g.columns]
        if present_hvi:
            use_afis = float(np.nansum(g[present_hvi].values.astype(float))) == 0.0

        # 对 HVI/AFIS 进行“按列除以列和”的归一化
        g = self._norm_cols_by_sum(g, self.base_hvi_cols)
        g = self._norm_cols_by_sum(g, self.afis_cols)

        feat_cols = self.afis_cols if use_afis else self.base_hvi_cols
        feat_cols_present = [c for c in feat_cols if c in g.columns]
        if len(feat_cols_present) != 5:
            # 不足就补0列，保证维度稳定
            for c in feat_cols:
                if c not in g.columns:
                    g[c] = 0.0
            feat_cols_present = feat_cols

        # 棉等级 one-hot + 5 维特征 => 9 维
        if "棉等级" not in g.columns:
            g["棉等级"] = "UNKNOWN"
        degree_oh = np.stack([self._get_onehot(x) for x in g["棉等级"].values], axis=0)  # [L, d_deg]
        feat_vals = g[feat_cols_present].values.astype(np.float32)                        # [L, 5]
        feature1 = np.concatenate([degree_oh, feat_vals], axis=1).astype(np.float32)      # [L, 9]
        feature1 = torch.from_numpy(feature1).float()

        # 梳棉/精梳 one-hot 拼接 => 20 维（假设每个是 10 维）
        comber_parts = []
        for c in self.comber_cols:
            if c not in g.columns:
                g[c] = "UNKNOWN"
            oh = np.stack([self._get_onehot(x) for x in g[c].values], axis=0)
            comber_parts.append(oh)
        comber_feature = np.concatenate(comber_parts, axis=1).astype(np.float32)          # [L, 20]
        comber_feature = torch.from_numpy(comber_feature).float()

        # 配比 p
        if RATIO_COL not in g.columns:
            g[RATIO_COL] = 0.0
        p = g[RATIO_COL].values.astype(np.float32)
        # 让每个批次和为1（更符合 DirichletAttentionMixer 的假设）
        s = float(np.sum(p))
        if s > 0:
            p = p / s
        p = torch.from_numpy(p).float()

        # sp_ps：纺纱方式 one-hot + ps（根据 is_twist 选择）
        if "纺纱方式" not in g.columns:
            g["纺纱方式"] = "UNKNOWN"
        sp = self._get_onehot(g["纺纱方式"].values[0]).astype(np.float32)

        # ps
        if "纺纱纱支" not in g.columns:
            g["纺纱纱支"] = 0.0

        if self.is_twist:
            # is_twist=True：ps 只用纱支（训练时 d_yc 常为 3）
            ps = np.array([float(g["纺纱纱支"].values[0])], dtype=np.float32)
        else:
            # is_twist=False：ps 用 [纱支, 纺纱股数, 单纱捻度]（训练时 d_yc 常为 5）
            for c in self.single_twist_cols:
                if c not in g.columns:
                    g[c] = 0.0
            ps = g[["纺纱纱支"] + self.single_twist_cols].values[0].astype(np.float32)

        sp_ps = np.concatenate([sp, ps], axis=0).astype(np.float32)
        sp_ps = torch.from_numpy(sp_ps).float()

        # twist（仅 is_twist=True 时需要）
        for c in self.ply_twist_cols:
            if c not in g.columns:
                g[c] = 0.0
        twist = g[self.ply_twist_cols].values[0].astype(np.float32)
        twist = torch.from_numpy(twist).float()

        # y dummy（label 缩放后的）
        y = float(g["_y_scaled"].values[0])
        y = torch.tensor(y).float()

        # padding
        pad1 = tnn.ZeroPad2d(padding=(0, 0, 0, (self.max_len - length)))
        feature1 = pad1(feature1)
        comber_feature = pad1(comber_feature)

        pad2 = tnn.ZeroPad2d(padding=(0, (self.max_len - length)))
        p = pad2(p)

        if self.is_twist:
            return feature1, p, comber_feature, sp_ps, twist, y
        return feature1, p, comber_feature, sp_ps, y


def _build_mydataset(DS, csv_path: str, max_len: int, is_twist: bool):
    import inspect
    sig = inspect.signature(DS.__init__)
    params = sig.parameters
    kwargs = {}
    if "max_len" in params:
        kwargs["max_len"] = int(max_len)
    if "is_twist" in params:
        kwargs["is_twist"] = bool(is_twist)
    return DS(csv_path, **kwargs)


def dl_predict_from_df(
    df_raw: pd.DataFrame,
    ckpt_paths: List[Path],
    one_hot_map_path: Path,
    batch_size: int = 16,
    device: Optional[str] = None,
    output_scale: float = 100.0,
) -> Dict[str, float]:
    """
    返回：{纱批 -> 预测纱强力(原始量纲)}，默认多权重平均。
    优先使用 dataset_use.MyDataset（与训练一致）；失败则回退内置 Dataset。
    """
    import torch
    from torch.utils.data import DataLoader
    from model_i import Blendmapping

    device = _resolve_device(device)

    if len(ckpt_paths) == 0:
        raise FileNotFoundError("没有找到任何 DL checkpoint（.pth）。请检查 --dl_ckpt_dir。")

    max_len_needed = int(df_raw.groupby(BATCH_COL).size().max())

    all_preds: Dict[str, List[float]] = {}

    # 写一次临时 csv 给 dataset_use.MyDataset
    tmp_csv = None
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            tmp_csv = f.name
            df_raw.to_csv(tmp_csv, index=False)
    except Exception:
        tmp_csv = None

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

        used_fallback = False

        # 1) dataset_use
        try:
            from dataset_use import MyDataset as DS  # type: ignore
            if tmp_csv is None:
                raise RuntimeError("无法创建临时 CSV，不能使用 dataset_use.MyDataset。")
            ds = _build_mydataset(DS, tmp_csv, max_len=max_len, is_twist=is_twist)
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

        except Exception as e:
            used_fallback = True
            print(f"[DL] WARN: dataset_use.MyDataset 失败，回退内置 Dataset。原因: {type(e).__name__}: {e}")

        # 2) fallback internal
        if used_fallback:
            one_hot_map = pd.read_pickle(one_hot_map_path)
            ds = YarnDLInferDataset(df_raw, one_hot_map=one_hot_map, max_len=max_len, is_twist=is_twist, d_yc=int(ckpt["d_yc"]))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

            with torch.no_grad():
                start = 0
                for batch in loader:
                    if is_twist:
                        x, prop, comber, sp_ps, twist, _y = batch
                        out = model(x.to(device), prop.to(device), comber.to(device), sp_ps.to(device), twist.to(device))
                    else:
                        x, prop, comber, sp_ps, _y = batch
                        out = model(x.to(device), prop.to(device), comber.to(device), sp_ps.to(device))

                    out = out.view(-1).detach().cpu().numpy().astype(float) * float(output_scale)
                    bs = len(out)
                    names = ds.unique_batches[start:start+bs]
                    start += bs
                    for name, p in zip(names, out):
                        all_preds.setdefault(str(name), []).append(float(p))

    if tmp_csv:
        try:
            Path(tmp_csv).unlink(missing_ok=True)
        except Exception:
            pass

    return {k: float(np.mean(v)) for k, v in all_preds.items()}


# ----------------- 主流程 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--artifact_dir", type=str, required=True)
    ap.add_argument("--dl_ckpt_dir", type=str, default="./cv_models")
    ap.add_argument("--dl_batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--one_hot_map", type=str, default="../one_hot_map.pkl")
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    gate_art = joblib.load(artifact_dir / "gate_model.pkl")
    gate_model = gate_art["gate_model"]
    gate_cols = gate_art["gate_feature_cols"]

    ml_bundle = joblib.load(artifact_dir / "ml_refit_bundle.pkl")
    num_features = list(ml_bundle.get("num_features", []))
    cat_features = list(ml_bundle.get("cat_features", []))

    df_test_raw = pd.read_csv(args.test_csv, low_memory=False)
    print(f"[Test] raw rows={len(df_test_raw)}, unique batches={df_test_raw[BATCH_COL].nunique()}")

    # meta
    meta_test = batch_meta_features(df_test_raw).set_index(BATCH_COL, drop=False)

    # ML：清洗+聚合
    df_test_clean = clean_invalid_rows(df_test_raw, HVI_COLS_ALL, AFIS_COLS_ALL)
    batch_test = aggregate_by_batch_for_ml(df_test_clean).set_index(BATCH_COL, drop=False)

    # ML 预测
    # 补齐列
    for c in num_features:
        if c not in batch_test.columns:
            batch_test[c] = 0.0
    for c in cat_features:
        if c not in batch_test.columns:
            batch_test[c] = ""

    X = batch_test[num_features + cat_features].copy()
    pre = ml_bundle["preprocessor"]
    model = ml_bundle["model"]
    pred_ml = pd.Series(model.predict(pre.transform(X)), index=batch_test.index, dtype=float)

    # DL 预测（多个 ckpt 平均）
    ckpts = _list_ckpts(Path(args.dl_ckpt_dir))
    pred_dl_map = dl_predict_from_df(
        df_raw=df_test_raw,
        ckpt_paths=ckpts,
        one_hot_map_path=Path(args.one_hot_map),
        batch_size=args.dl_batch_size,
        device=args.device,
    )
    pred_dl = pd.Series(pred_dl_map, dtype=float)
    pred_dl.index.name = BATCH_COL

    # 对齐并融合
    common = meta_test.index.intersection(pred_ml.index).intersection(pred_dl.index)
    meta_i = meta_test.loc[common].reset_index(drop=True)

    X_gate = build_gate_features(meta_i, pred_ml.loc[common], pred_dl.loc[common])
    for c in gate_cols:
        if c not in X_gate.columns:
            X_gate[c] = 0.0
    X_gate = X_gate[gate_cols]

    w_hat = np.clip(gate_model.predict(X_gate), 0.0, 1.0)
    y_fused = w_hat * pred_dl.loc[common].values + (1.0 - w_hat) * pred_ml.loc[common].values

    out_df = pd.DataFrame({
        BATCH_COL: common,
        "pred_ml": pred_ml.loc[common].values,
        "pred_dl": pred_dl.loc[common].values,
        "w_hat": w_hat,
        "pred_fused": y_fused,
    })

    # 若测试集有真实值，输出统计
    if TARGET_COL in batch_test.columns:
        y_true = batch_test.loc[common, TARGET_COL].values.astype(float)
        out_df["y_true"] = y_true
        out_df["rel_err_fused"] = rel_err(y_true, y_fused)
        out_df["rel_err_ml"] = rel_err(y_true, out_df["pred_ml"].values)
        out_df["rel_err_dl"] = rel_err(y_true, out_df["pred_dl"].values)

        print(f"[Test] Within5 ML   : {within5(y_true, out_df['pred_ml'].values):.4f}")
        print(f"[Test] Within5 DL   : {within5(y_true, out_df['pred_dl'].values):.4f}")
        print(f"[Test] Within5 Fused: {within5(y_true, y_fused):.4f}")

    excel_path = artifact_dir / "test_predictions.xlsx"
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="predictions")

        if "y_true" in out_df.columns:
            total = len(out_df)
            count_5 = int(np.sum(out_df["rel_err_fused"].values <= 0.05))
            summary_df = pd.DataFrame({
                "指标": ["总批次数(有真实值)", "偏差≤5%的批次数", "合格率"],
                "数量": [total, count_5, f"{count_5}/{total}={count_5/total:.4f}"],
            })
            summary_df.to_excel(writer, index=False, sheet_name="summary")

    print(f"[Test] saved: {excel_path}")


if __name__ == "__main__":
    main()