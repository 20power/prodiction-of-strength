
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PKL_DIR = Path("./pkl")
BUNDLE_PATH = PKL_DIR / "yarn_strength_model_fix_cpu.pkl"
TEST_DATA_PATH = "./without/single_test_data_without.csv"

HVI_COLS_ALL = ["MIC", "MAT%", "LEN(INCH)", "STR(CN/TEX)"]
AFIS_COLS_ALL = ["AFIS-细度(MTEX)", "AFIS-成熟度", "AFIS-平均长度mm", "SFC(N)-%"]


def clean_invalid_rows(df: pd.DataFrame,
                       hvi_cols: list,
                       afis_cols: list,
                       save_path: Path) -> pd.DataFrame:
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

    cond1 = all_hvi_zero & any_afis_zero
    cond2 = all_afis_zero & any_hvi_zero
    remove_mask = cond1 | cond2

    removed_df = df[remove_mask].copy()
    kept_df = df[~remove_mask].copy()

    if not removed_df.empty:
        removed_df.to_excel(save_path, index=False)
    print(f"[预测] 清洗无效行：共删除 {len(removed_df)} 行，保留 {len(kept_df)} 行。")

    return kept_df


def weighted_mean_ignore_zero(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = (values != 0) & ~np.isnan(values)
    if not np.any(mask):
        return 0.0
    v = values[mask]
    w = weights[mask]
    w = w / w.sum()
    return float(np.sum(v * w))


def aggregate_by_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df["物料名称使用比例"].max() > 1.5:
        df["物料名称使用比例"] = df["物料名称使用比例"] / 100.0

    groups = df.groupby("纱批")
    agg_rows = []

    hvi_cols = [c for c in HVI_COLS_ALL if c in df.columns]
    afis_cols = [c for c in AFIS_COLS_ALL if c in df.columns]
    feature_cols = hvi_cols + afis_cols

    for batch, g in groups:
        g = g.copy()
        total_ratio = g["物料名称使用比例"].sum()
        if total_ratio <= 0:
            g["norm_p"] = 1.0 / len(g)
        else:
            g["norm_p"] = g["物料名称使用比例"] / total_ratio

        row = {"纱批": batch}

        for col in feature_cols:
            vals = g[col].values
            weights = g["norm_p"].values
            row[col] = weighted_mean_ignore_zero(vals, weights)

        if "纺纱纱支" in g.columns:
            row["纺纱纱支"] = g["纺纱纱支"].iloc[0]

        if "实测单纱捻度" in g.columns:
            row["捻度"] = g["实测单纱捻度"].iloc[0]
        elif "捻度" in g.columns:
            row["捻度"] = g["捻度"].iloc[0]

        row["纺纱方式"] = g["纺纱方式"].iloc[0] if "纺纱方式" in g.columns else ""
        row["梳棉工艺名"] = g["梳棉工艺名"].iloc[0] if "梳棉工艺名" in g.columns else ""
        row["精梳工艺名"] = g["精梳工艺名"].iloc[0] if "精梳工艺名" in g.columns else ""

        if "纱强力" in g.columns:
            row["纱强力"] = g["纱强力"].iloc[0]
        else:
            row["纱强力"] = np.nan

        agg_rows.append(row)

    batch_df = pd.DataFrame(agg_rows)
    print(f"[预测] 按纱批聚合完成，共 {len(batch_df)} 个批次。")
    return batch_df


def main():
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    num_features = bundle["num_features"]
    cat_features = bundle["cat_features"]

    df = pd.read_csv(TEST_DATA_PATH)
    print(f"原始测试数据：{df.shape[0]} 行")

    removed_path = PKL_DIR / "removed_test_rows.xlsx"
    df_clean = clean_invalid_rows(df, HVI_COLS_ALL, AFIS_COLS_ALL, removed_path)

    batch_df = aggregate_by_batch(df_clean)

    all_feat_cols = num_features + cat_features
    X = batch_df[all_feat_cols].copy()
    y_true = batch_df["纱强力"].values

    X_enc = preprocessor.transform(X)
    y_pred = model.predict(X_enc)

    eps = 1e-8
    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + eps)

    mask_valid = ~np.isnan(y_true)
    valid_rel_err = rel_err[mask_valid]

    total_valid = int(mask_valid.sum())
    count_5 = int(np.sum(valid_rel_err <= 0.05))
    count_7 = int(np.sum(valid_rel_err <= 0.07))
    count_10 = int(np.sum(valid_rel_err <= 0.10))

    print("\n===== 预测偏差统计（按纱批） =====")
    print(f"总批次数（有真实值）: {total_valid}")
    print(f"偏差 ≤5% 的批次数:  {count_5}")
    print(f"偏差 ≤7% 的批次数:  {count_7}")
    print(f"偏差 ≤10% 的批次数: {count_10}")

    PKL_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = PKL_DIR / "predict_results.xlsx"

    results_df = pd.DataFrame({
        "纱批": batch_df["纱批"],
        "真实纱强力": y_true,
        "预测纱强力": y_pred,
        "相对误差": rel_err,
    })

    summary_df = pd.DataFrame({
        "指标": ["总批次数(有真实值)", "偏差≤5%的批次数", "偏差≤7%的批次数", "偏差≤10%的批次数"],
        "数量": [total_valid, count_5, count_7, count_10],
    })

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        results_df.to_excel(writer, index=False, sheet_name="predictions")
        summary_df.to_excel(writer, index=False, sheet_name="summary")

    print(f"\n预测结果和统计已保存到: {excel_path.resolve()}")


if __name__ == "__main__":
    main()
