
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import joblib

try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor  # type: ignore
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

TRAIN_DATA_PATH = "./without/single_train_data_without.csv"
PKL_DIR = Path("./pkl")
PKL_DIR.mkdir(parents=True, exist_ok=True)

BUNDLE_PATH = PKL_DIR / "yarn_strength_model.pkl"

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
    print(f"clean_invalid_rows: removed {len(removed_df)} rows, kept {len(kept_df)} rows.")

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

        row["纱强力"] = g["纱强力"].iloc[0]

        agg_rows.append(row)

    batch_df = pd.DataFrame(agg_rows)
    print(f"aggregate_by_batch: {len(batch_df)} batches.")
    return batch_df


def eval_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)
    within5 = np.mean(rel_err <= 0.05)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Within5": within5}


def within5_score(y_true, y_pred):
    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)
    return float(np.mean(rel_err <= 0.05))


def pso_optimize_model(
    model_name: str,
    model_ctor,
    param_bounds: dict,
    X_enc,
    y,
    n_splits=5,
    n_particles=15,
    n_iter=20,
    random_state=42,
    fixed_kwargs=None,
):
    if fixed_kwargs is None:
        fixed_kwargs = {}

    rng = np.random.RandomState(random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    param_names = list(param_bounds.keys())
    dim = len(param_names)

    lb = np.array([param_bounds[k][0] for k in param_names], dtype=float)
    ub = np.array([param_bounds[k][1] for k in param_names], dtype=float)

    w = 0.7
    c1 = 1.5
    c2 = 1.5

    pos = rng.uniform(lb, ub, size=(n_particles, dim))
    vel = np.zeros_like(pos)

    def vec_to_params(vec):
        params = {}
        for i, name in enumerate(param_names):
            v = float(vec[i])
            low, high = param_bounds[name]
            v = min(max(v, low), high)
            if any(key in name for key in ["n_estimators", "max_depth", "min_samples", "num_leaves", "min_child", "min_data"]):
                params[name] = int(round(v))
            else:
                params[name] = v
        return params

    def objective(vec):
        params = vec_to_params(vec)
        scores = []
        for train_idx, val_idx in kf.split(X_enc, y):
            X_tr, X_val = X_enc[train_idx], X_enc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = model_ctor(**fixed_kwargs, **params)
            model.fit(X_tr, y_tr)
            y_val_pred = model.predict(X_val)
            scores.append(within5_score(y_val, y_val_pred))

        mean_within5 = float(np.mean(scores))
        return 1.0 - mean_within5, mean_within5

    pbest_pos = pos.copy()
    pbest_loss = np.full(n_particles, np.inf)
    pbest_score = np.zeros(n_particles, dtype=float)

    gbest_pos = None
    gbest_loss = np.inf
    gbest_score = 0.0

    for i in range(n_particles):
        loss, sc = objective(pos[i])
        pbest_loss[i] = loss
        pbest_score[i] = sc
        if loss < gbest_loss:
            gbest_loss = loss
            gbest_score = sc
            gbest_pos = pos[i].copy()

    print(f"{model_name} PSO init best Within5 = {gbest_score:.4f}")

    for it in range(1, n_iter + 1):
        for i in range(n_particles):
            r1 = rng.rand(dim)
            r2 = rng.rand(dim)

            vel[i] = (
                w * vel[i]
                + c1 * r1 * (pbest_pos[i] - pos[i])
                + c2 * r2 * (gbest_pos - pos[i])
            )
            pos[i] = pos[i] + vel[i]
            pos[i] = np.minimum(np.maximum(pos[i], lb), ub)

            loss, sc = objective(pos[i])
            if loss < pbest_loss[i]:
                pbest_loss[i] = loss
                pbest_score[i] = sc
                pbest_pos[i] = pos[i].copy()

                if loss < gbest_loss:
                    gbest_loss = loss
                    gbest_score = sc
                    gbest_pos = pos[i].copy()

        print(f"{model_name} PSO iter {it:02d}: best Within5 = {gbest_score:.4f}")

    best_params = vec_to_params(gbest_pos)

    print(f"\n{model_name} PSO best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"{model_name} PSO CV Within5 = {gbest_score:.4f}")

    return best_params, gbest_score


def main():
    df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"raw train rows = {df.shape[0]}")

    removed_path = PKL_DIR / "removed_train_rows_pso_all.xlsx"
    df_clean = clean_invalid_rows(df, HVI_COLS_ALL, AFIS_COLS_ALL, removed_path)

    batch_df = aggregate_by_batch(df_clean)

    hvi_cols = [c for c in HVI_COLS_ALL if c in batch_df.columns]
    afis_cols = [c for c in AFIS_COLS_ALL if c in batch_df.columns]
    num_features = hvi_cols + afis_cols
    if "纺纱纱支" in batch_df.columns:
        num_features.append("纺纱纱支")
    if "捻度" in batch_df.columns:
        num_features.append("捻度")

    cat_features = []
    for c in ["纺纱方式", "梳棉工艺名", "精梳工艺名"]:
        if c in batch_df.columns:
            cat_features.append(c)

    X = batch_df[num_features + cat_features].copy()
    y = batch_df["纱强力"].values

    print("num_features:", num_features)
    print("cat_features:", cat_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    transformers = []
    if cat_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))
    if num_features:
        transformers.append(("num", StandardScaler(), num_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    print("fitting preprocessor on train...")
    preprocessor.fit(X_train)
    X_train_enc = preprocessor.transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    results = {}
    models = {}

    # GBR
    print("\n====== PSO + 5-fold for GradientBoostingRegressor ======")
    gbr_param_bounds = {
        "n_estimators": (200, 800),
        "learning_rate": (0.01, 0.10),
        "max_depth": (2, 5),
        "min_samples_leaf": (1, 30),
        "min_samples_split": (2, 40),
        "subsample": (0.6, 1.0),
    }
    gbr_fixed = {"random_state": 42}
    gbr_best_params, gbr_cv_within5 = pso_optimize_model(
        model_name="GBR",
        model_ctor=GradientBoostingRegressor,
        param_bounds=gbr_param_bounds,
        X_enc=X_train_enc,
        y=y_train,
        n_splits=5,
        n_particles=15,
        n_iter=20,
        random_state=42,
        fixed_kwargs=gbr_fixed,
    )
    gbr_final = GradientBoostingRegressor(**gbr_fixed, **gbr_best_params)
    gbr_final.fit(X_train_enc, y_train)
    y_pred_gbr = gbr_final.predict(X_test_enc)
    metrics_gbr = eval_regression(y_test, y_pred_gbr)
    results["GBR_PSO"] = metrics_gbr
    models["GBR_PSO"] = gbr_final
    print("GBR_PSO test metrics:", metrics_gbr)
    print(f"GBR_PSO CV Within5 ≈ {gbr_cv_within5*100:.2f}%")

    # XGBoost
    if HAS_XGB:
        print("\n====== PSO + 5-fold for XGBRegressor ======")
        xgb_param_bounds = {
            "n_estimators": (200, 800),
            "learning_rate": (0.01, 0.20),
            "max_depth": (3, 8),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "min_child_weight": (1, 10),
            "reg_lambda": (0.0, 10.0),
        }
        xgb_fixed = {
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        xgb_best_params, xgb_cv_within5 = pso_optimize_model(
            model_name="XGB",
            model_ctor=XGBRegressor,
            param_bounds=xgb_param_bounds,
            X_enc=X_train_enc,
            y=y_train,
            n_splits=5,
            n_particles=15,
            n_iter=20,
            random_state=52,
            fixed_kwargs=xgb_fixed,
        )
        xgb_final = XGBRegressor(**xgb_fixed, **xgb_best_params)
        xgb_final.fit(X_train_enc, y_train)
        y_pred_xgb = xgb_final.predict(X_test_enc)
        metrics_xgb = eval_regression(y_test, y_pred_xgb)
        results["XGB_PSO"] = metrics_xgb
        models["XGB_PSO"] = xgb_final
        print("XGB_PSO test metrics:", metrics_xgb)
        print(f"XGB_PSO CV Within5 ≈ {xgb_cv_within5*100:.2f}%")
    else:
        print("\nXGBoost not installed, skip XGBRegressor PSO.")

    # LightGBM
    if HAS_LGBM:
        print("\n====== PSO + 5-fold for LGBMRegressor ======")
        lgb_param_bounds = {
            "n_estimators": (200, 800),
            "learning_rate": (0.01, 0.20),
            "num_leaves": (31, 127),
            "max_depth": (3, 8),
            "min_data_in_leaf": (10, 80),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.5, 1.0),
        }
        lgb_fixed = {
            "objective": "regression",
            "random_state": 42,
        }
        lgb_best_params, lgb_cv_within5 = pso_optimize_model(
            model_name="LGBM",
            model_ctor=LGBMRegressor,
            param_bounds=lgb_param_bounds,
            X_enc=X_train_enc,
            y=y_train,
            n_splits=5,
            n_particles=15,
            n_iter=20,
            random_state=62,
            fixed_kwargs=lgb_fixed,
        )
        lgb_final = LGBMRegressor(**lgb_fixed, **lgb_best_params)
        lgb_final.fit(X_train_enc, y_train)
        y_pred_lgb = lgb_final.predict(X_test_enc)
        metrics_lgb = eval_regression(y_test, y_pred_lgb)
        results["LGBM_PSO"] = metrics_lgb
        models["LGBM_PSO"] = lgb_final
        print("LGBM_PSO test metrics:", metrics_lgb)
        print(f"LGBM_PSO CV Within5 ≈ {lgb_cv_within5*100:.2f}%")
    else:
        print("\nLightGBM not installed, skip LGBMRegressor PSO.")

    print("\n===== Test metrics comparison =====")
    for name, m in results.items():
        print(
            f"{name:10s} | R2={m['R2']:.4f}  "
            f"RMSE={m['RMSE']:.4f}  "
            f"MAE={m['MAE']:.4f}  "
            f"Within5={m['Within5']*100:.2f}%"
        )

    best_name = max(results.items(), key=lambda kv: kv[1]["Within5"])[0]
    best_model = models[best_name]
    best_metrics = results[best_name]

    print(f"\nFinal chosen model: {best_name}")
    print("Final test metrics:", best_metrics)

    if HAS_SHAP:
        try:
            print("\nComputing SHAP importance for final model...")
            n_sample = min(500, X_train_enc.shape[0])
            X_bg = X_train_enc[:n_sample]
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_bg)

            try:
                feature_names_enc = preprocessor.get_feature_names_out()
            except Exception:
                feature_names_enc = [f"f{i}" for i in range(X_train_enc.shape[1])]

            cleaned_names = []
            for name in feature_names_enc:
                if name.startswith("cat__"):
                    new_name = name.replace("cat__", "")
                elif name.startswith("num__"):
                    new_name = name.replace("num__", "")
                else:
                    new_name = name
                cleaned_names.append(new_name)
            feature_names_enc = cleaned_names

            shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
            df_shap = pd.DataFrame({
                "feature": feature_names_enc,
                "mean_abs_shap": shap_abs_mean
            }).sort_values("mean_abs_shap", ascending=False)

            shap_csv_path = PKL_DIR / "shap_importance_pso_all.csv"
            df_shap.to_csv(shap_csv_path, index=False, encoding="utf-8-sig")
            print(f"SHAP importance saved to: {shap_csv_path}")

            try:
                import matplotlib.pyplot as plt
                from matplotlib import rcParams
                rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei']
                rcParams['axes.unicode_minus'] = False

                shap.summary_plot(shap_values, X_bg, feature_names=feature_names_enc, show=False)
                plt.tight_layout()
                shap_png_path = PKL_DIR / "shap_summary_pso_all.png"
                plt.savefig(shap_png_path, dpi=300)
                plt.close()
                print(f"SHAP summary plot saved to: {shap_png_path}")
            except Exception as e:
                print(f"Plot SHAP failed: {e}")
        except Exception as e:
            print(f"SHAP computation failed: {e}")
    else:
        print("\nshap not installed, skip SHAP analysis.")

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "metrics": best_metrics,
        "preprocessor": preprocessor,
        "num_features": num_features,
        "cat_features": cat_features,
    }
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"\nPSO-all best model bundle saved to: {BUNDLE_PATH}")


if __name__ == "__main__":
    main()
