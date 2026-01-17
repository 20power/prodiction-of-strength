# 纱线强力预测：ML + DL 门控融合（Gate Fusion）使用说明

> 适用场景：  
> 你已经有两个回归模型：  
> - 机器学习模型（LightGBM/XGBoost 等）在部分纱批上表现更好  
> - 深度学习模型（model_i + dataset）在另一部分纱批上表现更好  
> 但预测时没有真实标签，无法“先ML不合格再DL”。  
>  
> 本方案用一个 **门控器 Gate** 自动判断“更信 ML 还是更信 DL”，并输出融合预测：
> \[
> \hat y = w \cdot \hat y_{DL} + (1-w)\cdot \hat y_{ML}, \quad w\in[0,1]
> \]
> - w 越接近 1 → 更信 DL
> - w 越接近 0 → 更信 ML

---

## 1. 你需要准备的文件（你现在项目里已经基本都有）

### 1.1 数据文件
- 训练集：`single_train_data_without.csv`
- 测试集：`single_test_data_without.csv`

> 你说测试集共有 914 份纱批（以 “纱批” 列为单位）。

### 1.2 机器学习（ML）产物：bundle.pkl
你已经通过 `train_clean_pso_all.py` 得到了机器学习 bundle（通常路径类似）：
- `./pkl/yarn_strength_model.pkl`  
或你之前保存的其它 `*.pkl`（只要是 train_clean_pso_all 保存的 bundle 即可）

这个 bundle 一般包含：
- model（LightGBM/XGB 等）
- num_features / cat_features（特征列）
- metrics 等

### 1.3 深度学习（DL）产物：5 折最优权重 fold_*.pth
你已经通过 `train_blend_cross.py` 做了 5 折交叉验证训练，并保存了最优权重（要求命名类似）：
- `./cv_models/fold_1_best.pth`
- `./cv_models/fold_2_best.pth`
- `./cv_models/fold_3_best.pth`
- `./cv_models/fold_4_best.pth`
- `./cv_models/fold_5_best.pth`

> 注意：Gate 融合训练会用到 **每折对应 val 的 DL 预测**，因此最推荐你有这 5 个权重（越规范越好）。


### 1.4 代码依赖文件（必须能 import）
你项目根目录（或 python path）需要存在：
- `model_i.py`（包含 `Blendmapping`）
- `dataset.py`（包含 `MyDataset`）

> 因为推理时脚本会 `from model_i import Blendmapping`、`from dataset import MyDataset`。

---

## 2. 本仓库新增的两个脚本分别做什么？

你现在有两个脚本：

### 2.1 `train_gate_fusion.py`（训练）
它会做三件事：

1) **ML 的 OOF 预测**  
- 在训练集上做 5 折：每折训练 ML，再对该折 val 纱批预测  
- 拼起来得到每个训练纱批的 `pred_ml_oof`

2) **DL 的 OOF 预测**  
- 用 `fold_k_best.pth` 对第 k 折 val 纱批做预测  
- 拼起来得到每个训练纱批的 `pred_dl_oof`

3) **训练 Gate（门控器）**  
- Gate 学一个权重 w（0~1）  
- 输出融合预测 `pred_fused_oof`

最终保存：
- `gate_model.pkl`（门控器）
- `ml_refit_bundle.pkl`（用全训练集重新拟合后的 ML 模型，用于测试推理更稳）
- `train_oof_predictions.csv`（训练集 OOF 预测明细）

---

### 2.2 `predict_gate_fusion.py`（推理）
它会对 **测试集** 做：

1) 读 `ml_refit_bundle.pkl` → 计算测试集 `pred_ml`
2) 用 `cv_models` 下全部 `*.pth` 权重做 DL 预测并平均 → `pred_dl`
3) 读 `gate_model.pkl` → 输出权重 `w_hat`
4) 输出最终融合预测：
   \[
   pred\_fused = w\_hat*pred\_dl + (1-w\_hat)*pred\_ml
   \]

最后保存到：
- `gate_fusion_out/test_predictions.xlsx`

如果测试集 CSV 里存在真实列 `纱强力`，会自动统计：
- ML 合格率（误差≤5%）
- DL 合格率
- 融合合格率

---

## 3. 环境要求与安装

建议你在与你现有训练相同的环境运行（因为需要 torch、你的 dataset/model 文件）。

### 3.1 必需依赖
- Python 3.8+
- numpy, pandas
- scikit-learn
- joblib
- torch
- xlsxwriter（写 Excel）

安装示例：
```bash
pip install numpy pandas scikit-learn joblib xlsxwriter
