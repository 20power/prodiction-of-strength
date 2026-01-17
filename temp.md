python train_gate_fusion22.py --train_csv .\without\single_train_data_without.csv --ml_bundle .\pkl\yarn_strength_model_gpu.pkl --dl_ckpt_dir .\cv_models --out_dir .\gate_fusion_out --one_hot_map .\one_hot_map_without.pkl
python train_gate_fusion_v5.py --train_csv .\without\single_train_data_without.csv --ml_bundle .\pkl\yarn_strength_model_gpu.pkl --dl_ckpt_dir .\cv_models --out_dir .\gate_fusion_out --one_hot_map .\one_hot_map_without.pkl
python predict_gate_fusion_v5.py --test_csv .\without\single_test_data_without.csv --artifact_dir .\gate_fusion_out --dl_ckpt_dir .\cv_models --one_hot_map .\one_hot_map_without.pkl
python train_gate_classifier.py --train_csv .\without\single_train_data_without.csv --ml_bundle .\pkl\yarn_strength_model_gpu.pkl --dl_ckpt_dir .\cv_models  --out_dir .\gate_fusion_out  --one_hot_map .\one_hot_map_without.pkl
python predict_gate_classifier.py ^
  --test_csv .\without\single_test_data_without.csv ^
  --artifact_dir .\gate_fusion_out ^
  --dl_ckpt_dir .\cv_models ^
  --one_hot_map .\one_hot_map_without.pkl

运行代码：
训练 Gate 融合模型（线性）：
```bash
python train_gate_fusion_v5.py --train_csv .\without\single_train_data_without.csv --ml_bundle .\pkl\yarn_strength_model_gpu.pkl --dl_ckpt_dir .\cv_models --out_dir .\gate_fusion_out --one_hot_map .\one_hot_map_without.pkl
```
推理 Gate 融合模型（线性）：
```bash
python predict_gate_fusion_v5.py --test_csv .\without\single_test_data_without.csv --artifact_dir .\gate_fusion_out --dl_ckpt_dir .\cv_models --one_hot_map .\one_hot_map_without.pkl
```
训练 Gate 融合模型(gate内增加纱支和捻度)（线性）：
```bash
python train_gate_fusion_v6.py --train_csv .\without\single_train_data_without.csv --ml_bundle .\pkl\yarn_strength_model_gpu.pkl --dl_ckpt_dir .\cv_models --out_dir .\gate_fusion_out --one_hot_map .\one_hot_map_without.pkl
```
推理 Gate 融合模型(gate内增加纱支和捻度)（线性）：
```bash
python predict_gate_fusion_v6.py --test_csv .\without\single_test_data_without.csv --artifact_dir .\gate_fusion_out --dl_ckpt_dir .\cv_models --one_hot_map .\one_hot_map_without.pkl
```

训练 Gate 分类器：
```bash
python train_gate_classifier.py --train_csv .\without\single_train_data_without.csv --ml_bundle .\pkl\yarn_strength_model_gpu.pkl --dl_ckpt_dir .\cv_models  --out_dir .\gate_fusion_out_classifier  --one_hot_map .\one_hot_map_without.pkl
```
推理 Gate 分类器：
```bash
python predict_gate_classifier.py --test_csv .\without\single_test_data_without.csv --artifact_dir .\gate_fusion_out_classifier --dl_ckpt_dir .\cv_models --one_hot_map .\one_hot_map_without.pkl
```