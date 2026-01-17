import os
import time
import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torcheval.metrics.functional import r2_score, mean_squared_error
from model_i import Blendmapping
from dataset import MyDataset
import warnings

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(model, dataloader, device, is_test=False, is_twist=False):
    res = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            if is_twist:
                x, prop, x1, sp_ps, twist, y = batch
                twist = twist.to(device)
                pred = model(x.to(device), prop.to(device), x1.to(device), sp_ps.to(device), twist)
            else:
                x, prop, x1, sp_ps, y = batch
                pred = model(x.to(device), prop.to(device), x1.to(device), sp_ps.to(device))
            if is_test:
                y = y.numpy() * 100.
                pred = pred.view(-1).detach().cpu().numpy() * 100
                diff = np.abs(y - pred)
                res.extend(pred)
                labels.extend(y)
            else:
                res.extend(pred.view(-1).detach().cpu().numpy())
                labels.extend(y.view(-1).numpy())
    res = torch.tensor(res).to(device)
    labels = torch.tensor(labels).to(device)

    mse = mean_squared_error(res, labels).detach().cpu().numpy()
    r2 = r2_score(res, labels).detach().cpu().numpy()
    if is_test:
        return mse,r2,res.detach().cpu().numpy(),labels.detach().cpu().numpy()
    return mse,r2

def train_one_fold(fold_idx, train_loader, val_loader, config, max_patch):
    device = config['device']
    is_twist = config['is_twist']
    
    model = Blendmapping(
        config['d_model'],
        config['hvi_num'],
        config['comber_num'],
        config['d_yc'],
        config['d_y'],
        config['N'],
        config['heads'],
        dropout=config['dropout'],
        is_twist=is_twist,
        use_dirichlet=True
    )
    model = model.to(device)
    
    loss_func = torch.nn.MSELoss()
    kl_weight = 1e-3
    
    optimizer = AdamW(params=model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=100, factor=0.9, min_lr=7e-5
    )
    
    MSE = 1e8
    best_r2 = -1e8
    
    for epoch in range(config['num_epochs']):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            if is_twist:
                x, prop, x1, sp_ps, twist, y = batch
                x, prop, x1, sp_ps, twist, y = \
                    x.to(device), prop.to(device), x1.to(device), sp_ps.to(device), twist.to(device), y.to(device)

                if getattr(model, "use_dirichlet", False):
                    outputs, kl = model(x, prop, x1, sp_ps, twist, return_kl=True)
                    outputs = outputs.squeeze()
                    loss = loss_func(outputs, y) + kl_weight * kl
                else:
                    outputs = model(x, prop, x1, sp_ps, twist).squeeze()
                    loss = loss_func(outputs, y)
            else:
                x, prop, x1, sp_ps, y = batch
                x, prop, x1, sp_ps, y = \
                    x.to(device), prop.to(device), x1.to(device), sp_ps.to(device), y.to(device)

                if getattr(model, "use_dirichlet", False):
                    outputs, kl = model(x, prop, x1, sp_ps, return_kl=True)
                    outputs = outputs.squeeze()
                    loss = loss_func(outputs, y) + kl_weight * kl
                else:
                    outputs = model(x, prop, x1, sp_ps).squeeze()
                    loss = loss_func(outputs, y)

            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            mse, r2 = test(model, val_loader, device, is_twist=is_twist)
            print(f"Fold: {fold_idx} Epoch: {epoch}, testing mse:{mse:.5f}, R2score: {r2:.5f} LR: {optimizer.param_groups[0]['lr']}")
            
            if r2 > 0.92 and mse < MSE:
                MSE = mse
                best_r2 = r2
                # Save best model for this fold
                save_path = f'./cv_models/fold_{fold_idx}_best.pth'
                torch.save({
                    'model': model.state_dict(),
                    'd_model': config['d_model'], 'heads': config['heads'],
                    'N': config['N'], 'hvi_num': config['hvi_num'],
                    'comber_num': config['comber_num'],
                    'd_yc': config['d_yc'], 'd_y': config['d_y'],
                    'max_padding': max_patch, 'is_twist': is_twist
                }, save_path)
                
        # scheduler.step(metrics=MSE) # Kept commented out as in original code

    return MSE, best_r2

if __name__ == '__main__':
    seed_everything(42)
    
    os.makedirs('./cv_models', exist_ok=True)
    
    # Configuration
    DATA_PATH = r"./without1/single_train_data_without.csv"
    N_FOLDS = 5
    
    config = {
        'num_epochs': 800,
        'd_model': 192,
        'heads': 2,
        'N': 2,
        'is_twist': False,
        'learning_rate': 3e-4,
        'weight_decay': 1e-3,
        'hvi_num': 9,
        'comber_num': 20,
        'd_yc': 5, # 3 if is_twist else 5
        'd_y': 1,
        'batch_size': 8,
        'dropout': 0.05,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    }
    
    # 1. Load full data to get unique batches and max_patch
    full_dataset = MyDataset(DATA_PATH, max_len=None, is_twist=config['is_twist'])
    max_patch = full_dataset.get_max_patch_count()
    print(f"Max patch count: {max_patch}")
    
    df_full = pd.read_csv(DATA_PATH)
    unique_batches = df_full['纱批'].unique()
    print(f"Total unique batches: {len(unique_batches)}")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_batches)):
        print(f"\n========== Starting Fold {fold + 1}/{N_FOLDS} ==========")
        train_batches = unique_batches[train_idx]
        val_batches = unique_batches[val_idx]
        
        # Filter data
        df_train = df_full[df_full['纱批'].isin(train_batches)]
        df_val = df_full[df_full['纱批'].isin(val_batches)]
        
        # Create temporary CSV files for MyDataset
        train_csv = f'./temp_train_fold_{fold}.csv'
        val_csv = f'./temp_val_fold_{fold}.csv'
        df_train.to_csv(train_csv, index=False)
        df_val.to_csv(val_csv, index=False)
        
        # Create Datasets and Loaders
        train_dataset = MyDataset(train_csv, max_len=max_patch, is_twist=config['is_twist'])
        val_dataset = MyDataset(val_csv, max_len=max_patch, is_twist=config['is_twist'])
        
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        test_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        
        # Train
        best_mse, best_r2 = train_one_fold(fold + 1, train_dataloader, test_dataloader, config, max_patch)
        fold_results.append({'fold': fold + 1, 'mse': best_mse, 'r2': best_r2})
        
        # Clean up temp files
        if os.path.exists(train_csv):
            os.remove(train_csv)
        if os.path.exists(val_csv):
            os.remove(val_csv)
            
    # Summary
    print("\n========== Cross Validation Summary ==========")
    avg_mse = np.mean([r['mse'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    
    for res in fold_results:
        print(f"Fold {res['fold']}: MSE={res['mse']:.5f}, R2={res['r2']:.5f}")
        
    print(f"Average MSE: {avg_mse:.5f}")
    print(f"Average R2: {avg_r2:.5f}")