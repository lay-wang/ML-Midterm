import os, json, argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F

from .utils.seed import set_seed
from .utils.data import PairNPZDataset

class OptimizedCompareNet(nn.Module):
    """针对GPU优化的高效比较网络"""
    def __init__(self, feat_dim=256):
        super().__init__()
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            
            # 第二层
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            
            # 第三层
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        
        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # 比较头
        self.comparison_head = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # 差异头
        self.diff_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
    
    def forward(self, xa, xb):
        # 特征提取
        fa = self.feature_extractor(xa)
        fb = self.feature_extractor(xb)
        
        # 特征投影
        fa_feat = self.feature_proj(fa)
        fb_feat = self.feature_proj(fb)
        
        # 主要比较分支
        combined = torch.cat([fa_feat, fb_feat], dim=-1)
        main_logit = self.comparison_head(combined).squeeze(1)
        
        # 差异分支
        diff = torch.abs(fa_feat - fb_feat)
        diff_logit = self.diff_head(diff).squeeze(1)
        
        # 组合输出
        final_logit = main_logit + 0.3 * diff_logit
        
        return final_logit

class OptimizedAugmentedDataset(PairNPZDataset):
    """优化的数据增强数据集"""
    def __init__(self, path_npz, is_train=False):
        super().__init__(path_npz, is_train)
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=8),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = None
    
    def __getitem__(self, idx):
        img = self.x[idx]  # (28,56)
        xa = img[:, :28]
        xb = img[:, 28:]
        
        # 转换为tensor并归一化
        xa = torch.from_numpy(xa).float().unsqueeze(0) / 255.0
        xb = torch.from_numpy(xb).float().unsqueeze(0) / 255.0
        
        # 应用数据增强（仅训练时）
        if self.transform is not None:
            xa_pil = xa.squeeze(0).numpy() * 255
            xb_pil = xb.squeeze(0).numpy() * 255
            xa_pil = np.stack([xa_pil] * 3, axis=-1).astype(np.uint8)
            xb_pil = np.stack([xb_pil] * 3, axis=-1).astype(np.uint8)
            
            xa_aug = self.transform(xa_pil)[0:1]
            xb_aug = self.transform(xb_pil)[0:1]
            
            xa = xa_aug
            xb = xb_aug
        
        if self.y is None:
            return xa, xb, self.ids[idx]
        else:
            y = int(self.y[idx])
            return xa, xb, y

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xa, xb, y in loader:
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            
            logit = model(xa, xb)
            prob = torch.sigmoid(logit)
            pred = (prob >= 0.5).long()
            ys.append(y.long().cpu().numpy())
            ps.append(pred.cpu().numpy())
    
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = (y_true == y_pred).mean().item()
    
    # macro-F1
    f1s = []
    for cls in [0,1]:
        tp = np.sum((y_true==cls) & (y_pred==cls))
        fp = np.sum((y_true!=cls) & (y_pred==cls))
        fn = np.sum((y_true==cls) & (y_pred!=cls))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    f1_macro = float(np.mean(f1s))
    return acc, f1_macro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./outputs/gpu_optimized")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--feat_dim", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # GPU设置和优化
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # 设置GPU优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # 启用混合精度训练
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
    else:
        device = torch.device("cpu")
        print("使用CPU")
        use_amp = False
    
    # 创建模型
    model = OptimizedCompareNet(feat_dim=args.feat_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {n_params:,}")

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path   = os.path.join(args.data_dir, "val.npz")
    
    # 数据集
    train_ds = OptimizedAugmentedDataset(train_path, is_train=True)
    val_ds   = PairNPZDataset(val_path, is_train=False)

    # 优化DataLoader设置
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True if args.num_workers > 0 else False,
                             drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True if args.num_workers > 0 else False)

    # 优化器和调度器
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=args.lr/10)
    criterion = torch.nn.BCEWithLogitsLoss()

    best = {"acc":0.0, "f1":0.0, "epoch":-1}
    patience, bad = args.patience, 0

    print(f"开始训练，目标准确率: 80%")
    print(f"训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}")
    print(f"使用混合精度训练: {use_amp}")

    for epoch in range(1, args.epochs+1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for xa, xb, y in pbar:
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            
            optim.zero_grad()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    logit = model(xa, xb)
                    loss = criterion(logit, y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
            else:
                logit = model(xa, xb)
                loss = criterion(logit, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        val_acc, val_f1 = evaluate(model, val_loader, device)
        
        current_lr = optim.param_groups[0]['lr']
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val F1: {val_f1:.4f}, LR: {current_lr:.2e}")

        # 保存最佳模型
        if val_acc > best["acc"]:
            best = {"acc":val_acc, "f1":val_f1, "epoch":epoch}
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
            with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
                json.dump({
                    "best_val_acc": val_acc, 
                    "best_val_f1": val_f1, 
                    "best_epoch": epoch, 
                    "params": n_params,
                    "model_type": "optimized",
                    "use_amp": use_amp
                }, f, indent=2)
            bad = 0
            print(f"✓ 新的最佳模型保存! 准确率: {val_acc:.4f}")
        else:
            bad += 1
            if bad >= patience:
                print(f"早停触发，最佳准确率: {best['acc']:.4f}")
                break

    print(f"\n训练完成!")
    print(f"最佳结果 @ epoch {best['epoch']}: acc={best['acc']:.4f}, f1_macro={best['f1']:.4f}")
    print(f"目标准确率: 80%, 当前最佳: {best['acc']*100:.2f}%")
    
    if best['acc'] >= 0.8:
        print("🎉 达到目标准确率!")
    else:
        print(f"❌ 未达到目标，还需要提升 {0.8-best['acc']:.4f}")

if __name__ == "__main__":
    main()
