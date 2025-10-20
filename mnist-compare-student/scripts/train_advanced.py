import os, json, argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F

from .utils.seed import set_seed
from .utils.data import PairNPZDataset
from .models.improved_compare_cnn import AdvancedCompareNet, EfficientCompareNet, count_params

class AdvancedAugmentedDataset(PairNPZDataset):
    """高级数据增强数据集"""
    def __init__(self, path_npz, is_train=False):
        super().__init__(path_npz, is_train)
        if is_train:
            # 更激进的数据增强
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = None
    
    def __getitem__(self, idx):
        img = self.x[idx]  # (28,56)
        # split into two (28,28)
        xa = img[:, :28]
        xb = img[:, 28:]
        
        # 转换为tensor并归一化
        xa = torch.from_numpy(xa).float().unsqueeze(0) / 255.0
        xb = torch.from_numpy(xb).float().unsqueeze(0) / 255.0
        
        # 应用数据增强（仅训练时）
        if self.transform is not None:
            # 将单通道图像转换为3通道用于transforms
            xa_pil = xa.squeeze(0).numpy() * 255
            xb_pil = xb.squeeze(0).numpy() * 255
            xa_pil = np.stack([xa_pil] * 3, axis=-1).astype(np.uint8)
            xb_pil = np.stack([xb_pil] * 3, axis=-1).astype(np.uint8)
            
            xa_aug = self.transform(xa_pil)[0:1]  # 取第一个通道
            xb_aug = self.transform(xb_pil)[0:1]
            
            xa = xa_aug
            xb = xb_aug
        
        if self.y is None:
            return xa, xb, self.ids[idx]
        else:
            y = int(self.y[idx])
            return xa, xb, y

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xa, xb, y in loader:
            xa = xa.to(device); xb = xb.to(device); y = y.to(device).float()
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
    ap.add_argument("--out_dir", type=str, default="./outputs/advanced")
    ap.add_argument("--model", type=str, choices=["advanced", "efficient"], default="advanced")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--use_focal_loss", action="store_true", default=True)
    ap.add_argument("--use_mixup", action="store_true", default=True)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
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
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    # 选择模型
    if args.model == "advanced":
        model = AdvancedCompareNet(feat_dim=512).to(device)
    else:
        model = EfficientCompareNet(feat_dim=384).to(device)
    
    n_params = int(count_params(model))
    print(f"模型参数数量: {n_params:,}")

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path   = os.path.join(args.data_dir, "val.npz")
    
    # 使用高级数据增强
    train_ds = AdvancedAugmentedDataset(train_path, is_train=True)
    val_ds   = PairNPZDataset(val_path, is_train=False)

    # 优化DataLoader设置以更好地利用GPU
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True if args.num_workers > 0 else False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True if args.num_workers > 0 else False)

    # 优化器和调度器
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optim, max_lr=args.lr*10, epochs=args.epochs, 
                          steps_per_epoch=len(train_loader))
    
    # 损失函数
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    best = {"acc":0.0, "f1":0.0, "epoch":-1}
    patience, bad = args.patience, 0

    print(f"开始训练，目标准确率: 80%")
    print(f"训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}")
    print(f"使用Focal Loss: {args.use_focal_loss}")
    print(f"使用Mixup: {args.use_mixup}")

    def mixup_data(xa, xb, y, alpha=1.0):
        """Mixup数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = xa.size(0)
        index = torch.randperm(batch_size).to(xa.device)
        
        mixed_xa = lam * xa + (1 - lam) * xa[index, :]
        mixed_xb = lam * xb + (1 - lam) * xb[index, :]
        y_a, y_b = y, y[index]
        return mixed_xa, mixed_xb, y_a, y_b, lam

    for epoch in range(1, args.epochs+1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (xa, xb, y) in enumerate(pbar):
            xa = xa.to(device); xb = xb.to(device); y = y.to(device).float()
            
            # Mixup数据增强
            if args.use_mixup and np.random.random() < 0.5:
                mixed_xa, mixed_xb, y_a, y_b, lam = mixup_data(xa, xb, y, args.mixup_alpha)
                logit = model(mixed_xa, mixed_xb)
                loss = lam * criterion(logit, y_a) + (1 - lam) * criterion(logit, y_b)
            else:
                logit = model(xa, xb)
                loss = criterion(logit, y)
            
            optim.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optim.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

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
                    "model_type": args.model,
                    "use_focal_loss": args.use_focal_loss,
                    "use_mixup": args.use_mixup
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
