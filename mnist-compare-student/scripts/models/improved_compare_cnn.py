import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedTower(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        # 更深的卷积网络，使用残差连接
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 特征提取头
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

class AttentionModule(nn.Module):
    """注意力机制模块，用于增强特征表示"""
    def __init__(self, feat_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 4, feat_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights

class ImprovedCompareNet(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.tower = ImprovedTower(out_dim=feat_dim)
        self.attention = AttentionModule(feat_dim)
        
        # 更复杂的比较头
        in_dim = feat_dim * 2
        self.comparison_head = nn.Sequential(
            nn.Linear(in_dim, 512),
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
        
        # 添加绝对值差特征
        self.diff_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, xa, xb):
        # 提取特征
        fa = self.tower(xa)
        fb = self.tower(xb)
        
        # 应用注意力机制
        fa_attn = self.attention(fa)
        fb_attn = self.attention(fb)
        
        # 连接特征
        fuse = torch.cat([fa_attn, fb_attn], dim=-1)
        
        # 主要比较分支
        main_logit = self.comparison_head(fuse).squeeze(1)
        
        # 绝对值差分支
        diff = torch.abs(fa - fb)
        diff_logit = self.diff_head(diff).squeeze(1)
        
        # 组合两个分支
        combined_logit = main_logit + 0.3 * diff_logit
        
        return combined_logit

class ImprovedCompareNetV2(nn.Module):
    """另一个改进版本：使用更现代的网络架构"""
    def __init__(self, feat_dim=512):
        super().__init__()
        
        # 使用深度可分离卷积
        self.tower = nn.Sequential(
            # 第一层：深度可分离卷积
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, groups=32),  # 深度卷积
            nn.Conv2d(32, 64, 1),  # 逐点卷积
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
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # 多尺度特征融合
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, 1)
        )
        
        # 对比学习分支
        self.contrastive = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, xa, xb):
        fa = self.tower(xa)
        fb = self.tower(xb)
        
        fa_feat = self.feature_extractor(fa)
        fb_feat = self.feature_extractor(fb)
        
        # 特征融合
        fuse = torch.cat([fa_feat, fb_feat], dim=-1)
        main_logit = self.fusion(fuse).squeeze(1)
        
        # 对比学习
        contrast_logit = self.contrastive(torch.abs(fa_feat - fb_feat)).squeeze(1)
        
        return main_logit + 0.2 * contrast_logit

class AdvancedCompareNet(nn.Module):
    """高级比较网络：结合多种先进技术"""
    def __init__(self, feat_dim=512):
        super().__init__()
        
        # 使用ResNet风格的残差块
        self.tower = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(1, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            
            # 残差块1
            self._make_residual_block(64, 128, stride=2),  # 7x7
            
            # 残差块2
            self._make_residual_block(128, 256, stride=2),  # 4x4
            
            # 残差块3
            self._make_residual_block(256, 512, stride=2),  # 2x2
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        
        # 对比学习头
        self.contrastive_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # 相似度计算头
        self.similarity_head = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, xa, xb):
        # 特征提取
        fa = self.tower(xa)
        fb = self.tower(xb)
        
        fa_feat = self.feature_extractor(fa)
        fb_feat = self.feature_extractor(fb)
        
        # 多头注意力
        fa_attn, _ = self.attention(fa_feat.unsqueeze(1), fa_feat.unsqueeze(1), fa_feat.unsqueeze(1))
        fb_attn, _ = self.attention(fb_feat.unsqueeze(1), fb_feat.unsqueeze(1), fb_feat.unsqueeze(1))
        
        fa_attn = fa_attn.squeeze(1)
        fb_attn = fb_attn.squeeze(1)
        
        # 对比学习分支
        contrast_logit = self.contrastive_head(torch.abs(fa_attn - fb_attn)).squeeze(1)
        
        # 相似度分支
        similarity_input = torch.cat([fa_attn, fb_attn], dim=-1)
        similarity_logit = self.similarity_head(similarity_input).squeeze(1)
        
        # 融合两个分支
        combined = torch.stack([contrast_logit, similarity_logit], dim=-1)
        final_logit = self.fusion(combined).squeeze(1)
        
        return final_logit

class EfficientCompareNet(nn.Module):
    """高效比较网络：平衡性能和计算效率"""
    def __init__(self, feat_dim=384):
        super().__init__()
        
        # 使用EfficientNet风格的深度可分离卷积
        self.tower = nn.Sequential(
            # 第一层：标准卷积
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 深度可分离卷积块1
            self._make_separable_block(32, 64, stride=2),  # 14x14
            
            # 深度可分离卷积块2
            self._make_separable_block(64, 128, stride=2),  # 7x7
            
            # 深度可分离卷积块3
            self._make_separable_block(128, 256, stride=2),  # 4x4
            
            # 深度可分离卷积块4
            self._make_separable_block(256, 512, stride=2),  # 2x2
            
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 8, feat_dim),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feat_dim, feat_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_dim // 16, feat_dim, 1),
            nn.Sigmoid()
        )
        
        # 比较头
        self.comparison_head = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # 差异头
        self.diff_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
    
    def _make_separable_block(self, in_channels, out_channels, stride=1):
        """创建深度可分离卷积块"""
        return nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 逐点卷积
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, xa, xb):
        # 特征提取
        fa = self.tower(xa)
        fb = self.tower(xb)
        
        fa_feat = self.feature_extractor(fa)
        fb_feat = self.feature_extractor(fb)
        
        # 空间注意力
        fa_spatial = self.spatial_attention(fa_feat) * fa_feat
        fb_spatial = self.spatial_attention(fb_feat) * fb_feat
        
        # 通道注意力
        fa_channel = self.channel_attention(fa_spatial.unsqueeze(-1)).squeeze(-1) * fa_spatial
        fb_channel = self.channel_attention(fb_spatial.unsqueeze(-1)).squeeze(-1) * fb_spatial
        
        # 主要比较分支
        combined = torch.cat([fa_channel, fb_channel], dim=-1)
        main_logit = self.comparison_head(combined).squeeze(1)
        
        # 差异分支
        diff = torch.abs(fa_channel - fb_channel)
        diff_logit = self.diff_head(diff).squeeze(1)
        
        # 组合
        final_logit = main_logit + 0.4 * diff_logit
        
        return final_logit

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 为了向后兼容，保留原始类名
CompareNet = ImprovedCompareNet
