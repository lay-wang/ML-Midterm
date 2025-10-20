# ML-Midterm
Midterm
# MNIST 数字比较项目

这是一个基于深度学习的MNIST数字比较项目，用于判断两个MNIST数字图像中哪个数字更大。

## 项目概述

本项目实现了一个孪生网络（Siamese Network）架构，用于比较两个MNIST数字图像并判断哪个数字更大。项目包含完整的训练、推理和评估流程。

## 项目结构

```
mnist-compare-student/
├── data/                          # 数据文件
│   ├── meta.json                 # 元数据配置
│   ├── train.npz                 # 训练数据
│   ├── val.npz                   # 验证数据
│   ├── test_public.npz           # 公开测试数据
│   ├── test_public_labels.csv    # 公开测试标签
│   └── test_private.npz          # 私有测试数据
├── scripts/                       # 主要脚本
│   ├── models/                    # 模型定义
│   │   └── simple_compare_cnn.py # 比较网络模型
│   ├── utils/                     # 工具函数
│   │   ├── data.py               # 数据处理
│   │   ├── metrics.py            # 评估指标
│   │   ├── seed.py               # 随机种子设置
│   │   └── corruptions.py        # 数据增强
│   ├── train_baseline.py         # 训练脚本
│   ├── baseline_inference.py     # 推理脚本
│   ├── eval_public.py            # 公开测试评估
│   ├── check_submission.py       # 提交检查
│   └── requirements.txt          # 依赖包
└── README.md                     # 项目说明
```

## 模型架构

### CompareNet
- **Tower网络**: 用于提取单个数字图像的特征
  - 卷积层: 1→32→64→128 通道
  - 批归一化和ReLU激活
  - 最大池化和自适应平均池化
  - 全连接层输出128维特征向量

- **比较头**: 融合两个特征向量进行二分类
  - 拼接两个128维特征向量
  - 全连接层: 256→1
  - 使用Sigmoid激活输出概率

## 数据格式

- **输入**: 28×56像素的图像，包含两个并排的28×28 MNIST数字
- **标签**: 0或1，表示第一个数字是否大于第二个数字
- **数据增强**: 支持不同严重程度的数据增强

## 环境要求

```bash
torch>=2.2
torchvision>=0.17
numpy>=1.23
pandas>=1.5
tqdm>=4.65
```

## 使用方法

### 1. 安装依赖

```bash
cd scripts
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python -m scripts.train_baseline --data_dir ../data --out_dir ./outputs/baseline --epochs 10 --batch_size 128 --lr 1e-3
```

参数说明：
- `--data_dir`: 数据目录路径
- `--out_dir`: 输出目录
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率

### 3. 模型推理

```bash
python -m scripts.baseline_inference --data_dir ../data --ckpt ./outputs/baseline/model.pt --out ./pred_public.csv
```

参数说明：
- `--data_dir`: 数据目录路径
- `--ckpt`: 模型检查点路径
- `--out`: 预测结果输出文件
- `--private`: 是否使用私有测试集（True/False）

### 4. 评估结果

```bash
python -m scripts.eval_public --data_dir ../data --pred ./pred_public.csv --labels ../data/test_public_labels.csv
```

## 评估指标

- **准确率 (Accuracy)**: 正确预测的比例
- **宏平均F1分数 (Macro-F1)**: 两个类别的F1分数平均值

## 训练配置

- **优化器**: AdamW
- **损失函数**: BCEWithLogitsLoss
- **早停**: 验证集准确率3个epoch无提升时停止
- **数据增强**: 支持不同严重程度的增强
- **随机种子**: 42（可配置）

## 输出文件

训练完成后会生成：
- `model.pt`: 最佳模型权重
- `metrics.json`: 训练指标记录

推理完成后会生成：
- `pred_public.csv`: 预测结果文件（包含id和label列）

## 注意事项

1. 确保有足够的GPU内存进行训练
2. 数据文件需要放在正确的目录结构中
3. 推理时需要指定正确的模型检查点路径
4. 评估时需要提供对应的标签文件

## 项目特点

- 使用孪生网络架构进行数字比较
- 支持完整的训练、推理和评估流程
- 包含数据增强和正则化技术
- 提供详细的配置和日志记录
