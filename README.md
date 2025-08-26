# 多模态多普勒雷达人体检测技术方案

## 项目概述

本项目利用多模态学习(文本+图像)提升多普勒雷达人体检测性能，通过知识蒸馏实现轻量化部署。

**核心创新**: 训练时使用文本+图像，推理时只用图像，解决强干扰场景下的检测难题

## 技术架构

### 整体流程
- **训练阶段**: 多普勒图[3,64,25] + 文本描述 → CLIP-based Teacher → 高精度分类
- **蒸馏阶段**: Teacher知识 → 蒸馏 → 轻量Student模型  
- **推理阶段**: 仅多普勒图[3,64,25] → Student模型 → 最终预测

### 核心模块
- `TextProcessor`: 文件名解析和文本描述生成
- `DopplerSpectrumAdapter`: 多普勒图适配模块
- `DopplerCLIPTeacher`: 多模态Teacher模型
- `DopplerStudent`: 轻量级Student模型
- `DistillationTrainer`: 知识蒸馏框架
- `DopplerMultimodalDataset`: 数据加载器

## 数据格式

- **多普勒谱图**: [3, 64, 25] (3个range bins × 64频率bins × 25时间帧)
- **文件命名**: "1_1_设备高1.2m风吹绿植7m有人在1m前后踱步.npy"
- **标签**: 二分类 (1=有人, 0=无人)

## 训练策略

### 三阶段训练
1. **Stage 1**: CLIP多模态微调 (10 epochs)
2. **Stage 2**: 知识蒸馏训练 (20 epochs) 
3. **Stage 3**: Student微调 (10 epochs)

### 损失函数
```
总损失 = 0.7 × KL蒸馏损失 + 0.3 × 硬标签损失 + 0.1 × 特征对齐损失
温度参数 = 4.0
```

## 预期效果

- 整体准确率: 85% → 95%+
- 复杂场景召回率: 70% → 90%+
- 误报率: 15% → 5%

## 安装与使用

```bash
# 安装依赖
pip install -r requirements.txt

# 训练模型
python train.py --config configs/train_config.yaml

# 评估模型
python evaluate.py --config configs/eval_config.yaml
```

## 目录结构

```
.
├── README.md
├── requirements.txt
├── configs/
│   ├── train_config.yaml
│   └── eval_config.yaml
├── src/
│   ├── models/
│   ├── data/
│   ├── training/
│   └── utils/
├── train.py
├── evaluate.py
└── scripts/
```