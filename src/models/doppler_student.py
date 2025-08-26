import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class DepthwiseSeparableConv2d(nn.Module):
    """深度可分离卷积，减少参数量"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EfficientConvBlock(nn.Module):
    """高效卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_separable=True):
        super().__init__()
        padding = kernel_size // 2
        
        if use_separable and in_channels > 1:
            self.conv = DepthwiseSeparableConv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)  # ReLU6更适合量化
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SqueezeExcitation(nn.Module):
    """轻量级注意力机制"""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class DopplerStudent(nn.Module):
    """
    轻量级Student模型
    
    设计目标：
    - 参数量 < 563KB  
    - FLOPs < 1.5M
    - 保持CNN+LSTM架构以兼容DTNet
    - 能够从Teacher模型学习多模态知识
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 64, 25),
        num_classes: int = 2,
        hidden_dim: int = 32,  # 大幅减少隐藏层维度
        lstm_hidden: int = 16,  # 减少LSTM隐藏层
        lstm_layers: int = 1,   # 减少LSTM层数
        use_attention: bool = True,
        dropout_rate: float = 0.1,
        use_separable_conv: bool = True,
        teacher_feature_dim: int = 512,  # Teacher输出特征维度
    ):
        """
        初始化Student模型
        
        Args:
            input_shape: 输入多普勒谱图形状 [range_bins, freq_bins, time_frames]
            num_classes: 分类类别数
            hidden_dim: CNN隐藏层维度
            lstm_hidden: LSTM隐藏层维度  
            lstm_layers: LSTM层数
            use_attention: 是否使用轻量级注意力
            dropout_rate: Dropout比率
            use_separable_conv: 是否使用深度可分离卷积
            teacher_feature_dim: Teacher特征维度，用于特征对齐
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.range_bins, self.freq_bins, self.time_frames = input_shape
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.teacher_feature_dim = teacher_feature_dim
        
        # 1. 轻量级CNN特征提取器
        self.feature_extractor = self._build_cnn_backbone(use_separable_conv, use_attention)
        
        # 2. 计算CNN输出尺寸
        self.cnn_output_size = self._calculate_cnn_output_size()
        
        # 3. LSTM序列建模
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=False  # 单向LSTM减少参数
        )
        
        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 5. 特征对齐层 (用于知识蒸馏)
        self.feature_alignment = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, teacher_feature_dim)
        )
        
        # 6. 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 初始化权重
        self._initialize_weights()
        
        # 计算并打印模型信息
        self._print_model_info()
    
    def _build_cnn_backbone(self, use_separable_conv: bool, use_attention: bool) -> nn.Module:
        """构建轻量级CNN骨干网络"""
        layers = []
        
        # 第一层：处理3个range bins
        layers.append(EfficientConvBlock(
            self.range_bins, self.hidden_dim // 4, 
            kernel_size=3, stride=1, use_separable=False  # 第一层不用分离卷积
        ))
        
        # 第二层：空间下采样
        layers.append(EfficientConvBlock(
            self.hidden_dim // 4, self.hidden_dim // 2,
            kernel_size=3, stride=2, use_separable=use_separable_conv
        ))
        
        # 第三层：通道扩展
        layers.append(EfficientConvBlock(
            self.hidden_dim // 2, self.hidden_dim,
            kernel_size=3, stride=1, use_separable=use_separable_conv
        ))
        
        # 添加注意力机制
        if use_attention:
            layers.append(SqueezeExcitation(self.hidden_dim))
        
        # 最后一层：特征压缩
        layers.append(EfficientConvBlock(
            self.hidden_dim, self.hidden_dim // 2,
            kernel_size=1, stride=1, use_separable=False  # 1x1卷积不需要分离
        ))
        
        return nn.Sequential(*layers)
    
    def _calculate_cnn_output_size(self) -> int:
        """计算CNN输出的特征维度"""
        # 模拟前向传播计算输出尺寸
        dummy_input = torch.randn(1, *self.input_shape)
        with torch.no_grad():
            cnn_out = self.feature_extractor(dummy_input)
            # 输出形状: [batch, channels, freq_reduced, time_reduced]
            return cnn_out.size(1) * cnn_out.size(2)  # channels * freq_reduced
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 估算模型大小 (假设float32, 4 bytes per parameter)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        model_size_kb = (total_params * 4) / 1024
        
        print(f"=== DopplerStudent 模型信息 ===")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"模型大小: {model_size_kb:.1f} KB ({model_size_mb:.3f} MB)")
        print(f"目标大小: 563 KB")
        print(f"大小符合要求: {'✓' if model_size_kb <= 563 else '✗'}")
    
    def forward(
        self, 
        doppler_spectrum: torch.Tensor, 
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            doppler_spectrum: [batch_size, 3, 64, 25]
            return_features: 是否返回中间特征（用于蒸馏）
            
        Returns:
            outputs: 包含logits和特征的字典
        """
        batch_size = doppler_spectrum.size(0)
        
        # 1. CNN特征提取
        cnn_features = self.feature_extractor(doppler_spectrum)
        # cnn_features: [batch_size, channels, freq_reduced, time_reduced]
        
        # 2. 准备LSTM输入：将频率维度和通道维度合并，时间维度作为序列长度
        # 转换为: [batch_size, time_reduced, channels * freq_reduced]
        _, channels, freq_reduced, time_reduced = cnn_features.shape
        lstm_input = cnn_features.permute(0, 3, 1, 2).contiguous()  # [batch, time, channels, freq]
        lstm_input = lstm_input.view(batch_size, time_reduced, -1)  # [batch, time, features]
        
        # 3. LSTM序列建模
        lstm_out, (hidden, _) = self.lstm(lstm_input)
        # 取最后一个时间步的输出
        sequence_features = lstm_out[:, -1, :]  # [batch_size, lstm_hidden]
        
        # 4. 分类预测
        logits = self.classifier(sequence_features)
        
        outputs = {
            'logits': logits,
            'sequence_features': sequence_features
        }
        
        # 5. 返回用于蒸馏的特征
        if return_features:
            # 特征对齐到Teacher特征空间
            aligned_features = self.feature_alignment(sequence_features)
            outputs.update({
                'cnn_features': cnn_features,
                'lstm_features': lstm_out,
                'aligned_features': aligned_features,  # 用于与Teacher特征对齐
                'raw_features': sequence_features
            })
        
        return outputs
    
    def compute_loss(
        self, 
        doppler_spectrum: torch.Tensor, 
        labels: torch.Tensor,
        teacher_features: Optional[torch.Tensor] = None,
        temperature: float = 4.0,
        alpha: float = 0.7
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失（支持知识蒸馏）
        
        Args:
            doppler_spectrum: 输入数据
            labels: 真实标签
            teacher_features: Teacher模型的特征（可选）
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            
        Returns:
            loss_dict: 包含各种损失的字典
        """
        outputs = self.forward(doppler_spectrum, return_features=True)
        logits = outputs['logits']
        
        # 基础分类损失
        classification_loss = self.criterion(logits, labels)
        
        loss_dict = {
            'classification_loss': classification_loss,
            'total_loss': classification_loss
        }
        
        # 知识蒸馏损失
        if teacher_features is not None:
            student_features = outputs['aligned_features']
            
            # 特征对齐损失 (MSE)
            feature_alignment_loss = F.mse_loss(student_features, teacher_features)
            
            loss_dict['feature_alignment_loss'] = feature_alignment_loss
            
            # 总损失：分类损失 + 特征对齐损失
            total_loss = (1 - alpha) * classification_loss + alpha * feature_alignment_loss
            loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def predict(self, doppler_spectrum: torch.Tensor) -> torch.Tensor:
        """预测接口"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(doppler_spectrum)
            probabilities = F.softmax(outputs['logits'], dim=-1)
        return probabilities
    
    def get_features_for_distillation(self, doppler_spectrum: torch.Tensor) -> torch.Tensor:
        """获取用于蒸馏的特征"""
        with torch.no_grad():
            outputs = self.forward(doppler_spectrum, return_features=True)
            return outputs['aligned_features']
    
    def count_parameters(self) -> Tuple[int, int]:
        """计算参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def estimate_flops(self, input_shape: Tuple[int, int, int] = None) -> int:
        """估算FLOPs"""
        if input_shape is None:
            input_shape = self.input_shape
        
        def conv_flops(in_channels, out_channels, kernel_size, output_size):
            return in_channels * out_channels * kernel_size * kernel_size * output_size
        
        def linear_flops(in_features, out_features):
            return in_features * out_features
        
        batch_size = 1
        total_flops = 0
        
        # 模拟各层的FLOPs计算
        # 这是一个简化的估算，实际FLOPs可能有所不同
        
        # CNN层的FLOPs
        current_h, current_w = input_shape[1], input_shape[2]  # 64, 25
        current_channels = input_shape[0]  # 3
        
        # Layer 1: 3 -> hidden_dim//4
        out_channels = self.hidden_dim // 4
        output_size = current_h * current_w
        total_flops += conv_flops(current_channels, out_channels, 3, output_size)
        current_channels = out_channels
        
        # Layer 2: stride=2
        current_h, current_w = current_h // 2, current_w // 2
        out_channels = self.hidden_dim // 2  
        output_size = current_h * current_w
        if hasattr(self.feature_extractor[1].conv, 'depthwise'):  # 分离卷积
            # 深度卷积 + 逐点卷积
            total_flops += current_channels * 9 * output_size  # 深度卷积
            total_flops += current_channels * out_channels * output_size  # 逐点卷积
        else:
            total_flops += conv_flops(current_channels, out_channels, 3, output_size)
        current_channels = out_channels
        
        # Layer 3: 同样的输出尺寸
        out_channels = self.hidden_dim
        output_size = current_h * current_w
        if hasattr(self.feature_extractor[2].conv, 'depthwise'):
            total_flops += current_channels * 9 * output_size
            total_flops += current_channels * out_channels * output_size
        else:
            total_flops += conv_flops(current_channels, out_channels, 3, output_size)
        current_channels = out_channels
        
        # SE attention (简化)
        total_flops += current_channels * (current_channels // 4) * 2
        
        # Final conv layer
        out_channels = self.hidden_dim // 2
        total_flops += conv_flops(current_channels, out_channels, 1, output_size)
        
        # LSTM FLOPs (简化估算)
        lstm_input_size = out_channels * current_h
        lstm_flops = 4 * lstm_input_size * self.lstm_hidden * current_w  # 4个门 * 输入维度 * 隐藏维度 * 序列长度
        total_flops += lstm_flops
        
        # Classifier FLOPs
        total_flops += linear_flops(self.lstm_hidden, self.hidden_dim // 2)
        total_flops += linear_flops(self.hidden_dim // 2, self.num_classes)
        
        return total_flops


class CompactDopplerStudent(DopplerStudent):
    """
    更紧凑的Student模型版本
    专门针对极限参数约束优化
    """
    
    def __init__(self, **kwargs):
        # 使用更小的参数
        compact_kwargs = {
            'hidden_dim': 16,
            'lstm_hidden': 8,
            'lstm_layers': 1,
            'use_attention': False,  # 关闭注意力以节省参数
            'use_separable_conv': True,
            **kwargs
        }
        super().__init__(**compact_kwargs)


# 测试和验证代码
if __name__ == "__main__":
    print("=== DopplerStudent 模型测试 ===")
    
    # 标准版本
    model = DopplerStudent(
        input_shape=(3, 64, 25),
        num_classes=2,
        hidden_dim=32,
        lstm_hidden=16
    )
    
    # 测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 64, 25)
    test_labels = torch.tensor([1, 0, 1, 0])
    
    print(f"\n输入形状: {test_input.shape}")
    
    # 前向传播测试
    outputs = model.forward(test_input, return_features=True)
    print(f"输出logits形状: {outputs['logits'].shape}")
    print(f"对齐特征形状: {outputs['aligned_features'].shape}")
    
    # 损失计算测试
    loss_dict = model.compute_loss(test_input, test_labels)
    print(f"分类损失: {loss_dict['classification_loss'].item():.4f}")
    
    # 预测测试
    predictions = model.predict(test_input)
    print(f"预测概率形状: {predictions.shape}")
    
    # FLOPs估算
    flops = model.estimate_flops()
    print(f"估算FLOPs: {flops:,} ({flops/1e6:.2f}M)")
    print(f"目标FLOPs: 1.5M")
    print(f"FLOPs符合要求: {'✓' if flops <= 1.5e6 else '✗'}")
    
    # 测试紧凑版本
    print("\n=== CompactDopplerStudent 测试 ===")
    compact_model = CompactDopplerStudent()
    compact_outputs = compact_model.forward(test_input)
    compact_flops = compact_model.estimate_flops()
    
    total_params, _ = compact_model.count_parameters()
    model_size_kb = (total_params * 4) / 1024
    
    print(f"紧凑模型参数量: {total_params:,}")
    print(f"紧凑模型大小: {model_size_kb:.1f} KB")
    print(f"紧凑模型FLOPs: {compact_flops:,} ({compact_flops/1e6:.2f}M)")
    print(f"紧凑模型符合要求: {'✓' if model_size_kb <= 563 and compact_flops <= 1.5e6 else '✗'}")