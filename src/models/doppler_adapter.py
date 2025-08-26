import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class DopplerSpectrumAdapter(nn.Module):
    """
    多普勒谱图适配器
    
    解决核心技术难点：
    1. 将[3, 64, 25]多普勒谱图适配为CLIP ViT可处理的格式
    2. 保持物理意义：3个range bins的空间信息
    3. 无需resize到224x224，通过自定义patch embedding处理64x25尺寸
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 64, 25),  # (range_bins, freq_bins, time_frames)
        patch_size: int = 8,
        embed_dim: int = 768,  # 与CLIP ViT-Base一致
        use_learnable_position: bool = True,
        doppler_normalization: str = "adaptive",  # "adaptive", "minmax", "zscore"
        physical_encoding: bool = True
    ):
        """
        初始化多普勒适配器
        
        Args:
            input_shape: 输入多普勒谱图形状 [range_bins, freq_bins, time_frames]
            patch_size: patch大小，用于将64x25分割为patches
            embed_dim: embedding维度，需与CLIP ViT一致
            use_learnable_position: 是否使用可学习的位置编码
            doppler_normalization: 归一化方式
            physical_encoding: 是否添加物理先验编码
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.range_bins, self.freq_bins, self.time_frames = input_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.physical_encoding = physical_encoding
        self.doppler_normalization = doppler_normalization
        
        # 计算patch网格尺寸
        self.freq_patches = self.freq_bins // patch_size  # 64 // 8 = 8
        self.time_patches = self.time_frames // patch_size  # 25 // 8 = 3 (余1)
        
        # 处理不能整除的情况 - 使用padding
        self.freq_pad = (self.freq_patches * patch_size) - self.freq_bins  # 0
        self.time_pad = (self.time_patches * patch_size) - self.time_frames  # 24 - 25 = -1, 需要pad到24
        if self.time_pad < 0:
            self.time_patches = (self.time_frames + patch_size - 1) // patch_size  # 向上取整
            self.time_pad = (self.time_patches * patch_size) - self.time_frames  # 32 - 25 = 7
        
        self.num_patches = self.freq_patches * self.time_patches  # 8 * 4 = 32 patches
        
        # Range bin融合层 - 将3个range bin融合为单通道
        self.range_fusion = nn.Sequential(
            nn.Conv2d(self.range_bins, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # Patch embedding层 - 替代标准CLIP的patch embedding
        patch_dim = patch_size * patch_size  # 8*8 = 64
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        
        # 物理先验编码
        if self.physical_encoding:
            self.freq_encoding = self._create_frequency_encoding()
            self.time_encoding = self._create_temporal_encoding()
            self.range_encoding = self._create_range_encoding()
        
        # 位置编码
        if use_learnable_position:
            self.position_embedding = nn.Parameter(
                torch.randn(1, self.num_patches, embed_dim) * 0.02
            )
        else:
            self.position_embedding = self._create_sinusoidal_position_encoding()
        
        # 自适应归一化层
        if doppler_normalization == "adaptive":
            self.adaptive_norm = nn.AdaptiveAvgPool2d((1, 1))
            self.norm_scale = nn.Parameter(torch.ones(1))
            self.norm_bias = nn.Parameter(torch.zeros(1))
    
    def _create_frequency_encoding(self) -> nn.Parameter:
        """创建频率维度的物理编码 - 基于多普勒频移"""
        freq_encoding = torch.zeros(self.freq_bins, self.embed_dim // 4)
        
        # 多普勒频移公式：f_d = 2 * v * f_c / c
        # 这里我们假设频率bins对应不同的速度
        max_velocity = 10.0  # 假设最大检测速度10m/s
        velocities = torch.linspace(-max_velocity, max_velocity, self.freq_bins)
        
        for i, velocity in enumerate(velocities):
            # 使用正弦和余弦编码速度信息
            div_term = torch.exp(torch.arange(0, self.embed_dim // 4, 2) * 
                               -(math.log(10000.0) / (self.embed_dim // 4)))
            freq_encoding[i, 0::2] = torch.sin(velocity * div_term)
            freq_encoding[i, 1::2] = torch.cos(velocity * div_term)
        
        return nn.Parameter(freq_encoding, requires_grad=False)
    
    def _create_temporal_encoding(self) -> nn.Parameter:
        """创建时间维度的物理编码"""
        time_encoding = torch.zeros(self.time_frames, self.embed_dim // 4)
        
        # 时间位置编码
        position = torch.arange(0, self.time_frames).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim // 4, 2) * 
                           -(math.log(10000.0) / (self.embed_dim // 4)))
        time_encoding[:, 0::2] = torch.sin(position * div_term)
        time_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(time_encoding, requires_grad=False)
    
    def _create_range_encoding(self) -> nn.Parameter:
        """创建距离维度的物理编码"""
        range_encoding = torch.zeros(self.range_bins, self.embed_dim // 4)
        
        # 假设range bins对应不同距离
        distances = torch.tensor([1.0, 5.0, 10.0])  # 假设3个range bin对应1m, 5m, 10m
        
        for i, distance in enumerate(distances):
            div_term = torch.exp(torch.arange(0, self.embed_dim // 4, 2) * 
                               -(math.log(10000.0) / (self.embed_dim // 4)))
            range_encoding[i, 0::2] = torch.sin(distance * div_term)
            range_encoding[i, 1::2] = torch.cos(distance * div_term)
        
        return nn.Parameter(range_encoding, requires_grad=False)
    
    def _create_sinusoidal_position_encoding(self) -> nn.Parameter:
        """创建标准的正弦位置编码"""
        position_enc = torch.zeros(self.num_patches, self.embed_dim)
        
        position = torch.arange(0, self.num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * 
                           -(math.log(10000.0) / self.embed_dim))
        position_enc[:, 0::2] = torch.sin(position * div_term)
        position_enc[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(position_enc.unsqueeze(0), requires_grad=False)
    
    def _apply_doppler_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """应用多普勒谱图特定的归一化"""
        if self.doppler_normalization == "adaptive":
            # 自适应归一化 - 考虑谱图的动态范围
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            
            # 计算每个样本的统计量
            mean_vals = torch.mean(x_flat, dim=1, keepdim=True)
            std_vals = torch.std(x_flat, dim=1, keepdim=True) + 1e-8
            
            # 归一化
            x_flat = (x_flat - mean_vals) / std_vals
            x = x_flat.view_as(x)
            
            # 可学习的缩放和偏移
            x = x * self.norm_scale + self.norm_bias
            
        elif self.doppler_normalization == "minmax":
            # Min-Max归一化
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            min_vals = torch.min(x_flat, dim=1, keepdim=True)[0]
            max_vals = torch.max(x_flat, dim=1, keepdim=True)[0]
            x_flat = (x_flat - min_vals) / (max_vals - min_vals + 1e-8)
            x = x_flat.view_as(x)
            
        elif self.doppler_normalization == "zscore":
            # Z-score标准化
            mean_val = torch.mean(x)
            std_val = torch.std(x) + 1e-8
            x = (x - mean_val) / std_val
        
        return x
    
    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        从2D谱图中提取patches
        
        Args:
            x: [batch_size, 1, freq_bins, time_frames]
            
        Returns:
            patches: [batch_size, num_patches, patch_size*patch_size]
        """
        batch_size = x.size(0)
        
        # Padding到可被patch_size整除的尺寸
        if self.time_pad > 0:
            x = F.pad(x, (0, self.time_pad), mode='constant', value=0)
        if self.freq_pad > 0:
            x = F.pad(x, (0, 0, 0, self.freq_pad), mode='constant', value=0)
        
        # 使用unfold提取patches
        # x: [batch_size, 1, freq_bins_padded, time_frames_padded]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches: [batch_size, 1, freq_patches, time_patches, patch_size, patch_size]
        
        patches = patches.contiguous().view(
            batch_size, self.freq_patches * self.time_patches, -1
        )
        # patches: [batch_size, num_patches, patch_size*patch_size]
        
        return patches
    
    def _add_physical_encoding(self, x: torch.Tensor, original_spectrum: torch.Tensor) -> torch.Tensor:
        """添加物理先验编码"""
        if not self.physical_encoding:
            return x
        
        batch_size = x.size(0)
        device = x.device
        
        # 计算每个patch对应的物理特征
        physical_features = []
        
        for i in range(self.num_patches):
            freq_idx = i // self.time_patches
            time_idx = i % self.time_patches
            
            # 频率编码 (对应patch的频率范围)
            freq_start = freq_idx * self.patch_size
            freq_end = min(freq_start + self.patch_size, self.freq_bins)
            freq_feature = torch.mean(self.freq_encoding[freq_start:freq_end], dim=0)
            
            # 时间编码 (对应patch的时间范围)
            time_start = time_idx * self.patch_size
            time_end = min(time_start + self.patch_size, self.time_frames)
            time_feature = torch.mean(self.time_encoding[time_start:time_end], dim=0)
            
            # Range编码 (通过原始3D数据计算)
            range_weights = torch.mean(original_spectrum[:, :, freq_start:freq_end, time_start:time_end], dim=(2, 3))
            range_weights = F.softmax(range_weights, dim=1)  # [batch_size, 3]
            range_feature = torch.sum(range_weights.unsqueeze(-1) * self.range_encoding.unsqueeze(0), dim=1)
            # range_feature: [batch_size, embed_dim//4]
            
            # 组合物理特征
            combined_feature = torch.cat([
                freq_feature.expand(batch_size, -1),
                time_feature.expand(batch_size, -1), 
                range_feature
            ], dim=-1)  # [batch_size, embed_dim//4 * 3]
            
            # Padding到embed_dim
            if combined_feature.size(-1) < self.embed_dim:
                pad_size = self.embed_dim - combined_feature.size(-1)
                combined_feature = F.pad(combined_feature, (0, pad_size))
            
            physical_features.append(combined_feature)
        
        physical_encoding = torch.stack(physical_features, dim=1)  # [batch_size, num_patches, embed_dim]
        
        # 加权融合
        alpha = 0.1  # 物理编码的权重
        x = x + alpha * physical_encoding.to(device)
        
        return x
    
    def forward(self, doppler_spectrum: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            doppler_spectrum: [batch_size, 3, 64, 25] 多普勒谱图
            
        Returns:
            patch_embeddings: [batch_size, num_patches, embed_dim] 适配后的特征
        """
        batch_size = doppler_spectrum.size(0)
        original_spectrum = doppler_spectrum.clone()
        
        # 1. 归一化处理
        doppler_spectrum = self._apply_doppler_normalization(doppler_spectrum)
        
        # 2. Range bin融合：[batch_size, 3, 64, 25] -> [batch_size, 1, 64, 25]
        fused_spectrum = self.range_fusion(doppler_spectrum)
        
        # 3. 提取patches：[batch_size, 1, 64, 25] -> [batch_size, num_patches, patch_size^2]
        patches = self._extract_patches(fused_spectrum)
        
        # 4. Patch embedding：[batch_size, num_patches, patch_size^2] -> [batch_size, num_patches, embed_dim]
        patch_embeddings = self.patch_embedding(patches)
        
        # 5. 添加物理先验编码
        patch_embeddings = self._add_physical_encoding(patch_embeddings, original_spectrum)
        
        # 6. 添加位置编码
        patch_embeddings = patch_embeddings + self.position_embedding
        
        return patch_embeddings
    
    def get_patch_info(self) -> dict:
        """返回patch划分信息，用于调试和可视化"""
        return {
            'input_shape': self.input_shape,
            'patch_size': self.patch_size,
            'freq_patches': self.freq_patches,
            'time_patches': self.time_patches,
            'num_patches': self.num_patches,
            'freq_pad': self.freq_pad,
            'time_pad': self.time_pad,
            'embed_dim': self.embed_dim
        }


class DopplerPreprocessor(nn.Module):
    """
    多普勒数据预处理器
    处理原始numpy数据到tensor的转换和基础预处理
    """
    
    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (3, 64, 25),
        apply_log_transform: bool = True,
        apply_smoothing: bool = True,
        noise_reduction: bool = True
    ):
        super().__init__()
        self.target_shape = target_shape
        self.apply_log_transform = apply_log_transform
        self.apply_smoothing = apply_smoothing
        self.noise_reduction = noise_reduction
        
        if apply_smoothing:
            # 时间维度平滑滤波器
            self.time_smoother = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
            self.time_smoother.weight.data = torch.tensor([[[0.25, 0.5, 0.25]]])
            self.time_smoother.weight.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        预处理多普勒数据
        
        Args:
            x: [batch_size, 3, 64, 25] 或者其他尺寸的多普勒数据
            
        Returns:
            processed: [batch_size, 3, 64, 25] 预处理后的数据
        """
        # 确保数据格式正确
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 添加batch维度
        
        # Log变换减少动态范围
        if self.apply_log_transform:
            x = torch.log(torch.abs(x) + 1e-8)
        
        # 噪声减少
        if self.noise_reduction:
            # 简单的阈值去噪
            threshold = torch.quantile(x.abs(), 0.1)
            x = torch.where(x.abs() < threshold, torch.zeros_like(x), x)
        
        # 时间维度平滑
        if self.apply_smoothing:
            batch_size, range_bins, freq_bins, time_frames = x.shape
            x_reshaped = x.view(-1, 1, time_frames)  # [batch*range*freq, 1, time]
            x_smoothed = self.time_smoother(x_reshaped)
            x = x_smoothed.view(batch_size, range_bins, freq_bins, time_frames)
        
        # 尺寸调整到目标形状
        if x.shape[1:] != self.target_shape:
            x = F.interpolate(x, size=self.target_shape[1:], mode='bilinear', align_corners=False)
        
        return x


# 测试和使用示例
if __name__ == "__main__":
    # 创建适配器
    adapter = DopplerSpectrumAdapter(
        input_shape=(3, 64, 25),
        patch_size=8,
        embed_dim=768,
        physical_encoding=True
    )
    
    # 创建预处理器
    preprocessor = DopplerPreprocessor()
    
    # 测试数据
    batch_size = 4
    test_data = torch.randn(batch_size, 3, 64, 25)
    
    print("=== 多普勒适配器测试 ===")
    print(f"输入数据形状: {test_data.shape}")
    
    # 预处理
    processed_data = preprocessor(test_data)
    print(f"预处理后形状: {processed_data.shape}")
    
    # 适配
    adapted_features = adapter(processed_data)
    print(f"适配后特征形状: {adapted_features.shape}")
    
    # 打印patch信息
    patch_info = adapter.get_patch_info()
    print(f"Patch信息: {patch_info}")
    
    # 验证维度兼容性
    print(f"是否与CLIP ViT兼容: {adapted_features.shape[-1] == 768}")
    print(f"Patch数量: {adapted_features.shape[1]}")