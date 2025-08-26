import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPTextTransformer
import clip
from typing import Dict, Tuple, Optional, Union
import numpy as np

from .doppler_adapter import DopplerSpectrumAdapter, DopplerPreprocessor


class DopplerCLIPTeacher(nn.Module):
    """
    基于CLIP的多模态Teacher模型
    
    核心功能：
    1. 处理多普勒谱图[3,64,25] + 文本描述的多模态输入
    2. 通过CLIP架构学习图像-文本对齐关系
    3. 为后续知识蒸馏提供强监督信号
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        doppler_adapter_config: Optional[Dict] = None,
        freeze_text_encoder: bool = False,
        freeze_backbone_ratio: float = 0.7,
        num_classes: int = 2,
        classifier_dropout: float = 0.1,
        contrastive_temperature: float = 0.07,
        use_custom_classifier: bool = True
    ):
        """
        初始化多模态Teacher模型
        
        Args:
            clip_model_name: 预训练CLIP模型名称
            doppler_adapter_config: 多普勒适配器配置
            freeze_text_encoder: 是否冻结文本编码器
            freeze_backbone_ratio: 冻结backbone的比例
            num_classes: 分类类别数
            classifier_dropout: 分类器dropout率
            contrastive_temperature: 对比学习温度参数
            use_custom_classifier: 是否使用自定义分类器
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.contrastive_temperature = contrastive_temperature
        self.use_custom_classifier = use_custom_classifier
        
        # 1. 加载预训练CLIP模型
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # 获取模型配置
        self.config = self.clip_model.config
        self.vision_embed_dim = self.config.vision_config.hidden_size  # 768 for ViT-Base
        self.text_embed_dim = self.config.text_config.hidden_size     # 512 for CLIP text encoder
        self.projection_dim = self.config.projection_dim              # 512
        
        # 2. 多普勒谱图适配器
        if doppler_adapter_config is None:
            doppler_adapter_config = {
                'input_shape': (3, 64, 25),
                'patch_size': 8,
                'embed_dim': self.vision_embed_dim,
                'physical_encoding': True
            }
        
        self.doppler_adapter = DopplerSpectrumAdapter(**doppler_adapter_config)
        self.doppler_preprocessor = DopplerPreprocessor()
        
        # 3. 替换CLIP的视觉编码器的patch embedding
        # 保留transformer blocks，只替换patch embedding部分
        self.vision_transformer = self.clip_model.vision_model
        
        # 获取原始patch embedding的配置
        original_patch_embed = self.vision_transformer.embeddings.patch_embedding
        
        # 创建适配层连接多普勒adapter和ViT
        self.vision_adapter_layer = nn.Sequential(
            nn.LayerNorm(self.vision_embed_dim),
            nn.Dropout(0.1),
            nn.Linear(self.vision_embed_dim, self.vision_embed_dim)
        )
        
        # 4. 文本编码器
        self.text_transformer = self.clip_model.text_model
        if freeze_text_encoder:
            for param in self.text_transformer.parameters():
                param.requires_grad = False
        
        # 5. 投影层 (保持CLIP原有的投影层)
        self.visual_projection = self.clip_model.visual_projection
        self.text_projection = self.clip_model.text_projection
        
        # 6. 自定义分类器
        if use_custom_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(self.projection_dim * 2, 256),  # 拼接图像和文本特征
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(128, num_classes)
            )
        
        # 7. 冻结部分参数
        self._freeze_backbone_layers(freeze_backbone_ratio)
        
        # 8. 损失函数
        self.classification_criterion = nn.CrossEntropyLoss()
        self.contrastive_criterion = nn.CrossEntropyLoss()
    
    def _freeze_backbone_layers(self, freeze_ratio: float):
        """冻结部分backbone层"""
        if freeze_ratio <= 0:
            return
        
        # 冻结vision transformer的前几层
        vision_layers = self.vision_transformer.encoder.layers
        num_vision_layers = len(vision_layers)
        freeze_vision_layers = int(num_vision_layers * freeze_ratio)
        
        for i in range(freeze_vision_layers):
            for param in vision_layers[i].parameters():
                param.requires_grad = False
        
        # 冻结text transformer的前几层
        text_layers = self.text_transformer.encoder.layers
        num_text_layers = len(text_layers)
        freeze_text_layers = int(num_text_layers * freeze_ratio)
        
        for i in range(freeze_text_layers):
            for param in text_layers[i].parameters():
                param.requires_grad = False
        
        print(f"冻结了 {freeze_vision_layers}/{num_vision_layers} 个vision层和 {freeze_text_layers}/{num_text_layers} 个text层")
    
    def encode_doppler_image(self, doppler_spectrum: torch.Tensor) -> torch.Tensor:
        """
        编码多普勒谱图为视觉特征
        
        Args:
            doppler_spectrum: [batch_size, 3, 64, 25]
            
        Returns:
            image_features: [batch_size, projection_dim]
        """
        # 1. 预处理多普勒数据
        processed_spectrum = self.doppler_preprocessor(doppler_spectrum)
        
        # 2. 适配为patch embeddings
        patch_embeddings = self.doppler_adapter(processed_spectrum)  # [batch_size, num_patches, embed_dim]
        
        # 3. 适配层处理
        adapted_embeddings = self.vision_adapter_layer(patch_embeddings)
        
        # 4. 添加class token (模仿CLIP ViT)
        batch_size = adapted_embeddings.size(0)
        class_token = self.vision_transformer.embeddings.class_embedding.expand(batch_size, 1, -1)
        embeddings_with_cls = torch.cat([class_token, adapted_embeddings], dim=1)
        
        # 5. 添加位置编码 (需要处理位置编码尺寸不匹配的问题)
        num_patches = adapted_embeddings.size(1)
        original_pos_embed = self.vision_transformer.embeddings.position_embedding
        
        if original_pos_embed.size(1) != num_patches + 1:  # +1 for class token
            # 插值位置编码到匹配的尺寸
            pos_embed = self._interpolate_position_encoding(original_pos_embed, num_patches + 1)
        else:
            pos_embed = original_pos_embed
        
        embeddings_with_pos = embeddings_with_cls + pos_embed
        
        # 6. 通过vision transformer
        encoder_outputs = self.vision_transformer.encoder(
            embeddings_with_pos,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # 7. 取class token的输出
        pooled_output = encoder_outputs.last_hidden_state[:, 0]  # [batch_size, vision_embed_dim]
        
        # 8. 应用层归一化和投影
        pooled_output = self.vision_transformer.post_layernorm(pooled_output)
        image_features = self.visual_projection(pooled_output)  # [batch_size, projection_dim]
        
        return image_features
    
    def _interpolate_position_encoding(self, pos_embed: torch.Tensor, target_length: int) -> torch.Tensor:
        """插值位置编码到目标长度"""
        current_length = pos_embed.size(1)
        
        if current_length == target_length:
            return pos_embed
        
        # 分离class token的位置编码
        class_pos_embed = pos_embed[:, :1]  # [1, 1, embed_dim]
        patch_pos_embed = pos_embed[:, 1:]  # [1, current_patches, embed_dim]
        
        if target_length == 1:  # 只需要class token
            return class_pos_embed
        
        # 插值patch位置编码
        embed_dim = pos_embed.size(-1)
        current_patches = current_length - 1
        target_patches = target_length - 1
        
        # 计算原始patch grid尺寸
        current_grid_size = int(np.sqrt(current_patches))
        
        if current_grid_size * current_grid_size != current_patches:
            # 非方形grid，使用简单的线性插值
            patch_pos_embed_interp = F.interpolate(
                patch_pos_embed.transpose(1, 2),  # [1, embed_dim, current_patches]
                size=target_patches,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [1, target_patches, embed_dim]
        else:
            # 方形grid，使用2D插值
            patch_pos_embed_2d = patch_pos_embed.view(1, current_grid_size, current_grid_size, embed_dim)
            patch_pos_embed_2d = patch_pos_embed_2d.permute(0, 3, 1, 2)  # [1, embed_dim, grid_h, grid_w]
            
            target_grid_size = int(np.sqrt(target_patches))
            if target_grid_size * target_grid_size == target_patches:
                patch_pos_embed_interp = F.interpolate(
                    patch_pos_embed_2d,
                    size=(target_grid_size, target_grid_size),
                    mode='bilinear',
                    align_corners=False
                )
                patch_pos_embed_interp = patch_pos_embed_interp.permute(0, 2, 3, 1).view(1, target_patches, embed_dim)
            else:
                # 回退到线性插值
                patch_pos_embed_interp = F.interpolate(
                    patch_pos_embed.transpose(1, 2),
                    size=target_patches,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
        
        # 重新组合
        interpolated_pos_embed = torch.cat([class_pos_embed, patch_pos_embed_interp], dim=1)
        
        return interpolated_pos_embed
    
    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        编码文本为文本特征
        
        Args:
            text_inputs: tokenized text inputs
            
        Returns:
            text_features: [batch_size, projection_dim]
        """
        text_outputs = self.text_transformer(**text_inputs)
        pooled_output = text_outputs.pooler_output  # [batch_size, text_embed_dim]
        text_features = self.text_projection(pooled_output)  # [batch_size, projection_dim]
        
        return text_features
    
    def forward(
        self,
        doppler_spectrum: torch.Tensor,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            doppler_spectrum: [batch_size, 3, 64, 25]
            text_inputs: tokenized text inputs
            labels: [batch_size] ground truth labels
            return_dict: 是否返回字典格式
            
        Returns:
            outputs: 包含损失、logits、特征等信息
        """
        # 1. 编码图像
        image_features = self.encode_doppler_image(doppler_spectrum)  # [batch_size, projection_dim]
        
        outputs = {'image_features': image_features}
        
        # 2. 编码文本 (如果提供)
        if text_inputs is not None:
            text_features = self.encode_text(text_inputs)  # [batch_size, projection_dim]
            outputs['text_features'] = text_features
            
            # 3. 计算对比学习损失
            contrastive_loss = self.compute_contrastive_loss(image_features, text_features)
            outputs['contrastive_loss'] = contrastive_loss
            
            # 4. 多模态特征融合用于分类
            if self.use_custom_classifier:
                # 归一化特征
                image_features_norm = F.normalize(image_features, dim=-1)
                text_features_norm = F.normalize(text_features, dim=-1)
                
                # 拼接特征
                fused_features = torch.cat([image_features_norm, text_features_norm], dim=-1)
                logits = self.classifier(fused_features)
                outputs['logits'] = logits
                
                # 5. 计算分类损失
                if labels is not None:
                    classification_loss = self.classification_criterion(logits, labels)
                    outputs['classification_loss'] = classification_loss
                    
                    # 总损失
                    total_loss = classification_loss + 0.1 * contrastive_loss
                    outputs['loss'] = total_loss
        else:
            # 仅图像模式
            if self.use_custom_classifier:
                # 使用图像特征进行分类
                image_features_norm = F.normalize(image_features, dim=-1)
                # 简单地将图像特征重复一遍来匹配分类器输入维度
                fused_features = torch.cat([image_features_norm, image_features_norm], dim=-1)
                logits = self.classifier(fused_features)
                outputs['logits'] = logits
                
                if labels is not None:
                    classification_loss = self.classification_criterion(logits, labels)
                    outputs['classification_loss'] = classification_loss
                    outputs['loss'] = classification_loss
        
        if return_dict:
            return outputs
        else:
            return outputs.get('loss', None)
    
    def compute_contrastive_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算CLIP式的对比学习损失"""
        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算相似度矩阵
        logits_per_image = torch.matmul(image_features, text_features.T) / self.contrastive_temperature
        logits_per_text = logits_per_image.T
        
        # 创建标签 (对角线为正样本)
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # 计算双向对比损失
        loss_i2t = self.contrastive_criterion(logits_per_image, labels)
        loss_t2i = self.contrastive_criterion(logits_per_text, labels)
        
        contrastive_loss = (loss_i2t + loss_t2i) / 2
        
        return contrastive_loss
    
    def get_image_features(self, doppler_spectrum: torch.Tensor) -> torch.Tensor:
        """提取图像特征，用于知识蒸馏"""
        return self.encode_doppler_image(doppler_spectrum)
    
    def get_text_features(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取文本特征"""
        return self.encode_text(text_inputs)
    
    def predict(self, doppler_spectrum: torch.Tensor, text_inputs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """预测接口"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(doppler_spectrum, text_inputs)
            if 'logits' in outputs:
                return F.softmax(outputs['logits'], dim=-1)
            else:
                # 使用图像特征进行简单分类
                image_features = outputs['image_features']
                # 这里可以实现基于相似度的分类逻辑
                return torch.softmax(torch.randn(image_features.size(0), self.num_classes), dim=-1)


# 辅助函数
def create_doppler_clip_teacher(config: Dict) -> DopplerCLIPTeacher:
    """创建DopplerCLIP Teacher模型的工厂函数"""
    return DopplerCLIPTeacher(**config)


# 测试代码
if __name__ == "__main__":
    # 测试配置
    config = {
        'clip_model_name': "openai/clip-vit-base-patch32",
        'freeze_backbone_ratio': 0.7,
        'num_classes': 2,
        'use_custom_classifier': True
    }
    
    # 创建模型
    model = DopplerCLIPTeacher(**config)
    
    # 测试数据
    batch_size = 2
    doppler_data = torch.randn(batch_size, 3, 64, 25)
    labels = torch.tensor([1, 0])
    
    # 模拟文本输入
    text_descriptions = [
        "Device at 1.2m height. vegetation interference at 7.0m. human target detected at 1.0m. performing pacing.",
        "Device at 2.0m height. no human target detected."
    ]
    
    # 使用CLIP processor处理文本
    text_inputs = model.clip_processor(
        text=text_descriptions,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    print("=== DopplerCLIP Teacher 测试 ===")
    print(f"输入多普勒数据形状: {doppler_data.shape}")
    print(f"文本输入: {len(text_descriptions)} 个描述")
    
    # 前向传播
    outputs = model(doppler_data, text_inputs, labels)
    
    print(f"输出键: {list(outputs.keys())}")
    if 'logits' in outputs:
        print(f"分类logits形状: {outputs['logits'].shape}")
    if 'loss' in outputs:
        print(f"总损失: {outputs['loss'].item():.4f}")
    if 'image_features' in outputs:
        print(f"图像特征形状: {outputs['image_features'].shape}")
    if 'text_features' in outputs:
        print(f"文本特征形状: {outputs['text_features'].shape}")
    
    # 测试预测
    predictions = model.predict(doppler_data, text_inputs)
    print(f"预测结果形状: {predictions.shape}")
    print(f"预测概率: {predictions}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数比例: {(total_params - trainable_params) / total_params:.2%}")