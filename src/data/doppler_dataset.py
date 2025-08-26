import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from pathlib import Path
import json
import random
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor
import albumentations as A
import cv2

from ..utils.text_processor import TextProcessor, ParsedInfo


class DopplerAugmentation:
    """多普勒谱图数据增强"""
    
    def __init__(
        self,
        noise_std: float = 0.01,
        time_shift_max: int = 3,
        freq_shift_max: int = 2,
        amplitude_scale_range: Tuple[float, float] = (0.8, 1.2),
        enable_mixup: bool = False,
        mixup_alpha: float = 0.2
    ):
        """
        初始化数据增强
        
        Args:
            noise_std: 高斯噪声标准差
            time_shift_max: 时间维度最大偏移
            freq_shift_max: 频率维度最大偏移
            amplitude_scale_range: 幅度缩放范围
            enable_mixup: 是否启用mixup增强
            mixup_alpha: mixup参数
        """
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.freq_shift_max = freq_shift_max
        self.amplitude_scale_range = amplitude_scale_range
        self.enable_mixup = enable_mixup
        self.mixup_alpha = mixup_alpha
    
    def add_noise(self, spectrum: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, spectrum.shape)
        return spectrum + noise
    
    def time_shift(self, spectrum: np.ndarray) -> np.ndarray:
        """时间维度偏移"""
        if self.time_shift_max == 0:
            return spectrum
        
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
        if shift == 0:
            return spectrum
        
        shifted = np.roll(spectrum, shift, axis=-1)
        
        # 填充边界
        if shift > 0:
            shifted[:, :, :shift] = 0
        else:
            shifted[:, :, shift:] = 0
        
        return shifted
    
    def freq_shift(self, spectrum: np.ndarray) -> np.ndarray:
        """频率维度偏移"""
        if self.freq_shift_max == 0:
            return spectrum
        
        shift = np.random.randint(-self.freq_shift_max, self.freq_shift_max + 1)
        if shift == 0:
            return spectrum
        
        shifted = np.roll(spectrum, shift, axis=1)
        
        # 填充边界
        if shift > 0:
            shifted[:, :shift, :] = 0
        else:
            shifted[:, shift:, :] = 0
        
        return shifted
    
    def amplitude_scale(self, spectrum: np.ndarray) -> np.ndarray:
        """幅度缩放"""
        scale = np.random.uniform(*self.amplitude_scale_range)
        return spectrum * scale
    
    def apply_augmentation(self, spectrum: np.ndarray, apply_prob: float = 0.5) -> np.ndarray:
        """应用数据增强"""
        augmented = spectrum.copy()
        
        # 随机应用各种增强
        if np.random.random() < apply_prob:
            augmented = self.add_noise(augmented)
        
        if np.random.random() < apply_prob:
            augmented = self.time_shift(augmented)
        
        if np.random.random() < apply_prob:
            augmented = self.freq_shift(augmented)
        
        if np.random.random() < apply_prob:
            augmented = self.amplitude_scale(augmented)
        
        return augmented


class DopplerMultimodalDataset(Dataset):
    """
    多模态多普勒雷达数据集
    
    支持功能：
    1. 加载多普勒谱图数据 (.npy文件)
    2. 从文件名解析生成文本描述
    3. 数据增强
    4. 训练/验证/测试集划分
    5. 多模态数据对齐
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # 'train', 'val', 'test'
        text_processor: Optional[TextProcessor] = None,
        clip_processor: Optional[CLIPProcessor] = None,
        transform: Optional[DopplerAugmentation] = None,
        max_text_length: int = 77,
        target_shape: Tuple[int, int, int] = (3, 64, 25),
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        filter_complex_scenes: bool = False,
        min_interference_samples: int = 100,
        balance_classes: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            split: 数据集分割 ('train', 'val', 'test')
            text_processor: 文本处理器
            clip_processor: CLIP文本处理器
            transform: 数据增强器
            max_text_length: 最大文本长度
            target_shape: 目标数据形状
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
            filter_complex_scenes: 是否筛选复杂场景
            min_interference_samples: 最小干扰样本数
            balance_classes: 是否平衡类别
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_shape = target_shape
        self.max_text_length = max_text_length
        self.use_cache = use_cache
        self.filter_complex_scenes = filter_complex_scenes
        self.balance_classes = balance_classes
        
        # 初始化处理器
        self.text_processor = text_processor or TextProcessor()
        self.clip_processor = clip_processor
        self.transform = transform
        
        # 设置缓存目录
        if cache_dir is None:
            cache_dir = self.data_dir / 'cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 设置随机种子
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # 加载和分割数据
        self.data_list = self._load_data_list()
        self.train_files, self.val_files, self.test_files = self._split_dataset(
            train_ratio, val_ratio, test_ratio, random_seed
        )
        
        # 获取当前分割的文件列表
        if split == 'train':
            self.files = self.train_files
        elif split == 'val':
            self.files = self.val_files
        else:
            self.files = self.test_files
        
        # 类别平衡
        if balance_classes and split == 'train':
            self.files = self._balance_classes(self.files)
        
        # 复杂场景过滤
        if filter_complex_scenes:
            self.files = self._filter_complex_scenes(self.files)
        
        print(f"数据集 '{split}' 加载完成: {len(self.files)} 个样本")
        self._print_dataset_info()
    
    def _load_data_list(self) -> List[Dict]:
        """加载数据文件列表"""
        cache_file = self.cache_dir / 'data_list.json'
        
        if self.use_cache and cache_file.exists():
            print("从缓存加载数据列表...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print("扫描数据目录...")
        data_list = []
        
        # 遍历数据目录
        for file_path in self.data_dir.rglob('*.npy'):
            try:
                # 解析文件名
                filename = file_path.name
                parsed_info, text_description = self.text_processor.process(filename)
                
                # 验证标签
                if parsed_info.has_person is None:
                    print(f"警告: 无法解析标签 - {filename}")
                    continue
                
                # 检查文件是否存在且可读
                if not file_path.exists():
                    continue
                
                # 尝试加载数据验证格式
                try:
                    data = np.load(file_path)
                    if data.shape != self.target_shape:
                        print(f"警告: 数据形状不匹配 {data.shape} != {self.target_shape} - {filename}")
                        continue
                except Exception as e:
                    print(f"警告: 无法加载数据 - {filename}: {e}")
                    continue
                
                data_item = {
                    'file_path': str(file_path),
                    'filename': filename,
                    'label': int(parsed_info.has_person),
                    'text_description': text_description,
                    'parsed_info': {
                        'has_person': parsed_info.has_person,
                        'device_height': parsed_info.device_height,
                        'distances': parsed_info.distances,
                        'interference_types': parsed_info.interference_types,
                        'activities': parsed_info.activities
                    }
                }
                
                data_list.append(data_item)
                
            except Exception as e:
                print(f"处理文件时出错 {file_path}: {e}")
                continue
        
        print(f"成功加载 {len(data_list)} 个数据文件")
        
        # 保存缓存
        if self.use_cache:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        return data_list
    
    def _split_dataset(
        self, 
        train_ratio: float, 
        val_ratio: float, 
        test_ratio: float,
        random_seed: int
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """分割数据集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "分割比例之和必须为1"
        
        # 按标签分层分割
        labels = [item['label'] for item in self.data_list]
        
        # 首先分离训练集
        train_files, temp_files = train_test_split(
            self.data_list, 
            test_size=(1 - train_ratio),
            stratify=labels,
            random_state=random_seed
        )
        
        # 再分离验证集和测试集
        temp_labels = [item['label'] for item in temp_files]
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(1 - relative_val_ratio),
            stratify=temp_labels,
            random_state=random_seed
        )
        
        print(f"数据集分割: 训练集={len(train_files)}, 验证集={len(val_files)}, 测试集={len(test_files)}")
        
        return train_files, val_files, test_files
    
    def _balance_classes(self, files: List[Dict]) -> List[Dict]:
        """平衡类别分布"""
        # 统计各类别样本数
        class_counts = {}
        class_samples = {}
        
        for item in files:
            label = item['label']
            if label not in class_counts:
                class_counts[label] = 0
                class_samples[label] = []
            class_counts[label] += 1
            class_samples[label].append(item)
        
        print(f"原始类别分布: {class_counts}")
        
        # 找到最小类别的样本数
        min_count = min(class_counts.values())
        
        # 每个类别随机采样到相同数量
        balanced_files = []
        for label, samples in class_samples.items():
            if len(samples) > min_count:
                samples = random.sample(samples, min_count)
            balanced_files.extend(samples)
        
        # 打乱顺序
        random.shuffle(balanced_files)
        
        new_class_counts = {}
        for item in balanced_files:
            label = item['label']
            new_class_counts[label] = new_class_counts.get(label, 0) + 1
        
        print(f"平衡后类别分布: {new_class_counts}")
        
        return balanced_files
    
    def _filter_complex_scenes(self, files: List[Dict]) -> List[Dict]:
        """筛选复杂干扰场景"""
        complex_files = []
        
        for item in files:
            parsed_info = item['parsed_info']
            
            # 定义复杂场景的条件
            is_complex = False
            
            # 1. 有干扰且有人体目标
            if parsed_info['interference_types'] and parsed_info['has_person']:
                is_complex = True
            
            # 2. 距离差异大（近距离干扰 + 远距离目标）
            distances = parsed_info['distances']
            if len(distances) >= 2:
                min_dist = min(distances)
                max_dist = max(distances)
                if max_dist - min_dist > 5.0:  # 距离差大于5米
                    is_complex = True
            
            # 3. 多种干扰类型
            if len(parsed_info['interference_types']) >= 2:
                is_complex = True
            
            if is_complex:
                complex_files.append(item)
        
        print(f"筛选出 {len(complex_files)} 个复杂场景样本")
        
        return complex_files
    
    def _print_dataset_info(self):
        """打印数据集信息"""
        # 类别分布
        class_counts = {}
        interference_counts = {}
        
        for item in self.files:
            label = item['label']
            class_counts[label] = class_counts.get(label, 0) + 1
            
            # 干扰类型统计
            interference_types = item['parsed_info']['interference_types']
            for interference in interference_types:
                interference_counts[interference] = interference_counts.get(interference, 0) + 1
        
        print(f"=== {self.split.upper()} 数据集信息 ===")
        print(f"总样本数: {len(self.files)}")
        print(f"类别分布: {class_counts}")
        if interference_counts:
            print(f"干扰类型分布: {interference_counts}")
        
        # 距离分布统计
        all_distances = []
        for item in self.files:
            all_distances.extend(item['parsed_info']['distances'])
        
        if all_distances:
            print(f"距离范围: {min(all_distances):.1f}m - {max(all_distances):.1f}m")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """获取单个样本"""
        item = self.files[idx]
        
        # 加载多普勒数据
        try:
            doppler_data = np.load(item['file_path']).astype(np.float32)
        except Exception as e:
            print(f"加载数据失败 {item['file_path']}: {e}")
            # 返回零数据作为备选
            doppler_data = np.zeros(self.target_shape, dtype=np.float32)
        
        # 数据增强
        if self.transform is not None and self.split == 'train':
            doppler_data = self.transform.apply_augmentation(doppler_data)
        
        # 转换为tensor
        doppler_tensor = torch.from_numpy(doppler_data)
        
        # 准备返回数据
        sample = {
            'doppler_spectrum': doppler_tensor,
            'label': torch.tensor(item['label'], dtype=torch.long),
            'text_description': item['text_description'],
            'filename': item['filename'],
            'file_path': item['file_path']
        }
        
        # 处理文本输入（如果有CLIP processor）
        if self.clip_processor is not None:
            text_inputs = self.clip_processor(
                text=item['text_description'],
                return_tensors="pt",
                padding='max_length',
                max_length=self.max_text_length,
                truncation=True
            )
            
            # 移除batch维度
            for key, value in text_inputs.items():
                sample[f'text_{key}'] = value.squeeze(0)
        
        return sample
    
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重用于损失函数"""
        class_counts = {}
        for item in self.files:
            label = item['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self.files)
        num_classes = len(class_counts)
        
        weights = []
        for i in range(num_classes):
            if i in class_counts:
                weight = total_samples / (num_classes * class_counts[i])
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_complex_scene_subset(self) -> 'DopplerMultimodalDataset':
        """获取复杂场景子集"""
        complex_files = self._filter_complex_scenes(self.files)
        
        # 创建新的数据集实例
        subset = DopplerMultimodalDataset.__new__(DopplerMultimodalDataset)
        subset.__dict__.update(self.__dict__)
        subset.files = complex_files
        
        return subset


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    augmentation_config: Optional[Dict] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批大小
        num_workers: 工作进程数
        clip_model_name: CLIP模型名称
        augmentation_config: 数据增强配置
        **dataset_kwargs: 数据集额外参数
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 初始化处理器
    text_processor = TextProcessor()
    
    try:
        from transformers import CLIPProcessor
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    except Exception as e:
        print(f"警告: 无法加载CLIP processor: {e}")
        clip_processor = None
    
    # 初始化数据增强
    transform = None
    if augmentation_config:
        transform = DopplerAugmentation(**augmentation_config)
    
    # 创建数据集
    train_dataset = DopplerMultimodalDataset(
        data_dir=data_dir,
        split='train',
        text_processor=text_processor,
        clip_processor=clip_processor,
        transform=transform,
        **dataset_kwargs
    )
    
    val_dataset = DopplerMultimodalDataset(
        data_dir=data_dir,
        split='val',
        text_processor=text_processor,
        clip_processor=clip_processor,
        transform=None,  # 验证集不使用数据增强
        **dataset_kwargs
    )
    
    test_dataset = DopplerMultimodalDataset(
        data_dir=data_dir,
        split='test',
        text_processor=text_processor,
        clip_processor=clip_processor,
        transform=None,  # 测试集不使用数据增强
        **dataset_kwargs
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# 测试代码
if __name__ == "__main__":
    # 测试数据集
    print("=== DopplerMultimodalDataset 测试 ===")
    
    # 创建测试数据目录（如果不存在）
    test_data_dir = "./test_data"
    if not os.path.exists(test_data_dir):
        print("创建测试数据...")
        os.makedirs(test_data_dir)
        
        # 创建一些测试文件
        test_files = [
            "1_1_设备高1.2m风吹绿植7m有人在1m前后踱步.npy",
            "0_2_设备高2m无干扰10m处无人.npy",
            "1_3_植被干扰_有人走动_3m.npy",
            "0_1_设备高1.5m窗帘摆动无人.npy"
        ]
        
        for filename in test_files:
            filepath = os.path.join(test_data_dir, filename)
            # 创建假数据
            fake_data = np.random.randn(3, 64, 25).astype(np.float32)
            np.save(filepath, fake_data)
    
    try:
        # 创建数据集
        dataset = DopplerMultimodalDataset(
            data_dir=test_data_dir,
            split='train',
            balance_classes=True
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            # 测试获取样本
            sample = dataset[0]
            print(f"样本键: {list(sample.keys())}")
            print(f"多普勒数据形状: {sample['doppler_spectrum'].shape}")
            print(f"标签: {sample['label']}")
            print(f"文本描述: {sample['text_description']}")
            
            # 测试类别权重
            class_weights = dataset.get_class_weights()
            print(f"类别权重: {class_weights}")
        
        # 测试数据加载器创建
        if len(dataset) > 0:
            print("\n测试数据加载器创建...")
            try:
                train_loader, val_loader, test_loader = create_dataloaders(
                    data_dir=test_data_dir,
                    batch_size=2,
                    num_workers=0  # 避免多进程问题
                )
                
                print(f"训练集批次数: {len(train_loader)}")
                print(f"验证集批次数: {len(val_loader)}")
                print(f"测试集批次数: {len(test_loader)}")
                
                # 测试一个批次
                if len(train_loader) > 0:
                    batch = next(iter(train_loader))
                    print(f"批次数据形状: {batch['doppler_spectrum'].shape}")
                    print(f"批次标签形状: {batch['label'].shape}")
                
            except Exception as e:
                print(f"数据加载器测试失败: {e}")
        
    except Exception as e:
        print(f"数据集测试失败: {e}")
        import traceback
        traceback.print_exc()