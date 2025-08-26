# 示例数据说明

本目录包含用于演示和测试的示例多普勒雷达数据。

## 数据格式

所有数据文件均为NumPy数组格式 (.npy)，形状为 `[3, 64, 25]`：
- 第一维 (3): 3个range bins，对应不同的距离范围
- 第二维 (64): 64个频率bins，对应多普勒频移
- 第三维 (25): 25个时间帧，对应时间序列

## 文件命名规则

文件名包含丰富的场景信息，格式如下：
```
{有人标签}_{序号}_{设备配置}_{干扰描述}_{人体活动}.npy
```

示例：
- `1_1_设备高1.2m风吹绿植7m有人在1m前后踱步.npy`
  - 1: 有人
  - 1: 序号
  - 设备高1.2m: 雷达设备安装高度
  - 风吹绿植7m: 7米处有植被干扰
  - 有人在1m前后踱步: 1米处有人踱步

## 创建示例数据

如果需要创建更多示例数据用于测试，可以运行：

```python
import numpy as np
import os

def create_sample_data(filename, has_person=True, interference_level=0.1):
    """
    创建示例多普勒数据
    
    Args:
        filename: 文件名
        has_person: 是否包含人体信号
        interference_level: 干扰强度
    """
    # 基础噪声
    data = np.random.randn(3, 64, 25) * 0.1
    
    if has_person:
        # 添加人体多普勒信号（通常在低频部分）
        human_signal = np.random.randn(3, 10, 25) * 0.8
        data[:, 25:35, :] += human_signal
    
    # 添加干扰信号
    if interference_level > 0:
        interference = np.random.randn(3, 64, 25) * interference_level
        data += interference
    
    # 保存数据
    np.save(filename, data.astype(np.float32))
    return data

# 创建示例文件
sample_files = [
    ("1_1_设备高1.2m风吹绿植7m有人在1m前后踱步.npy", True, 0.2),
    ("0_2_设备高2m无干扰10m处无人.npy", False, 0.05),
    ("1_3_植被干扰_有人走动_3m.npy", True, 0.3),
    ("0_4_设备高1.5m窗帘摆动无人.npy", False, 0.15)
]

for filename, has_person, interference in sample_files:
    create_sample_data(filename, has_person, interference)
    print(f"创建示例文件: {filename}")
```

## 数据使用示例

```python
import numpy as np
from src.utils.text_processor import TextProcessor
from src.data.doppler_dataset import DopplerMultimodalDataset

# 加载数据
data = np.load("1_1_设备高1.2m风吹绿植7m有人在1m前后踱步.npy")
print(f"数据形状: {data.shape}")

# 解析文件名
processor = TextProcessor()
filename = "1_1_设备高1.2m风吹绿植7m有人在1m前后踱步.npy"
parsed_info, description = processor.process(filename)

print(f"解析结果: {parsed_info}")
print(f"文本描述: {description}")

# 使用数据集加载器
dataset = DopplerMultimodalDataset(
    data_dir="examples/sample_data",
    split="train"
)

sample = dataset[0]
print(f"样本键: {list(sample.keys())}")
```

## 注意事项

1. **数据大小**: 示例数据已经过压缩，实际数据可能更大
2. **真实性**: 这些是模拟数据，用于演示目的
3. **隐私**: 不包含任何真实的个人信息
4. **许可**: 示例数据可自由使用于研究和开发

## 获取真实数据

如需获取真实的多普勒雷达数据，请：
1. 联系相关研究机构
2. 使用公开的雷达数据集
3. 自行采集实验数据

请确保遵守相关的数据使用协议和隐私保护法规。