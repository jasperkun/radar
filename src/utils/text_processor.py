import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParsedInfo:
    """解析后的信息结构"""
    has_person: Optional[bool] = None
    device_height: Optional[float] = None
    distances: List[float] = None
    interference_types: List[str] = None
    activities: List[str] = None
    raw_filename: str = ""
    
    def __post_init__(self):
        if self.distances is None:
            self.distances = []
        if self.interference_types is None:
            self.interference_types = []
        if self.activities is None:
            self.activities = []


class TextProcessor:
    """
    文件名解析和文本描述生成器
    
    支持多种文件名格式的解析，包括：
    - 标准格式: "1_1_设备高1.2m风吹绿植7m有人在1m前后踱步.npy"
    - 简化格式: "有人_植被干扰_2m.npy" 
    - 扩展格式: "0_2_设备高2m无干扰10m处无人.npy"
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化文本处理器
        
        Args:
            config_path: 配置文件路径，包含自定义解析规则
        """
        self.patterns = self._init_patterns()
        self.interference_mapping = self._init_interference_mapping()
        self.activity_mapping = self._init_activity_mapping()
        
        if config_path:
            self._load_config(config_path)
    
    def _init_patterns(self) -> Dict[str, str]:
        """初始化正则表达式模式"""
        return {
            # 标签解析 (文件名开头的数字)
            'label': r'^(\d+)_',
            
            # 设备高度
            'device_height': r'设备高([0-9.]+)m',
            
            # 距离信息 (包括各种表述方式)
            'distances': r'([0-9.]+)m',
            
            # 干扰类型
            'interference': r'(风吹|摆动|晃动)?(绿植|植被|树木|灌木|草|叶子|枝条|窗帘|衣物|布料)',
            
            # 活动类型
            'activities': r'(踱步|走动|行走|站立|坐下|跳跃|挥手|转身|弯腰|静止|移动)',
            
            # 人的状态
            'person_status': r'(有人|无人|人体|目标)',
            
            # 位置描述
            'position': r'在([0-9.]+)m(处|前|后|左|右)',
        }
    
    def _init_interference_mapping(self) -> Dict[str, str]:
        """初始化干扰类型映射"""
        return {
            '绿植': 'vegetation',
            '植被': 'vegetation', 
            '树木': 'tree',
            '灌木': 'bush',
            '草': 'grass',
            '叶子': 'leaves',
            '枝条': 'branches',
            '窗帘': 'curtain',
            '衣物': 'clothing',
            '布料': 'fabric',
            '风吹': 'wind_blown',
            '摆动': 'swaying',
            '晃动': 'shaking'
        }
    
    def _init_activity_mapping(self) -> Dict[str, str]:
        """初始化活动类型映射"""
        return {
            '踱步': 'pacing',
            '走动': 'walking',
            '行走': 'walking', 
            '站立': 'standing',
            '坐下': 'sitting',
            '跳跃': 'jumping',
            '挥手': 'waving',
            '转身': 'turning',
            '弯腰': 'bending',
            '静止': 'stationary',
            '移动': 'moving'
        }
    
    def _load_config(self, config_path: str):
        """从配置文件加载自定义规则"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if 'patterns' in config:
                self.patterns.update(config['patterns'])
            if 'interference_mapping' in config:
                self.interference_mapping.update(config['interference_mapping'])
            if 'activity_mapping' in config:
                self.activity_mapping.update(config['activity_mapping'])
                
        except Exception as e:
            print(f"警告: 无法加载配置文件 {config_path}: {e}")
    
    def parse_filename(self, filename: str) -> ParsedInfo:
        """
        解析文件名，提取结构化信息
        
        Args:
            filename: 文件名 (如 "1_1_设备高1.2m风吹绿植7m有人在1m前后踱步.npy")
            
        Returns:
            ParsedInfo: 解析后的结构化信息
        """
        parsed = ParsedInfo(raw_filename=filename)
        
        # 去除文件扩展名
        name_without_ext = filename.replace('.npy', '').replace('.NPY', '')
        
        # 1. 解析标签 (有人/无人)
        label_match = re.search(self.patterns['label'], name_without_ext)
        if label_match:
            parsed.has_person = bool(int(label_match.group(1)))
        else:
            # 备用方案：从文件名中直接查找
            if '有人' in name_without_ext:
                parsed.has_person = True
            elif '无人' in name_without_ext:
                parsed.has_person = False
        
        # 2. 解析设备高度
        height_match = re.search(self.patterns['device_height'], name_without_ext)
        if height_match:
            parsed.device_height = float(height_match.group(1))
        
        # 3. 解析距离信息
        distance_matches = re.findall(self.patterns['distances'], name_without_ext)
        parsed.distances = [float(d) for d in distance_matches if d]
        
        # 4. 解析干扰类型
        interference_matches = re.findall(self.patterns['interference'], name_without_ext)
        for match in interference_matches:
            if isinstance(match, tuple):
                # 处理复合匹配 (如 "风吹绿植")
                for part in match:
                    if part and part in self.interference_mapping:
                        english_term = self.interference_mapping[part]
                        if english_term not in parsed.interference_types:
                            parsed.interference_types.append(english_term)
            else:
                if match in self.interference_mapping:
                    english_term = self.interference_mapping[match]
                    if english_term not in parsed.interference_types:
                        parsed.interference_types.append(english_term)
        
        # 5. 解析活动类型
        activity_matches = re.findall(self.patterns['activities'], name_without_ext)
        for activity in activity_matches:
            if activity in self.activity_mapping:
                english_term = self.activity_mapping[activity]
                if english_term not in parsed.activities:
                    parsed.activities.append(english_term)
        
        return parsed
    
    def generate_text_description(self, parsed_info: ParsedInfo) -> str:
        """
        根据解析信息生成英文文本描述
        
        Args:
            parsed_info: 解析后的信息
            
        Returns:
            str: 英文文本描述
        """
        components = []
        
        # 1. 设备配置
        if parsed_info.device_height is not None:
            components.append(f"Device mounted at {parsed_info.device_height}m height")
        
        # 2. 干扰信息
        if parsed_info.interference_types:
            interference_str = ", ".join(parsed_info.interference_types)
            if parsed_info.distances:
                # 尝试匹配干扰距离 (通常是较远的距离)
                far_distances = [d for d in parsed_info.distances if d > 3.0]
                if far_distances:
                    max_dist = max(far_distances)
                    components.append(f"{interference_str} interference at {max_dist}m")
                else:
                    components.append(f"{interference_str} interference present")
            else:
                components.append(f"{interference_str} interference present")
        
        # 3. 人体目标信息
        if parsed_info.has_person is not None:
            if parsed_info.has_person:
                # 找到人体可能的距离 (通常是较近的距离)
                if parsed_info.distances:
                    close_distances = [d for d in parsed_info.distances if d <= 3.0]
                    if close_distances:
                        min_dist = min(close_distances)
                        components.append(f"human target detected at {min_dist}m")
                    else:
                        components.append("human target detected")
                else:
                    components.append("human target detected")
                
                # 添加活动信息
                if parsed_info.activities:
                    activity_str = ", ".join(parsed_info.activities)
                    components.append(f"performing {activity_str}")
            else:
                components.append("no human target detected")
        
        # 4. 场景复杂度描述
        if len(parsed_info.interference_types) > 0 and parsed_info.has_person:
            components.append("challenging detection scenario with interference")
        
        # 组合所有组件
        if components:
            description = ". ".join(components).strip()
            # 确保以句号结尾
            if not description.endswith('.'):
                description += '.'
            return description
        else:
            return "Doppler radar detection scenario."
    
    def process(self, filename: str) -> Tuple[ParsedInfo, str]:
        """
        完整处理流程：解析文件名并生成文本描述
        
        Args:
            filename: 输入文件名
            
        Returns:
            Tuple[ParsedInfo, str]: (解析信息, 文本描述)
        """
        parsed_info = self.parse_filename(filename)
        text_description = self.generate_text_description(parsed_info)
        return parsed_info, text_description
    
    def batch_process(self, filenames: List[str]) -> List[Tuple[ParsedInfo, str]]:
        """
        批量处理文件名
        
        Args:
            filenames: 文件名列表
            
        Returns:
            List[Tuple[ParsedInfo, str]]: 处理结果列表
        """
        results = []
        for filename in filenames:
            try:
                result = self.process(filename)
                results.append(result)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                # 返回默认结果
                default_info = ParsedInfo(raw_filename=filename)
                default_desc = "Doppler radar detection scenario."
                results.append((default_info, default_desc))
        
        return results
    
    def validate_parsing(self, filename: str, expected_label: Optional[int] = None) -> bool:
        """
        验证解析结果的正确性
        
        Args:
            filename: 文件名
            expected_label: 期望的标签 (0 or 1)
            
        Returns:
            bool: 解析是否正确
        """
        parsed_info, _ = self.process(filename)
        
        if expected_label is not None:
            expected_has_person = bool(expected_label)
            return parsed_info.has_person == expected_has_person
        
        # 基本验证：至少应该能解析出有人/无人信息
        return parsed_info.has_person is not None


# 使用示例和测试
if __name__ == "__main__":
    processor = TextProcessor()
    
    # 测试用例
    test_cases = [
        "1_1_设备高1.2m风吹绿植7m有人在1m前后踱步.npy",
        "0_2_设备高2m无干扰10m处无人.npy", 
        "1_3_植被干扰_有人走动.npy",
        "0_1_设备高1.5m窗帘摆动无人.npy"
    ]
    
    print("=== 文件名解析测试 ===")
    for filename in test_cases:
        parsed_info, description = processor.process(filename)
        print(f"\n文件名: {filename}")
        print(f"解析结果: {parsed_info}")
        print(f"文本描述: {description}")
        print("-" * 80)