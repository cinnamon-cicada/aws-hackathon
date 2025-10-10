# 主函数详细指南 - Main Function Guide

## 🎯 主函数位置

**文件**: `backend/disaster_analysis_main.py`
**入口点**: 第547-548行
```python
if __name__ == "__main__":
    sys.exit(main())
```

## 🔧 主函数结构

### main() 函数 (第413行)
```python
def main():
    """主函数 - 解析参数并运行完整分析"""
    # 1. 参数解析 (第415-434行)
    # 2. 输入验证 (第480-512行)
    # 3. 创建分析器 (第529行)
    # 4. 运行完整分析 (第530-537行)
    # 5. 错误处理 (第542-544行)
```

## 📊 参数解析

### 输入选项
```python
# 互斥参数组
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--example4", action="store_true")  # 使用测试数据
input_group.add_argument("--before", type=str)              # 自定义图片
```

### 分析参数
```python
parser.add_argument("--output-dir", default="outputs")      # 输出目录
parser.add_argument("--grid-size", type=int, default=32)    # 网格大小
parser.add_argument("--damage-weight", type=float, default=0.6)  # 损害权重
parser.add_argument("--density-weight", type=float, default=0.4) # 密度权重
parser.add_argument("--bbox", type=str)                     # 地理边界框
parser.add_argument("--verbose", action="store_true")       # 详细日志
```

## 🏗️ DisasterAnalyzer 类

### 类初始化
```python
class DisasterAnalyzer:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.results = {}
```

### 核心方法

#### 1. analyze_building_density() - 建筑密度分析
```python
def analyze_building_density(self, before_image_path: str):
    """Step 1: 从灾难前图片生成建筑密度热力图"""
    # 加载图片
    before_image = cv2.imread(before_image_path)
    
    # 生成建筑密度热力图
    density_heatmap = create_light_gray_heatmap(before_image, radius=60, sigma=30.0)
    
    # 保存热力图和叠加图
    # 返回结果字典
```

#### 2. analyze_damage_assessment() - 灾难影响评估
```python
def analyze_damage_assessment(self, before_image_path: str, after_image_path: str):
    """Step 2: 对比灾难前后图片生成损害评估"""
    # 处理图片对
    damage_geojson, patch_results_raw = process_image_pair(
        before_path, after_path, geojson_path, grid_size, bbox
    )
    
    # 创建损害热力图
    damage_heatmap = create_damage_heatmap(patch_features, after_image.shape)
    
    # 保存结果
```

#### 3. fuse_analysis_results() - 数据融合
```python
def fuse_analysis_results(self, damage_weight: float = 0.6, density_weight: float = 0.4):
    """Step 3: 融合建筑密度和灾难影响数据"""
    # 获取之前的结果
    density_heatmap = self.results['building_density']['heatmap']
    damage_heatmap = self.results['damage_assessment']['heatmap']
    
    # 融合热力图
    fused_heatmap = fuse_heatmaps(damage_heatmap, density_heatmap, weights)
    
    # 计算最终紧急程度
    final_patches = calculate_urgency(patches_with_density, weights)
```

#### 4. create_interactive_map() - 生成HTML地图
```python
def create_interactive_map(self, bbox: Optional[List[float]] = None):
    """Step 4: 创建交互式HTML地图"""
    # 获取融合结果
    patch_features = self.results['fusion']['patch_features']
    
    # 创建交互式地图
    create_fusion_map(patch_features, image_path, bbox, html_path)
```

#### 5. run_complete_analysis() - 完整分析流程
```python
def run_complete_analysis(self, before_image_path: str, after_image_path: str):
    """运行完整的灾难分析工作流"""
    # Step 1: 建筑密度分析
    density_results = self.analyze_building_density(before_image_path)
    
    # Step 2: 灾难影响评估
    damage_results = self.analyze_damage_assessment(before_image_path, after_image_path)
    
    # Step 3: 数据融合
    fusion_results = self.fuse_analysis_results(damage_weight, density_weight)
    
    # Step 4: 生成交互式地图
    html_map_path = self.create_interactive_map(bbox)
    
    # 返回所有结果
    return self.results
```

## 🔄 执行流程

### 完整流程
```
命令行输入
    ↓
main() 函数 (第413行)
    ↓
参数解析和验证
    ↓
创建 DisasterAnalyzer 实例 (第529行)
    ↓
调用 run_complete_analysis() (第530行)
    ↓
├── analyze_building_density() - 建筑密度分析
├── analyze_damage_assessment() - 灾难影响评估
├── fuse_analysis_results() - 数据融合
└── create_interactive_map() - 生成HTML地图
    ↓
输出结果到指定目录
```

### 模块调用关系
```
disaster_analysis_main.py (主控制器)
    ├── house.py (建筑密度检测)
    │   └── create_light_gray_heatmap()
    ├── mapping.py (变化检测)
    │   └── process_image_pair()
    └── fusion.py (数据融合)
        ├── create_damage_heatmap()
        ├── fuse_heatmaps()
        ├── calculate_urgency()
        └── create_fusion_map()
```

## 🎯 使用示例

### 基本使用
```bash
python disaster_analysis_main.py --example4
```

### 自定义图片
```bash
python disaster_analysis_main.py --before before.png --after after.png
```

### 完整参数
```bash
python disaster_analysis_main.py \
    --before before.png \
    --after after.png \
    --output-dir my_results \
    --grid-size 64 \
    --damage-weight 0.7 \
    --density-weight 0.3 \
    --bbox "-86.8,36.1,-86.7,36.2" \
    --verbose
```

## 📊 输出结果

### 生成的文件
```
outputs/
├── heatmaps/
│   ├── building_density_heatmap.png    # 建筑密度热力图
│   ├── damage_heatmap.png              # 灾难影响热力图
│   └── fused_heatmap.png               # 融合紧急程度热力图
├── overlays/
│   ├── building_density_overlay.png    # 建筑密度叠加图
│   ├── damage_overlay.png              # 灾难影响叠加图
│   └── fused_overlay.png               # 融合叠加图
├── geojson/
│   └── damage_assessment.geojson       # 详细分析数据
└── html_maps/
    └── disaster_analysis_map.html      # 交互式地图
```

### 结果说明
- **建筑密度热力图**: 显示建筑分布密度
- **灾难影响热力图**: 显示损害程度
- **融合紧急程度热力图**: 最终的综合评估
- **交互式HTML地图**: 包含网格和图层切换功能

## 🔧 技术特点

### 模块化设计
- **清晰的职责分离**: 每个方法负责特定功能
- **统一的接口**: 标准化的方法签名
- **易于扩展**: 可以轻松添加新功能

### 错误处理
- **输入验证**: 检查图片文件是否存在
- **异常捕获**: 完善的错误处理机制
- **详细日志**: 提供详细的执行信息

### 灵活配置
- **参数化**: 支持自定义分析参数
- **权重调整**: 可调整损害和密度权重
- **输出控制**: 可指定输出目录和格式

## 🐛 故障排除

### 常见问题
1. **图片加载失败**: 检查图片路径和格式
2. **内存不足**: 减小 `--grid-size` 参数
3. **分析结果异常**: 检查图片质量和对比度
4. **地图显示空白**: 确保有网络连接

### 调试模式
```bash
python disaster_analysis_main.py --example4 --verbose
```

## 📞 技术支持

### 系统要求
- Python >= 3.7
- 所有依赖包已安装
- 图片文件存在且可读
- 输出目录有写入权限

### 依赖包
```
numpy
opencv-python
scikit-image
shapely
folium
rasterio (可选)
```

---

**主函数入口**: `backend/disaster_analysis_main.py` 第547行
