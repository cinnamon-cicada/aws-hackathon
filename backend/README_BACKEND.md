# 后端核心系统 - Backend Core System

## 🎯 系统概述

这是灾难分析系统的后端核心，包含完整的图像分析功能。

## 📁 核心文件结构

```
backend/
├── disaster_analysis_main.py      # 🎯 主执行脚本
├── fusion.py                      # 数据融合模块
├── house.py                       # 建筑密度检测
├── mapping.py                     # 变化检测模块
├── config.py                      # 配置文件
├── requirements.txt               # 依赖包
├── test_data/                     # 测试数据
│   └── mapping examples/
│       ├── 4pre.png              # 灾难前图片
│       └── 4post.png             # 灾难后图片
└── outputs/                       # 输出结果
    ├── heatmaps/                  # 热力图文件
    ├── overlays/                  # 叠加图片
    ├── geojson/                   # 地理数据
    └── html_maps/                 # 交互式地图
```

## 🚀 主函数说明

### 入口点
**`disaster_analysis_main.py`** - 系统主入口

### 核心类
```python
class DisasterAnalyzer:
    """灾难分析器 - 协调整个分析流程"""
    
    def analyze_building_density(self):      # 建筑密度分析
    def analyze_damage_assessment(self):     # 灾难影响评估
    def fuse_analysis_results(self):         # 数据融合
    def create_interactive_map(self):        # 生成HTML地图
    def run_complete_analysis(self):         # 完整分析流程
```

### 模块依赖
```python
from mapping import process_image_pair           # 变化检测
from house import create_light_gray_heatmap      # 建筑密度
from fusion import (                             # 数据融合
    create_damage_heatmap,
    fuse_heatmaps,
    create_fusion_map
)
```

## 🔧 核心功能流程

### 1. 建筑密度分析
```python
# 调用 house.py
density_heatmap = create_light_gray_heatmap(before_image)
```

### 2. 灾难影响评估
```python
# 调用 mapping.py
damage_geojson, patch_results = process_image_pair(
    before_path, after_path, geojson_path, grid_size, bbox
)
```

### 3. 数据融合
```python
# 调用 fusion.py
fused_heatmap = fuse_heatmaps(damage_heatmap, density_heatmap, weights)
```

### 4. 生成HTML地图
```python
# 调用 fusion.py
create_fusion_map(patch_features, image_path, bbox, html_path)
```

## 🎯 使用方法

### 快速开始
```bash
cd backend
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
    --output-dir results \
    --grid-size 64 \
    --damage-weight 0.7 \
    --density-weight 0.3 \
    --verbose
```

## 📊 输出结果

### 生成文件
- **热力图**: `heatmaps/` - 纯热力图图片
- **叠加图**: `overlays/` - 热力图叠加在原始图片上
- **GeoJSON**: `geojson/` - 详细分析数据
- **HTML地图**: `html_maps/` - 交互式地图

### 核心输出
1. **建筑密度热力图** - 显示建筑分布
2. **灾难影响热力图** - 显示损害程度
3. **融合紧急程度热力图** - 最终综合评估
4. **交互式HTML地图** - 包含网格和图层切换

## 🎨 HTML地图功能

### 交互式特性
- **网格显示**: 分析网格边界
- **图层切换**: 4个可切换图层
- **点击交互**: 查看详细信息
- **图层控制**: 右上角控制面板

### 可切换图层
1. **Post-Disaster Satellite Image** - 灾难后图片
2. **Damage Assessment** - 灾难影响评估
3. **Building Density** - 建筑密度
4. **Fused Urgency** - 融合紧急程度

## 🔧 技术特点

### 模块化设计
- **清晰的职责分离**: 每个模块负责特定功能
- **统一的接口**: 标准化的方法签名
- **易于扩展**: 可以轻松添加新功能

### 核心优势
- **完整流程**: 从图片分析到可视化
- **灵活配置**: 支持自定义参数
- **多种输出**: 热力图、叠加图、HTML地图
- **交互式**: 支持图层切换和详细信息

## 🐛 故障排除

### 常见问题
1. **图片加载失败**: 检查路径和格式
2. **内存不足**: 减小grid-size参数
3. **分析结果异常**: 检查图片质量
4. **地图显示空白**: 确保网络连接

### 调试模式
```bash
python disaster_analysis_main.py --example4 --verbose
```

## 📞 技术支持

### 系统要求
- Python >= 3.7
- 依赖包已安装
- 图片文件可读
- 输出目录可写

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

**开始使用**: `python disaster_analysis_main.py --example4`
