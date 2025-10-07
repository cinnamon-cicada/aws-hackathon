# 🚀 Quick Start Guide - Mapping Module

## 快速开始（中文）

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 运行测试示例

最简单的方式 - 生成测试数据并自动处理：

```bash
python3 mapping.py --generate-test
```

这会：
- ✅ 生成模拟的灾前/灾后图像
- ✅ 运行完整的变化检测流程
- ✅ 输出 GeoJSON 损伤地图
- ✅ 创建交互式 HTML 可视化

输出文件在 `test_data/` 目录：
- `before.png` - 灾前图像
- `after.png` - 灾后图像（中心区域有损伤）
- `damage.geojson` - 损伤评分数据
- `damage.html` - 交互式地图（用浏览器打开）

### 3. 处理真实图像

#### 方式 A：GeoTIFF 图像（推荐）

如果你的卫星图像是 GeoTIFF 格式（带地理坐标信息）：

```bash
python3 mapping.py \
    --before data/before.tif \
    --after data/after.tif \
    --out results/damage.geojson \
    --visualize
```

#### 方式 B：普通图像（PNG/JPG）

如果是普通图像，需要手动指定边界框：

```bash
python3 mapping.py \
    --before data/before.png \
    --after data/after.png \
    --out results/damage.geojson \
    --bbox "-86.8,36.1,-86.7,36.2" \
    --visualize
```

边界框格式：`"最小经度,最小纬度,最大经度,最大纬度"`

### 4. 调整参数

#### 调整网格大小（影响精度和速度）

```bash
# 更小的网格 = 更精细的分析，但更慢
python3 mapping.py --before before.tif --after after.tif --out damage.geojson --grid 16

# 更大的网格 = 更快，但分辨率低
python3 mapping.py --before before.tif --after after.tif --out damage.geojson --grid 64
```

#### 调整损伤评分权重

```bash
# 更重视结构变化（边缘检测）
python3 mapping.py --before before.tif --after after.tif --out damage.geojson \
    --weights '{"edge":0.7,"mean":0.2,"var":0.1}'

# 更重视亮度变化（适合检测火灾/烟雾）
python3 mapping.py --before before.tif --after after.tif --out damage.geojson \
    --weights '{"edge":0.3,"mean":0.6,"var":0.1}'
```

### 5. 运行单元测试

```bash
pytest tests/test_mapping.py -v
```

所有 14 个测试应该全部通过 ✅

---

## Python API 使用

```python
from mapping import process_image_pair, visualize_geojson_on_map

# 处理图像对
geojson = process_image_pair(
    before_path='before.tif',
    after_path='after.tif',
    output_geojson_path='damage.geojson',
    grid_size=32,
    bbox=[-86.8, 36.1, -86.7, 36.2],  # 可选
    cloud_thresh=230,
    normalize=True
)

# 生成可视化
visualize_geojson_on_map(
    geojson_path='damage.geojson',
    output_html_path='map.html'
)

# 访问结果
for feature in geojson['features']:
    props = feature['properties']
    print(f"Grid {props['grid_id']}: damage_score={props['damage_score']:.3f}")
```

---

## 输出数据说明

### GeoJSON 格式

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon1,lat1], [lon2,lat2], ...]]
      },
      "properties": {
        "grid_id": "10_5",           // 网格ID（行_列）
        "lat": 36.1627,              // 中心纬度
        "lon": -86.7816,             // 中心经度
        "damage_score": 0.82,        // 损伤评分（0-1，越高越严重）
        "mean_before": 123.4,        // 灾前平均亮度
        "mean_after": 34.2,          // 灾后平均亮度
        "var_before": 45.2,          // 灾前方差
        "var_after": 78.3,           // 灾后方差
        "edge_diff": 12.3,           // 边缘差异
        "cloud_mask": false          // 是否被云遮挡
      }
    }
  ]
}
```

### 损伤评分含义

- **0.8 - 1.0** 🔴 **严重损伤**：建筑倒塌、严重破坏
- **0.6 - 0.8** 🟠 **高度损伤**：明显结构变化
- **0.4 - 0.6** 🟡 **中度损伤**：可见变化
- **0.2 - 0.4** 🟢 **轻度损伤**：轻微变化
- **0.0 - 0.2** 🟢 **基本无损**：几乎无变化

---

## 集成到其他模块

### 从 fusion.py 调用

```python
from mapping import process_image_pair

# 在 fusion.py 中调用 mapping
damage_data = process_image_pair(
    before_path=s3_download_path('before.tif'),
    after_path=s3_download_path('after.tif'),
    output_geojson_path='temp/damage.geojson'
)

# 提取损伤评分用于融合
damage_scores = {
    f['properties']['grid_id']: f['properties']['damage_score']
    for f in damage_data['features']
    if not f['properties']['cloud_mask']
}
```

---

## 常见问题

### Q: 为什么显示 "rasterio not available" 警告？

**A:** 这个是可选依赖。如果你只用普通图像（PNG/JPG），可以忽略。如果需要处理 GeoTIFF：

```bash
pip install rasterio
```

### Q: 生成的地图打开后是空白的？

**A:** 检查：
1. 确保有网络连接（地图瓦片需要从网络加载）
2. 查看浏览器控制台是否有错误
3. 确认 GeoJSON 文件不为空

### Q: 损伤评分都是 0 或 1？

**A:** 
- 都是 0：两张图像可能完全相同
- 都是 1：可能需要调整权重或云层阈值
- 尝试 `--verbose` 查看详细日志

### Q: 如何处理大图像？

**A:** 
- 增大 `--grid` 参数（如 64 或 128）
- 或者先将图像分割成小块分别处理
- 未来版本会支持分块处理

---

## 下一步

1. ✅ **集成到 FastAPI**：创建 REST API 端点
2. ✅ **连接 Firebase**：实时推送数据
3. ✅ **融合其他数据**：结合人口密度、幸存者检测
4. ✅ **前端展示**：React + Mapbox 可视化

查看完整文档：[README.md](README.md)

