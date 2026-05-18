# 05 — 社区资源：GitHub / 论坛中的林业 AI 工具与数据集

## 1. 热门 GitHub 仓库

| 仓库 | Stars | 语言 | 功能 | 预训练权重 |
|------|-------|------|------|-----------|
| **DeepForest** (weecology) | 738 | Python | 树冠检测，支持自定义微调 | ✅ HuggingFace Hub |
| **lidR** (Jean-Romain) | 692 | R | LiDAR 点云林业处理全流程 | N/A (算法包) |
| **awesome-forests** (brycefrank) | 370 | — | 社区维护的林业 ML 数据集清单 | — |
| **NeonTreeEvaluation** (weecology) | 161 | Python | RGB+HSI+LiDAR 多模态树冠检测基准 | ✅ 评估用 |
| **PercepTreeV1** | 108 | Python | 43k 合成森林图像 + 预训练模型 | ✅ 提供 |
| **DeepForest-utils** | 89 | Python | DeepForest 数据预处理工具 | — |
| **FOR-instance** | 56 | Python | 森林实例分割数据集 | ✅ 提供 |
| **FORTLS** | 30 | R | 地基 LiDAR 森林清查 | ✅ 提供 |
| **IDS-TLS** | 28 | Python | TLS 单木分割 | — |
| **Quebec Trees** | 22 | Python | 魁北克树种数据集 | ✅ 提供 |

## 2. 预训练模型可用性

| 模型 | 框架 | 输入 | 任务 | 数据来源 |
|------|------|------|------|---------|
| DeepForest v2.1.0 | PyTorch | RGB (NAIP/UAV) | 树冠检测 | NEON + 标注 |
| PercepTreeV1 | PyTorch | 合成森林图像 | 树种分类 | 43k 合成图像 |
| SegmentAnyTree (SAT) | PyTorch | TLS/MLS 点云 | 多层冠层分割 | 欧洲 TLS |
| GlobalGeoTree | PyTorch + CLIP | Sentinel-2 + 气候 | 21k 树种零样本 | iNaturalist |

## 3. 可通过代码获取的数据源

| 数据源 | 内容 | 覆盖 | 获取方式 |
|--------|------|------|---------|
| **NEON** | RGB + HSI + LiDAR 多时空 | 81 美国站点 | Python API (neonUtilities) |
| **FIA** (Forest Inventory & Analysis) | ~130K 美国样地 | 全美 | R / Python 包 |
| **NAIP** | 1m 航拍 RGB | 美国全国 | Google Earth Engine |
| **GEDI** | 全波形 LiDAR 25m 足迹 | 全球 | NASA DAAC |
| **Sentinel-2** | 10m 多光谱 | 全球 | Copernicus / Google Earth Engine |
| **TreeSatAI** | 15 树种 50K 影像 + Sentinel | 德国 | HuggingFace |
| **FOR-species20K** | 20K 树 TLS 点云 33 树种 | 欧洲 | 论文附属数据 |

## 4. 论坛讨论热点（DeepForest Issues / Discussions）

- **自定义训练问题**：用户最常问的是如何在自有数据上微调，说明开箱即用的泛化能力有限
- **遮挡处理**：密集冠层下的检测召回率被多次报告偏低（<60% in closed-canopy）
- **大区域处理**：内存管理和瓦片拼接是工程部署的主要障碍
- **多传感器适配**：缺少 LiDAR 和 RGB 融合的官方支持
- **模型输出后处理**：需要 community 开发的额外工具完成树冠分割到生态参数的一步

## 5. 关键发现

1. **DeepForest 是唯一的工业化林业 DL 框架**（738★，持续维护），其余多为研究项目（<200★，不定期更新）
2. **预训练权重集中在 RGB 航拍**：缺 LiDAR、多光谱、热红外的预训练权重
3. **多个高质量数据集（FOR-species20K、FOR-instance、Quebec Trees）未进入主流学术引用闭环**
4. **社区讨论揭示了论文对比基准掩盖的问题**：真实森林部署的泛化能力比报告精度差 15-30 pp
