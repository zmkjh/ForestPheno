# 林业特有公开数据集与基准 (Forestry-Specific Public Datasets & Benchmarks)

> 检索日期：2026-05-18  |  来源：OpenAlex + arXiv
> 侧重在通用农业/遥感搜索中**容易被遗漏**的林业专用资源

---

## 1. NEON (National Ecological Observatory Network)

| 属性 | 描述 |
|------|------|
| **规模** | 81个陆地观测站 + 26个水生站，覆盖美国全部主要生态区 |
| **数据类型** | 机载LiDAR (ALS)、高光谱 (HSI)、RGB正射影像、叶面积指数、物候相机 |
| **时间跨度** | 2013-至今（持续采集） |
| **访问** | https://data.neonscience.org/ |
| **代码/工具** | neonUtilities (R包)、pyNEON (Python)、**DeepForest** 模型 |

### 关键论文（OpenAlex + arXiv 检索）

| 论文 | 年份 | 引用 | DOI | 代码 |
|------|------|------|-----|------|
| ZS-TreeSeg: Zero-Shot Tree Crown Instance Segmentation | 2026 | new | arXiv:2602.00470 | yes |
| Crown-CAM: XAI for tree crown detection on NEON | 2022 | 33 | arXiv:2211.13126 | -- |
| Individual canopy tree species maps for NEON (Weinstein et al.) | 2024 | 13 | 10.1371/journal.pbio.3002700 | data+code |
| The shape of trees: reimagining forest ecology in 3D (Lines et al.) | 2022 | 104 | 10.1111/1365-2745.13944 | review |
| Tree Crown Detection from UAV RGB (Gan et al.) | 2023 | 68 | 10.3390/rs15030778 | -- |
| Review of RS Forest AGB Estimation (Tian et al., Nanjing Forestry U.) | 2023 | 177 | 10.3390/f14061086 | review |

### 评价

- **优势**: 数据标准化程度最高、多模态融合机会最大、代码生态最活跃（DeepForest）
- **劣势**: 限于美国、热带/亚热带缺失、非专门物种分类标注
- **与 PureForest 互补性**: Strong -- NEON provides airborne LiDAR+HSI, PureForest provides ground TLS+species labels

---

## 2. 全球森林数据集

### 2.1 GEDI (Global Ecosystem Dynamics Investigation)

| 属性 | 描述 |
|------|------|
| **类型** | 星载全波形LiDAR (ISS搭载) |
| **覆盖** | 全球 (51.6N-51.6S) |
| **时间** | 2019-2023 |
| **数据** | 冠层高度 (RH metrics)、地上生物量密度 (L4A) |
| **访问** | https://gedi.umd.edu/ |

| 年份 | 引用 | 关键论文 | DOI |
|------|------|----------|-----|
| 2023 | 60 | NY州10m冠层高度&AGB (GEDI+S1+S2融合) | 10.1016/j.ecoinf.2023.102404 |
| 2023 | 48 | FORMS: 法国森林高度/体积/生物量 10-30m | 10.5194/essd-15-4927-2023 |
| 2024 | 29 | 南亚+中非热带森林LiDAR AGB参考图 | 10.1038/s41597-024-03162-x |
| 2025 | 31 | 多源GEDI+卫星+ML 森林AGB | 10.1016/j.ecoinf.2025.103052 |

### 2.2 ForestGEO (Forest Global Earth Observatory)

| 属性 | 描述 |
|------|------|
| **规模** | 77个大型森林动态样地（每样地>10ha，定期每木调查） |
| **核心数据** | 物种ID、胸径、位置坐标（逐棵树）、重复测量 |
| **覆盖** | 29个国家，热带/温带均有 |
| **访问** | https://forestgeo.si.edu/ （数据需申请） |

| 年份 | 引用 | 关键论文 | DOI |
|------|------|----------|-----|
| 2022 | 109 | 热带林TLS+UAV激光扫描融合 (Terryn et al.) | 10.1016/j.rse.2022.112912 |
| 2020 | 87 | TLS结构特征树种分类 (Terryn et al.) | 10.1016/j.isprsjprs.2020.08.009 |
| 2023 | 65 | Mask R-CNN 热带林ITC (Ball et al.) | 10.1002/rse2.332 |

### 2.3 FOR-species20K

| 属性 | 描述 |
|------|------|
| **类型** | TLS点云树种分类基准 (2025新发布) |
| **规模** | 20,000+棵树，来自全球11个国家多个森林类型 |
| **模态** | 地面激光扫描 (TLS) 点云 |
| **论文** | Puliti et al., Methods in Ecology and Evolution 2025 |
| **DOI** | 10.1111/2041-210x.14503 |

> Most complementary: ground LiDAR perspective -- perfect complement to PureForest (aerial)

### 2.4 ReforesTree

| 属性 | 描述 |
|------|------|
| **类型** | 热带森林碳储量估算数据集 |
| **规模** | 厄瓜多尔6个农场、100+棵树木的实测AGB |
| **模态** | 无人机正射影像 + 实测地上生物量 |
| **论文** | Reiersen et al., AAAI 2022, cited 28 |
| **DOI** | 10.1609/aaai.v36i11.21471 |
| **GitHub** | https://github.com/gyrrei/ReforesTree |

### 2.5 TreeSatAI Benchmark Archive

| 属性 | 描述 |
|------|------|
| **类型** | 多传感器/多标签树种分类基准 |
| **规模** | 50K+ 影像块 (Sentinel-2 + 航片)，33个欧洲常见树种 |
| **论文** | Ahlswede et al., ESSD 2023, cited 59 |
| **DOI** | 10.5194/essd-15-681-2023 |
| **Zenodo** | 10.5281/zenodo.6598390 (code+data) |

### 2.6 VHRTrees

| 属性 | 描述 |
|------|------|
| **类型** | 超高分辨率卫星影像树木检测基准 |
| **论文** | Topgul et al., Front. For. Glob. Change 2025 |
| **DOI** | 10.3389/ffgc.2024.1495544 |

### 2.7 IDTReeS
- 集成 UAV + ALS + TLS 多平台树木检测
- 使用 NEON 和多个独立数据集，关注算法可迁移性

### 2.8 FORMS
- 法国全境 10-30m 分辨率森林高度/体积/生物量图
- 融合 Sentinel-1/2 + GEDI + 法国NFI

### 2.9 Tallo
- 全球树木实测数据库（树干直径、物种、位置）
- 61,000+ 群落、61万+条记录，适合作为分类模型的 ground truth

---

## 3. 中国林业数据集

### 3.1 关键论文

| 年份 | 引用 | 论文 | 机构 | DOI |
|------|------|------|------|-----|
| 2023 | 39 | 时序约束DL人工林树种制图 (Huang et al.) | 中科院 | 10.1016/j.isprsjprs.2023.09.009 |
| 2022 | 19 | Beyond tree cover: 中国南方森林特征 (Li et al.) | 中科院 | 10.1002/rse2.292 |
| 2021 | 12 | Deep Fusion uNet 多时序树种制图 (Guo et al.) | 中国林科院 | 10.3390/rs13183613 |
| 2024 | 9 | PointNet++ 真彩色点云树种分类 (Liu et al.) | -- | 10.1080/01431161.2024.2377837 |
| 2022 | 30 | 多时序CNN个体树种分类 (Guo et al.) | 中科院 | 10.3390/s22093157 |

### 3.2 中国特有数据集/平台

| 数据集/平台 | 描述 | 状态 |
|-------------|------|------|
| 中国森林资源清查 (CNFI) | 全国40万+固定样地，每5年重复调查 | 不完全公开 |
| Chinese Academy of Forestry | 多篇论文使用自采集无人机+卫星数据 | data upon request |
| 南方人工林 | 桉树、杉木、马尾松为主要目标树种 | 论文少量样本 |
| 中科院遥感所 | 时序Sentinel-2 + 高分卫星 | 部分产品公开 |
| Tsinghua 10m土地覆盖 | 宫鹏团队全球土地覆盖（含森林类型） | 完全公开 |

### 3.3 评价
- **可用性**: 大规模标准化公开数据集极少
- **代码**: 多数论文未释放代码
- **与 PlantD 互补性**: 中国南方人工林数据是 PlantD 最有前景的扩展方向

---

## 4. 时序物候数据集

### 4.1 PhenoCam Network

| 属性 | 描述 |
|------|------|
| **规模** | 600+ 台固定相机，北美+欧洲为主 |
| **数据** | 每日RGB影像（近景），计算GCC时间序列 |
| **时间** | 2000-至今 |
| **访问** | https://phenocam.sr.unh.edu/ |

| 年份 | 引用 | 论文 | DOI |
|------|------|------|-----|
| 2022 | 27 | DL+HMM 每日物候分类 (Taylor & Browning) | 10.3390/rs14020286 |
| 2021 | 44 | 超像素DL热带物候监测 (Song et al., HKU) | 10.1016/j.isprsjprs.2021.10.023 |
| 2018 | 146 | PhenoCam vs MODIS 物候过渡 (Richardson et al.) | 10.1038/s41598-018-23804-6 |
| 2025 | 1 | RESformer: Acer 物候细粒度识别 (Jing et al.) | 10.1007/s11676-025-01843-w |

### 4.2 其他物候/时序资源

| 资源 | 类型 | 用途 |
|------|------|------|
| MODIS MCD12Q2 | 500m 全球物候指标 | 大尺度物候趋势 |
| Sentinel-2 NDVI 时序 | 10m 多光谱 | 中尺度树种物候分异 |
| Phenology Eyes Network | 日本鱼眼相机网络 | 补充 PhenoCam 地理覆盖 |
| Landtrendr / CCDC | Landsat 时序算法 | 森林扰动/恢复检测 |

---

## 5. 数据集综合对比

| 数据集 | 模态 | 规模 | 物种标签 | 代码 | 地理 | 互补性 |
|--------|------|------|----------|------|------|--------|
| NEON | ALS+HSI+RGB | 81站点 | 部分 | DeepForest | 美国 | Strong |
| TreeSatAI | S2+Aerial | 50K块 | 33种 | Zenodo | 欧洲 | Medium |
| ReforesTree | UAV RGB | 6农场 | 少数 | GitHub | 热带 | Medium |
| FOR-species20K | TLS点云 | 20K树 | 多树种 | Yes | 11国 | Strong |
| FORMS | S1+S2+GEDI | 法国 | No | -- | 法国 | Low |
| ForestGEO | 地面实测 | 77样地 | 全树种 | Apply | 29国 | Medium |
| GEDI | 星载LiDAR | 全球 | No | -- | 全球 | Medium |
| PhenoCam | RGB时序 | 600站 | No | -- | 北美/欧 | Medium |
| VHRTrees | 高分卫星 | 新 | Yes | TBD | TBD | Low |
| BAMFOREST | UAV/ALS | -- | -- | Partial | TBD | Low |
| CNFI | 地面实测 | 40万样地 | Yes | No | 中国 | Strong |
| IDTReeS | 多平台 | 多集 | Yes | -- | 多区域 | Medium |

---

## 6. 关键发现与建议

### 6.1 被遗漏的核心资源

1. **FOR-species20K** (2025) -- 最新最大TLS树种分类基准，地面视角与PureForest形成数据级联
2. **ReforesTree** -- 唯一有公开代码的热带森林碳储量DL数据集
3. **FORMS** -- 法国国家级产品，GEDI+Sentinel融合最高水准
4. **VHRTrees** -- 最新高分卫星树木检测基准
5. **BAMFOREST** -- ZS-TreeSeg论文使用的跨传感器数据集
6. **ZS-TreeSeg** (2026) -- 零样本树冠分割，Cellpose-SAM框架，无需训练即可泛化

### 6.2 与 PureForest/PlantD 互补性

| PureForest缺失维度 | 可补充数据集 |
|---------------------|-------------|
| 地面LiDAR视角 | FOR-species20K (TLS点云树种分类) |
| 物候时序信息 | PhenoCam + Landtrendr |
| 热带生物群系 | ReforesTree + ForestGEO |
| 大规模验证标签 | ForestGEO + Tallo |
| 中国特有物种 | CNFI (需合作) |
| 全波形LiDAR | GEDI L2A |

### 6.3 数据可用性分级
- Green (完全开放): NEON, GEDI, PhenoCam, TreeSatAI, ReforesTree, FOR-species20K
- Yellow (需申请): ForestGEO, Tallo, CNFI
- Red (不可公开获取): 多数中国论文的实验数据

### 6.4 开源代码与模型
- **DeepForest** (Weecology Lab): RetinaNet树冠检测，预训练于NEON
- **Detectree2** (Ball et al.): Mask R-CNN热带林木检测
- **ZS-TreeSeg** (Chen et al. 2026): 零样本树冠分割 (Cellpose-SAM)
- **TreeSatAI repo**: Zenodo 完整数据+基线代码
- **ReforesTree repo**: GitHub 碳储量估算pipeline