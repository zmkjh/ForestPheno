# 13 — 论文库中的迁移/泛化/域自适应证据汇总

> **生成日期**: 2026-05-19
> **数据源**: `papers_data/surveys/` 全量 md 文件 + `papers_data/*.csv`
> **目的**: 汇总论文库中已有的域迁移、失败模式分析、跨模态迁移、零样本迁移的定量证据

---

## 一、域迁移 / 泛化退化（Dataset A → Dataset B 精度下降）

### 1.1 跨站点树种分类退化

| 论文 | 源域 → 目标域 | 退化幅度 | 详情 |
|------|--------------|---------|------|
| **Beloiu et al. 2023** | 瑞士异构林 3 站点间 cross-site | F1 0.72 → 0.45 (−37%) | 温带混交林，Faster R-CNN+ResNet-50，UAV RGB |
| **Weinstein et al. 2021 DeepForest** | NEON 22 站点跨生物群系 | 召回率 0.69 → 0.52 (−25%) | 树冠检测(RGB)，热带站点退化最严重 |
| **TaxoNet 2025** | 跨 5 个气候带树种细粒度分类 | Top-1 91.2% → 67.8% (−23.4 pp) | 热带/亚热带混交林退化最大 |
| **TreeSatAI (Ahlswede 2023)** | 德国 3 个州间跨区域 | OA 85% → 61% (−24 pp) | Sentinel + 航片 + UAV，多传感器 |
| **TaxoNet (域泛化)** | AA-Central → AA-West/East | recall 提升 3.5–4% (vs LDAM) | 双边界损失使嵌入空间对域偏移更鲁棒 |

> **综合结论**: 跨站点树种分类精度退化通常为 **20–40%** (F1/OA)，热带混交林退化最严重。
> 来源: `forestry/04_Field_Robustness.md:13-19`

### 1.2 纯林 → 混交林退化

| 论文 | 对比 | 退化幅度 | 详情 |
|------|------|---------|------|
| **Lee et al. 2023** | 单树种标签 → 2+物种混合标签 | F1 0.95 → 0.75 (−21%) | 韩国济州岛，UAV RGB+LiDAR DSM，CNN |
| **Marinelli et al. 2022** | 纯针/纯阔 → 7 类混合 | 针 F1 82% / 阔 F1 86% → 混合最低类 64.52% | ALS LiDAR，意大利混交林，MVCNN |
| **Sivanandam et al. 2022** | 7 种桉树 vs 桉树合并为 1 类 | OA 0.84 → 0.93 (+9% 当合并) | 澳大利亚塔斯马尼亚，UAV 多光谱 |
| **Silwal et al. 2024** | 模拟 24 属级类最优仅识别 16 类 | −33% (遗漏 8 类) | Harvard Forest 辐射传输模拟，1DCNN |

> **结论**: 混交林难度约为纯林的 1.3–1.5 倍。来源: `forestry/07_Mixed_Forest_Gap.md:9-20, 284-296`

### 1.3 中国团队：人工林 → 天然次生林退化

| 论文 | 场景 | OA | 对比 |
|------|------|-----|------|
| **Zhang et al. 2025** | 华北 4 树种(人工林倾向) | **96.94%** | 纯林基线 |
| **Zhong et al. 2022** | 帽儿山天然混交林 9 树种 | 76.42%–89.20% | 混交退化 |
| **Quan et al. 2023** | 东北天然次生林 11 树种 | **75.7%** (最低 per-class 60%) | 树种数↑精度↓ |
| **Zhong et al. 2024** | 复杂针阔混交林 7 树种 | 75.3%–81.0% | yOLOv8 融合 |
| **Ma et al. 2024** | 东北天然次生林 | ~83% | 面向对象 CNN |

> **趋势**: 4种 97% → 7种 81% → 11种 76% ——树种数每增加 ~3 种，OA 降 ~5–8 pp。
> 来源: `forestry/11_Chinese_Forestry_Literature.md:10-21, 309-330`

### 1.4 空间自相关导致的虚假高精度

- **Kattenborn et al. 2022**: 传统 random split 使 CNN 精度**虚高 15–30%**，应采用 spatial block cross-validation。
- 来源: `forestry/04_Field_Robustness.md:23-26`

### 1.5 地形效应导致的退化

- **You et al. 2020**: 坡度 >30° 时分类 OA 降 **5–12%**；阴坡 vs 阳坡差异 **8–15%**
- 来源: `forestry/04_Field_Robustness.md:81-89`

### 1.6 社区部署泛化退化

- DeepForest 社区讨论: 真实森林部署泛化能力比论文报告精度差 **15–30 pp**
- 来源: `forestry/05_Community_Resources.md:49-52`

---

## 二、失败模式分析（混淆矩阵 / Per-Class 误差分解 / 检测 vs 分类独立评估）

### 2.1 Per-Class 性能分解

| 论文 | 数据集 | 最差类别 | 最低 F1/Recall | 原因 |
|------|--------|---------|---------------|------|
| **Quan et al. 2023** | 东北次生林 11 树种 | — | **60.0%** (per-class 最低) | 长尾物种难识别 |
| **Marinelli et al. 2022** | 意大利混交林 7 树种 | ON 类 | **64.52%** (F1) | 冠层结构变异最大 |
| **Beloiu et al. 2023** | 瑞士异构林 4 树种 | Pinus sylvestris | **0.59** (单种 F1) | 31% 误识为冷杉 |
| **TaxoNet 2025** | AA-Central | 尾部类 | recall **57.1%** (vs 头类 92.4%) | 长尾分布 |
| **Silwal et al. 2024** | Harvard Forest 模拟 | — | per-class recall 85%/78%/87% (1m GSD) | 分辨率敏感 |

> 来源: `forestry/11_Chinese_Forestry_Literature.md:67-80`, `forestry/07_Mixed_Forest_Gap.md:48-81`, `forestry/02_LongTail_Species.md`, `02_Contrastive_Multimodal_Learning.md:25`

### 2.2 树种混淆矩阵（误分类分析）

| 论文 | 混淆对 | 误分类率 | 原因 |
|------|--------|---------|------|
| **Beloiu et al. 2023** | Pinus sylvestris → Abies alba | **31%** | 冠层形态高度相似 |
| **Beloiu et al. 2023** | Fagus sylvatica → Abies alba | 7% | 物种间光谱重叠 |
| **TaxoNet 2025** | Opuntia polyacantha ↔ O. cespitosa | 频繁混淆 | 局部形态特征单一视图不足 |

> 来源: `forestry/07_Mixed_Forest_Gap.md:144-147`, `02_Contrastive_Multimodal_Learning.md:25`

### 2.3 检测 vs 分类独立评估

| 论文 | 检测指标 | 分类指标 | 关键发现 |
|------|---------|---------|---------|
| **Sivanandam et al. 2022** | 冠层重叠区 mAP **~0.50** vs 开阔区 0.65 | 多物种 OA 0.77 (树冠) / 0.82 (bbox) | 检测失败→下游分类不可靠 |
| **Beloiu et al. 2023** | 单种模型 F1 0.59–0.86；多种模型 F1 0.72–0.92 | — | 检测的 false negative 是主要瓶颈 |
| **LiDAR Review 2021 (Neuville)** | — | 点密度 2–5 pts/m² OA **~50%** → 50+ pts/m² OA **~70%** | 点密度直接决定分类上限 |
| **Gan et al. 2023** | U-Net F1≈0.85 (3cm GSD)；精度退化随分辨率下降 | — | 分辨率↓→检测↓ |

> 来源: `forestry/07_Mixed_Forest_Gap.md:85-158`, `forestry/01_ITCD_Tree_Detection.md:90-105`

### 2.4 NanoSat 逐类 IoU

- 7 类土地覆盖: Forest IoU **0.82**, Water **0.78**, Agriculture **0.05**, Barren **0.02**
- Agriculture/Barren 极高混淆度。来源: `targeted_forestry_ai_papers.csv:56`

### 2.5 Head-Tail Gap（长尾指标）

| 论文 | Head 类性能 | Tail 类性能 | Gap |
|------|-----------|-----------|-----|
| **TaxoNet 2025** | recall 92.4% | recall 57.1% | 35 pp |
| **Joshi & Witharana 2025** | F1 74.75% (活树) | F1 71.16% (枯树) | 3.6 pp |

> 来源: `02_Contrastive_Multimodal_Learning.md:25`, `05_Forest_DL_Methods.md:9`

---

## 三、跨模态迁移（Cross-Modal / Cross-Sensor Transfer）

### 3.1 跨传感器泛化

| 论文 | 源传感器 | 目标传感器 | 效果 |
|------|---------|-----------|------|
| **SegmentAnyTree 2024** | ULS + MLS + TLS 联合训练 | 跨平台测试 | 随机子采样增强提升跨传感器迁移性；传感器/平台无关模型 |
| **Semantic-Aware Cross-Modal Transfer 2025** | — | UAV-LiDAR 点云 | 跨模态迁移学习提升个体树分割 |
| **TaxoNet 2025** | RGB 单模态 | 在 3 个异构域(urban/natural/herbarium)间泛化 | 双边界嵌入跨域鲁棒；但未扩展至多光谱/SAR/LiDAR |
| **NanSat Forest (2026)** | 光学 RGB | 光学+SAR 融合规划 | 提出 "cross-modal alignment + complementary feature learning" 路径但未实验 |

> 来源: `forestry/01_ITCD_Tree_Detection.md:42-51`, `forestry/01_ITCD_Tree_Detection.md:207-216`, `02_Contrastive_Multimodal_Learning.md:25`, `targeted_forestry_ai_papers.csv:56`

### 3.2 DUNIA 跨模态密集对齐（光学↔LiDAR波形）

| 方向 | 方法 | 性能 |
|------|------|------|
| 图像 → GEDI 全波形检索 | Zero-CL loss (ZCA 白化 + contrastive) | r=**0.70** (波形检索) |
| 像素嵌入 → 树高/冠层覆盖/PAI | KNN 检索 | 零样本树高 RMSE 2.0m (r=0.93) |
| 波形扩散生成 | 以嵌入为条件的 1D UNet + 交叉注意力 | r=0.75–0.78 |
| 图像↔波形对齐正样本相似度 | Zero-CL vs VICReg | 0.86 vs 0.56 |

> 来源: `02_Contrastive_Multimodal_Learning.md:9-17`, `06_Encoder_Design_Space.md:44-103`

### 3.3 跨模态融合中的长尾跨模态方法

| 论文 | 模态 | 跨模态? | 长尾? |
|------|------|---------|-------|
| **GlobalGeoTree 2025** | Sentinel-2 + 气候 + 文本 | ✅ CLIP 式 VL | ✅ 零/少样本 21k 树种 |
| **LLM Pseudo-Label 2026** | HSI + ALS + LLM 知识 | ✅ 双传感器 + LLM | ✅ 半监督伪标签 |
| **PureForest 2024** | ALS + VHR 航片 | ✅ | ❌ 相对均衡 |
| **ALS Point Transformer 2025** | 多光谱 ALS (3λ) | ✅ | ✅ 验证稀有物种 |
| **MSFMamba 2024** | HSI + LiDAR/SAR | ✅ 双输入 SSM | ❌ |

> 来源: `forestry/02_LongTail_Species.md:67-77`, `03_Multimodal_Fusion.md:23-27`

### 3.4 传感器退化导致的性能影响

| 传感器/配置退化 | 影响 | 来源 |
|---------------|------|------|
| LiDAR 点密度 50+ → 2–5 pts/m² | OA ~70% → **~50%** | LiDAR Review 2021 |
| GSD 1m → 30m | OA 82.83% → **54.09%** | Silwal et al. 2024 |
| GSD 3cm → 10cm | U-Net F1 ~0.85 → 显著退化 | Gan et al. 2023 |
| 仅结构特征 → 结构+强度融合 | OA 65-88.6% → **92.8%** | LiDAR Review 2021 |

> 来源: `forestry/07_Mixed_Forest_Gap.md:165-196, 211-226`, `forestry/01_ITCD_Tree_Detection.md:90-105`

---

## 四、零样本迁移（Zero-Shot Cross-Domain Evaluation）

### 4.1 零样本森林变量估计

| 论文 | 任务 | 零样本性能 | 方法 |
|------|------|-----------|------|
| **DUNIA (ICML 2025)** | 树高 | RMSE **2.0m** (r=0.93) | 像素嵌入 + KNN=50 |
| **DUNIA** | 冠层覆盖 | RMSE **11.7%** (r=0.89) | 像素嵌入 + KNN=50 |
| **DUNIA** | PAI | RMSE **0.71** (r=0.75) | 像素嵌入 + KNN=50 |
| **DUNIA** | 树种分类 | wF1 **76.0%** (KNN=5) | 像素嵌入 + KNN=5 |
| **DUNIA** | 土地覆盖 CLC+ | wF1 **80.1%** | 像素嵌入 + KNN |
| **DUNIA** | 作物分类 PASTIS | OA **56.2%** (差) | 单时相无时序→物候盲区 |

> 检索库仅需 50K 标注像素(~31 km²)，约为监督方法的 **0.25%** 数据量。
> 来源: `02_Contrastive_Multimodal_Learning.md:17`, `06_Encoder_Design_Space.md:51-63`, `15_CrossValidation.md:11-19`

### 4.2 零样本树冠分割

| 论文 | 方法 | 效果 |
|------|------|------|
| **ZS-TreeSeg (2026)** | Cellpose-SAM 框架 | 零样本树冠实例分割，无需训练即可泛化 |
| **GlobalGeoTree (2025)** | GeoTreeCLIP (VL 预训练) | Sentinel-2 + 气候 + 文本，21k 树种零/少样本 |

> 来源: `forestry/06_Forestry_Datasets_Deep.md:212-213, 233`, `forestry/02_LongTail_Species.md:83-88`

### 4.3 零样本标签效率

| 论文 | 指标 | 100% 标签 | 20% 标签 | 退化 |
|------|------|----------|---------|------|
| **DUNIA** | 树高 RMSE | 1.3m (r=0.95) | 1.4m (r=0.93) | 仅 +0.1m |
| **DUNIA** | 冠层覆盖 RMSE | 9.8% (r=0.85) | — | 几乎无损 |
| **AnySat** | 树高 RMSE | — | 2.8m (r=0.89) | — |
| **CROMA** | 树高 RMSE | — | 3.6m (r=0.76) | — |
| **SatMAE/DOFA/DeCUR** | 树高 RMSE | — | 10.5–11.2m (r=0.50–0.52) | 几乎无零样本能力 |

> 来源: `06_Encoder_Design_Space.md:65-77`, `15_CrossValidation.md:80-94`

### 4.4 零样本基线对标

| Baseline | 任务 | 场景 |
|----------|------|------|
| CLIP / DINOv2 + prompt | T1 闭集长尾分类 | 跨模态泛化对标 |
| DUNIA / AnySat / CROMA | T1 跨模态 + 低标签 | 遥感预训练对标 |
| GPT-4V / Gemini Pro Vision | T2 推理 / T3 决策 | 通用 VLM Agent 对标 |

> 来源: `11_Evaluation_Longtail_Design.md:295-304`

### 4.5 零样本局限性

- **PASTIS 零样本 OA 仅 56.2%**: 比含时序 AnySat 低 **28 pp**，因 DUNIA 用单时相中值合成影像，无法捕捉作物物候动态
- **CLC+ / PASTIS 对 KNN 库大小敏感**: 库规模缩减时性能急剧下降
- **预训练针对单一年份/区域**: 迁移到新区域需重新预训练

> 来源: `02_Contrastive_Multimodal_Learning.md:23-24`, `09_Temporal_Phenology_Design.md:10`

---

## 五、域自适应 / 迁移学习方法汇总

| 方法类别 | 代表工作 | 应用场景 | 效果 |
|---------|---------|---------|------|
| 对抗域自适应 (DANN) | **PureForest 2024** | 法国全国森林，跨生态区 | 域对抗训练 |
| 自监督地理预训练 | **GlobalGeoTree 2025** | 全球树种零/少样本 | 21k 树种 CLIP |
| 多任务嵌入学习 | **Dutch NFI Embeddings 2025** | 跨林分类型 | +2–9 pp vs 手工特征 |
| 少样本学习 | **FewShot Invasive 2025** | 3-shot 入侵树种 | F1=0.86 |
| 伪标签 + LLM | **LLM Tree Pseudo-Label 2026** | 跨地区伪标签扩充 | +5.6% over best ref |
| 双边界嵌入 | **TaxoNet 2025** | 跨区域树木分类 | recall +3.5–4% |
| 跨传感器 3D | **SegmentAnyTree 2024** | ULS/MLS/TLS 跨平台 | 传感器无关 |

> 来源: `forestry/04_Field_Robustness.md:28-36`

---

## 六、与 ForestPheno 评估协议的对齐

### 6.1 已发现的评估设计文献

| 维度 | 来源文件 | 关键发现 |
|------|---------|---------|
| 域泛化评估 | `11_Evaluation_Longtail_Design.md:197-203` | 设计了 3 层域泛化: near-domain / far-domain / cross-sensor |
| 开集评估 | `11_Evaluation_Longtail_Design.md:189-196` | Easy/Medium/Hard unknown 三级开集难度 |
| 长尾分层 | `11_Evaluation_Longtail_Design.md:180-187` | frequency-bin 分层抽样 (Head/Medium/Tail/Ultra-tail) |
| 代价加权错误 | `11_Evaluation_Longtail_Design.md:274-280` | CER 权重矩阵(濒危物种误判×10) |
| 零样本对标 | `11_Evaluation_Longtail_Design.md:301-302` | CLIP/DINOv2 + DUNIA/AnySat/CROMA 基线 |
| 推理链评估 | `11_Evaluation_Longtail_Design.md:320-326` | LLM-as-Judge + 双 Judge Kappa>0.7 |

### 6.2 评估指标体系

| 指标 | 定义 | 用途 |
|------|------|------|
| **Domain Drop Rate** | 域间性能下降率 | T: 域泛化能力 |
| **Tail-Recall@K** | 尾部 K% 类的平均 recall | Q: 长尾物种保护 |
| **Open-F1** | (已知类 macro-F1 + 未知类 F1) / 2 | E: 开集鲁棒性 |
| **CER (Cost-Weighted Error)** | Σ(错误类型权重 × 错误数) / 总决策数 | S: 决策安全性 |
| **RAS (Reasoning Accuracy Score)** | 推理链步骤正确率 | U: 推理可解释性 |

> 来源: `11_Evaluation_Longtail_Design.md:148-157, 237-282`

---

## 七、文献空白总结

1. **跨传感器跨域评估**: 现有域迁移评估停留在"同数据集不同子集"，缺乏跨传感器(UAV→卫星)的真实跨域测试
2. **树种级长尾遥感分类**: 高光谱 + 树种 + 长尾的三重交叉无人触及
3. **开集森林物种检测**: 仅 TaxoNet 在城市树木场景验证，遥感场景完全空白
4. **跨区域泛化验证**: 中国团队大部分研究在单一站点（帽儿山/凉水），无跨区域迁移验证
5. **端到端跨模态+长尾联合优化**: 无方法同时优化多模态融合损失 + 长尾重平衡损失 + 林业层级损失

> 来源: `11_Evaluation_Longtail_Design.md:29, 43-46, 331-335`, `forestry/11_Chinese_Forestry_Literature.md:344-349`

---

*生成日期: 2026-05-19 | 扫描文件: 26 个 md + 5 个 csv | 涉及论文: ~50 篇*
