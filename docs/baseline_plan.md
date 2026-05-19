# 混交林树种分类跨域迁移 Baseline — 实验方案

> **场景**: 纯林公开数据（PureForest/NEON）上预训练的树种分类模型，不做 fine-tune，直接迁移到本地混交林数据，诊断失败模式。
> **约束**: 单人操作，单 GPU，零自研代码。

---

## 前置步骤：数据准备

### 输入
- 一块混交林的无人机 RGB 航片（GSD ≤ 10 cm）
- 同区域的 LiDAR 点云（点密度 > 5 pts/m²）
- 样地实测记录（树种、坐标、胸径、树高）

### 操作
1. **空间对齐**：航片、点云、样地坐标统一到同一坐标系（UTM）
2. **树冠标注**：将样地实测的每棵树坐标与航片中可见树冠对应。标注 ~200 棵树的最小外接矩形作为 ground-truth bbox，记录树种标签
3. **LiDAR 点云裁剪**：对每棵标注树提取点云子集
4. **斑块分类**：根据样地树种组成，将剖面划分为纯林斑块（单树种 > 80%）和混交林斑块

### 备选数据源（如样地实测不可用）
- PureForest 公开数据集（法国纯林，ALS + 航片 + 树种标签）— 作为源域训练数据
- TreeSatAI（德国，多传感器 + 树种标签）
- NEON（美国，多站点时序数据）

---

## 实验 A：域内基线

**目的**：确定本地混交林数据自身的可分性上限。

**输入**：
- 本地混交林航片 + 标注树冠 bbox + 树种标签

**操作**：
1. 用 OpenCLIP ViT-L/14 对每个标注树冠提取 768-d 特征向量
2. 5 次 spatial block split 重复（50m×50m 块，块间 ≥ 30m 隔离），固定随机种子
3. 每次 split 取 50% 树冠训练 SVM（RBF kernel），另外 50% 测试
4. 报告 mean ± std：OA、macro-F1、per-species F1

**关键说明**：
- 使用 spatial block split，禁止 random split（同一树冠不同像素泄露会虚高 10-20 pp）
- 固定 random_state=42，确保可复现；在不同 split 间改变分割 seed
- 多次重复是必需的——SVM 对 split 敏感，单次结果不可靠

**产出**：域内分类精度上限 + 域内 top-5 混淆对

---

## 实验 B：检测迁移

**目的**：评估预训练树冠检测器在本地混交林中的零样本性能。

**输入**：
- 本地混交林航片 + 标注树冠 bbox

**操作**：
1. **GSD 归一化**：用 rasterio 将航片重采样至 0.1 m GSD（DeepForest 训练分辨率）。不做此步骤会导致 recall 下降被错误归因为模型差
2. DeepForest（NEON 航片预训练）直接推理，不做 fine-tune
3. 与标注 bbox 做 IoU ≥ 0.5 匹配，统计 recall、precision、F1
4. 分层统计：林内 vs 林缘、开阔冠区 vs 郁闭冠区、大冠 vs 小冠

**产出**：检测器在不同林相条件下的 dropout 模式

---

## 实验 C：分类跨域迁移

**目的**：获得核心定量指标——纯林训练的物种分类器在混交林中的退化幅度。

**输入**：
- 源域（公开纯林数据，如 PureForest 航片子集或 TreeSatAI 纯林斑块）
- 目标域（本地混交林标注树冠 crops，使用 ground-truth bbox 裁出，排除检测误差）

**操作**：
1. 源域 crown crops 与目标域 crown crops 统一预处理：bbox 外扩 10% padding，固定 resize 至 224×224
2. 用 OpenCLIP 对源域和目标域分别提取特征
3. 设置两组分类器：
   - **Source SVM**：源域 CLIP 特征训练，目标域推理（跨域迁移，核心指标）
   - **Target linear probe**：目标域 CLIP 特征 + SVM 训练+测试（特征可迁移性上限，对照）
4. 计算 Domain Drop Rate = (target-probe OA − source-SVM OA) / target-probe OA
5. Per-species 精度对比 + 可选 CLIP vs DINOv2 特征对比

**关键说明**：
- Target linear probe 是必须的对照：如果目标域 CLIP 特征本身可分（target-probe OA 高），但 source-SVM OA 低 → 分类器决策边界不适应，不是特征不行
- 如果 target-probe OA 已经很低 → 特征本身对这片混交林无效，已到天花板

**产出**：整体和 per-species 跨域退化幅度，源域→目标域混淆矩阵变化

---

## 实验 D：失败归因

**目的**：回答"没找到树"、"找到了但认错了"、"多找了不存在的树"各占多少。

**输入**：实验 B 的检测结果 + 实验 C 的分类结果 + 标注 bbox

**操作**：
1. 端到端流水线：DeepForest 检测 → CLIP+SVM 分类
2. 错误三分法（遥感论文标准写法）：
   - **Missed tree**：树未被检测到（GT bbox 无任何预测框 IoU ≥ 0.5）
   - **False positive**：预测框无对应 GT（IoU < 0.5 且分类结果存在）
   - **Misclassification**：IoU ≥ 0.5 但分类标签 ≠ 真值
3. 计算三类失败占比
4. 对 Misclassification 子集绘制条件混淆矩阵，列出 top-5 跨域混淆对
5. **跨域混淆矩阵 diff**：ΔConfusion = Conf_target − Conf_source，高亮仅在混交林中出现的混淆对
6. 对 top-5 混淆对裁出典型样本做可视化

**产出**：误差归因饼图 + 条件混淆矩阵 + 典型误分类样本

---

## 工具清单

```
pip install deepforest          # 树冠检测（NEON 预训练 RetinaNet）
pip install open_clip_torch     # CLIP ViT-L/14 特征提取
pip install scikit-learn        # SVM + 评估指标
pip install torch torchvision   # PyTorch + DINOv2（备选 backbone）
pip install rasterio            # 栅格读写
pip install geopandas           # 空间数据处理
pip install matplotlib seaborn  # 可视化
```

---

## 可选增强（有余力时追加，全部零自研）

| 增强 | 操作 | 成本 | 价值 |
|------|------|------|------|
| CLIP vs DINOv2 backbone 对比 | 实验 C 中并行提取两组特征，报告跨域退化幅度差异 | 低 | 强（消融 backbone 影响） |
| Genus-level 重算 | 将 species 标签上卷为 genus，重跑实验 C 的 OA/Drop Rate | 低 | 高（消除物种不重叠干扰） |
| t-SNE / UMAP 可视化 | 源域 + 目标域 CLIP 特征联合降维，按域着色 | 低 | 高（图好看，直观展示域偏移） |
| 冠层高度作为辅助特征 | 从 LiDAR CHM 提取每个 crown 的 max height，作为 SVM 额外特征列 | 低 | 中（验证 LiDAR 结构信息价值） |

---

## 潜在风险与应对

| 风险 | 应对 |
|------|------|
| 样地坐标与影像偏移 > 5m | 先做人工 co-registration，挑 20 棵明显树对特征点 |
| 样地实测记录只有林分统计，无单树坐标 | 改用目视解译手工标注 200 棵树作为 ground-truth |
| 纯林公开数据物种与本地完全不重叠 | 降级为 genus-level 分类，或只用形态特征不依赖物种匹配 |
| DeepForest 零样本检测率极低 | 不 fine-tune，如实报告退化幅度，检测结果作为"上限对比"而非流水线依赖 |
| 混交林内纯林斑块不够大 | 降级为"全混交林域内 baseline + 跨域退化"，不做纯林 vs 混交林内部分层 |
| 源域与目标域 crown crop 方式不一致 | 统一预处理：bbox 外扩 10% padding + 固定 resize 224×224，杜绝 crop 分布偏移噪声 |

---

## 核心产出逻辑

4 个实验串联回答一条链：

1. 域内能做到多好？（上限，mean ± std over 5 splits）
2. 预训练检测器能找到多少棵树？（检测退化，分层统计）
3. 纯林分类器放混交林退化多少？（跨域迁移 + target linear probe 对照）
4. 退化中，Missed tree / False positive / Misclassification 各占多少？（三类归因 + 混淆矩阵 diff）

四个答案合起来，就是一份完整的跨域迁移诊断报告。
