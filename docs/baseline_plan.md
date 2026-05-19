# 混交林树种分类跨域迁移 Baseline — MVP 实验方案

> **场景**: 纯林公开数据（PureForest/NEON）上预训练的 SOTA 树种分类模型，不做 fine-tune，直接迁移到导师的混交林航片，诊断失败模式。
> **约束**: 单人操作 6-8 周，单 RTX 3060+，零自研代码，数据待从导师获取。

---

## 工具选择

| 环节 | 工具 | 获取 | 理由 |
|------|------|------|------|
| 树冠检测 | **DeepForest** | `pip install deepforest` | NEON 预训练 RetinaNet，738★，生态学社区标准 |
| 树种特征提取 | **CLIP (OpenCLIP)** | `pip install open_clip_torch` | ViT-L/14 预训练，开放权重，通用视觉特征 |
| 分类器 | **sklearn SVM / LogisticRegression** | `pip install scikit-learn` | 简单、可解释、无 GPU 开销 |
| 辅助 backbone | **DINOv2** | `pip install torchvision` + torch hub | 备选特征提取器，对比用 |

---

## 实验 A：域内基线

**目的**: 知道"在这片混交林里，数据自身的可分性上限是多少"——为跨域退化提供参照系。

**输入数据**:
- 导师提供的混交林航片（RGB，GSD ≤10 cm）+ 对应树种标签（crown-level）

**具体操作**:
1. 用 DeepForest 在航片上推断树冠 bbox（不 fine-tune，纯推理）
2. 对每个树冠 crop，用 OpenCLIP ViT-L/14 提取特征向量（image encoder 输出 pooler，768-d）
3. 取 50% 树冠的 CLIP 特征训练 SVM（RBF kernel），另外 50% 测试
4. 划分方式：spatial block split（50m×50m 块，块间 ≥30m 隔离），拒绝 random split（防虚高）
5. 报告：OA、macro-F1、per-species F1、混淆矩阵

**预期产出**:
- 混交林域内 OA / macro-F1（参照上限）
- Per-species F1 排序（识别 head/tail 物种）
- Spatial block split vs random split 精度差异（预期 10-20 pp）
- 混淆矩阵（域内 top-5 混淆对）

**参考方法**: Kattenborn et al. (2022) 空间块验证；DomainBed (2021) ERM 基线；`01_ITCD_Tree_Detection.md` DeepForest 生态学社区标准

**时间**: 1 周

---

## 实验 B：检测迁移

**目的**: 知道"预训练树检测器能不能在我们的林子里找到树"——量化检测器的域迁移退化。

**输入数据**:
- 混交林航片 + crown-level ground-truth bbox（从导师标注或手工标注 ~200 棵树）

**具体操作**:
1. DeepForest（NEON 预训练）直接跑在混交林航片上，不 fine-tune
2. 与 ground-truth bbox 做 IoU≥0.5 匹配，统计 recall、precision、F1
3. 分层统计（按标注或目视分类）：
   - 林内 vs 林缘（距离林缘 <10m vs ≥10m）
   - 开阔冠区 vs 郁闭冠区（目视冠层重叠程度）
   - 大冠 vs 小冠（以像素面积中位数二分）
4. 输出分层 recall/precision 表

**预期产出**:
- 全局检测 recall/precision/F1
- 分层检测召回率对比表（预期林缘低于林内，郁闭区低于开阔区，小冠低于大冠）
- 典型漏检/误检样本的 png 截图（各 10 张）
- 回答：DeepForest 零样本在我们的林子里能找回多少棵树？

**参考方法**: Sivanandam et al. (2022) 冠层重叠区 mAP 0.50 vs 开阔区 0.65；Beloiu et al. (2023) false negative 瓶颈分析

**时间**: 1 周

---

## 实验 C：分类跨域迁移

**目的**: 得到核心数字——"纯林训的分类器在混交林上退化多少"。

**输入数据**:
- 源域（公开纯林）：建议 PureForest 公开航片子集，或 TreeSatAI 中纯林斑块的 crown crops + 树种标签
- 目标域（混交林）：实验 A 中用 ground-truth bbox 裁出的树冠 crops（避免检测误差干扰分类评估）

**具体操作**:
1. 从公开纯林数据中提取 crown crops，用 OpenCLIP 提取特征
2. 在纯林 CLIP 特征上训练 SVM 分类器
3. 在混交林 crown crops（ground-truth bbox）上直接推理，不做任何 fine-tune
4. 计算 Domain Drop Rate = (域内 OA − 跨域 OA) / 域内 OA
5. 输出 per-species 跨域精度对比（识别退化最严重的树种）
6. 可选对比：CLIP 特征 vs DINOv2 特征，看 backbone 选择是否影响跨域退化幅度

**预期产出**:
- Domain Drop Rate（整体和 per-species）
- 源域→目标域混淆矩阵变化图（域内混淆矩阵 vs 跨域混淆矩阵的差异可视化）
- Top-5 跨域退化最严重的树种
- 可选：CLIP vs DINOv2 跨域退化幅度对比

**参考方法**: Beloiu et al. (2023) F1 0.72→0.45；TaxoNet (2025) Top-1 91.2%→67.8%；`04_Field_Robustness.md:19` 预期退化 20-40%

**时间**: 2 周

---

## 实验 D：失败分解

**目的**: 回答核心问题——"失败是检测的问题还是分类的问题"。

**输入数据**:
- 实验 B 的检测结果 + 实验 C 的分类结果 + ground-truth bbox

**具体操作**:
1. 构建端到端流水线：DeepForest 检测 → CLIP+SVM 分类
2. 将错误拆分为两类：
   - **检测失败**：树没被检测到（DeepForest 漏检，IoU<0.5 或完全未命中）
   - **分类失败**：树被检测到但分错了（IoU≥0.5，但分类标签错误）
3. 计算误差比例：检测失败占比 vs 分类失败占比
4. 对于分类失败子集，绘制**条件混淆矩阵**（仅考虑被正确检测到的树），列出 top-5 跨域混淆对
5. 对 top-5 混淆对，从原始影像中裁出典型误分类样本，做可视化检查
6. 如有时间：按冠层重叠程度分层，看检测失败比例是否随重叠程度增加而上升

**预期产出**:
- 误差归因饼图：检测失败 % / 分类失败 %
- 条件混淆矩阵（仅检测正确的树），top-5 混淆对
- 典型误分类样本图集（每对 3-5 张）
- 分层分析表（如果做了第 6 步）

**参考方法**: Sivanandam et al. (2022) 检测-分类分离评估；`07_Mixed_Forest_Gap.md` 混交林误分类模式

**时间**: 1 周

---

## 甘特图（6-8 周）

```
                W1  W2  W3  W4  W5  W6  W7  W8
────────────────────────────────────────────────
数据获取+预处理  ████
Exp A: 域内基线      ████
Exp B: 检测迁移      ████
Exp C: 跨域迁移          ████████
Exp D: 失败分解                      ████
论文撰写+图表整理                       ████▌
────────────────────────────────────────────────
```

**说明**: 数据获取与预处理需要等导师提供。如果数据在第 1 周末到手，总周期 6 周。如果数据延迟，Exp B/C 可调整顺序（先做 C 再用 B 的结果做 D）。

---

## 工具清单（pip install 即可）

```
# 核心
pip install deepforest          # 树冠检测（NEON 预训练）
pip install open_clip_torch     # CLIP 特征提取
pip install scikit-learn        # SVM + 评估指标
pip install torch torchvision   # PyTorch + DINOv2

# 数据处理
pip install rasterio            # 栅格读写
pip install geopandas           # 矢量/坐标处理

# 实验管理
pip install mlflow              # 实验追踪

# 可视化
pip install matplotlib seaborn
```

---

## 论文故事线

全文围绕一个核心数字展开：**纯林→混交林跨域迁移的 Domain Drop Rate = X%**，然后通过 4 个实验逐层拆解：

1. **Exp A 设定上限**：即使在同一片混交林里训练和测试，精度也只有 Y%（域内上限）
2. **Exp B 检测先行**：先回答"树找得到吗"，分离检测退化
3. **Exp C 给出核心数字**：纯林训的分类器放混交林里退化 Z pp（Domain Drop Rate）
4. **Exp D 收束**：退化中 X% 是检测没找到树（Exp B），Y% 是找到了但分错了（条件混淆矩阵）

> 结论锚点：跨域退化主要来自分类混淆（光谱/形态相似树种对）还是检测失效（冠层重叠漏检），给后续方法改进（数据增强 / 域自适应 / LiDAR 融合）提供靶向方向。
