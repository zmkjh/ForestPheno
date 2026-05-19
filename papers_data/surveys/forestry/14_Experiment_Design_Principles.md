# 14 — 单站点跨域迁移诊断实验：原则与实验方案

> **生成日期**: 2026-05-19
> **融合材料**: `12_Benchmark_Methodology.md`, `13_Our_Library_Transfer_Methods.md`, `04_Field_Robustness.md`, `01_ITCD_Tree_Detection.md`, `02_LongTail_Species.md`, `07_Mixed_Forest_Gap.md`, `11_Chinese_Forestry_Literature.md`, `02_Contrastive_Multimodal_Learning.md`, `01_Agent_Orchestration.md`
>
> **场景摘要**: 预训练 SOTA 树种分类模型（PureForest/NEON 纯林），不做 fine-tune，直接迁移到单一混交林站点，诊断失败模式。

---

## Part 1：核心原则（8 条）

### 原则 1：检测与分类独立评估（Detection vs. Classification Decouple）

- **来源**: Sivanandam et al. (2022, 塔斯马尼亚桉树混交林) — 冠层重叠区树检测 mAP 仅 0.50，明确区分了"树没找到"（检测）与"找到了但认错"（分类）。Beloiu et al. (2023) 进一步确认 false negative（检测漏检）是异构林中的主要瓶颈。
- **适用原因**: 域迁移（纯林→混交林）导致的精度下降中，检测失效和分类失效必须被区分——混交林的冠层重叠首先影响检测，而纯林训练的模型在冠层边界清晰的场景下可能检测更好。
- **具体操作**:
  - 部署两阶段流水线：树检测器（Faster R-CNN / YOLOv8 / DeepForest）→ 树种分类器（CNN / ViT / 集成方法）
  - 对检测器报告 recall、precision、mAP（按冠层重叠程度分层）
  - 对分类器报告条件分类准确率（在 ground-truth bbox 之上评估，排除检测误差）
  - 输出"检测-分类混淆矩阵"：行=真实存在/不存在，列=检测到/未检测到+分类正确/错误

### 原则 2：Diversity Shift 与 Correlation Shift 双重分解

- **来源**: OoD-Bench (Ye et al., 2022, CVPR, `2106.03721`) — 将 OOD 失败分解为两个正交维度：Diversity Shift（训练分布未覆盖测试分布的某些子模式）和 Correlation Shift（训练中学到的虚假背景-目标关联在测试域被打破）。
- **适用原因**: 纯林→混交林的域迁移中，两类 Shift 同时存在但性质根本不同——Diversity Shift 来自混交林中出现纯林训练集中不存在的物种组合/冠层结构模式，Correlation Shift 来自背景植被、光照-地形耦合关系在混交林中的改变。
- **具体操作**:
  - Diversity Shift 诊断：对每个误分类样本，判断该物种（或该冠层形态）在训练集中是否充分出现 → 定义 Diversity Shift Score = 训练集未覆盖类别（物种/冠层形态 bin）上的总错误数占比
  - Correlation Shift 诊断：对每个误分类样本，判断训练集中该物种与该背景/光照/密度组合是否曾正确共现 → 定义 Correlation Shift Score = 背景特征发生显著统计偏移的样本上的错误率
  - 使用 CLIPood (`2302.00864`) 的文本-视觉对齐方法验证"背景-树种"的虚假相关是否在源域存在（如源域中某树种总出现在特定密度林分中）

### 原则 3：Leave-One-Domain-Out 与纯→混渐进式域定义

- **来源**: OoD-Bench (2022) 的 Leave-One-Domain-Out CV + Cross-city Matters (Hong et al., 2023, RSE) 的"按城市划分域"协议 + WILDS (Koh et al., 2021) 的 iWildCam 按地理位置划分域 + DomainBed (2021) 的标准化域泛化评估。
- **适用原因**: 域迁移实验的核心风险在于"域"的定义模糊。我们的场景需要明确定义三种域：① 源域（纯林站点）、② 目标域（混交林站点）、③ 域内留出测试（混交林的空间块留出）。WILDS 的"按地理位置划分训练/测试域"直接对应我们的"按林区/林分类型划分域"。
- **具体操作**:
  - 定义三组域：D_pure（源纯林站点）、D_mixed（目标混交林站点）、D_mixed_holdout（混交林空间隔离留出）
  - 训练协议 A（域内基线）：训练 D_mixed 训练部分 → 测试 D_mixed_holdout
  - 训练协议 B（跨域零样本）：训练 D_pure → 测试 D_mixed（全部）
  - 训练协议 C（跨域少样本）：训练 D_pure + D_mixed 5-10% → 测试 D_mixed 剩余
  - 若有多纯林站点：Leave-One-Site-Out 交叉验证，每次留出一个纯林站点作为验证
  - 超参选择仅使用训练域验证集，测试域仅运行一次（DomainBed 黄金标准）

### 原则 4：空间块交叉验证防虚高

- **来源**: Kattenborn et al. (2022, ISPRS Open Journal) — 传统 random split 使 CNN 精度虚高 15-30%，空间自相关是高分辨率遥感中最大的评估陷阱。
- **适用原因**: 混交林场景下，空间相邻的树木在冠层结构、光照条件、土壤类型上高度相关。随机划分会打破空间自相关结构，使验证精度严重高估，进而低估真正的域迁移退化幅度。
- **具体操作**:
  - 在混交林站点内使用 spatial block K-fold cross-validation（块大小 ≥ 50m × 50m）
  - 训练块与测试块在空间上完全不相邻（最小间隔 ≥ 30m）
  - 报告 random split 与 spatial block split 的精度差异作为"空间自相关效应"（预期差 15-30%）
  - 参考 DomainBed (2021) 的严格协议：在所有实验中使用相同的空间块划分，不因模型不同而改变
  - 源纯林站点同样使用 spatial block split 以确保域内基线也是诚实评估

### 原则 5：Per-Species 分层——Head、Tail、Unknown 三级

- **来源**: TaxoNet (2025, `2512.18994`) 的双边界损失 + 范数引导样本选择揭示了 Head-Tail recall 差距 35 pp + Change is Hard (Yang et al., 2023, `2302.12254`) 的子群体偏移框架中的"类别不平衡偏移"类别 + WILDS 的 Worst-Class Accuracy 指标。
- **适用原因**: 纯林（1-2 物种，均衡）→混交林（5-20+ 物种，极端长尾）导致域迁移带来的退化在不同物种间不均衡——纯林中训练充分的优势树种退化小，混交林中新增的尾部树种退化大甚至无法识别。
- **具体操作**:
  - 按训练集出现频率将混交林树种分为三级：
    - **Head**（训练集中出现 ≥100 样本/或已在纯林中出现）：直接迁移类
    - **Tail**（训练集中出现 1-99 样本）：少样本迁移类
    - **Unknown**（训练集中完全不出现）：零样本/开集类
  - 对每级分别报告 recall、precision、F1
  - 对 Unknown 类额外报告开集检测率（模型是否输出"未知"而不是强行分到已知类）——参考 TaxoNet 的 TNR (True Negative Rate) 指标
  - 使用 TaxoNet 的范数引导策略识别"嵌入范数低"的样本（通常对应困难样本/尾部类），追踪它们在域迁移后的嵌入位移

### 原则 6：不确定性量化分离域内困难与域外失败

- **来源**: UQ Survey (Gawlikowski et al., 2023, `10.1007/s10462-023-10562-9`) — 系统综述 DNN 中 Aleatoric（数据固有不确定性）与 Epistemic（模型知识不足）不确定性的区分方法。DISC (Wu et al., 2023, `2305.00650`) 的"不稳定概念发现"机制可辅助识别域迁移后失效的视觉特征。
- **适用原因**: "冠层遮挡导致分类困难"（域内困难，Aleatoric 主导）与"混交林中新树种无法识别"（域外样本，Epistemic 主导）是两种性质根本不同的错误，但它们都可能表现为低置信度预测。不确定性量化可以自动区分二者，无需人工标注每个错误的原因。
- **具体操作**:
  - 部署 MC Dropout / Deep Ensemble 对分类器进行不确定性估计
  - 将每个测试样本分配到四象限：
    - 高 Aleatoric + 低 Epistemic → 域内困难样本（如重度遮挡、低光照）
    - 低 Aleatoric + 高 Epistemic → 域外/未见过模式（如新物种、新冠层结构）
    - 双高 → 复杂混淆
    - 双低 → 模型自信（可能正确或静默失败）
  - 对每个象限统计错误率，验证 Aleatoric 是否真的对应冠层遮挡/低光照，Epistemic 是否真的对应新物种/新林分结构
  - 计算 Expected Calibration Error (ECE) 检查模型在域迁移后是否变得更不校准

### 原则 7：多模态消融——光学 vs LiDAR 的错误缓解谱

- **来源**: Lee et al. (2023) 展示了 UAV RGB + LiDAR DSM 融合对混交林分类的提升效应；LiDAR Review (Neuville et al., 2021) 系统总结了点密度对 OA 的决定性影响（2-5 pts/m² ~50% → 50+ pts/m² ~70%）；Zhong et al. (2022) 在中国混交林上展示多源融合 OA 提升 ~12%。
- **适用原因**: LiDAR 提供冠层结构信息（高度、形态、垂直分层），在理论上应缓解"形态相似性导致的分类混淆"（如 Beloiu 2023 中松树→冷杉 31% 误分类），但无法解决"光谱重叠导致的混淆"。系统性消融可以精确刻画 LiDAR 的受益边界。
- **具体操作**:
  - 三组对比：RGB-only / LiDAR-only / RGB+LiDAR fused，统一 backbone 和训练协议
  - 对每组，按错误类型分层报告性能：
    - 形态混淆型错误（冠层结构相似、光谱不同的树种对）→ 预期 LiDAR 缓解
    - 光谱混淆型错误（光谱相似、冠层结构不同的树种对）→ 预期光学缓解
    - 双模态混淆型（冠层和光谱均相似）→ 预期均无效
  - 按冠层重叠程度（开阔 vs 轻度重叠 vs 严重重叠）分层 → 验证 LiDAR 的冠层穿透优势
  - 按 LiDAR 点密度分层（有效点密度在高遮挡区自动降低）→ 绘制点密度-精度曲线

### 原则 8：诚实模型选择协议与统计显著性

- **来源**: DomainBed (Gulrajani & Lopez-Paz, 2021, `2007.01434`) — 证明了在严格调参协议下 ERM 与专用域泛化方法表现相当，揭示了之前文献中"域泛化增益"的虚高来源。Towards Reliable DG (2023, `2309.06142`) 提供了 DG 实验规范检查清单。
- **适用原因**: 这是一个诊断实验，不是方法创新实验——但"诊断的正确性"同样依赖于诚实的评估协议。如果我们在超参选择中泄露了测试域信息，诊断结论（如"LiDAR 缓解了 X% 的失败"）将是无效的。
- **具体操作**:
  - 超参搜索仅使用源域（纯林）验证集，完全禁止接触目标域（混交林）的任何信息
  - 每个实验配置使用 ≥5 个随机种子，报告均值和标准差
  - 使用 Wilcoxon signed-rank test (α=0.05) 比较不同消融配置之间的差异显著性
  - 参考 DomainBed 的"模型选择协议清单"逐项核查
  - 报告训练域验证精度与测试域精度的 Spearman 排序相关性——低相关性说明超参在域迁移后失效（本身即诊断信号）

---

## Part 2：实验方案

### 实验 0：数据准备与环境搭建

- **目的**: 建立可复现的实验环境，包括数据集选择、预处理、标注对齐
- **方法**:
  - **源域数据集**（二选一）:
    - **PureForest** (arXiv `2404.12064`): 法国全国纯林/人工林，ALS 点云 + VHR 航片，18 语义类（13 树种）。覆盖多种生态区，利于构建"跨生态区基线"。
    - **NEON Tree Crowns** (Weinstein et al. 2021): 美国 22 个 NEON 站点，RGB + LiDAR + 高光谱。可按站点选择纯林为主的站点作为源域。
  - **目标域数据集**（候选）:
    - **TreeSatAI** (Ahlswede et al. 2023): 德国 3 州温带混交林，20 树种，Sentinel-1/2 + 航片 + UAV。
    - **FOR-instance** (Puliti et al. 2023): 欧洲混交林，UAV-LiDAR 点云，含林下植被标注。
    - **中国团队数据**: 帽儿山天然次生林（Zhong 2022/2024, Quan 2023, Ma 2024），东北混交林，UAV LiDAR + 高光谱。
  - 首选组合：PureForest（源）→ TreeSatAI（目标），因为两者都是欧洲温带森林，传感器模态重叠（ALS/UAV LiDAR + 航片），且 TreeSatAI 是公开基准。
  - **预处理**: 统一空间分辨率（如重采样至 10 cm GSD）、统一分类体系（映射到共同的树种/属级分类）、标注格式统一（coco JSON 或 YOLO 格式）。
- **输出**: 标准化的 train/val/test 数据集 splits，数据统计卡片（物种数、样本分布、域划分）
- **参考来源**: DomainBed 的数据标准化流程，Cross-city Matters 的跨域数据集构建协议

---

### 实验 1：域内基线校准（Mixed→Mixed）

- **目的**: 回答"模型在混交林本身的识别上限是多少？"——为域迁移退化提供相对基准
- **方法**:
  - 在混交林目标站点上，使用 spatial block K-fold CV（原则 4），训练并测试树种分类模型
  - 检测器 + 分类器两阶段评估（原则 1）
  - 使用与实验 2 完全相同的模型架构和训练协议
  - K=5，块大小 50m×50m，块间最小间隔 30m
- **输出**:
  - 混交林域内全局 OA / macro-F1 / per-species F1
  - 检测 recall 按冠层重叠程度分层
  - Per-species 混淆矩阵
  - Random split vs Spatial block split 精度差异（预期差 15-30%）
  - 域内 Head-Tail Gap（原则 5）
- **参考来源**: DomainBed 的域内 ERM 基线协议，Kattenborn et al. (2022) 的空间块对比

---

### 实验 2：纯林→混交林零样本迁移（Pure→Mixed Zero-Shot）

- **目的**: 回答"不做任何适应，纯林预训练模型在混交林中表现多差？"——量化域迁移的总退化量
- **方法**:
  - 使用 PureForest（或 NEON 纯林站点）上预训练的模型，直接（不 fine-tune）在混交林目标站点上推理
  - 推理前对输入数据进行最小必要的域对齐（如直方图匹配、通道归一化），但不修改模型权重
  - 分别评估检测器和分类器的迁移表现（原则 1）
  - 与实验 1 的域内基线对比，计算：
    - **Domain Drop Rate** = (域内 F1 − 跨域 F1) / 域内 F1（来源: `11_Evaluation_Longtail_Design.md:255`）
- **输出**:
  - 全局 Domain Drop Rate（预期 20-40%，来源: `04_Field_Robustness.md:19`）
  - Per-species Domain Drop Rate（识别哪些物种退化最严重）
  - 混淆矩阵变化图（域内混淆 → 跨域混淆的方向性迁移）
  - 源域验证精度 vs 目标域精度 Spearman ρ（原则 8）
- **参考来源**: Beloiu et al. (2023) F1 0.72→0.45, TaxoNet (2025) Top-1 91.2%→67.8%, TreeSatAI OA 85%→61%

---

### 实验 3：检测失效 vs 分类失效分解

- **目的**: 回答"错误中多大比例是'树没找到'、多大比例是'找到了但认错了'？"
- **方法**:
  - 使用混交林 ground-truth 标注的树冠边界框（bbox）作为"完美检测器"
  - 对每个 ground-truth bbox，用预训练分类器进行树种识别 → 得到"纯分类迁移精度"
  - 对比使用自动检测器（从纯林预训练）时的端到端精度
  - 差异 Δ_det = 端到端精度 − 纯分类迁移精度 → 归因于检测失效
  - 剩余退化 Δ_cls = 域内分类精度 − 纯分类迁移精度 → 归因于分类失效
- **输出**:
  - 检测贡献误差率 vs 分类贡献误差率的比例
  - 按冠层重叠程度分层的检测失效比例（预期重叠区检测失效占主导）
  - 按物种分层的分类失效比例（预期形态相似/光谱混淆的物种对分类失效占主导）
  - 冠层重叠 vs 物种混淆的交互效应
- **参考来源**: Sivanandam et al. (2022) 的冠层重叠区 mAP 0.50 vs 开阔区 0.65；Beloiu et al. (2023) 的 false negative 瓶颈分析；`13_Our_Library_Transfer_Methods.md:89-98` 的检测/分类独立评估汇总

---

### 实验 4：错误根因归因——Diversity Shift vs Correlation Shift

- **目的**: 回答"域迁移错误是因为混交林有新物种/新模式（多样性不足），还是学到了错误的背景关联？"
- **方法**:
  - 对每个误分类的测试样本，执行根因判定（原则 2）：
    - 步骤 1：检查该样本的树种（或冠层形态 bin）在源域训练集中是否存在 → 若不存在，标记为 **Diversity Shift**
    - 步骤 2：检查该样本的背景特征组合（光照等级、冠层密度等级、地形坡度-坡向组合）在源域中是否与该树种共现 → 若源域中共现良好但目标域仍错误，标记为 **Correlation Shift**
    - 步骤 3：剩余样本标记为 **Ambiguous**（可能是域内困难，需实验 6 进一步区分）
  - 使用 DISC (`2305.00650`) 的"不稳定概念发现"思想：在源域和目標域的中间层特征上训练一个"域判别器"，找出跨域最不稳定的特征通道（对应该通道的视觉概念可能是虚假相关）
- **输出**:
  - 错误归因饼图：Diversity Shift % / Correlation Shift % / Ambiguous %
  - 每类 Shift 的 Top-5 代表物种
  - "不稳定概念"通道索引及其对应的视觉语义（如"林下灌丛纹理"或"坡向阴影"）
  - Diversity Shift 占比高 → 结论：需扩大训练数据覆盖混交林物种
  - Correlation Shift 占比高 → 结论：需域自适应/不变表征学习
- **参考来源**: OoD-Bench (`2106.03721`) 的二维分解框架；DISC (`2305.00650`) 的概念发现机制；Change is Hard (`2302.12254`) 的子群体偏移分类

---

### 实验 5：LiDAR 消融——结构信息能修复什么、不能修复什么

- **目的**: 回答"加入 LiDAR 后，哪些错误类型被修复、哪些依然存在？"
- **方法**:
  - 在混交林目标站点上，构建三组输入配置（原则 7）：
    - **RGB-only**：仅光学影像（RGB 或多光谱）
    - **LiDAR-only**：仅 LiDAR 衍生特征（CHM、强度、回波特征、点云体素）
    - **RGB+LiDAR**：多模态融合（如早融合/晚融合两种策略）
  - 三组使用相同的主干网络（如 ResNet-50 / Swin Transformer）+ 相同的训练协议
  - 对三组分别执行实验 2（跨域零样本）和实验 3（检测/分类分解）
  - 计算 LiDAR 增益 = RGB+LiDAR 指标 − RGB-only 指标
- **输出**:
  - LiDAR 增益按错误类型分层：
    - 形态混淆型错误对 → LiDAR 高增益（预期 +10-20 pp）
    - 光谱混淆型错误对 → LiDAR 低/零增益
    - 冠层重叠严重区检测 → LiDAR 中高增益
    - 低光照/阴影区 → LiDAR 中增益（LiDAR 不受光照影响）
  - LiDAR 不能修复的错误列表（如纯光谱区分的树种对、训练集中完全缺失的物种）
  - 点密度-精度曲线（按目标域 LiDAR 有效点密度分层）
  - 融合策略对比（早融合 vs 晚融合）
- **参考来源**: Lee et al. (2023) 的 LiDAR DSM + RGB 融合对混合标签的提升效应；Marinelli et al. (2022) 的点密度-OA 关系；Zhong et al. (2022/2024) 的中国混交林多源融合证据；Neuville et al. (2021) LiDAR 综述

---

### 实验 6：域内困难与域迁移的混淆剥离

- **目的**: 回答"精度下降中，多少是因为混交林本来就难（遮挡、光照差），多少是纯的域迁移效应？"
- **方法**:
  - 在混交林目标域内，对测试集按三个物理因子分层：
    - **冠层遮挡等级**（通过 LiDAR 穿透率/冠层间隙分数估计）：开阔 / 轻度遮挡 / 重度遮挡
    - **光照条件**（通过 RGB 图像亮度直方图 / 阴影指数）：良好光照 / 中等 / 低光照/阴影
    - **树密度**（单位面积树冠数）：低密度 / 中密度 / 高密度
  - 在每一层内，对比"域内训练模型的精度"（实验 1）与"跨域零样本模型的精度"（实验 2）
  - 域内困难效应 = 域内模型在困难层与简单层之间的精度差（如遮挡严重 vs 开阔）
  - 域迁移效应 = 跨域模型与域内模型在同一困难层上的精度差
- **输出**:
  - 困难分层精度表（3 因子 × 3 等级 × 2 模型来源 = 18 个精度值）
  - 域内困难效应大小 vs 域迁移效应大小的对比
  - 交互效应：域迁移是否在困难层上放大退化（如遮挡+迁移的双重打击）
  - 结论示例："纯林→混交林下降 30 pp 中，约 12 pp 是混交林本身的困难（遮挡/密度），约 18 pp 是纯的域迁移"
- **参考来源**: Beloiu et al. (2023) 的 site conditions（林分结构）vs illumination 的分离分析；Silwal et al. (2024) 的 GSD/分辨率控制变量实验范式；`04_Field_Robustness.md:86-99, 113-118` 的地形和混合林难度因素

---

### 实验 7：不确定性引导的透明失败检测

- **目的**: 回答"能否在不看 ground-truth 的情况下，预测模型会在哪些样本上出错？"
- **方法**:
  - 使用 Deep Ensemble（5 个独立训练的模型）估计 Epistemic Uncertainty（模型间预测方差）
  - 使用 MC Dropout (T=50) 或 Test-Time Augmentation 估计 Aleatoric Uncertainty（单模型预测熵 + 增强扰动下的方差）
  - 将测试样本按不确定性等级分桶（低/中/高），计算每桶的精度
  - 对高不确定性样本进行手工审查，验证其是否对应预期失败模式（原则 6 的四象限）
  - 计算 Prediction Rejection Ratio：如果按不确定性排序丢弃最高 20% 的预测，剩余预测的精度提升多少
- **输出**:
  - 不确定性-精度曲线（各桶精度）
  - 高 Epistemic 样本的可视化图集（验证是否确实为新物种/新模式）
  - 高 Aleatoric 样本的可视化图集（验证是否确实为重度遮挡/低光照）
  - ECE 对比（源域校正误差 vs 目标域校正误差）
  - 预测拒绝比（如丢弃 20% 最高不确定度的预测后 OA 从 61% 提升至 74%）
- **参考来源**: UQ Survey (`10.1007/s10462-023-10562-9`) 的方法分类学；DUNIA (`02_Contrastive_Multimodal_Learning.md:9-17`) 的 KNN 检索不确定性；WILDS 的 ECE 指标

---

### 实验 8：长尾物种的域迁移脆弱性

- **目的**: 回答"域迁移对常见树种和稀有树种的打击是否对等？"
- **方法**:
  - 按源域（纯林）训练集中出现频率将目标域树种分桶（原则 5）：
    - Head: 训练集中与该物种相同/最相似物种 ≥100 样本
    - Tail: 1-99 样本
    - Unknown: 0 样本（训练集中无任何近缘种）
  - 对每一桶计算 Domain Drop Rate、检测 recall、分类 precision
  - 分析 Unknown 类的预测分布（模型将它们错误分配到了哪些已知类？错误分配是否有模式？）
  - 如果实验 7 中使用了不确定性方法：检查 Unknown 类样本的平均 Epistemic Uncertainty 是否显著高于 Head 类
- **输出**:
  - Head-Tail-Unknown 三级 Domain Drop Rate 对比
  - Unknown 类的 Top-5 误分类目标（如"新物种 X 被系统性地分类为已知物种 Y"）
  - 尾部类中哪些在域迁移后"幸存"（退化小），哪些"崩溃"（退化大）
  - Tail-Recall@K 指标（K=最稀有 20% 物种的平均 recall）
- **参考来源**: TaxoNet (2025) 的 Head 92.4% vs Tail 57.1% recall gap；EKDC-Net (`2601.16498`) 的 +6.42% 长尾校准；Quan et al. (2023) 的 per-species 最低 60%；WILDS 的 Worst-Class Accuracy 指标

---

### 实验 9（可选）：少样本桥接——最少需要多少混交林标注？

- **目的**: 回答"如果允许在混交林标注极少样本（如每物种 1-5 棵），能恢复多少精度？"
- **方法**:
  - 在混交林站点标注极少样本：1-shot / 3-shot / 5-shot / 10-shot per species
  - 用这些少样本在纯林预训练模型上进行适配（方法选项：① 仅重训练分类头 ② LoRA 微调 ③ KNN 检索增强）
  - 对比零样本迁移（实验 2）和域内全量训练（实验 1）之间的精度恢复曲线
- **输出**: 标注量-精度恢复曲线，标注效率分析（每增加 1 个标注样本的边际精度增益）
- **参考来源**: DUNIA 仅需 0.25% 标注量即接近 SOTA；FewShot Invasive Trees 3-shot F1=0.86；TaxoNet 的跨域少样本 recall 提升；`13_Our_Library_Transfer_Methods.md:192-200` 的标签效率数据

---

## Part 3：工具和方法清单

### 一、数据集

| 数据集 | 覆盖范围 | 传感器 | 树种数 | 获取方式 | 适用实验 |
|--------|---------|--------|--------|---------|---------|
| **PureForest** | 法国全国 | ALS + VHR 航片 | 18 类 (13 树种) | arXiv `2404.12064`，联系作者 | 源域训练 (Exp 1,2) |
| **NEON Tree Crowns** | 美国 22 站点 | RGB + LiDAR + 高光谱 | 多物种 | NEON Data Portal | 源域训练备选 |
| **TreeSatAI** | 德国 3 州 | Sentinel-1/2 + 航片 + UAV | 20 树种 | Zenodo / ESD 公开 | 目标域测试 (全实验) |
| **FOR-instance** | 欧洲混交林 | UAV-LiDAR | 多物种 | arXiv `2309.01279` | 目标域测试备选（LiDAR 重点） |
| **IDTReeS** | 美国多生物群系 | RGB + LiDAR | 多物种 | idtrees.org | 额外泛化测试 |
| **StreetTree** | 全球城市 | RGB (街景) | 8,300+ 种 | arXiv `2602.19123` | 长尾开集评估 |

### 二、树检测开源工具

| 工具 | 方法 | 输入 | 代码 | 备注 |
|------|------|------|------|------|
| **DeepForest** | RetinaNet (RGB 树冠检测) | RGB 航片/UAV | `github.com/weecology/DeepForest` | NEON 预训练，生态学社区标准 |
| **Detectron2** | Mask R-CNN / Faster R-CNN | RGB / 多光谱 | `github.com/facebookresearch/detectron2` | 通用目标检测，需自行标注训练 |
| **YOLOv8 / YOLOv11** | Anchor-free 检测 | RGB / 多光谱 | `github.com/ultralytics/ultralytics` | Zhong et al. (2024) 在混交林中使用 |
| **SegmentAnyTree** | 3D CNN (PointGroup 架构) | ULS/MLS/TLS 点云 | 未开源（论文描述可复现） | 跨传感器 3D 分割，仅有的多层冠层评估 |
| **SAM-2** | 提示式分割 | RGB | `github.com/facebookresearch/sam2` | 可零样本泛化至树冠分割 |

### 三、树种分类基础模型/预训练权重

| 模型 | 预训练数据 | 模态 | 代码/权重 | 用途 |
|------|-----------|------|----------|------|
| **TaxoNet** | Auto-Arborist + iNat + NAFlora | RGB | 论文描述，无公开权重 | 长尾树种分类基线，双边界损失 |
| **GeoTreeCLIP** | Sentinel-2 + 气候 + 文本 | 多光谱 + 文本 | 权重需联系作者 | 21k 树种零/少样本，VL 式分类 |
| **DUNIA** | Sentinel-1/2 + GEDI | 光学 + SAR + LiDAR 波形 | 论文描述，零样本检索框架 | 跨模态密集对齐，零样本森林变量 |
| **SatMAE** | Sentinel-2 (全球) | 多光谱 | `github.com/sustainlab-group/SatMAE` | 遥感 MAE 预训练 backbone |
| **Prithvi** | HLS (Landsat + Sentinel-2) | 多光谱 | Hugging Face (NASA-IMPACT) | 遥感基础模型，支持多下游任务 |
| **DINOv2** | 大规模网络图像 | RGB | `github.com/facebookresearch/dinov2` | 通用视觉特征，可 LoRA 适配 |
| **CLIP (OpenCLIP)** | 大规模图像-文本 | RGB + 文本 | `github.com/mlfoundations/open_clip` | 零样本跨模态基线 |
| **EKDC-Net** | CU-Tree102 | RGB | arXiv `2601.16498` | 即插即用长尾校准模块（0.08M 参数） |

### 四、域泛化/域自适应工具

| 工具/方法 | 功能 | 代码 | 参考 |
|----------|------|------|------|
| **DomainBed** | 标准化 DG 算法评估套件（14 种算法 + 7 数据集） | `github.com/facebookresearch/DomainBed` | Gulrajani & Lopez-Paz (2021) |
| **WILDS** | 分布外泛化基准（10 数据集 + 标准化协议） | `github.com/p-lambda/wilds` | Koh et al. (2021) |
| **DANN (Domain-Adversarial NN)** | 对抗域自适应 | PyTorch 实现广泛可用 | PureForest (2024) 中用于跨生态区 |
| **CORAL** | 协方差对齐 | `github.com/VisionLearningGroup/CORAL` | 简单有效的特征对齐 |

### 五、错误分析与可解释性工具

| 工具 | 功能 | 代码 | 用途 |
|------|------|------|------|
| **SHAP** | 特征归因 (Shapley 值) | `github.com/shap/shap` | 单样本错误解释 — 哪些像素/特征导致误分类？ |
| **Captum** | PyTorch 可解释性工具箱 | `github.com/pytorch/captum` | 梯度归因、遮挡分析、Integrated Gradients |
| **TCAV** | 概念激活向量 | `github.com/tensorflow/tcav` | 测试"背景植被类型"等概念对分类的影响 |
| **Yellowbrick** | 混淆矩阵可视化 | `github.com/DistrictDataLabs/yellowbrick` | Per-class error 可视化 |
| **Uncertainty Toolbox** | 不确定性校准分析 | `github.com/uncertainty-toolbox/uncertainty-toolbox` | ECE、可靠性曲线、不确定性分解 |

### 六、遥感预处理与数据分析

| 工具 | 功能 | 代码/来源 |
|------|------|----------|
| **GDAL / rasterio** | 栅格数据读写与预处理 | `gdal.org` / `github.com/rasterio/rasterio` |
| **PDAL** | 点云数据处理 | `pdal.io` (支持 ALS/UAV LiDAR) |
| **Open3D** | 3D 点云可视化与处理 | `github.com/isl-org/Open3D` |
| **laspy** | Python LAS/LAZ 点云读写 | `github.com/laspy/laspy` |
| **s2cloudless** | Sentinel-2 云掩膜 | `github.com/sentinel-hub/sentinel2-cloud-detector` |
| **FMask** | Landsat/Sentinel-2 云/阴影检测 | `github.com/GERSL/Fmask` |
| **scikit-learn** | ML 评估指标、统计检验、LODO CV | `scikit-learn.org` |
| **SciPy** | Wilcoxon signed-rank test | `docs.scipy.org` |

### 七、实验管理与追踪

| 工具 | 功能 | 代码 |
|------|------|------|
| **MLflow** | 实验追踪、指标记录、模型注册 | `github.com/mlflow/mlflow` |
| **Weights & Biases** | 实验可视化、超参搜索 | `wandb.ai` |
| **Hydra** | 配置管理（YAML 配置 + 命令行覆盖） | `github.com/facebookresearch/hydra` |
| **DVC** | 数据版本控制 | `github.com/iterative/dvc` |

### 八、评估指标体系汇总

| 指标 | 公式/定义 | 来源 | 适用实验 |
|------|----------|------|---------|
| **Domain Drop Rate** | (域内 F1 − 跨域 F1) / 域内 F1 | `11_Evaluation_Longtail_Design.md:255` | Exp 2, 8 |
| **Detection Recall @ Overlap** | 按冠层重叠程度分层的检测 recall | Sivanandam 2022 | Exp 1, 3 |
| **Conditional Classification Accuracy** | 在 ground-truth bbox 上的分类精度 | Beloiu 2023 | Exp 3 |
| **Diversity Shift Score** | 未覆盖子模式上的错误占比 | OoD-Bench 2022 | Exp 4 |
| **Correlation Shift Score** | 虚假相关破裂样本上的错误占比 | OoD-Bench 2022 | Exp 4 |
| **LiDAR Gain** | F1(RGB+LiDAR) − F1(RGB-only) | 原则 7 | Exp 5 |
| **Tail-Recall@K** | 最稀有 K% 物种的平均 recall | TaxoNet 2025 | Exp 8 |
| **ECE (Expected Calibration Error)** | 置信度-准确率校准偏差 | WILDS 2021 | Exp 7 |
| **Prediction Rejection Ratio** | 丢弃高不确定度预测后的精度提升比 | UQ Survey 2023 | Exp 7 |
| **Worst-Class Accuracy** | 所有类别中最低的 per-class F1 | WILDS 2021 | Exp 2, 8 |
| **Worst-Domain Accuracy** | 若多站点：最差站点的 F1 | WILDS 2021 | Exp 2 |
| **CER (Cost-Weighted Error)** | Σ(错误权重 × 错误数) / 总决策数 | `11_Evaluation_Longtail_Design.md:258` | Exp 8 (濒危物种×10) |

---

## 附录：实验优先级与依赖关系

```
实验 0 (数据准备)
  ├─→ 实验 1 (域内基线)
  │     └─→ 实验 6 (域内困难剥离) ← 需要实验 1 的域内模型
  │           └─→ 实验 4 (Diversity/Correlation Shift) ← 需要实验 6 的困难分层
  ├─→ 实验 2 (零样本迁移)
  │     ├─→ 实验 3 (检测/分类分解) ← 需要实验 2 的零样本预测
  │     ├─→ 实验 8 (长尾脆弱性) ← 需要实验 2 的 per-species 结果
  │     └─→ 实验 5 (LiDAR 消融) ← 需要实验 2+3 的框架
  ├─→ 实验 7 (不确定性分析) ← 可并行
  └─→ 实验 9 (少样本桥接) ← 可选，依赖实验 2 基线
```

**建议执行顺序**：实验 0 → 实验 1 + 实验 2（并行）→ 实验 3 + 实验 6（并行，各依赖 1 或 2）→ 实验 4 + 实验 5 + 实验 7 + 实验 8（并行）→ 实验 9。

---

*生成日期: 2026-05-19 | 融合材料数: 9 | 涉及论文: ~50+ 篇*
