# 森林/植物AI系统的长尾识别、开集检测与评估协议设计

## 1. 林业场景长尾/开集问题的文献全貌

### 1.1 长尾分类的通用方法演进

长尾识别（Long-Tailed Recognition, LTR）的核心矛盾在于：模型在头部类（majority）上获得充分训练，但在尾部类（minority）上因样本稀缺而产生严重的决策边界偏移。现有方法分为三条技术路线：

| 路线 | 代表方法 | 核心思想 | 林业适用性 |
|------|---------|---------|-----------|
| **数据重采样** | MES (Cheng et al., 2025), Decoupling (Kang et al., 2020) | 少数类过采样/多数类欠采样 | 简单但易过拟合尾部类 |
| **损失重加权** | LDAM (Cao et al., 2019), CBLoss (Cui et al., 2019) | 按类频率倒数加权损失 | 梯度不稳定，对极端长尾无效 |
| **表示学习重平衡** | TaxoNet (arXiv 2025), BCL (Zhu et al., 2022) | 在嵌入空间中施加类间margin约束 | 林业最有效路径 |

**TaxoNet 的核心贡献**（参见 `02_Contrastive_Multimodal_Learning.md`）：双边界惩罚损失（Dual-Margin Penalization）——对目标类引入大 margin（增强尾部类 intra-class compactness），对非目标尾部类原型引入 margin 抑制头类主导的排斥梯度。该梯度以 exp(m_yi - m_c) 指数衰减，数学上保证尾部类不被头部类"淹没"。配合范数引导样本选择（优先选取低范数嵌入的尾部类样本）和 AugMix 增强，在三个植物数据集上实现 macro-recall 增益：Google Auto-Arborist +6%、iNaturalist-Plantae +3%、NAFlora-Mini +1%。

### 1.2 遥感场景的长尾识别

遥感图像的长尾分布具有独特性质：

- **空间相关性**：相邻像素往往属于同一类别，简单地按像素频率重采样会破坏空间上下文
- **类内光谱方差大**：同一土地覆盖类型在不同季节/光照下光谱差异巨大，长尾类的类内方差甚至超过头类
- **类别定义模糊**：如"稀疏森林"与"灌木"之间的界限主观，尾部类标注噪声高

**DECOR (IEEE GRSL 2024)**：针对遥感场景的长尾识别提出了分解式学习框架。核心思路是将特征空间分解为"类共享"子空间和"类特定"子空间，前者捕获通用纹理/形状特征，后者捕获细粒度类间差异。实验覆盖 NWPU-RESISC45-LT 和 AID-LT 两个长尾遥感基准，以及自建的 BIT-AFGR50-LT 数据集。

**MES (Agriculture 2025)**：面向 UAV 遥感作物分类的少数类增强采样器。核心原理是将重采样概率与像素频率关联，在采样时对少数类像素实施"密集重采样"，同时施加在线数据增强（随机旋转、色彩抖动）防止过拟合。在四个语义分割骨干网络（PSPNet、DeepLabv3+、SegFormer、UPerNet）上验证，mIoU 提升 +1.54% ~ +7.08%。该方法的"plug-and-play"特性使其可直接嵌入 ForestPheno 的语义分割管线。

**关键空白**：现有遥感长尾研究几乎全部聚焦土地覆盖/作物分类场景，**专门针对树种级别（species-level）长尾识别的遥感研究几乎空白**。树种分类不仅面临样本数量长尾，还面临"光谱混淆长尾"——外观相似的不同树种在遥感影像中的光谱特征高度重叠（如云杉 vs 冷杉）。

### 1.3 开集识别（Open-Set Recognition, OSR）

开集识别要求模型既能准确分类已知类别，又能拒绝未知类别样本。林业场景中的开集需求源自：

1. **新树种出现**：引种/入侵树种未在训练集中出现
2. **新病虫害**：未知病原体导致的异常表型
3. **新物候状态**：训练集未覆盖的极端气候条件下的异常表型

**TaxoNet 的 OSR 能力**：在 Google Auto-Arborist 数据集上实现 TNR（True Negative Rate）91.3%，证明其双边界机制天然具备开集判别能力——尾部类原型的大 margin 为"未知"区域预留了决策空间。具体而言，当测试样本的嵌入与所有已知类别原型的距离均超过阈值时，标记为"未知"。

**领域泛化（Domain Generalization）**：TaxoNet 在 AA-Central → AA-West/East 的跨区域迁移中 recall 提升 3.5-4%，说明双边界损失学习的嵌入空间对分布偏移具有鲁棒性。这对 ForestPheno 尤为重要——模型需要在训练区域A的森林中学习识别树种，然后在未见过的区域B的森林中部署。

**现有方法的不足**：
- 目前没有同时处理**长尾 + 开集 + 遥感多模态**的端到端框架
- 开集检测的阈值设定缺乏林业场景的先验知识指导
- 域迁移评估停留在"同数据集不同子集"，缺乏跨传感器（UAV→卫星）的真实跨域测试

### 1.4 小样本/零样本稀有物种识别

| 方法 | 场景 | 关键技术 | 性能 |
|------|------|---------|------|
| DUNIA (ICML 2025) | 遥感零样本检索 | 像素级嵌入 + KNN | 树种 wF1=76.0% (KNN=5) |
| TaxoNet | 植物细粒度分类 | 双边界损失 + 范数选择 | 尾部类 recall=57.1% |
| ProtoNet (Snell et al., 2017) | 通用小样本 | 原型网络度量学习 | - |

DUNIA 证明了**像素级嵌入的零样本潜力**——在仅 20% 标签量下垂直结构（树高、冠层覆盖）性能几乎无损。这对 ForestPheno 极为重要：可在有限的地面实测数据下构建联合嵌入检索库，实现新获取影像的零样本森林参数估算。

---

## 2. 现有评估方法的不足

### 2.1 PhenoAssistant 式的"手动评估"

PhenoAssistant 当前的评估范式属于 **基于人工审核的定性评估**：
- **任务设计**：给定影像+问题，Agent 输出答案 + 推理链
- **指标**：人工判断"正确/部分正确/错误"，统计准确率
- **局限**：
  1. **不可规模化**：每个测试用例需人工审核，无法支撑大规模回归测试
  2. **主观偏差**：植物分类学专家的判断标准不一致（见 `05_Forest_DL_Methods.md` 中 Joshi & Witharana 的标注者差异问题）
  3. **缺乏分解评估**：无法区分"检索错误"（Agent 调用了错误的工具/知识）vs "推理错误"（Agent 有正确信息但推理逻辑错误）
  4. **无开集评估**：人工审核不会主动设计"不可能回答"或"超出知识范围"的测试用例
  5. **无长尾意识**：测试集通常平衡采样，掩盖了尾部类性能

### 2.2 手动评估 vs 自动化评估的对比

| 维度 | 手动评估（PhenoAssistant 现状） | 自动化评估（目标状态） |
|------|-------------------------------|---------------------|
| 规模 | ~50-100 样本/轮 | 1000+ 样本/轮 |
| 可复现性 | 低（人工判断波动） | 高（程序化指标） |
| 覆盖度 | 主观选择，偏头类 | 分层采样，覆盖尾部 |
| 成本 | 1-2 人天/轮 | 秒级 |
| 开集测试 | 无 | 可系统化引入 |
| 细粒度诊断 | 仅最终答案 | 可分解到每个推理步骤 |

### 2.3 机器学习社区的标准评估 vs 科学 Agent 评估的鸿沟

传统 ML 评估（ImageNet 范式）与科学 Agent 评估之间存在本质差异：

| 差异维度 | ML 分类器评估 | 科学 Agent 评估 |
|---------|-------------|---------------|
| 输出空间 | 固定类别集合 | 开放的自然语言 + 结构化数据 |
| 评估粒度 | 单次预测 | 多步推理链 |
| 正确性定义 | Top-1 match | 语义等价 + 科学严谨性 |
| 部分正确 | 不存在 | 常见且需要细粒度评分 |
| 知识来源 | 仅从数据学习 | 检索 + 推理 + 先验知识 |
| 错误代价 | 均匀 | 高度不均匀（珍稀物种误判 > 常见物种） |

这意味着 ForestPheno 的评估不能简单套用分类指标（accuracy, macro-F1），需要设计**层次化、分维度的评估协议**。

---

## 3. AI Agent 系统评估的现有框架

### 3.1 通用 Agent 评估基准

| 基准 | 时间 | 任务类型 | 评估方式 | 规模 | 与 ForestPheno 的相关性 |
|------|------|---------|---------|------|----------------------|
| **SWE-bench** (Jimenez et al., 2023) | 2023 | 代码修复 | 自动化测试通过率 | 2294 个 GitHub issue | 低（软件工程，非科学任务） |
| **GAIA** (Mialon et al., 2023) | 2023 | 通用问答 | 精确字符串匹配 | 466 个问题 | 中（推理链设计可借鉴） |
| **MLAgentBench** (Huang et al., 2024) | 2024 | ML 研究任务 | 自动化指标 + 人工审核 | 13 个任务 | 中（Agent + 科学工作流） |
| **ScienceAgentBench** (2024) | 2024 | 科学数据分析 | 自动化验证 | - | 高（最接近的领域基准） |
| **SWE-agent** (Yang et al., 2024) | 2024 | 代码修复 | 自动化测试 | - | 低 |

### 3.2 医疗 AI 评估的 QUEST 框架 (npj Digital Medicine 2024)

QUEST 框架是迄今最系统的 AI Agent 人工评估框架，源自对 142 项医疗 LLM 评估研究的系统综述。其核心结构：

**三阶段工作流**：
1. **Planning（规划）**：定义评估维度、采样策略、评估者资格要求
2. **Implementation & Adjudication（实施与裁定）**：双盲评估 + 分歧仲裁
3. **Scoring & Review（评分与审查）**：统计分析 + 偏差审查

**五项评估原则**（QUEST = Quality, Understanding, Expression, Safety, Trust）：
1. **Quality of Information**：信息准确性、完整性、时效性
2. **Understanding & Reasoning**：逻辑推理的连贯性、因果链的可追溯性
3. **Expression Style & Persona**：输出风格的专业性与可操作性
4. **Safety & Harm**：错误信息的潜在危害（分层加权）
5. **Trust & Confidence**：置信度校准（模型是否知道它不知道）

**对 ForestPheno 的启示**：
- 可将 QUEST 的五原则映射为 ForestPheno 的五维评估轴（见第4节）
- "Safety & Harm" 在林业场景中可转化为"生态决策风险"——对珍稀物种的误判代价远高于常见物种
- "Trust & Confidence" 对于开集检测尤为重要：Agent 必须明确表达"我不确定这是哪个物种"

### 3.3 现有框架的共同不足

1. **缺乏领域特异性长尾评估**：SWE-bench、GAIA 等不考虑类别不平衡
2. **无开集场景设计**：测试集通常封闭在已知类别内
3. **无多模态评估**：单一文本模态评估无法覆盖 ForestPheno 的视觉-光谱-结构多模态需求
4. **无生态决策链评估**：从影像到物种识别到管理建议的端到端链路无评估基准

**核心缺口**：目前**没有任何论文**提出过"如何系统评估一个面向森林/植物表型分析的 AI Agent 系统"的框架。这是 ForestPheno 评估协议设计的核心贡献空间。

---

## 4. ForestPheno 评估协议的设计原则

### 4.1 五维评估轴（改编自 QUEST + 林业领域知识）

| 评估轴 | 定义 | 核心问题 | 对应指标 |
|--------|------|---------|---------|
| **Q: 物种识别质量** | 树种/表型分类的准确性 | 它认对了吗？ | Per-class F1, Macro-F1, Tail-Recall@K |
| **U: 推理可解释性** | 推理链的生态科学合理性 | 它的推理逻辑对吗？ | 推理链正确率（RAS: Reasoning Accuracy Score） |
| **E: 开集鲁棒性** | 对未知物种/表型的检测能力 | 它知道什么不知道吗？ | AUROC, TNR@95, Open-F1 |
| **S: 决策安全性** | 错误对生态管理的影响程度 | 它犯错时有多危险？ | 代价加权错误率（Cost-Weighted Error Rate） |
| **T: 域泛化能力** | 跨季节/地点/传感器的迁移能力 | 换一个林子它还准吗？ | 域间性能下降率（Domain Drop Rate） |

### 4.2 测试场景设计的六维度

为确保评估的**全面性和生态真实性**，测试场景应沿六个维度系统变化：

| 变化维度 | 变量 | 示例 |
|---------|------|------|
| **D1: 样本频率** | 头类 → 尾部类 → 超尾部（singleton） | 松树(5000) / 冷杉(500) / 紫薇(5) |
| **D2: 季节/物候** | 春季 → 夏季 → 秋季 → 冬季 | 同一树种在不同季节的光谱变化 |
| **D3: 光照条件** | 正午强光 → 阴天漫射 → 低角度背光 | 冠层阴影变化 |
| **D4: 传感器** | UAV RGB → 卫星多光谱 → 地面相机 | 空间分辨率从 0.05m 到 10m |
| **D5: 地理区域** | 训练区域A → 未见区域B → 未见区域C | 纬度/海拔/气候差异 |
| **D6: 开集程度** | 全闭集 → 部分开集（25%未知） → 高开集（50%未知） | 外来入侵物种测试 |

### 4.3 数据划分策略

针对长尾 + 开集场景，传统的随机划分（random split）会产生严重偏差。推荐以下分层策略：

```
┌─────────────────────────────────────────────────────────┐
│  ForestPheno 数据划分协议                                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 闭集识别 (Closed-Set)                               │
│     ├── Train: 按 frequency-bin 分层抽样                │
│     │   ├── Head (≥100 samples):  80% for train         │
│     │   ├── Medium (20-99):       60% for train         │
│     │   ├── Tail (5-19):          50% for train         │
│     │   └── Ultra-tail (1-4):     1 for train (rest val)│
│     ├── Val: 每个 bin 固定数量样本                      │
│     └── Test: 按原始分布采样，确保每个类至少1个          │
│                                                         │
│  2. 开集识别 (Open-Set)                                 │
│     ├── Closed-set classes: 80% of species → Train      │
│     ├── Open-set classes: 20% of species → Only in Test │
│     │   ├── Easy unknowns: 不同属的树种                  │
│     │   ├── Medium unknowns: 同属不同种                  │
│     │   └── Hard unknowns: 同种不同亚种/栽培品种         │
│     └── 开集程度分级: 0% / 15% / 30% / 50%             │
│                                                         │
│  3. 域泛化 (Domain Generalization)                      │
│     ├── Source domain: 固定的训练地理区域                │
│     └── Target domains: 3+ 个未见区域                   │
│         ├── Near-domain: 相邻区域（同气候带）            │
│         ├── Far-domain: 远距离区域（异气候带）            │
│         └── Cross-sensor: 不同传感器（UAV → 卫星）      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**关键设计理由**：
- 按 frequency-bin 分层抽样确保尾部类在训练/验证/测试中均有代表
- Ultra-tail 类（singleton）的 1 个样本放入训练集（不评估这类），因为该类在真实部署中同样不可见
- 开集难度分级（Easy/Medium/Hard unknown）比简单的"已知/未知"二分法提供更丰富的诊断信息

---

## 5. 评估基准框架的具体设计

### 5.1 三层任务结构

```
┌─────────────────────────────────────────────┐
│           T1: 原子能力层 (Atomic)            │
│  树种分类 | 表型识别 | 冠层分割 | 树高估计   │
├─────────────────────────────────────────────┤
│           T2: 组合推理层 (Composite)         │
│  "这是哪种树 + 它健康吗" | "这个林分的主导种  │
│   + 郁闭度 + 生物量"                        │
├─────────────────────────────────────────────┤
│           T3: Agent 决策层 (Agentic)         │
│  "巡护路线规划" | "优先保护树种识别"          │
│  "火灾风险评估" | "物种入侵预警"             │
└─────────────────────────────────────────────┘
```

**设计动机**：分解评估可以精准定位 ForestPheno 在哪个层次失效——是基础视觉识别不行（T1），还是多步推理链断裂（T2），还是决策策略错误（T3）。

### 5.2 具体指标

#### T1: 闭集长尾分类指标

| 指标 | 公式/定义 | 用途 |
|------|----------|------|
| **Overall Accuracy** | 所有测试样本的正确率 | 基础对标 |
| **Macro-F1** | 每类 F1 的算术平均 | 反映尾部类是否被忽略 |
| **Tail-Recall@K** | 尾部 K% 类的平均 recall | 聚焦最稀有物种 |
| **Head-Tail Gap** | Recall_head - Recall_tail | 不平衡程度的直接度量 |
| **G-mean** | (∏_c Recall_c)^{1/C} | 对低 recall 类惩罚大 |

#### T1: 开集检测指标

| 指标 | 定义 |
|------|------|
| **AUROC** | 已知/未知二分类的 ROC 曲线下面积 |
| **TNR@95** | 在 TPR=95% 约束下的 True Negative Rate |
| **Open-F1** | (已知类 macro-F1 + 未知类 F1) / 2 |
| **OSCR** | Open-Set Classification Rate = 正确分类的已知样本 + 正确拒绝的未知样本 / 总样本 |

#### T2: 推理链质量指标

| 指标 | 定义 |
|------|------|
| **RAS (Reasoning Accuracy Score)** | 推理链中每个关键步骤的正确率（由专家按步骤标注） |
| **RCS (Reasoning Coherence Score)** | 推理步骤间逻辑连贯性（1-5 分） |
| **HC (Hallucination Count)** | 推理链中虚构的物种特征或生态事实的数量 |
| **SES (Scientific Evidence Score)** | 推理中引用的科学事实的准确率 |

#### T3: Agent 决策层指标

| 指标 | 定义 |
|------|------|
| **TAR (Task Achievement Rate)** | 复杂多步任务的完成率 |
| **DER (Decision Error Rate)** | 导致生态误导的决策比例 |
| **CER (Cost-weighted Error Rate)** | Σ(错误类型权重 × 该类错误数) / 总决策数 |
| **LTC (Long-Tail Coverage)** | Agent 在决策中是否考虑了稀有物种 |

**CER 权重矩阵示例**（林业专家定义）：
- 珍稀濒危树种误判为常见树种：权重 10.0
- 常见树种误判为珍稀树种：权重 5.0
- 健康误判为病害：权重 8.0
- 病害误判为健康：权重 3.0
- 相同属内的物种混淆：权重 2.0
- 不同属的物种混淆：权重 1.0

### 5.3 评估数据集构建蓝图

基于现有可用资源，建议构建分阶段评估数据集：

| 阶段 | 数据来源 | 样本量 | 物种数 | 长尾比例 | 用途 |
|------|---------|--------|--------|---------|------|
| **Phase 1: 闭集长尾** | NAIP (1m) + 野外照片 | 10K+ | 50-100 | 10:1~100:1 | T1 核心评估 |
| **Phase 2: 开集检测** | 留出 20% 物种 | 2K | 50 闭 + 10 开 | - | T1 开集评估 |
| **Phase 3: 域泛化** | 多区域 + 多季节 | 3K+ | 30 | - | T1/T2 跨域评估 |
| **Phase 4: Agent 决策** | 合成场景 + 专家设计 | 200 | - | - | T3 决策评估 |

### 5.4 Baseline 选择

| Baseline 类型 | 具体方法 | 评估任务 | 代表性 |
|--------------|---------|---------|--------|
| **传统分类器** | ResNet-50 + CE loss | T1 闭集 | 基础对标 |
| **长尾分类器** | ResNet-50 + LDAM / CBLoss | T1 闭集 | 长尾方法对标 |
| **遥感长尾分类器** | DECOR | T1 闭集 | 遥感领域对标 |
| **植物长尾分类器** | TaxoNet | T1 闭集 + 开集 | 植物领域最佳对标 |
| **基础模型零样本** | CLIP / DINOv2 + prompt | T1 零样本 | 跨模态泛化对标 |
| **遥感基础模型** | DUNIA / AnySat / CROMA | T1 跨模态 | 遥感预训练对标 |
| **VLM Agent** | GPT-4V / Gemini Pro Vision | T2 推理 / T3 决策 | 通用 Agent 对标 |
| **PhenoAssistant v1** | 当前系统 | 全部 | 自身迭代对标 |

### 5.5 评估流程自动化设计

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  测试用例生成 │ → │  Agent 推理   │ → │  自动评分     │
│  (自动化)     │    │  (被测系统)   │    │  (程序化)     │
└──────────────┘    └──────────────┘    └──────────────┘
       ↑                                       │
       │           ┌──────────────┐            │
       └───────────│  差异分析     │←───────────┘
                   │  (诊断报告)   │
                   └──────────────┘
```

**自动评分规则引擎**：
- T1 分类指标：标准 sklearn metrics，完全自动化
- T2 推理评估：基于关键实体匹配 + LLM-as-Judge（使用 GPT-4 或 Claude 进行推理链语义等价判断；参考 GAIA 的 validation 模式）
- T3 决策评估：预定义正确答案的离散选项 + 部分正确评分矩阵

**关键设计**：LLM-as-Judge 在 T2 推理评估中需要双重验证——一份推理链由两个独立的 Judge LLM 分别评分，Kappa 系数 > 0.7 视为可信，否则触发人工仲裁。

---

## 6. 核心结论与建议路线图

### 6.1 文献空白总结

1. **树种级长尾遥感分类**是一个几乎空白的交叉领域——高光谱 + 树种 + 长尾的三重耦合无人触及
2. **开集森林物种检测**仅有 TaxoNet 在城市树木场景（Google Auto-Arborist）验证，遥感场景完全空白
3. **Agent 系统科学评估框架**在医疗领域（QUEST）和通用编程（SWE-bench）已有探索，但**生态/林业 AI Agent 评估基准为零**

### 6.2 ForestPheno 评估协议设计优先级

| 优先级 | 任务 | 时间预估 | 产出 |
|--------|------|---------|------|
| **P0** | 构建闭集长尾树种分类测试集 + TaxoNet baseline | 2 周 | 可复现的 T1 评估 |
| **P0** | 实现 Tier-1 自动化评分 pipeline | 1 周 | metrics.py + eval config |
| **P1** | 设计开集测试用例 + Open-F1 指标实现 | 2 周 | T1 开集评估 |
| **P1** | 构建跨域泛化测试集（至少 2 个 target domain） | 3 周 | T1/T2 跨域评估 |
| **P2** | 设计 T2 推理链评估方案 + LLM-as-Judge 验证 | 3 周 | 推理评估框架 |
| **P2** | 设计 T3 Agent 决策场景 + 专家审核代价矩阵 | 4 周 | 决策评估框架 |
| **P3** | 完整 ForestPheno Benchmark v1.0 发布 | 2 周 | 可引用的评估基准 |

### 6.3 论文发表建议

建议将评估协议部分整理为 **"ForestPheno-Bench: A Comprehensive Evaluation Protocol for Forest Phenotyping AI Agents"**，贡献点：

1. **首个**面向森林表型 AI Agent 的系统性评估框架
2. 提出 **QUEST-Forest 五维评估轴**（在 QUEST 框架基础上的领域适配创新）
3. 提出**代价加权错误率（CER）**作为生态决策安全的统一度量
4. 提供 **Tail-Recall@K** 作为长尾物种保护的核心指标
5. 开源评估数据集 + 自动评分工具链

---

## 参考文献

1. Loeffler, H.H. et al. "Reinvent 4: Modern AI–driven generative molecule design." *Journal of Cheminformatics*, 2024.
2. Wang, Y. et al. "QUEST: A framework for human evaluation of large language models in healthcare." *npj Digital Medicine*, 2024.
3. Cheng, J. et al. "A Minority Sample Enhanced Sampler for Crop Classification in UAV Remote Sensing Images with Class Imbalance." *Agriculture*, 2025.
4. Sivasubramanian, A. et al. "Transformer based ensemble deep learning approach for remote sensing natural scene classification." *International Journal of Remote Sensing*, 2024.
5. DECOR: Decomposition-Based Framework for Long-Tailed Remote Sensing Scene Recognition. *IEEE GRSL*, 2024.
6. TaxoNet: Plant Taxonomy Classification with Dual-Margin Contrastive Learning. *arXiv*, 2025.
7. DUNIA: Dense Unsupervised Cross-Modal Alignment for Earth Observation. *ICML*, 2025.
8. Jimenez, C.E. et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" *ICLR*, 2024.
9. Mialon, G. et al. "GAIA: a benchmark for General AI Assistants." *NeurIPS*, 2023.
10. Yang, J. et al. "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." *arXiv*, 2024.
11. Caicedo, J.C. et al. "Improving the learning of treatment effects from high-throughput imaging with Cell Painting CNN." *Nature Communications*, 2024.
12. Sun, Z. et al. "A review of AI applications in remote sensing." *Remote Sensing*, 2023.
13. Bejide, M. "Multi-modal Inconsistency Diagnosis in Forest Monitoring: An NLP-assisted Systematic Review." 2026.
14. Weinstein, B.G. et al. "Individual tree-crown detection in RGB imagery using semi-supervised deep learning." *Methods in Ecology and Evolution*, 2019.
15. Joshi, R. & Witharana, C. "Assessing annotator variability in tree crown delineation." *Remote Sensing*, 2025.
