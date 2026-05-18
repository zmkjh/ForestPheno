# 15 — Cross-Validation: Design-Space (D) vs. Traditional Survey (T)

Cross-validated on 2026-05-18. D = `综述_design_space_oriented.md`, T = `综述_traditional_survey.md`.

---

## 一致项

| 数字 | D | T | 状态 |
|------|---|---|------|
| DUNIA 零样本树高 RMSE | 2.0 m (r=0.93) | 2.0 m (r=0.93) | ✓ |
| DUNIA 微调树高 RMSE | 1.3 m (r=0.95) | 1.3 m (r=0.95) | ✓ |
| DUNIA 零样本树冠覆盖 RMSE | 11.7% (r=0.89) | 11.7% (r=0.89) | ✓ |
| DUNIA 零样本 PAI RMSE | 0.71 (r=0.75) | 0.71 (r=0.75) | ✓ |
| DUNIA 零样本树种 wF1 | 76.0% (KNN=5) | 76.0% (KNN=5) | ✓ |
| DUNIA 微调树种 wF1 (PureForest) | 82.2% | 82.2% | ✓ |
| DUNIA 零样本 PASTIS OA | 56.2% | 56.2% | ✓ |
| DUNIA 微调 PASTIS wF1 | 77.0% | 77.0% | ✓ |
| DUNIA 零样本 land cover wF1 | — | 80.1% | △ (仅T) |
| DUNIA 微调 land cover wF1 | 90.3% (Table 2 row) | 90.3% | ✓ |
| DUNIA 微调冠层覆盖 RMSE | — | 9.8% (r=0.85) | △ (仅T) |
| 监督 SOTA 树高 RMSE | 5.2 m, FORMS | 5.2 m (r=0.77) | ✓ |
| AnySat 微调树高 RMSE | 2.8 m (r=0.89) | 2.8 m (r=0.89) | ✓ |
| AnySat 微调树种 wF1 | 82.3% | 82.3% | ✓ |
| AnySat 微调 PASTIS wF1 | 81.1% | 81.1% | ✓ |
| AnySat 推理 (20 km²) | 177 s | ~177 s | ✓ |
| DUNIA 推理 (20 km²) | 4.22 s | 4.22 s | ✓ |
| CROMA 微调树高 RMSE | 3.5 m (r=0.78) | 3.5 m (r=0.78) | ✓ |
| CROMA 微调 land cover wF1 | — | 86.4% | △ (仅T) |
| CROMA 微调 PASTIS wF1 | — | 73.3% | △ (仅T) |
| SatMAE 微调树高 RMSE | 10.5 m (r=0.52) | 10.5 m (r=0.52) | ✓ |
| SatMAE 微调树种 wF1 | 78.8% | 78.8% | ✓ |
| SatMAE 微调 PASTIS wF1 | — | 55.2% | △ (仅T) |
| DOFA 微调树高 RMSE | 11.0 m (r=0.51) | 11.0 m (r=0.51) | ✓ |
| DOFA 微调树种 wF1 | 79.8% | 79.8% | ✓ |
| DOFA 微调 PASTIS wF1 | — | 54.5% | △ (仅T) |
| DeCUR 微调树高 RMSE | 11.0 m (r=0.55) | 11.0 m (r=0.55) | ✓ |
| DeCUR 微调树种 wF1 | 78.9% | 78.9% | ✓ |
| MSFMamba Houston2013 OA | 92.86% | 92.86% (AA 93.77%, Kappa 92.25%) | ✓ |
| MSFMamba Houston2018 OA | 92.38% | 92.38% (AA 95.51%) | ✓ |
| MSFMamba Berlin OA | 76.92% | 76.92% (AA 64.88%) | ✓ |
| MSFMamba Augsburg OA | 91.38% | 91.38% (AA 63.31%) | ✓ |
| MSFMamba 参数 | 1.53M | 1.53M | ✓ |
| MSFMamba 推理 | 0.175 s | 0.175 s (RTX 4090) | ✓ |
| DCMNet Houston2013 OA | 95.11% (Kappa 94.69%) | 95.11% (AA 95.74%, Kappa 94.69%) | ✓ |
| DCMNet Trento OA | 98.96% | 98.96% (AA 97.55%, Kappa 98.61%) | ✓ |
| DCMNet Houston2018 OA | — | 93.27% (AA 96.33%, Kappa 91.33%) | △ (仅T) |
| DCMNet 参数 | 3.83M | 3.83M | ✓ |
| DCMNet 推理 | 0.010 s | 0.0097 s | ✓ (四舍五入) |
| DFFNet Houston2013 OA | 92.35% (Kappa 91.70%) | 92.35% (AA 93.48%, Kappa 91.70%) | ✓ |
| DFFNet 参数 | 1.28M | 1.28M | ✓ |
| DFFNet 推理 | 0.239 s | 0.239 s | ✓ |
| IFGNet Houston2013 OA | 99.37% (Kappa 99.32%) | 99.37% (AA 99.50%, Kappa 99.32%) | ✓ |
| FusDreamer Houston2013 OA (few-shot) | 89.24% (Kappa 90.15%) | 89.24% (AA 88.35%, Kappa 90.15%) | ✓ |
| FusDreamer Trento OA (few-shot) | 96.36% | 96.36% (AA 95.12%, Kappa 93.81%) | ✓ |
| PhenoAssistant 工具选择精度 | 100% (50/50 tasks) | 100% (50 tasks) | ✓ |
| PhenoAssistant 工具匹配精度 | 100% (20/20 tasks) | 100% (20 tasks) | ✓ |
| PhenoAssistant 数据分析精度 | 85% (17/20) | 85% (17/20) | ✓ |
| PhenoAssistant 链合理性 (无Critic) | 4.25/5 | 4.25/5 | ✓ |
| PhenoAssistant 工具存在性 | 5.00/5 | 5.00/5 | ✓ |
| PhenoAssistant 工具适当性 (无Critic) | 4.65/5 | 4.65/5 | ✓ |
| PhenoAssistant 参数正确性 (无Critic) | 4.30/5 | 4.30/5 | ✓ |
| PhenoAssistant (有Critic) 链合理性 | 4.35/5 | 4.35/5 | ✓ |
| PhenoAssistant (有Critic) 工具适当性 | 4.90/5 | 4.90/5 | ✓ |
| PhenoAssistant (有Critic) 参数正确性 | 4.40/5 | 4.40/5 | ✓ |
| PureForest 面积 | 339 km² | 339 km² | ✓ |
| PureForest 物种数 | 18 (→13 语义类) | 18 (→13 语义类) | ✓ |
| PureForest VHR ResNet OA | 73.1% | 73.1% | ✓ |
| PureForest LiDAR+高程 OA | 83.6% | 83.6% | ✓ |
| PureForest LiDAR 纯几何 OA | — | 80.3% | △ (仅T) |
| PlantD 样本数 | 2.26M | 2,264,747 | ✓ |
| PlantD 物种数 | 64 | 64 | ✓ |
| PlantD Sentinel-2 macro-F1 | ~62% | ~62% | ✓ |
| DUNIA Zero-CL cosine similarity | 0.86 | 0.86 | ✓ |
| VICReg cosine similarity | 0.56 | 0.56 | ✓ |
| GEDI 波形数 | 19M | 19 million | ✓ |
| 预训练 patches | 836K | 836,000 | ✓ |
| 训练步数 | 250K | 250,000 | ✓ |
| GPU | A6000 48GB | A6000 48GB | ✓ |
| 零样本检索数据库 | 50K labeled pixels, ~31 km², ~0.25% | 50,000 labeled pixels, ~31 km², ~0.25% | ✓ |
| SAGE 总体提升 | 16.2 pp | 16.2 pp | ✓ |
| SAGE agentic pipeline vs few-shot | 8.1 pp | 8.1 pp | ✓ |
| Bejide 时间异步占比 | 7.8% | 7.8% | ✓ |
| Bejide 生态异质性 | — | 30.2% | △ (仅T) |
| Bejide 尺度不匹配 | — | 19.8% | △ (仅T) |
| Bejide 传感器饱和 | — | 18.1% | △ (仅T) |

---

## 不一致项

| 数字 | D 篇 | T 篇 | 差异 | 可能原因 |
|------|------|------|------|----------|
| **DUNIA 20%标签树高 RMSE** | **1.4 m** (r=0.93) | **2.1 m** (r=0.92) | 0.7 m (方向相反) | T 篇将零样本基线(2.0m)误当作 20%标签起点，声称 "从2.0m退化到2.1m"，这与"微调应提升"的逻辑矛盾。D 篇的 1.4m 符合标签效率假设(全量 1.3m → 20% 1.4m，仅退化 0.1m)。T 篇极可能转录错误。 |
| **SAGE 大豆精度 (baseline)** | 31.1% → **48.6%** (+17.5 pp) | 31.1% → **62.2%** (+31.1 pp) | +13.6 pp | 明显转录错误。D 篇四作物的平均提升 = (17.5+18.2+23.8+5.0)/4 ≈ 16.125 pp，与总体 16.2 pp 吻合。T 篇大豆 +31.1 pp 远超平均。 |
| **SAGE 大豆精度 (k=8)** | **48.6%** | **62.2%** | 13.6 pp | 同上，T 篇错误。 |
| **SAGE 玉米精度 (k=8)** | **60.2%** | **61.4%** | 1.2 pp | 轻微差异，D 更精确。 |
| **SAGE 番茄 baseline** | **52.3%** | **57.5%** | 5.2 pp | 两者 baseline 不一致。D 篇 (76.1−52.3=23.8 pp) 与 T 篇 (72.6−57.5=15.1 pp) 差距明显。 |
| **SAGE 番茄精度 (k=8)** | **76.1%** | **72.6%** | 3.5 pp | 同上，T 篇偏低。 |
| **SAGE 芒果 baseline** | **92.5%** | **82.5%** | 10.0 pp | 大幅差异。D 篇的 92.5% baseline 与芒果仅 4 类、难度最低一致；T 篇 82.5% 偏低。 |
| **PASTIS gap vs AnySat** | **24.9 pp** (81.1−56.2) | **28 pp** ("28 points below AnySat") | 3.1 pp | T 篇将 SOTA 差距(84.2−56.2=28 pp)错误地套用到 AnySat。正确的数学是 81.1−56.2=24.9 pp。 |
| **AnySat 推理耗时** | "177 s (~40× slower than DUNIA)" | "roughly 40× slower" | 0 | 一致；D 篇详述 4.22s vs 177s。 |

---

## 仅在 D 篇出现的陈述（T 篇未覆盖）

| 类别 | 内容 | D 篇位置 |
|------|------|----------|
| 设计空间框架 | 五维正交设计空间(Encoder, Fusion, Agent, Temporal, Quality) | §3 |
| 跨维度依赖分析 | D1–D6 六条依赖关系及其影响 | §4.1 |
| 集成路线图 | Phase 1/2/3 三阶段集成方案 | §4.3 |
| 组件可行性评估 | 6 个组件的开源/框架/集成风险/缓解策略(Table 6) | §4.4 |
| QUEST-Forest 评估框架 | 含 Cost-Weighted Error Rate, Tail-Recall@K, Open-Set TNR@95, Head-Tail Gap, Pheno-Stage Accuracy, Modality-Inconsistency Score | §5.2 |
| 分层任务结构 | T1(原子)/T2(组合)/T3(智能体决策) | §5.3 |
| 基线方法表 | ResNet-50+CE, TaxoNet, DUNIA zero-shot, CLIP/DINOv2, GPT-4V, PhenoAssistant v1 | §5.4 |
| 验证实验 V-1~V-5 | 含假设、实验设计、成功标准的可证伪实验 | §6.2 |
| G-1~G-5 差距表 | 时间、质量映射、Agent-Fusion接口、长尾跨模态、评估协议 | §6.1 |
| 质量→融合映射接口 | GAP-1/2/3：质量阈值、多维融合、质量感知训练数据 | §3.5.3 |
| DCMNet Houston2018 OA | 提及但未给数值 | §3.2.1 |
| TaxoNet 长尾增益(+5.1 pp) | AA-Central macro-recall vs LDAM | §3.4 (提及) |
| TaxoNet tail recall 57.1%, head 92.4% | Auto-Arborist 数据集 | §6.2 V-4 |
| LEMON 基准 | MMLU, GSM8K, AQuA, MultiArith, SVAMP, HumanEval | §3.3.1 |

---

## 仅在 T 篇出现的陈述（D 篇未覆盖）

| 类别 | 内容 | T 篇位置 |
|------|------|----------|
| ITCD 传统方法 | CHM 分水岭、局部极大值、区域生长；F1 值 | §6.1 |
| ITCD 深度学习方法 | Mask R-CNN (F1≈0.75), YOLOv7 (mAP≈75.8%), SegFormer (F1=0.85) | §6.1 |
| ITCD 死/活树 F1 | 死树 71.16%, 活树 74.75% | §6.1 |
| 生物量估算 | 异速方程、半监督 434K 噪声标签 | §6.2 |
| Bejide 完整七因子 | 生态异质性 30.2%/尺度不匹配 19.8%/传感器饱和 18.1%/基线不稳定 13.8%/时间异步 7.8%/结构-功能解耦 5.2%/数据覆盖 3.4% | §6.3 |
| Bejide 文献统计 | 94% 报告不确定性, 4.3% 做光谱-结构对比, 49.1% 多传感器融合, 50.9% 集成 ML, 11.2% 不确定性传播 | §6.3 |
| TreeSatAI 数据集 | 15 树种、德国、50K 影像、60 m | §3.4 |
| PureForest RandLA-Net 纯LiDAR OA 80.3% (mIoU 55.1%) | 纯几何特征基线 | §3.1 |
| SAGE 规模 | 335 作物, 1,251 病害, 839K 影像 | §7.2 |
| SAGE 推理流程 | 5 阶段(器官识别→解剖过滤→KB症状匹配→顺序参考影像→预测) | §7.2 |
| 额外 Agent 架构 | APWA, Swarm Skills, Dynamic Tiered AgentRunner, GraphFlow (97.08% 完成率) | §7.3 |
| 时间编码范式 | RNN/LSTM, TCN, InceptionTime, ChangeMamba | §8.1 |
| DUNIA 预训练硬件 | 单卡 A6000 48GB, 250K steps, batch=60, Lion optimizer, lr=5e-5 | §4.1 |
| DUNIA 架构细节 | 16-layer ViT, 4 blocks×4, 8 heads, d=512, GeGLU, patch size 8, 64-dim output | §4.1 |
| DUNIA 波形扩散模型 r | 0.75–0.78 | §9.9 |
| MSFMamba FLOPs | 0.038 GFLOPs | §5.1 |
| DCMNet FLOPs | 0.046 GFLOPs | §5.2 |
| DFFNet FLOPs | 0.0303 GFLOPs | §5.3 |
| FusDreamer 推理 | Trento 16.4s, Houston2013 30.1s | §5.5 |
| "No normative recommendations" | 纯描述性声明 | §10 |
| SAGE 大豆 25 类, 玉米 30 类, 番茄 20 类, 芒果 4 类 | 各类别数量 | §7.2 |

---

## 综合评估

### D 篇的潜在风险项

| 风险 | 严重度 | 说明 |
|------|--------|------|
| DUNIA 20%标签树高 1.4 m 与 T 篇 2.1 m 冲突 | **高** | 需回溯 DUNIA 原文 Table/Figure 确认。D 篇的 1.4 m 逻辑自洽(1.3→1.4→2.0)，但 T 篇的 2.1 m 可能是另一实验设定的结果(如非微调而是 zero-shot+20%检索库) |
| SAGE 番茄 baseline 52.3% vs T 57.5% | **中** | 需对照 SAGE 原论文。D 篇数值使平均提升恰好 16.2 pp，可能是 D 篇为凑平均数对原文进行了取舍 |
| D篇大量引用 DUNIA "Table 2" | **低** | D 篇 Table 2 声称"来自 DUNIA 论文统一实验设定"，但对比表含 DUNIA 未发表的数值(20%标签列、20 km² 推理)。需核实是否来自原作者个人通信 |

### T 篇的潜在风险项

| 风险 | 严重度 | 说明 |
|------|--------|------|
| DUNIA 20%标签 2.1 m 大概率转录错误 | **高** | "从 2.0m 退化至 2.1m"与标签效益逻辑矛盾；D 篇、DUNIA 论文的结论均为"标签高效" |
| SAGE 大豆 k=8 → 62.2% 与整体平均 16.2 pp 矛盾 | **高** | 大豆 (+31.1 pp) 若属实，剩余三作物平均需 ≈ +10 pp 才能拉回 16.2，但 T 篇并无相应数据 |
| SAGE 芒果 baseline 82.5% 偏低 | **中** | 4 类任务的简单基线应接近 90%；D 篇的 92.5% 更合理 |
| PASTIS gap "28 pp vs AnySat" 是算术错误 | **中** | 81.1 − 56.2 = 24.9，不是 28。T 篇可能将 28 pp (vs SOTA) 和 24.9 pp (vs AnySat) 混用 |

### 建议修正

1. **T 篇必须修正**：DUNIA 20%标签树高 RMSE（当前 2.1m → 应改为 1.4m 或注明另一实验设定）
2. **T 篇必须修正**：SAGE 大豆(62.2% → 应为 48.6%)、番茄 baseline(57.5% → 52.3%, 最终 72.6% → 76.1%)、芒果 baseline(82.5% → 92.5%)
3. **T 篇应修正**：PASTIS gap vs AnySat(28 pp → 24.9 pp)
4. **D 篇应核实**：DUNIA Table 2 中的 "20% Labels Height 1.4 m" 是否在原始 DUNIA 论文中明确出现；如仅为衍生计算，应标注"estimated from DUNIA data"
5. **双方应统一**：DUNIA 微调性能表格(dimension vs method)的列，D 篇缺失 canopy cover、land cover、PASTIS 等列；T 篇有但仅 6 行。建议双方对齐到同一全维度表
