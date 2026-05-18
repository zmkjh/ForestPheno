# FactCheck: 综述_design_space_oriented.md — PhenoAssistant / SAGE / 数据集 / TaxoNet 数字核查

查阅日期: 2026-05-18
核对方式: 逐条与原文 PDF 文本提取比对

---

## PhenoAssistant [4] — Key_PhenoAssistant_NatureComms2026.pdf

### 1. 工具链合理性 4.25/5 (4.35/5 with Critic) — 50 个任务

**⚠️ 表值正确，但任务数错误**

综述原文 (L173): "Evaluation on 50 manually designed tasks yielded: tool chain rationality 4.25/5..."

原文证据:
- 原文 Table 1 (L1070–1094): Manager Overall chain = 4.25, Manager + Critic Overall chain = 4.35。**分值正确**。
- 但原文明确写 "To conduct this evaluation, we create 20 tasks" (L369), "The average scores across the 20 tasks are reported in Table 1" (L383)。**任务数是 20 而非 50**。
- 50 tasks 是 Vision Model Selection I (L421: "we create 50 plant tasks")。
- 综述将两个不同评估（工具链 20 任务 vs 模型选择 50 任务）的任务数混用。

**⚠️ 综述将表值正确写入，但评估任务数张冠李戴了，不是 50 个任务。**

---

### 2. 100% (50/50) 模型类型选择，100% (20/20) 模型精确匹配

**✅ 正确**

原文证据:
- Vision Model Selection I: "PhenoAssistant achieves 100% accuracy" (L426)，50 task 级别 (L421) → 50/50 ✓
- Vision Model Selection II: "It achieves 100% accuracy... for the 15 tasks with corresponding model identifiers, it provides the correct match; for the remaining 5 tasks without a direct match, it also correctly identifies the unavailability" (L449–452) → 20/20 ✓

---

### 3. 数据分析准确率 85% (17/20)，失败全部是 Plot Analyser

**✅ 正确**

原文证据 (L460–467):
"PhenoAssistant achieves 85% (17/20) accuracy in this evaluation. All failed cases fall within the plot analysis category... This suggests that fine-grained visual reasoning remains a bottleneck for LLMs."

---

### 4. Manager 用 GPT-4o, temperature=0.1

**✅ 正确**

原文证据 (L637–639):
"The Manager is implemented using GPT-4o (version: 2024-08-06, via OpenAI Azure API)... with the model temperature set to 0.1"

---

### 5. 基于 AutoGen 框架

**✅ 正确**

原文证据 (L660):
"PhenoAssistant is implemented in Python, with tool invocation and multi-agent orchestration supported by AutoGen."

---

### 6. 视觉模型动物园包含 Mask2Former, Leaf-only SAM, DINOv2-base

**✅ 正确**

原文证据:
- Mask2Former: L678 "we integrate Mask2Former for segmenting individual Arabidopsis leaves"
- Leaf-only SAM: L680 "we integrate Leaf-only SAM for potato leaf instance segmentation"
- DINOv2-base: L708 "we employ the DINOv2-base model as the pre-trained model"

---

## SAGE [11] — Key_SAGE_CropDiseaseAgent_arXiv2026.pdf

### 7. KB 引入后精度平均 +16.2pp (k=8)

**⚠️ 数值存在但含义被误导性表述**

原文证据 (Table 2, L800–810):
均值 +16.2 pp 确实出现在原文 Table 2 的 Mean ∆(pp) 行 (Agent + internet KB, k=8 列)。

**但**这 +16.2 是相对于 k=0 no-KB baseline 的提升，不是 "引入 KB 带来的纯增益"。在同为 k=8 条件下，KB vs no-KB 的实际增益为:
- Soybean: 48.6% - 45.9% = +2.7 pp
- Corn: 60.2% - 52.3% = +7.9 pp
- Tomato: 76.1% - 67.0% = +9.1 pp
- Mango: 97.5% - 92.5% = +5.0 pp
- 均值仅 +6.2 pp (不是 16.2)

原文明确将此 +16.2 表述为 "The full pipeline (KB, k=8) improves on the no-KB k=0 baseline by 16.2 percentage points" (L741)。综述简化为 "KB引入后精度平均+16.2pp" 具有误导性——实际 KB 在控制参考预算条件下的增益仅约 +6.2pp。

---

### 8. 大豆 31.1%→62.2%, 玉米 42.0%→61.4%, 番茄 57.5%→72.6%, 芒果 82.5%→97.5%

**❌ 数字严重不匹配原文 Table 2**

原文 Table 2 (Sonnet, internet KB, k=8):
| Crop | k=0 (baseline) | k=8 + KB |
|------|----------------|----------|
| Soybean | 31.1% | **48.6%** (非 62.2%) |
| Corn | 42.0% | **60.2%** (非 61.4%) |
| Tomato | **52.3%** (非 57.5%) | **76.1%** (非 72.6%) |
| Mango | **92.5%** (非 82.5%) | 97.5% ✓ |

- Soybean 差 13.6pp、Tomato 基线差 5.2pp、Mango 基线差 10pp。
- 这些数值在 Table 2 中不存在于任何模型/KB 组合下，疑似来源不明。

**❌ 综述中 Per-crop 数字严重偏离原文，需要撤回重写。**

---

### 9. Agentic vs Few-shot +8.1pp

**⚠️ 无法完全验证（需查附录）**

原文正文仅提及 "Against a single-pass few-shot baseline (Appendix Table 6), the agent wins at matched k" (L919–920)。具体 +8.1pp 数值须查阅 Appendix Table 6，该附录未在本次提取文本中获得。注：综述声称此值，但无法在原文字提取中独立验证。

---

### 10. 1251 个病害类别, 83.9 万图像

**✅ 正确**

原文证据 (Table 1, L277–278): "SAGE (ours): 335 crops, 1,251 diseases, 838,936 images"
+ 正文 (L25–26): "335 crops, 1,251 disease classes, and approximately 839K images"
综述 "83.9万" ≈ "839K" ✓

---

## PureForest [3] — Ref_PureForest_LiDAR_arXiv2024.pdf

### 11. 339km², 18 树种, 40pts/m²

**✅ 正确**

原文证据:
- "spans 339 km²" (L20) ✓
- "18 tree species grouped into 13 semantic classes" (L19) ✓ (综述写 18 树种正确)
- "40 pts/m²" (L34) / "10 pulses/m², or about 40 pts/m²" (L172) ✓

---

### 12. RandLA-Net LiDAR + elevation OA 83.6%

**✅ 正确**

原文证据 (Table III, L853–858):
"Lidar + Elevation, OA=83.6" ✓

---

### 13. VHR 航片 ResNet18 OA 73.1%

**✅ 正确**

原文证据:
- Table III (L862): "Aerial Imagery, OA=73.1" ✓
- 模型规格确认 (L808): "a ResNet18 encoder" ✓

---

## PlantD [14] — Ref_PlantD_ForestDataset_arXiv2024.pdf

### 14. 5 颗卫星, 64 树种, 2.26M 样本, 41 国

**✅ 正确**

原文证据:
- 5 satellites: Sentinel-1, Sentinel-2, Landsat-7, ALOS-2, MODIS (Table 2, L262–311) ✓
- 64 classes: "64 tree label classes (46 genera and 40 species)" (L25) ✓
- 2,264,747 总样本 (L340, L383) ✓ → "2.26M" 准确
- 41 countries (L339) ✓

---

### 15. Video ViT 最优

**⚠️ 表述不够精确**

原文使用 Video Vision Transformer (ViViT-style 3D patch embedding) 作为 baseline 模型 (L441–443): "extended to multi-temporal 3D data inputs as in video Arnab et al. (2021)"。但 Table 4 显示不同卫星组合效果不同 (S1-S2 macro F1 62.2 最佳)，且没有进行非-ViT 架构的对比实验。原文存在的主要对比是单模态 vs 多模态融合、early/mid/late fusion 策略。
架构本身确为 Video ViT，但原文并未声称非 ViT 架构建模 optimality，"最优"属于综述的主观判断。

---

## TaxoNet [10] — Ref_TaxoNet_PlantTaxonomy_arXiv2025.pdf

### 16. +6% macro-recall on Auto-Arborist

**⚠️ 略过度声称，实际 +5.05pp**

原文 Table 1 (L515–618):
- LDAM AA-Central Recall: 67.85
- TaxoNet (best: w/ B, E, F) AA-Central Recall: 72.90
- 实际增益: 72.90 - 67.85 = **5.05 pp**

跨 Auto-Arborist 全部三区域平均: (5.05 + 4.85 + 2.75) / 3 ≈ 4.22 pp。原文有声称 "outperforming LDAM (the strongest SOTA baseline) by 6% on Google AA" (L498–499)，但实际数值不支持。综述沿用此说法，属于放大原始结论。

---

### 17. 开集 TNR 91.3%

**✅ 正确**

原文 Table 3 (L711): TaxoNet AA-Central TNR = 91.28 ≈ 91.3% ✓

---

### 18. 尾部 recall 57.1%, 头类 92.4%

**✅ 正确**

原文 Table 4 Genus-level (L720–748):
- Head Recall: 92.37 ≈ 92.4% ✓
- Tail Recall: 57.06 ≈ 57.1% ✓

---

## 总结

| # | 内容 | 结果 | 说明 |
|---|------|------|------|
| 1 | PhenoAssistant 工具链 4.25/5 (Critic 4.35) | ⚠️ | 分数正确，但评估任务数是 20 而非 50 |
| 2 | VMS-I 100% (50/50), VMS-II 100% (20/20) | ✅ | 完全正确 |
| 3 | 数据分析 85% (17/20) | ✅ | 完全正确 |
| 4 | GPT-4o temperature=0.1 | ✅ | 完全正确 |
| 5 | AutoGen 框架 | ✅ | 完全正确 |
| 6 | Mask2Former, Leaf-only SAM, DINOv2-base | ✅ | 完全正确 |
| **7** | **KB +16.2pp (k=8)** | **⚠️** | **数值存在于原文但被误导性表述：指全管线 vs k=0 baseline，纯 KB 增益仅 +6.2pp** |
| **8** | **大豆/玉米/番茄/芒果 per-crop 数字** | **❌** | **严重错误，与 Table 2 不匹配，疑似来源不明** |
| 9 | Agentic vs Few-shot +8.1pp | ⚠️ | 正文引用附录但未获附录文本，无法独立验证 |
| 10 | 1251 病害类别, 839K 图像 | ✅ | 完全正确 |
| 11 | PureForest 339km²/18 种/40pts/m² | ✅ | 完全正确 |
| 12 | LiDAR+elevation OA 83.6% | ✅ | 完全正确 |
| 13 | VHR ResNet18 OA 73.1% | ✅ | 完全正确 |
| 14 | PlantD 5 卫星/64 种/2.26M/41 国 | ✅ | 完全正确 |
| 15 | Video ViT "最优" | ⚠️ | 架构确为 ViT 但原文未做非-ViT 对比，"最优"为主观 |
| 16 | TaxoNet +6% macro-recall | ⚠️ | 实际约 +5.05pp (AA-Central)，原文也有过度声称 |
| 17 | 开集 TNR 91.3% | ✅ | 完全正确 |
| 18 | Tail 57.1% / Head 92.4% | ✅ | 完全正确 |

**关键发现**:
- **SAGE per-crop 数字 (Claim 8) 是严重错误**，需要立即修正。
- SAGE KB +16.2pp 的表述需要重新定性，明确指 combined effect 而非纯 KB 增益。
- PhenoAssistant 工具链评估的 "50 tasks" 应为 "20 tasks"。
- TaxoNet +6% 需要修正为实际数值 +5.05pp 或标注为原文自身 claims（非已证实事实）。
