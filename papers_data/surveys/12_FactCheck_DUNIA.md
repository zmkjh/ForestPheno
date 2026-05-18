# DUNIA 综述数字与方法核查报告

> 核查对象：`papers_data/综述_design_space_oriented.md`
> 核查依据：DUNIA 原文 `Ref_DUNIA_CrossModal_arXiv2025.pdf`（ICML 2025, arXiv:2502.17066v2, 共32页）

---

## 数字核查

### 1. 零样本树高 RMSE 2.0m, r=0.93 — 原文是否这样写？

**✅ 正确**

- DUNIA 原文 Table 1（第8页）：Wrh, KNN=50 → RMSE 2.0, r=.93
- 原文 Table G1（附录第21页）：Wrh, KNNb=50, 100% S → 2.0 (.93)
- 综述引用无偏差

---

### 2. 微调后树高 RMSE 1.3m, r=0.95 — 原文？

**✅ 正确**

- DUNIA 原文 Table 2（第9页）：Wrh, DUNIA 微调 (5K im) → RMSE 1.3, r=.95
- 综述引用准确

---

### 3. 树种 wF1 76.0%（零样本, KNN=5）— 原文？

**✅ 正确**

- DUNIA 原文 Table 1（第8页）：PF (PureForest), KNN=5 → wF1 76.0
- 原文 Table G1（第21页）：PF KNNb=5 → wF1 76.0
- 原文正文也明确指出："exceeds the baseline by 1.4% when decreasing the number of neighbors (KNN=5)"
- 综述引用准确

---

### 4. 树种 wF1 82.2%（微调）— 原文？

**✅ 正确**

- DUNIA 原文 Table 2（第9页）：PF, DUNIA (S=50K l) → wF1 82.2
- 综述引用准确

---

### 5. Zero-CL positive-pair cosine similarity 0.86 vs VICReg 0.56 — 原文？

**✅ 正确**

- DUNIA 原文 Table H1（附录第21页，消融实验）：
  - VICReg: +P↔W (positive pair pixel-waveform) = **0.56**
  - Zero-CL: +P↔W = **0.86**
- 原文正文 H.1 节明确说明："cosine similarity (CS) using the VICReg loss for the pixel-waveform alignment only reaches a maximum of 0.56 for the positive pairs ... On the other hand, using the ZERO-CL loss, the CS between the positive pairs reaches 0.86"
- 综述引用准确

---

### 6. PASTIS 零样本 OA 56.2% — 原文？

**✅ 正确**

- DUNIA 原文 Table 1（第8页）：PASTIS, KNN=50 → OA 56.2
- 原文 Table G1（第21页）：PASTIS KNNb=50 → OA 56.2
- 综述引用准确

---

### 7. 20% 标签下树高 RMSE 2.1m — 原文？

**❌ 错误**

- 综述 Table 2 "20% Labels Height (RMSE)" 行标注 DUNIA 为 **2.1 m (r=0.92)**
- DUNIA 原文 Table G2（第21页，微调 20% 标签 = 1K im vs fully 5K im）：DUNIA Wrh = **1.4 m (r=0.93)**，不是 2.1
- 2.1 的实际来源：DUNIA 原文 Table G1 的零样本检索 reduced database（10% S / 5% S）结果为 2.1 (.92)，而非微调标签减量结果
- **其他 5 个模型的 20% Labels 行数值全部与 Table G2 一致**（AnySat 2.8, CROMA 3.6, SatMAE 10.5, DOFA 11.2, DeCUR 11.1），唯独 DUNIA 的值张冠李戴
- 这造成对 DUNIA 在低标签场景下性能的错误低估（实际 1.4m 优于综述声称的 2.1m）
- **严重程度**：数据拼凑错误，破坏了 Table 2 列间可比性

---

### 8. 推理 4.22s（20km²） vs AnySat 177s — 原文？

**✅ 正确**

- DUNIA 原文 Table H6（附录第23页）：
  - DUNIA (256K DB, KNN=100): Forward 2.52 + Retrieval 0.36 + Classification 1.34 = **Total 4.22s**
  - AnySat: **Total 177.37s**
- 原文说明区域为 20.48×20.48 km，10m 分辨率 = 4,194,304 像素
- 综述引用准确

---

### 9. AnySat 树高 RMSE 2.8m — 这是 DUNIA 原文报告的吗？还是拼凑的？

**✅ DUNIA 原文报告**

- DUNIA 原文 Table 2（第9页）：AnySat Wrh RMSE 2.8 (r=.89), S=5K im
- DUNIA 原文 Table G2（第21页）：AnySat Wrh RMSE 2.8 (r=.89), S=1K im
- 所有竞争模型的对比数字均来自 DUNIA 原文的统一实验设置（所有模型用相同数据集预训练 250K 步）
- 综述引用准确

---

### 10. CROMA 树高 RMSE 3.5m — DUNIA 原文报告的？

**✅ 是**

- DUNIA 原文 Table 2（第9页）：CROMA Wrh RMSE 3.5 (r=.78), S=5K im
- 综述 Table 2 中 CROMA 的 Fine-tuned Height 3.5 与原文一致
- （注意：Table G2 中 1K im 时 CROMA 为 3.6 — 3.5 对应满量微调结果）

---

### 11. SatMAE 树高 RMSE 10.5m — DUNIA 原文报告的？

**✅ 是**

- DUNIA 原文 Table 2（第9页）：SatMAE Wrh RMSE 10.5 (r=.52)
- 综述引用准确

---

### 12. DOFA, Scale-MAE, DeCUR 的对比数字 — 都来自 DUNIA 原文的统一实验吗？

**⚠️ 部分正确，Scale-MAE 不是**

- **DOFA**：
  - Table 2 (5K im)：Wrh 11.0 (.51), PF wF1 79.8 ✅
  - Table G2 (1K im)：Wrh 11.2 (.50) ✅
  - 来自 DUNIA 原文统一实验

- **DeCUR**：
  - Table 2 (5K im)：Wrh 11.0 (.55), PF wF1 78.9 ✅
  - Table G2 (1K im)：Wrh 11.1 (.52) ✅
  - 来自 DUNIA 原文统一实验

- **Scale-MAE**：
  - DUNIA 原文仅在 Related Work 中提及 Scale-MAE（第2页），**未将其纳入任何实验对比表**
  - 综述 Table 2 中 Scale-MAE 的 Fine-tuned Height 和 20% Labels Height 均标记为 "—"，此处理正确
  - 但综述 Table 2 仍将 Scale-MAE 列在对比表中，可能给读者造成"来自 DUNIA 统一对比"的误导

---

## 方法核查

### 13. "双解码器架构：OV（垂直结构）解码器 + OH（水平结构）解码器" — 原文是否这样描述？

**✅ 正确**

- 原文第3页："two independent convolutional decoders"、"The two decoders are designed to disentangle horizontal and vertical structure understanding: one focuses on spatial (horizontal) relationships, the other on vertical structure"
- 原文第4页明确 OV 和 OH 的输出定义："The two decoders transform the encoder outputs into two sets of pixel-sized embeddings (OV, OH ∈ R^{H×W×Dp}) for pixel-waveform and pixel-pixel alignment respectively"
- **OV 的实际输出**：像素级 embedding，与 GEDI 波形 embedding (OW) 做 cross-modal alignment（Zero-CL loss），用于树高/冠层覆盖/PAI 等垂直结构任务
- **OH 的实际输出**：像素级 embedding，与多时相 AE 输出 (OT) 做 pixel-pixel alignment（VICReg loss），用于土地覆盖/树种/作物分类等水平结构任务
- 综述描述与原文一致

---

### 14. "Zero-CL loss: ZCA 白化后计算对比损失" — 公式对吗？白化是必须的吗？

**✅ 方法描述正确，白化是 Zero-CL 的核心机制**

- 原文附录 B（第17页）给出完整公式：
  - H^Ins = W^Ins Z, 其中 W^Ins = E^Ins Λ^{-1/2}_S E^{⊤}（ZCA 白化矩阵）
  - L_Ins = Σ_i (1 - Σ_d H^{A,Ins}_{i,d} · H^{B,Ins}_{i,d})²
  - L_Fea = Σ_d (1 - Σ_i H^{A,Fea}_{i,d} · H^{B,Fea}_{i,d})²
  - L_Zero-CL = L_Fea + L_Ins
- **ZCA 白化是必须的**：原文第5页说明 Zero-CL "replaces the alignment and uniformity terms in negative-pair-based contrastive losses with an instance-wise contrastive loss (L_Ins) and a feature-wise contrastive loss (L_Fea)"，而 ZCA 白化正是实现这一替代的核心计算步骤。不经过 ZCA 白化，Zero-CL 退化为普通的实例/特征对比损失
- 综述仅在文字中描述"ZCA 白化后计算对比损失"，未重复公式，方法描述准确

---

### 15. "编码器输入：Sentinel-1/2 单期中值合成影像，10m 分辨率" — 原文是否有输入规格说明？

**✅ 正确**

- 原文第4页："I is a median composite image generated from S-1 & 2 observations over several dates"
- 原文第6-7页详细说明：
  - 单期 leaf-on season mosaic（April-September 2020）
  - S-2 bands at 10m and 20m，"the latter upscaled to 10 m"
  - S-1 data "resampled to 10 m resolution"
  - 输入为 14 通道（stacked S-1 & 2）
- 注意细微差别：原文说"several dates"的中值合成，综述写"单期中值合成影像"。原文的"several dates"指同一 leaf-on 季内多个日期的中值合成，本质上仍是单期（mono-temporal），综述表述准确

---

### 16. "256K/512K个GEDI波形" — 原文的训练数据量是多少？

**⚠️ 数字属实但上下文易混淆**

- DUNIA 原文的训练数据量：
  - 预训练图像：**836K** 张 64×64 S-1&2 影像
  - GEDI 波形总数：**≈19 million**（1900万）个波形
  - 平均每张影像 26 个 GEDI 波形
- 综述中"256K/512K"的来源：
  - 原文 Table H6（第23页，推理速度测试）：检索数据库大小 DB=**256K** 或 **512K** 条目
  - 这指的是 KNN 检索时的数据库规模，**并非训练数据量**
- 综述在推理效率上下文中引用 256K/512K 是准确的，但如果脱离上下文单独说"256K/512K 个 GEDI 波形"会误导读者以为这是训练数据量
- 综述正确地在第84行标注了训练数据量："836K Sentinel-1/2 patches + 19M GEDI waveforms, 250K steps"

---

### 17. "64-dim output projection" — 原文是否提到这个维度？

**✅ 是**

- 原文第6页（Section 4 开头）："The embedding dimension is set to 64 (i.e., Dp = 64)"
- 原文附录 E.1（第19页）："the decoded image feature map to 64 embedding dimensions (i.e. Dp = 64)"
- 原文多处提及 Dp=64 这一关键超参数
- 综述第105行："pixel-level cross-modal embeddings (64-dim output projection, 10 m/pixel)" — 准确

---

## 额外核查发现

### A. 综述 Table 2 中 AnySat Fine-tuned Species 标注为 **82.3（加粗最佳）**

- DUNIA 原文 Table 2：AnySat PF wF1 = 82.3, DUNIA PF wF1 = 82.2
- 差值仅 0.1 pp，AnySat 确实比 DUNIA 高 0.1
- 综述将 82.3 加粗标注为最佳，**技术上正确**
- 但综述选型理由（Section 3.1.2）强调 DUNIA 为推荐的 base encoder，这是合理的——0.1 pp 的差异在统计上不显著，而 DUNIA 在高度估计上有压倒性优势

### B. 综述声称"FORMS 树高 RMSE 5.2m"

- 综述第21行和第99行两次提到"supervised SOTA FORMS at 5.2 m (r = 0.77)"
- DUNIA 原文 Table 1 SOTA 列：Schwartz et al. (2023) 5.2 (.77) — 这是 FORMS 论文的结果
- **✅ 正确引用**

### C. 综述"预训练 250K 步、单卡 A6000 48GB"

- 原文第19页："pre-trained on a single NVIDIA A6000 48GB GPU with a batch size of 60 for 250K steps"
- **✅ 完全一致**

### D. 综述声称零样本冠层覆盖 RMSE 11.7% 和 PAI RMSE 0.71

- DUNIA 原文 Table 1：Wc RMSE 11.7 (r=.89), Wpai RMSE 0.71 (r=.75), KNN=50
- **✅ 准确**

### E. 综述声称 DUNIA 微调后 PASTIS wF1 77.0

- DUNIA 原文 Table 2：PASTIS DUNIA wF1 77.0 (S=1.5K im)
- 综述第201行同样引用："DUNIA—using a single median composite—achieves only 77.0 under fine-tuning"
- **✅ 准确**

---

## 总结

| 编号 | 核查项 | 结果 | 问题说明 |
|------|--------|------|----------|
| 1 | 零样本树高 2.0m, r=0.93 | ✅ | — |
| 2 | 微调树高 1.3m, r=0.95 | ✅ | — |
| 3 | 树种 wF1 76.0% (KNN=5) | ✅ | — |
| 4 | 树种 wF1 82.2% (微调) | ✅ | — |
| 5 | Zero-CL cos 0.86 vs VICReg 0.56 | ✅ | — |
| 6 | PASTIS 零样本 OA 56.2% | ✅ | — |
| **7** | **20% 标签树高 RMSE 2.1m** | **❌** | **应为 1.4m (r=0.93)，2.1 实为降量零样本检索结果** |
| 8 | 推理 4.22s vs AnySat 177s | ✅ | — |
| 9 | AnySat 树高 2.8m | ✅ | 来自 DUNIA 原文 Table 2 |
| 10 | CROMA 树高 3.5m | ✅ | 来自 DUNIA 原文 Table 2 |
| 11 | SatMAE 树高 10.5m | ✅ | 来自 DUNIA 原文 Table 2 |
| 12 | DOFA/Scale-MAE/DeCUR 对比 | ⚠️ | DOFA/DeCUR ✅；Scale-MAE 未在 DUNIA 实验中 |
| 13 | 双解码器 OV+OH | ✅ | — |
| 14 | Zero-CL + ZCA 白化 | ✅ | — |
| 15 | S-1/2 单期中值合成 10m | ✅ | — |
| 16 | 256K/512K 波形 | ⚠️ | 搜库大小非训练量，综述上下文使用正确 |
| 17 | 64-dim projection | ✅ | — |

**共发现 1 项实质性错误（Item 7）和 2 项需注意的表述问题（Item 12、16）。**
