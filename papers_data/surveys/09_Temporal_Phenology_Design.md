# ForestPheno 时序物候建模设计方案

## 1. 现有物候建模方法总览

### 1.1 核心问题

DUNIA 编码器使用**单期中值合成影像**作为输入，导致两个根本性缺陷：

- **无法感知物候动态**：同一树冠在萌芽期、展叶期、盛叶期、落叶期的光谱特征完全不同，但 DUNIA 将它们编码为独立实例，丢失了"同一实体在不同时间"的 identity 联系。
- **作物分类性能差**：在 PASTIS 数据集上零样本 OA 仅 56.2%，远低于 AnySat 等含时序输入的方法——因为作物区分高度依赖时序物候轨迹。

**Bejide (2026)** 的 NLP 综述进一步揭示了时序建模缺失的深层后果：扰动后光谱-结构-功能三维度部分解耦（"green desert"现象），即光谱恢复的冠层绿度掩盖了持续的结构退化和功能丧失。94% 的研究报告了生物量/碳估测不确定性，而结构-功能解耦在扰动后随时间演变——单时相模型完全无法诊断这类时序依赖的生态过程。

### 1.2 检索到的相关论文

#### A. 物候+时序深度学习方法

| # | 论文 | 年份 | 时序编码方式 | 卫星数据 | 物候阶段数 | 精度 |
|---|------|------|------------|---------|-----------|------|
| 1 | **TSP-Former**: A Phenology-Guided Transformer for Tobacco Mapping Using SITS | 2025 | 物候引导的 Temporal Transformer + 交叉注意力 | Sentinel-2 时序 | 烟草物候关键期 3-5 个 | — |
| 2 | **Efficient Spatio-Temporal Vegetation Pixel Classification with ViT** | 2026 | ViT + 时序位置编码（3D patch embedding） | UAV RGB 时序 | 植被物候全周期（逐像素分类） | — |
| 3 | **Adaptive Multi-Scale Feature Refinement for Wheat Phenology Recognition** | 2026 | 跨尺度注意力机制 + CNN backbone | UAV/近地面 RGB | 小麦多物候阶段 | — |
| 4 | **Spatio-Temporal Transformers for Long-Term NDVI Forecasting** | 2026 | Spatio-Temporal Transformer（时间+空间联合注意力） | MODIS NDVI 时序 | N/A（预测任务） | — |
| 5 | **Knowledge-Guided Multi-Source Time-Series for Crop Type Classification** | 2026 | LSTM + 先验知识引导的时序嵌入 | Sentinel-1/2 时序 | 多作物类型分类 | — |
| 6 | **TerraFlow**: Multimodal, Multitemporal Representation Learning for EO | 2026 | 时序训练目标（sequence-aware learning），可变长度输入鲁棒 | 多卫星多时相 | N/A（通用预训练） | — |
| 7 | **Comparative Analysis of Dual-Form Networks for Live Land Monitoring** | 2026 | 双形态网络对比（CNN vs Transformer）处理多模态 SITS | Sentinel-1/2 SITS | 土地覆盖变化检测 | — |

#### B. 时序+多模态融合

| # | 论文 | 年份 | 方法 | 模态组合 |
|---|------|------|------|---------|
| 8 | **ChangeMamba**: Spatiotemporal State Space Model for CD | 2024 | Mamba (SSM) + 3D 时空卷积 | 双时相 SAR/Optical |
| 9 | **CROMA**: Contrastive Radar-Optical Masked Autoencoders | 2023 | 跨模态 MAE + 对比学习 | Sentinel-1 + Sentinel-2（单时相） |
| 10 | **OmniSat**: Self-supervised Modality Fusion for EO | 2024 | 多模态自监督融合 | Sentinel-1/2 + 地形 |
| 11 | **TerraFlow** (同上) | 2026 | 多模态多时相联合预训练 | Sentinel-1/2 + 时序 |

#### C. 物候标注数据集

| # | 数据集 | 年份 | 尺度 | 覆盖 | 物候阶段 |
|---|--------|------|------|------|---------|
| 12 | **DeepPhenoTree - Apple Edition** | 2026 | 单株（RGB 图像） | 瑞士、比利时、西班牙、意大利 4 站点 | 3 个（开花/幼果/果实） |
| 13 | **PASTIS** | 2021 | 地块级 | 法国 | 作物类型（非物候阶段） |
| 14 | **NortheastChinaMaizeYield10m** | 2026 | 10m 像素 | 中国东北 | N/A（产量估计） |

#### D. 精读论文

| # | 论文 | 年份 | 核心贡献 | 对物候建模的启示 |
|---|------|------|---------|----------------|
| 15 | **Bejide - Multidimensional Inconsistency in Forest Ecosystem Representation** | 2026 | 7 类不一致性驱动因素（生态异质性 30.2%、尺度错配 19.8%、传感器饱和 18.1%、基准不稳定 13.8%、**时间恢复异步性 7.8%**、结构-功能解耦 5.2%、数据覆盖 3.4%）；"green desert" 概念 | **时间恢复异步性**是核心不一致性来源之一，物候建模可显式解决这一问题 |
| 16 | **Gauli et al. - Fire Radiative Power Dynamics in Nepal Himalayan Forests** | 2026 | 13 年 VIIRS 时序 + RF/XGBoost + SHAP 归因；跨生态区火强度驱动因子分析 | 证明了时序 ML 在复杂地形森林监测中的可行性；**空-时交互特征工程**（terrain-wind interaction）为物候建模提供了可迁移的特征设计范式 |

**Bejide (2026) 论文的关键诊断框架**：

```
不一致性维度:
┌──────────────┬────────────────┬─────────────────┐
│   生态层面   │   方法层面     │   结构-功能层面  │
├──────────────┼────────────────┼─────────────────┤
│ 地形复杂性   │ 尺度错配       │ 结构-功能解耦    │
│ 生态异质性   │ 传感器饱和     │ 时间恢复异步性   │
│ 物种组成     │ 基准不稳定     │ (green desert)   │
└──────────────┴────────────────┴─────────────────┘
```

**关键发现**：结构-功能解耦在扰动后随时间演变，这意味着**时序建模不是可选项，而是诊断森林真实健康状态的必要条件**。单时相模型看到的"绿色"可能来自快速恢复的草本层或先锋树种，而非目标树种的真正恢复。

**Gauli (2026) 论文的火行为时空分析范式**：
- 使用 0.25° 网格 + 2 天容差的时空聚类，将 569,136 个火点聚合为 11,595 个离散火事件
- 构造**地形-风交互特征**（drought-fuel interaction index），超越原始风速的重要性
- 模型跨生态区性能：R² = 0.683（High Mountains）~ 0.757（Siwaliks）
- **方法迁移潜力**：时空聚类 → 物候事件聚合；交互特征工程 → 光谱-气象-地形交互建模

### 1.3 时序编码方法分类

```
时序编码方式
├── 循环神经网络 (RNN/LSTM/GRU)
│   ├── 代表: Knowledge-Guided Multi-Source Time-Series (2026)
│   ├── 优点: 原生序列建模，处理不规则时间间隔
│   └── 缺点: 难以并行化，长序列梯度消失
│
├── 1D-CNN / Temporal CNN (TCN)
│   ├── 代表: InceptionTime, TempCNN
│   ├── 优点: 并行计算，固定感受野
│   └── 缺点: 只捕获局部时序依赖
│
├── Temporal Transformer / ViT 3D Patch
│   ├── 代表: TSP-Former (2025), Efficient ViT (2026)
│   ├── 核心: 时间维度位置编码 + 时空联合自注意力
│   │   公式: PE(t) = [sin(t/10000^{2i/d}), cos(t/10000^{2i+1/d})]
│   ├── 优点: 全局时序依赖，灵活的感受野
│   └── 缺点: O(T²) 复杂度，需充分预训练
│
├── State Space Models (Mamba/S4)
│   ├── 代表: ChangeMamba (2024)
│   ├── 优点: O(T) 线性复杂度，擅长长序列
│   └── 缺点: 新兴架构，生态还未充分验证
│
└── 混合架构
    ├── 代表: TerraFlow (2026) - 时序训练目标 + 多模态融合
    ├── CNN 提取空间特征 → Transformer 建模时序依赖
    └── 平衡效率与表达能力
```

---

## 2. 时序编码方案对比

### 2.1 方案 A：输入层——多时相拼接（Baseline）

**架构**：
```
T₁ 影像 → ┐
T₂ 影像 → ├→ Channel Concat → DUNIA Encoder → Embedding
T₃ 影像 → ┘
```

- 将 3-12 期 Sentinel-2 影像按波段维度拼接，形成 `[B, T×C, H, W]` 张量
- Encoder 不变（ViT/ResNet），将时间维度压缩为"额外的通道"
- **AnySat 采用此方案**，在 PASTIS 上取得 SOTA

**优点**：
- 实现最简单，复用现有编码器
- 计算量增加可控（仅通道数倍增）

**缺点**：
- 无法显式建模时序依赖——不同时间步之间没有特征交互，CNN 的局部卷积只看到相邻时间步的像素
- 对时间顺序不敏感——如果打乱期次顺序，表征几乎不变
- 没有时间距离的概念——T₁ 到 T₂ 是 5 天还是 50 天，模型无法区分

**适用场景**：时序期次少（≤3 期）、时间间隔均匀、不需区分子季相变化的场景

### 2.2 方案 B：编码器内部——Temporal Transformer 层（推荐）

**架构**：
```
T₁ → Spatial Encoder → z₁ ┐
T₂ → Spatial Encoder → z₂ ├→ Temporal Transformer → z_fused
T₃ → Spatial Encoder → z₃ ┘
       (Siamese / shared weights)
```

具体设计：

```
输入: X ∈ R^{T×C×H×W}   (T 期 Sentinel-2 影像)

阶段 1: 空间编码 (共享权重 ViT/ResNet)
  for t in 1..T:
    z_t = SpatialEncoder(X_t)   # z_t ∈ R^{N_patches × d}

阶段 2: 时序位置编码
  PE(t, 2i)   = sin(DOY(t) / 365 * 10000^{2i/d})
  PE(t, 2i+1) = cos(DOY(t) / 365 * 10000^{2i+1/d})
  z'_t = z_t + PE(t)            # 注入儒略日 (Day of Year) 位置信息

阶段 3: Temporal Self-Attention
  Z = [z'_1, z'_2, ..., z'_T]   # Z ∈ R^{T × N_patches × d}
  Q, K, V = ZW_Q, ZW_K, ZW_V
  Attention(Q,K,V) = softmax(QK^T/√d_k + M)V
  
  其中 M 为 Temporal Mask:
  - 双向: 全部可见（离线分析）
  - 因果: 仅过去可见（在线监测）
  
阶段 4: 时间池化
  z_fused = TemporalPooling(Z)   # mean/max/attention pooling
  z_fused ∈ R^{N_patches × d}
```

**公式**（核心时序对比损失在方案 C 中详述）：

**Temporal Self-Attention with DOY Encoding**：
```
α_{ij}^{(h)} = softmax( (z'_i W_Q^{(h)}) (z'_j W_K^{(h)})^T / √d_k )

z_i' = Concat(head_1, ..., head_H) W_O

where z'_i = z_i + PE(DOY_i),  PE(doy) = [sin(ω_k·doy), cos(ω_k·doy)]_{k=0}^{d/2-1}
```

**优点**：
- 显式建模时序依赖：全局注意力捕获任意两个时间步之间的长距离依赖
- 时间感知：DOY 位置编码让模型知道每期影像的绝对季节位置
- 灵活：可以处理可变长度序列（任意 T 值）
- 可以与 DUNIA 的 Zero-CL 损失无缝集成

**缺点**：
- O(T²) 复杂度，当 T > 12 时计算量显著
- 需要一定量的时序预训练数据

**复杂度对比**：

| T | 方案 A (Concat) | 方案 B (Temporal TF) | 方案 C (Post-hoc) |
|---|----------------|---------------------|-------------------|
| 3 | 1× | 1.3× | 1× |
| 6 | 2× | 2.0× | 1× |
| 12 | 4× | 4.5× | 1× |

### 2.3 方案 C：嵌入后——时序对齐损失（轻量级）

**架构**：
```
T₁ → DUNIA Encoder → e₁ ┐
T₂ → DUNIA Encoder → e₂ ├→ Temporal Alignment Loss
T₃ → DUNIA Encoder → e₃ ┘
```

- 编码器不变，仅在嵌入层施加时序一致性约束
- 每个时间步独立编码，通过损失函数注入时序先验

**损失函数**（核心创新）：

```python
def temporal_alignment_loss(e, doy, tau=0.5, alpha=0.1):
    """
    e: [B, T, d] - embeddings for T timesteps
    doy: [B, T] - day of year
    目标: 相邻时间步的嵌入应平滑变化，不应突变超过阈值
    """
    # 1. Smoothness: 相邻时间步不应差异过大
    L_smooth = Σ_t ||e_t - e_{t+1}||² · exp(-α·|doy_t - doy_{t+1}|)
    
    # 2. Cyclic: 年首年末接近 (对常绿树种)
    L_cyclic = ||e_1 - e_T||² · exp(-α·(365 - |doy_T - doy_1|))
    
    # 3. Identity preservation: 同一树冠的"本质"表征不应因季节剧烈改变
    e_identity = mean(e, dim=1)  # 时序中心
    L_identity = Σ_t ||e_t - e_identity||²
    
    # 4. Contrastive: 不同树冠在任一时间步都应可区分
    # 正样本: 同一树冠不同时间步 → 拉近
    # 负样本: 不同树冠同一时间步 → 推远
    L_contrast = NTXent(same_canopy_diff_time, diff_canopy_same_time)
    
    return L_smooth + lambda_cyclic * L_cyclic + L_identity + L_contrast
```

**时序对齐的几何解释——"物候管"（Phenology Tube）概念**：

```
嵌入空间中的物候轨迹:
                           
     盛叶期 ●────────────● 盛叶期 (树冠 A)
            ╱              ╲
           ╱                ╲
     展叶期 ●                ● 变色期
          ╱                  ╲
         ╱                    ╲
     萌芽期 ●══════════════════● 落叶期
         ┊  "物候管" (管径 τ)  ┊
         ┊                      ┊
     萌芽期 ●══════════════════● 落叶期  (树冠 B)
          ╲                    ╱
           ╲                  ╱
     展叶期 ●                ● 变色期
            ╲              ╱
     盛叶期 ●────────────● 盛叶期

约束条件:
  1. 同一树冠在不同物候期的嵌入应沿连续轨迹移动 (L_smooth)
  2. 轨迹应形成(近似)年周期闭环 (L_cyclic)  
  3. 不同树冠的"物候管"不应相交 (L_contrast)
  4. 管径 τ 控制允许的物候变化幅度
```

**优点**：
- 不改动编码器，实现最快
- 可插拔——可与现有预训练模型配合
- 几何上直观可解释

**缺点**：
- 编码器内部仍无法感知时序，时序信息仅通过损失间接注入
- 对下游任务的时序区分能力有限
- 如果基础编码器对单期影像的表征能力弱，时序损失无法弥补

### 2.4 方案对比总结

| 维度 | 方案 A: 输入拼接 | 方案 B: 时序 Transformer | 方案 C: 时序对齐损失 |
|------|-----------------|------------------------|---------------------|
| 编码器改动量 | 无（仅改输入层） | 大（新增时序模块） | 无（仅改损失函数） |
| 时序依赖建模 | 隐式（CNN局部） | 显式（全局注意力） | 无（仅在损失层面） |
| 可变长度序列 | ❌ | ✅ | ✅ |
| 计算开销 | 低 (~1×) | 中 (~1.5-4×) | 低 (~1×) |
| 时间位置感知 | ❌ | ✅ (DOY PE) | 部分 (通过损失) |
| 与 DUNIA Zero-CL 兼容 | ✅ | ✅ (需适配) | ✅ (并行使用) |
| 推荐阶段 | 基线实验 | **核心方案** | 辅助正则化 |

---

## 3. 核心设计问题解答

### 3.1 时序编码应该在哪个阶段加入？

**推荐答案：方案 B（编码器内部 Temporal Transformer）+ 方案 C（时序对齐损失）作为辅助正则化**

**分阶段部署路径**：

```
Phase 1 (立即): 方案 A (多时相拼接) 
  → 验证时序信息是否带来增益
  → 数据准备: 2-3 期 Sentinel-2 影像

Phase 2 (3-6 月): 方案 B (Temporal Transformer)
  → 核心架构升级
  → 数据准备: 6-12 期 Sentinel-2 完整年周期影像

Phase 3 (6-12 月): 方案 B + C (Temporal Transformer + 物候管损失)
  → 联合优化
  → 引入物候先验知识
```

**为什么不在输入层**：输入层拼接抹平了时间维度，CNN 对通道的顺序不敏感。实验表明，一个在日期 X 训练的拼接模型，对日期 Y（物候不同）的影像会给出完全不同的表征——这不是迁移学习，而是**领域漂移**。

**为什么不在纯嵌入层**：嵌入后对齐是"事后补救"——编码器在特征提取阶段没有时序感知，导致编码出的单期特征已经丢失了部分光谱-物候耦合信息（例如，同一反射率值在春季可能对应"健康萌芽"，在秋季可能对应"提前衰老"——编码器无法区分，因为缺乏时间上下文）。

### 3.2 如何设计跨时相的对比学习目标？

**核心原则**：同一树冠在不同季节的编码向量应满足三个约束：

1. **同一性约束（Identity）**：时序平均嵌入（"树种本征向量"）应稳定，不受物候影响
2. **平滑性约束（Smoothness）**：相邻物候期的嵌入变化应连续、可预测
3. **可分性约束（Discriminability）**：同一物候期内，不同树冠应可区分

**具体损失设计**：

```
给定:
  树冠 i 在时间 t 的嵌入: e_i^t
  DOY(t): 时间 t 的儒略日

损失组件:

1. Intra-canopy Temporal Contrastive (ITC):
   正样本: (e_i^t, e_i^{t+Δt})  — 同一树冠，不同时间
   负样本: (e_i^t, e_j^{t+Δt})  — 不同树冠，不同时间
   
   关键: 不使用 hard contrastive (InfoNCE)，而是 soft positive:
   s_pos = e_i^t · e_i^{t+Δt} 
   target_sim = 1 - β · |DOY(t+Δt) - DOY(t)| / 180
   L_ITC = MSE(s_pos, target_sim)
   
   解释: 时间越近，相似度目标越高（期望接近 1）；间隔半年（180天），
         相似度目标降到 1-β（允许更大的物候差异）

2. Inter-canopy Discriminative (ICD):
   正样本: (e_i^t, e_i^t) 同一树冠数据增强
   负样本: (e_i^t, e_j^t) 不同树冠同一时间
   L_ICD = InfoNCE(e_i^t, e_j^t)  — 标准对比损失

3. Cross-time Identity (CTI):
   将时序平均嵌入作为"树种身份向量":
   e_i^identity = (1/T) · Σ_t e_i^t
   L_CTI = Σ_t ||e_i^t - e_i^identity||²

总损失: L = λ_ITC·L_ITC + λ_ICD·L_ICD + λ_CTI·L_CTI
```

**关键设计选择**：使用 **soft positive similarity target** 而非 hard binary contrastive loss。原因：hard contrastive 假设正样本对完全等价——但冬季落叶期的树冠和夏季盛叶期的树冠显然不"等价"。让相似度目标随 DOY 距离衰减，更符合生态现实。

### 3.3 物候先验知识如何注入？

**三层次先验注入策略**：

| 层次 | 方式 | 载体 | 示例 |
|------|------|------|------|
| **L1: 结构先验** | 知识图谱约束 | 损失函数正则项 | 落叶阔叶树种 11月→3月 NDVI 必须 < 0.4 |
| **L2: 参数先验** | 预训练权重初始化 | 模型参数 | 用 MODIS 多年 NDVI 时序预训练时序编码器 |
| **L3: 任务先验** | 多任务学习 | 辅助任务头 | 同时预测 NDVI 未来值、物候阶段标签 |

**L1 具体方案——物候知识图谱**：

```yaml
# 每条规则编码为可微约束
phenology_rules:
  - type: "deciduous_broadleaf"
    constraint: "NDVI_leaves_off < 0.35"
    time_window: [11-01, 03-15]
    penalty: "hinge_loss"
  
  - type: "evergreen_conifer"
    constraint: "NDVI_min > 0.5"
    time_window: [01-01, 12-31]
    penalty: "soft_boundary"
  
  - type: "early_successional"
    constraint: "growth_rate_spring > 0.02 NDVI/day"
    time_window: [03-15, 05-15]
    penalty: "log_barrier"

# 损失函数
L_knowledge = Σ_r w_r · penalty(prediction, rule.constraint)
```

**L2 具体方案——预训练策略**：

```
阶段 1: 时序编码器预训练
  数据: 10 年 MODIS/Sentinel-2 NDVI/EVI 时序 (全球样本)
  任务: 掩码时序重建 (Masked Temporal Modeling)
        - 随机 Mask 30% 的时间步
        - 编码器预测被 Mask 位置的 NDVI 值
  输出: 预训练的 Temporal Transformer 权重

阶段 2: 跨模态微调
  在已预训练的时序编码器上叠加 DUNIA 的空间编码器
  联合微调，使用 Zero-CL 损失 + 时序对齐损失
```

**L3 具体方案——多任务头**：

```
编码器输出 z_t 
  ├→ 任务头 1: 物候阶段分类 (softmax → {萌芽, 展叶, 盛叶, 变色, 落叶})
  ├→ 任务头 2: NDVI 回归 (预测当前 + 下一时间步的植被指数)
  ├→ 任务头 3: 跨模态对齐 (DUNIA Zero-CL with GEDI waveform)
  └→ 任务头 4: 树种识别 (KNN / linear probe)
```

### 3.4 最少需要几期影像？

**量化分析**：

| 影像期数 | 能否区分物候阶段 | 时序编码有效吗 | 推荐架构 | 典型场景 |
|---------|----------------|--------------|---------|---------|
| **1 期** | ❌ | ❌ | DUNIA 现状 | 单次普查 |
| **2 期** | 部分（仅区分 peak vs off-peak） | ⚠️ 有限 | 方案 A (拼接) | 只有两季清晰影像 |
| **3 期** | 基础（spring/peak/fall） | ✅ 开始有效 | 方案 B (轻量 TF) | 三季存档数据 |
| **4-6 期** | 良好（区分展叶/盛叶/变色） | ✅ 显著 | 方案 B (标准) | Sentinel-2 季度合成 |
| **12 期** | 完整月度物候 | ✅ 最优 | 方案 B (全量) | Sentinel-2 月合成 |
| **24+ 期** | 年际变化+异常检测 | ✅+ | Mamba 架构 | 多年度长期监测 |

**2-3 期影像的可行策略**：

当仅有 2-3 期时，采用以下补偿机制：

1. **学习型时间编码**：不依赖长序列全局注意力，而是用可学习的物候先验嵌入向量
   ```
   z_t = Encoder(X_t) + PhenologyEmbedding(DOY(t))
   ```
   其中 `PhenologyEmbedding` 是一个轻量 MLP，输入儒略日输出 d 维偏移向量，在大规模数据上预训练

2. **光谱-时间耦合特征**：显式计算植被物候指数（如 NDVI 变化率 ΔNDVI/Δt）作为额外输入通道

3. **知识蒸馏**：用 12 期模型（教师）蒸馏到 3 期模型（学生），让学生网络学会从稀疏时序推断完整物候轨迹

4. **时序数据增强**：对已有期次做时间插值，生成模拟中间期次的合成影像，增加训练信号的时序密度

**实验验证建议**：在现有档案数据上，对同一树冠取 2-3 期不同季节的 Sentinel-2 影像，比较方案 A（拼接）vs 方案 C（对齐损失）vs 仅单期，以差异显著性确定最小期次。

---

## 4. 对 ForestPheno 的具体建议

### 4.1 架构集成方案

**ForestPheno 时序增强版编码器架构**：

```
                         ┌──────────────────────────┐
                         │  Temporal Transformer     │
                         │  (方案 B: 核心模块)        │
                         │                          │
S2_T1 ──→ ┐              │  DOY PE → Self-Attention  │
S2_T2 ──→ ├─ Siamese ──→ │  → Cross-Time Pooling    │──→ z_temporal
S2_T3 ──→ ┘  Spatial     │                          │
              Encoder     └──────────────────────────┘
              (DUNIA)                │
S1_T1 ──→ ┐                         │
S1_T2 ──→ ├─ Siamese ──→  ┌────────┴─────────────────┐
S1_T3 ──→ ┘  Spatial       │  Cross-Modal Alignment    │
              Encoder       │  (DUNIA Zero-CL + 时序)   │
              (DUNIA)       │                           │
              │             │  L_total =                 │
GEDI ────────→ Waveform     │    L_ZeroCL(z_opt, z_sar,  │
              Encoder       │            gedi_wave)     │
              (1D UNet)     │    + L_temporal(z_t1..z_tT)│
                            │    + L_phenology_prior     │
                            └───────────────────────────┘
```

### 4.2 分阶段实施路线图

```
Phase 1: 数据准备与基线验证 (Month 1-2)
├── 从 Sentinel-2 档案中提取 3-6 期同一 ROI 的云掩膜后影像
├── 实现方案 A (多时相拼接) baseline
├── 在已有 GEDI footprint 上做零样本检索对比 (单期 vs 多期)
└── 指标: 树高 RMSE, 冠层覆盖 RMSE, NDCG@K

Phase 2: Temporal Transformer 集成 (Month 3-4)
├── 实现 DOY Position Encoding
├── 实现 Temporal Self-Attention 模块
├── 设计交叉时间对比损失
├── 与 DUNIA Zero-CL 联合训练
└── 指标: 物候阶段分类精度, 树种识别 wF1 提升

Phase 3: 物候先验注入 (Month 5-6)
├── 构建森林物候知识图谱 (树种→物候规则)
├── 实现知识引导损失函数
├── 大规模 NDVI 时序预训练
└── 指标: 跨年份泛化性能, 异常检测 AUROC

Phase 4: 多模态时序融合 (Month 7-9)
├── SAR 时序 (Sentinel-1) 与光学时序联合编码
├── 引入 GEDI 波形的时序变化 (如多期对比)
├── 结构-光谱不一致性诊断模块
└── 指标: 抗云鲁棒性, 物候异常检出率
```

### 4.3 数据需求

**最小可行数据集（MVD）**：

| 数据项 | 规格 | 来源 | 期次 |
|--------|------|------|------|
| Sentinel-2 L2A | 10m, 10 波段 | Copernicus CDSE / GEE | 3-6 期/年 |
| Sentinel-1 GRD | 10m, VV+VH | Copernicus CDSE | 3-6 期/年 |
| GEDI L2A/L2B | Footprint 级 | NASA LP DAAC | 同期 |
| 物候标签 | 树种+日期+阶段 | PhenoCam / 地面调查 | 3-5 阶段 |
| ERA5-Land | 气象格点 | ECMWF | 日尺度 |

**可选增强数据**：
- MODIS MCD43A4 (500m, daily) — 用于大规模预训练
- PlanetScope (3m, daily) — 用于精细物候验证
- PhenoCam 网络 — 地面真值物候状态

---

## 5. 现在能做的验证实验

### 5.1 实验 1：单期 vs 多期零样本检索

**目标**：验证时序信息是否提升森林变量零样本检索精度

**数据**：已有 Sentinel-2 档案 + GEDI 匹配 footprints

**设置**：
- 从 GEDI 过境日期前后各选 1 期 Sentinel-2 影像（共 3 期）
- 分别使用 1 期（中值合成）、3 期（方案 A 拼接）、3 期（方案 C 时序对齐）进行零样本检索
- 查询: GEDI 波形 → 最近邻检索 → Sentinel-2 像素嵌入

**预期**：3 期方案在作物/草地/落叶林区域的检索精度显著优于单期

```
指标矩阵:
┌─────────────┬──────────┬──────────┬──────────┐
│   方法      │ 树高RMSE │ 冠层覆盖 │ 树种 wF1 │
├─────────────┼──────────┼──────────┼──────────┤
│ 单期(DUNIA) │  2.0m    │  11.7%   │  76.0%   │
│ 3期(拼接)   │   ?      │   ?      │   ?      │
│ 3期(对齐)   │   ?      │   ?      │   ?      │
└─────────────┴──────────┴──────────┴──────────┘
```

### 5.2 实验 2：物候阶段聚类分析

**目标**：验证时序嵌入空间是否自然形成物候相关的几何结构

**数据**：同一地理区域的多期 Sentinel-2 影像

**方法**：
1. 使用现有 DUNIA 编码器对每期影像独立编码
2. 对同一像素的时序嵌入做 PCA/t-SNE 可视化
3. 计算"物候管"指标：
   - **管径** `τ = max_t ||e_t - e_identity||` — 反映同一像素的物候变化幅度
   - **轨迹连续性** `σ = mean_t ||e_t - e_{t+1}||` — 反映相邻时间步嵌入变化是否平滑
4. 按土地覆盖类型分组，分析不同类型的管径分布

**预期发现**：
- 常绿针叶林：管径小（<0.3），轨迹紧凑
- 落叶阔叶林：管径大（>0.5），轨迹呈年周期
- 农田/草地：管径最大（>0.8），轨迹受管理活动影响

### 5.3 实验 3：光谱-结构不一致性诊断（受 Bejide 2026 启发）

**目标**：验证单时相模型是否产生 "green desert" 误判

**数据**：包含扰动后恢复序列的 Sentinel-2 + GEDI 配对数据

**方法**：
1. 用 DUNIA 分别从光学影像预测树高和从 GEDI 波形检索树高
2. 计算预测差值的时序演变：
   ```
   ΔH(t) = H_predicted_from_optical(t) - H_GEDI(t)
   ```
3. 如果 ΔH(t) 随时间增大（光学预测偏高），则检测到 "green desert" 现象
4. 在时序编码方案中加入 ΔH(t) 作为不一致性诊断指标

**预期**：单时相模型在扰动后 2-3 年出现系统性正偏差（光学高估恢复程度），时序模型能缓解该偏差。

### 5.4 实验 4：Temporal Transformer 消融研究

| 消融条件 | 配置 | 预期影响 |
|---------|------|---------|
| Full Model | DOY PE + Temporal SA + 时序损失 | 最佳 |
| - DOY PE | 无时间位置编码 | 无法区分春季 vs 秋季（光谱可能相似） |
| - Temporal SA | 仅均值池化时序嵌入 | 丢失时序依赖，退化为方案 A |
| - 时序损失 | 仅用 Zero-CL | 时序模块无梯度引导 |
| - Siamese | 每期独立编码器（不共享权重） | 参数爆炸，训练不稳 |

---

## 6. 参考文献

1. **TSP-Former** (2025) - A Phenology-Guided Transformer for Tobacco Mapping Using Satellite Image Time Series. *IEEE JSTARS*. DOI: 10.1109/jstars.2025.3645265
2. **Efficient Spatio-Temporal ViT** (2026) - Efficient Spatio-Temporal Vegetation Pixel Classification with Vision Transformers. *arXiv:2605.00296*
3. **TerraFlow** (2026) - Multimodal, Multitemporal Representation Learning for Earth Observation. *arXiv:2603.12762*
4. **ChangeMamba** (2024) - Remote Sensing Change Detection With Spatiotemporal State Space Model. *IEEE TGRS*. DOI: 10.1109/tgrs.2024.3417253
5. **CROMA** (2023) - Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders. *arXiv:2311.00566*
6. **OmniSat** (2024) - Self-supervised Modality Fusion for Earth Observation. *ECCV 2024*
7. **DeepPhenoTree** (2026) - Multi-site Apple Phenology RGB Annotated Dataset. *ResearchSquare*. DOI: 10.21203/rs.3.rs-8977752/v1
8. **Adaptive Multi-Scale Wheat Phenology** (2026) - Adaptive Multi-scale Feature Refinement for Wheat Phenology Recognition. *Frontiers in Plant Science*. DOI: 10.3389/fpls.2026.1730706
9. **Knowledge-Guided Time-Series** (2026) - Knowledge-Guided Multi-Source Time-Series for Crop Type Classification. *Applied Sciences*. DOI: 10.3390/app16094194
10. **NDVI Forecasting Transformer** (2026) - Spatio-Temporal Transformers for Long-Term NDVI Forecasting
11. **Bejide** (2026) - Multidimensional Inconsistency in Forest Ecosystem Representation: An NLP-Assisted Thematic Review. *EarthArXiv*. DOI: 10.31223/x5fn4h
12. **Gauli et al.** (2026) - Fire Radiative Power Dynamics in Nepal Himalayan Forests. *ResearchSquare*. DOI: 10.21203/rs.3.rs-9716417/v1
13. **Dual-Form SITS** (2026) - Comparative Analysis of Dual-Form Networks for Live Land Monitoring Using Multi-Modal Satellite Image Time Series. *arXiv:2603.24109*
14. **Spatio-Temporal NDVI Forecasting** (2026) - Spatio-Temporal Transformers for Long-Term NDVI Forecasting
15. **AnySat** - Self-supervised Multimodal Satellite Image Time Series Analysis (referred in DUNIA paper)
16. **DUNIA** (2025) - Dense Unsupervised Cross-Modal Alignment for Earth Observation. *ICML 2025*

---

## 附录 A：DOY Position Encoding 实现参考

```python
import torch
import math

class DOYPositionalEncoding(torch.nn.Module):
    """Day-of-Year sinusoidal position encoding."""
    
    def __init__(self, d_model: int, max_freq: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        
        # 频率: 从 365 天周期到 1 天周期
        freq = torch.exp(
            torch.linspace(0, math.log(max_freq), d_model // 2)
        )
        self.register_buffer('freq', freq)  # [d_model/2]
    
    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            doy: [B, T] day of year (0-365)
        Returns:
            pe: [B, T, d_model]
        """
        B, T = doy.shape
        doy = doy.unsqueeze(-1)  # [B, T, 1]
        
        # 年周期项: sin/cos(2π * DOY/365 * freq)
        angle = 2 * math.pi * doy / 365.0 * self.freq  # [B, T, d/2]
        pe = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        
        # 添加季节差异项: sin/cos(4π * DOY/365 * freq) - 双峰
        angle_2 = 4 * math.pi * doy / 365.0 * self.freq
        pe_seasonal = torch.cat(
            [torch.sin(angle_2), torch.cos(angle_2)], dim=-1
        )
        
        return torch.cat([pe, pe_seasonal], dim=-1)[:, :, :self.d_model]


class TemporalTransformerLayer(torch.nn.Module):
    """Single temporal self-attention layer for time-series patch embeddings."""
    
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4 * d_model, d_model),
        )
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: [N_patches × B, T, d_model]
            mask: [T, T] attention mask
        Returns:
            [N_patches × B, T, d_model]
        """
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        mlp_out = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_out))
        return x
```

## 附录 B：时序对齐损失实现参考

```python
def temporal_contrastive_loss(
    embeddings: torch.Tensor,    # [B, T, d]
    doy: torch.Tensor,           # [B, T]
    temperature: float = 0.07,
    beta: float = 0.5,           # 控制物候变化允许幅度
) -> torch.Tensor:
    """
    跨时相对比损失
    
    核心思想: 同一树冠不同时间的相似度目标随 DOY 距离衰减。
    间隔 0 天 → target_sim = 1.0 (完全一致)
    间隔 90 天 → target_sim = 1-0.5*90/180 = 0.75
    间隔 180 天 → target_sim = 1-0.5 = 0.5 (最小)
    """
    B, T, d = embeddings.shape
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    
    # 计算所有时间步对的 DOY 距离
    doy_diff = (doy.unsqueeze(2) - doy.unsqueeze(1)).abs()  # [B, T, T]
    doy_diff = torch.min(doy_diff, 365 - doy_diff)  # 考虑年周期
    
    # Soft target: 相似度随 DOY 距离衰减
    target_sim = 1.0 - beta * doy_diff / 180.0  # [B, T, T]
    target_sim = target_sim.clamp(min=0.3)  # 最低相似度阈值
    
    # 实际相似度
    sim = torch.bmm(embeddings, embeddings.transpose(1, 2))  # [B, T, T]
    sim = sim / temperature
    
    # MSE between predicted sim and target sim
    loss = torch.nn.functional.mse_loss(sim, target_sim)
    
    return loss


def phenology_tube_loss(
    embeddings: torch.Tensor,  # [B, T, d]
    doy: torch.Tensor,         # [B, T]
    tau: float = 0.5,          # tube radius
) -> torch.Tensor:
    """
    物候管损失: 惩罚过分偏离时序中心的嵌入
    
    鼓励同一树冠的嵌入保持在半径为 tau 的超球内，
    同时允许沿物候轨迹的平滑变化。
    """
    B, T, d = embeddings.shape
    
    # 时序中心
    identity = embeddings.mean(dim=1, keepdim=True)  # [B, 1, d]
    
    # 管径惩罚
    deviation = torch.norm(embeddings - identity, dim=-1)  # [B, T]
    tube_loss = torch.nn.functional.relu(deviation - tau).pow(2).mean()
    
    # 平滑性惩罚
    smoothness = torch.norm(
        embeddings[:, 1:] - embeddings[:, :-1], dim=-1
    ).mean()
    
    return tube_loss + 0.1 * smoothness
```
