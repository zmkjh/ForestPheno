# 遥感多模态动态融合设计空间：Agent 可控融合策略蓝图

> 面向系统设计者的工程级技术对比，聚焦"具体怎么做"和"如何改造为 Agent 可控制"。

---

## 1 核心问题定义

**目标**：设计一个融合模块，其行为不仅依赖于多模态输入（HSI + LiDAR/SAR），还能接受外部 Agent 发出的控制信号（如物候阶段、任务描述、不确定性估计），实现**由 Agent 控制融合策略**的范式。

**五个备选技术路径**：

| 方案 | 核心机制 | 对"Agent 控制"的友好度 |
|------|---------|---------------------|
| A: 网络内部动态路由 (DCMNet) | 可学习门控选择计算路径 | 中等 — 路由门接受外部条件 |
| B: 频率域动态滤波 (DFFNet/IFGNet) | 频域自适应核/隐式聚合 | 低 — 滤波核依赖自身特征 |
| C: Mamba 状态空间融合 (MSFMamba) | 选择性SSM + 双输入交叉参数化 | 高 — SSM天然接受额外条件 |
| D: 世界模型 + CLIP引导 (FusDreamer) | 扩散生成 + 文本Prompt引导 | 最高 — 文本即Agent指令 |

---

## 2 方案A: 网络内部动态路由 (DCMNet)

### 2.1 架构描述

```
HSI patch ──→ [3D Conv Encoder] ──→ {F¹_h, F²_h, F³_h}
                                             ↓
LiDAR patch ─→ [2D Conv Encoder] ──→ {F¹_l, F²_l, F³_l}
                                             ↓
        ┌──────────────── Routing Space (3-layer fully-connected) ────────────────┐
        │                                                                           │
        │  Layer k: [BSAB] ═══ [BCAB] ═══ [ICB]    (parallel, fully-connected)    │
        │              ↘ gating ↙                                                   │
        │              Routing Gate: W^k_i = δ(FC(ReLU(FC(F_h+F_l+X^k_i))))        │
        │              δ(x) = max(0, Tanh(x))                                      │
        │              X^{k+1}_i = Σ_j w^k_{j,i} · H^k_j                          │
        │                                                                           │
        │  Block types:                                                            │
        │    BSAB: Spatial-wise bilinear cross-attention (V_h ⊙ Q_l, V_l ⊙ Q_h)  │
        │    BCAB: Channel-wise bilinear cross-attention                           │
        │    ICB:  Simple Conv(F_h + F_l + X) — "easy sample" fast path          │
        └───────────────────────────────────────────────────────────────────────────┘
                                             ↓
                                    [Aggregation + FC] → Classification
```

### 2.2 关键公式

**路由门控**（每个块的内部路径选择器）：

```
W^k_i = max(0, Tanh(FC_2(ReLU(FC_1(F_h + F_l + X^k_i)))))
```
- 输入：`F_h + F_l`（编码器特征）+ `X^k_i`（上层传入信号）
- 输出：3维向量 `{w_{i,1}, w_{i,2}, w_{i,3}}`，表示到下一层各块的概率
- 激活：restricted Tanh，确保非负（关键设计！）

**路由聚合**：

```
X^{k+1}_i = Σ_{j=1}^{3} w^k_{j,i} · H^k_j
```
- H^k_j 是第 k 层第 j 个块的计算输出（经过其计算单元处理后）
- w^k_{j,i} 是从 j 块到 i 块的路径权重

**双线性交叉注意力**（BSAB/BCAB 核心）：

```
CA_h = softmax(Q_h K_h^T / √d) · (V_h ⊙ Q_l)    # HSI query, LiDAR value bias
CA_l = softmax(Q_l K_l^T / √d) · (V_l ⊙ Q_h)    # LiDAR query, HSI value bias
```
- BSAB: attention along **spatial** dimension
- BCAB: attention along **channel** dimension

### 2.3 性能数据

| 数据集 | OA | AA | Kappa | 训练样本/类 |
|--------|-----|------|-------|------------|
| Trento | 98.96 | 97.55 | 98.61 | ~150 |
| Houston 2013 | 95.11 | 95.74 | 94.69 | ~190 |
| Houston 2018 | 93.27 | 96.33 | 91.33 | ~1000 |

**计算开销**：Params 3.83M，FLOPs 0.046G，Inference 0.0097s/sample (RTX 4090)

### 2.4 改造为"Agent 可控"的方案

**问题**：当前路由门 `W = f(F_h, F_l, X)` 仅依赖输入特征，不接受外部信号。

**改造点1：注入 Agent 条件向量到路由门**

```python
# 原始：W = FC(ReLU(FC(F_h + F_l + X)))
# 改造后：
agent_ctx = MLP(agent_signal)  # agent_signal: [物候期编码, 区域类型, 任务ID, ...]
W = FC(ReLU(FC(F_h + F_l + X) + agent_ctx))  # FiLM-style 条件注入
```

具体修改位置：`DCMNet` 中每个 BSAB/BCAB/ICB 的 `RoutingGate.forward()`。

**改造点2：添加可学习的"Agent override"函数**

```python
# Agent 可以发出 "override" 信号强制选择某条路径
agent_mask = sigmoid(Linear(agent_signal))  # 3-dim, 对应3个block
W_final = agent_mask * W_learned + (1 - agent_mask) * agent_routing
```

**改造点3：Agent 控制路由空间的层数/拓扑**

- Agent 根据任务难度决定使用多少层路由
- 简单场景（如"纯森林"）→ 1层路由 → ICB 直通
- 困难场景（如"森林-城市过渡带"）→ 3层路由 → 全连接

**难度评估**：★★★☆☆（中等）
- 路由门接受单个向量输入，注入条件不改变整体架构
- 需要重新训练，但改动量小（~10行代码）
- 潜在问题：Agent 信号维度与特征维度不对齐可能导致优化困难

**推理速度预估**：Agent 条件注入后推理增加 <5%（仅多一次 FC 加法）

---

## 3 方案B: 频率域动态融合 (DFFNet / IFGNet)

### 3.1 DFFNet 架构（动态频域滤波核）

```
HSI ──→ [3D Conv] ──→ ┐
                       ├→ [DFFM × 2] ──→ [FC] → Classification
LiDAR/SAR → [2D Conv] ┘

DFFM 内部：
  Input split ──┬──→ [DFB: Dynamic Filter Block]
                │        │
                │        ├─ 2D FFT → f(u,v) in frequency domain
                │        ├─ Dynamic Filter Generation:
                │        │     K(X) = Softmax(MLP(GAP(X_in))) ⊗ F_base
                │        │     F_base = {F₁,...,F_N} learnable basis filters
                │        ├─ f̂(u,v) = K(X_in) ⊙ f(u,v)   # frequency filtering
                │        ├─ 2D IFFT → X_out (back to spatial)
                │        └─ FFN: [identity branch] + [spatial Conv branch]
                │              + [frequency Conv branch]  # dual-domain
                │
                └──→ [SSAFB: Spectral-Spatial Adaptive Fusion Block]
                         ├─ F'_h = F_h + C₁×₁(ReLU(C₁×₁(GAP(F_h))))  # channel attn
                         ├─ F'_x = F_x + C₅×₅([GAP(F_x), GMP(F_x)])  # spatial attn
                         ├─ Channel Shuffle: 交替排列 HSI 和 LiDAR channels
                         └─ Fo = C₁×₁(CS([F'_h, F'_x]))
```

**动态滤波核生成**（核心创新）：

```
K(X) = Softmax(MLP(GAP(X))) ⊗ F_base
```
- F_base: N 个可学习的基础滤波器，每个与 FFT 输出同维度
- MLP 根据全局平均池化的输入特征生成 N 个权重
- Softmax 归一化后加权组合基础滤波器
- 结果：输入数据自适应选择频率分量

### 3.2 IFGNet 架构（KAN + 隐式频率聚合）

```
HSI patch ──→ [KAN Encoder] ──→ F_HSI ∈ R^{P²×D}
                                      │
LiDAR patch ─→ [KAN Encoder] ──→ F_LiDAR ∈ R^{P²×D}
                                      │
                    ┌─────────────────┤
                    ↓                 ↓
          [SIAU: Spatial]    [Freq Implicit Agg]
          LiDAR-guided       DFT → aggregate real
          implicit agg       and imag components
          with KAN           → IDFT
                    │                 │
                    ↓                 ↓
               FSpa  ─── + ───  FFre
                              │
                         Lightweight Head → Classification
```

**SIAU 核心公式**（KAN 驱动的邻域隐式聚合）：

```
v(k)_q = Φ_KAN([f_HSI_{xk}, f_LiDAR_q, q - xk])  # 候选特征 + 权重
α(k)_q = Softmax(v(k)_q[-1])                        # 归一化权重
FSpa_q = Σ α(k)_q · v(k)_q[:D]                      # 加权聚合
```

- Φ_KAN 是 B-spline 函数参数化的非线性映射
- 输入包括：邻域 HSI 特征 + 查询点 LiDAR 特征 + 相对坐标偏移
- KAN 的优势：连续可微的样条函数比 MLP 更适合建模几何不连续性

**频率域隐式聚合**：

```
F_real, F_imag = DFT(F_HSI), DFT(F_LiDAR)
FFre_real = SIA(F_HSI_real, F_LiDAR_real)     # 复用同一 SIA 算子
FFre_imag = SIA(F_HSI_imag, F_LiDAR_imag)
FFre = IDFT(FFre_real, FFre_imag)
```

### 3.3 性能数据

**DFFNet**：

| 数据集 | OA | AA | Kappa | Params | FLOPs(G) | Inference(s) |
|--------|-----|------|-------|--------|----------|-------------|
| Houston 2013 | 92.35 | 93.48 | 91.70 | 1.28M | 0.0303 | 0.2387 |
| Berlin | 75.42 | 64.85 | 63.22 | 1.28M | 0.0303 | 0.2387 |

**IFGNet**：

| 数据集 | OA | AA | Kappa |
|--------|-----|------|-------|
| Houston 2013 | 99.37 | 99.50 | 99.32 |
| MUUFL | 92.67 | 94.47 | 90.45 |

### 3.4 频域操作在物候分析中的潜力

1. **物候阶段 = 不同频率分量激活**
   - 生长季：高频（纹理信息，叶面积变化）
   - 休眠季/落叶季：低频（平滑冠层，稀疏结构）
   - DFFNet 的动态滤波核可学习为物候期依赖的带通滤波器

2. **多时相频率分析**
   - 可将 DFFNet 的 2D FFT 扩展为沿时间轴的 3D FFT
   - `f(u, v, t)` → 时间-频率-空间联合滤波
   - 对应的物候模式：年际周期信号

3. **IFGNet 的 KAN 隐式聚合在物候中的优势**
   - B-spline 函数的局部支持特性天然适合建模物候渐变
   - 邻居特征聚合可捕捉相邻像素的物候一致性
   - 频率域独立处理实部/虚部，适合提取周期模式

### 3.5 改造为"Agent 可控"的方案

**DFFNet 改造**：

```python
# 原始：K = Softmax(MLP(GAP(X_in))) ⊗ F_base
# 改造：Agent 条件信号注入到 MLP
agent_feat = MLP_agent(agent_signal)
K = Softmax(MLP(GAP(X_in) + agent_feat)) ⊗ F_base
# 或使用 FiLM：K = Softmax(MLP(GAP(X_in)) * γ(agent) + β(agent)) ⊗ F_base
```

**IFGNet 改造（更有前景）**：

```python
# Agent 信号可作为 SIAU 中 KAN 的额外输入
# 原始：v = Φ_KAN([f_HSI_xk, f_LiDAR_q, q - xk])
# 改造：
v = Φ_KAN([f_HSI_xk, f_LiDAR_q, q - xk, AgentEncoding])  
# AgentEncoding 例如：物候期 one-hot / 任务嵌入
```

**难度评估**：★★★☆☆（中等）
- 滤波核生成依赖 GAP 压缩的全局特征，注入额外条件可行但可能稀释信息
- KAN 改造更自然——B-spline 本身接受任意维度输入
- FFT 操作对分辨率敏感，Agent 需要知道自己面对的传感器分辨率

---

## 4 方案C: Mamba 状态空间融合 (MSFMamba)

### 4.1 架构描述

```
HSI ──→ [PCA] ──→ ┐
                   ├→ [Spatial-Spectral Mamba Module] × L ──→ [FC] → Class
LiDAR/SAR ────────┘

每个 Module 包含三个 Block：

┌─ MSpa-Mamba Block ────────────────────────────────────┐
│  X → [Linear] → [DWConv] → [SiLU] → [MSpa-SSM] → [LN] ─┐
│  X → [Linear] → [SiLU] ──────────────────────────────────┤
│                          X1 ⊙ X2 → [Linear] → Xout       │
│                                                          │
│  MSpa-SSM:                                              │
│    DWConv(stride=1) → Z₁(R^{H×W×C})  ─→ SSM(4扫描) ─┐  │
│    DWConv(stride=2) → Z₂(R^{H/2×W/2×C}) → SSM(4扫描) ─┤  │
│    Z' = Z'₁ + Z'₂ + Interp(Z'₃ + Z'₄)                   │  │
│    扫描方向：行优先→列优先 + 逆方向 (共4路)               │  │
└─────────────────────────────────────────────────────────┘

┌─ Spe-Mamba Block (光谱SSM) ───────────────────────────┐
│  同 MSpa-Mamba 结构                                     │
│  关键差异：扫描沿 Channel 维度 (C → sequence length)     │
│  reshape: R^{H×W×C} → R^{C×HW} → SSM(2方向扫描)        │
└─────────────────────────────────────────────────────────┘

┌─ Fus-Mamba Block (双输入SSM融合) ──────────────────────┐
│  HSI特征 F_h ─→ [Linear] → [DWConv] → [SiLU] → [Fus-SSM] ─→ F_ho │
│  LiDAR特征 F_x → [Linear] → [DWConv] → [SiLU] → [Fus-SSM] ─→ F_xo │
│                                                          │
│  核心：Fus-SSM 的双输入参数化：                          │
│    Algorithm: F_ho from F_h with F_x 生成 A,B,C,Δ        │
│                                                          │
│    // 上分支：F_h 是被处理序列，F_x 是条件模态           │
│    A, D ← 随机初始化参数                                 │
│    B = Linear(F_x)       // 投影矩阵由 LiDAR 生成        │
│    C = Linear(F_x)       // 投影矩阵由 LiDAR 生成        │
│    Δ = softplus(Linear(F_x) + Parameter)  // 时间尺度    │
│    A_bar = exp(Δ ⊗ A)                                   │
│    B_bar = Δ ⊗ B                                         │
│    F_ho = SSM(A_bar, B_bar, C, D)(F_h)  // 用 LiDAR 参  │
│    数处理 HSI 序列                                       │
│                                                          │
│    // 下分支对称：F_xo 用 F_h 生成参数处理 F_x           │
│    B', C', Δ' = from Linear(F_h)                         │
│    F_xo = SSM(A'_bar, B'_bar, C', D')(F_x)               │
└─────────────────────────────────────────────────────────┘
```

### 4.2 关键公式

**选择性扫描机制**（Mamba 核心）：

```
h_t = A_bar · h_{t-1} + B_bar · x_t     # 隐状态更新
y_t = C · h_t + D · x_t                  # 输出
```
- A_bar: R^{C×N}，状态转移矩阵（现在由对端模态控制）
- B_bar: R^{C×N}，输入投影
- C: R^{N}，输出投影
- 关键：B、C、Δ 都是 **input-dependent**（选择性扫描）
- Fus-Mamba：B、C、Δ 来自 **对端模态** 而非被处理序列自身

**多尺度空间 SSM**（MSpa-SSM）：

```
Z₁ = DWConv_stride1(X)  # 原始分辨率
Z₂ = DWConv_stride2(X)  # 1/2 分辨率
[Y₁,Y₂] = SSM(scan(Z₁, dir=正向+逆向))  # 2路原始分辨率
[Y₃,Y₄] = SSM(scan(Z₂, dir=正向+逆向))  # 2路降采样分辨率
Output = Y₁ + Y₂ + Interp(Y₃ + Y₄)
```
- 总计生成 4 路扫描序列，含 2 种空间分辨率
- 降采样路径减少冗余（是论文的主要贡献）

### 4.3 性能数据

| 数据集 | OA | AA | Kappa | 模态 |
|--------|-----|------|-------|------|
| Berlin | 76.92 | 64.88 | 64.88 | HSI+SAR |
| Augsburg | 91.38 | 63.31 | 87.45 | HSI+SAR |
| Houston 2018 | 92.38 | 95.51 | 90.16 | HSI+LiDAR |
| Houston 2013 | 92.86 | 93.77 | 92.25 | HSI+LiDAR |

**计算开销**（Augsburg）：Params 1.53M，FLOPs 0.0377G，Inference 0.1747s (RTX 4090)

### 4.4 改造为"Agent 可控"的方案（最简洁）

**核心洞察**：Fus-Mamba 中，一个模态的 SSM 参数（B, C, Δ）由**另一个输入**通过 `Linear()` 投影生成。这天然允许**第三个输入源**生成融合参数。

**改造点1：三输入 Fus-SSM（Agent 作为第三控制器）**

```python
# 原始 Fus-SSM：
#   B, C, Δ = from Linear(F_x)  # LiDAR 控制 HSI 的 SSM 参数
# 改造后 (Agent-Fus-SSM)：
agent_feat = MLP_agent(agent_signal)  # Agent 条件编码

# 方式A: 加性注入
B = Linear(F_x) + Linear_agent_B(agent_feat)
C = Linear(F_x) + Linear_agent_C(agent_feat)
Δ = softplus(Linear(F_x) + Linear_agent_Δ(agent_feat) + Param)

# 方式B: 门控注入 (Gating)
g = sigmoid(Linear_gate(agent_feat))
B = (1 - g) * Linear(F_x) + g * Linear_agent_B(agent_feat)
# agent 可控制"多少比例由 LiDAR 决定，多少由 Agent 指令决定"
```

**改造点2：Agent 控制扫描策略**

```python
# Agent 可根据任务决定扫描分辨率和方向
# 原始：固定 2 分辨率 + 4 方向
# 改造：
if agent_mode == "fine_detail":
    scans = [full_res_4dir]           # 完整分辨率，4方向
elif agent_mode == "coarse_overview":
    scans = [down2x_2dir]             # 降采样，快速扫描
else:  # 自适应
    scans = [full_res_4dir, down2x_2dir]  # 混合
```

**改造点3：Agent 条件注入到 Δ 时间尺度**

- Δ 控制 SSM 的"记忆步长"
- Agent 可通过调整 Δ 来控制融合的"时间尺度"
- 物候分析中：生长季需要短记忆（局部变化快），休眠季需要长记忆（全局一致性）

**难度评估**：★★☆☆☆（较低）
- Fus-Mamba 本来就是双输入设计→扩展为三输入最自然
- 不需要改变 SSM 底层实现（只是多一个 Linear 投影）
- O(n) 线性复杂度不变

**推理速度预估**：Agent 注入后推理增加 <3%

---

## 5 方案D: 世界模型 + CLIP 引导 (FusDreamer)

### 5.1 架构描述

```
┌────────────────── LaMG: Latent Multimodal Generation ──────────────────┐
│                                                                         │
│  HSI patch  ──→ [RDE: ResBlock×3 + Downsample] ──→ X'_hsi             │
│                                                     ↘                  │
│  LiDAR patch ─→ [RDE: ResBlock×3 + Downsample] ──→ X'_lid  → [RDAF]  │
│                                           Adaptive Weighted Fusion      │
│                                           M_hsi, M_lid = Softmax(       │
│                                              Conv(Concat(X'_hsi,X'_lid)))│
│                                           F_fus_en = Conv(Concat(       │
│                                              X'_hsi⊙M_hsi, X'_lid⊙M_lid))│
│                                                     ↓                   │
│  ┌─ [RDD: Symmetric Decoder + Upsample + Skip-connections] ─┐          │
│  │  F_fus_en → X''_hsi, X''_lid                              │          │
│  │  → 3× ResBlock + Upsample → X'''_hsi, X'''_lid           │          │
│  │  → N'_hsi, N'_lid (predicted noise)                       │          │
│  │  → F_fus_de = Concat(Conv(X'''_hsi), Conv(X'''_lid))     │          │
│  └───────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                                          ↓
┌────────────────── OK-CP: Open-World Knowledge ──────────────────────────┐
│                                                                         │
│  F_fus_de ──→ [MFE: 3D Residual Conv] ──→ F_fus ──→ [FC] → ŷ          │
│                                             │                           │
│                                             ├─→ F'_fus (projection)     │
│                                             │                           │
│  ┌─ MPE: Multi-attribute Prompts Encoder ──┐    ┌─ Contrastive ──┐    │
│  │                                         │    │                 │    │
│  │  Self-categorical prompts:              │    │ Cor_{m2c} =     │    │
│  │    "A HSI-LiDAR data of <class>"        │    │   F'_c ⊗ F'_fus │    │
│  │         ↓ CLIP Tokenizer ↓              │    │ Cor_{c2m} =     │    │
│  │    Fc = CLIP(T_c)  →  Fc' = Transf(Fc)  │    │   F'_fus ⊗ F'_c│    │
│  │                                         │    │                 │    │
│  │  Differentiated physical prompts:       │    │ L_mc =          │    │
│  │    "The apple trees appear khaki"       │    │   CE(Cor, y)    │    │
│  │    "Vineyard is a regular rectangle"    │    │                 │    │
│  │    "The buildings are next to road"     │    │ L_md = Σ CE(...)│    │
│  │         ↓ CLIP Tokenizer ↓              │    │                 │    │
│  │    F_di = CLIP(T_di) → Transformer      │    │ L_M =           │    │
│  │                                         │    │ α·L_mc+         │    │
│  └─────────────────────────────────────────┘    │ (1-α)·L_md     │    │
│                                                  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                          ↓
┌── MuCO: Multitask Combinatorial Optimization ───────────────────────────┐
│  L = λ₁·L_C + λ₂·L_N + λ₃·L_M                                          │
│  L_C: CrossEntropy(ŷ, y)          # 分类损失                            │
│  L_N: MSE(N_hsi,N'_hsi) + MSE(N_lid,N'_lid)  # 噪声预测                │
│  L_M: α·L_mc + (1-α)·L_md        # 图文一致性                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 CLIP Prompt 引导融合的具体机制

**Prompt 类别定义**（以Trento为例）：

| 类别 | Self-categorical | Differentiated Physical (×3) |
|------|-----------------|------------------------------|
| Apple trees | "A HSI-LiDAR data of apple trees" | ① "The apple trees appear khaki and green"<br>② "The height of apple trees is lower than that of vineyard"<br>③ "There is ground in the middle of the gap between apple trees" |
| Buildings | "A HSI-LiDAR data of buildings" | ① "The buildings are more distant from the ground than from the road"<br>② "The buildings are well spaced"<br>③ "The height of buildings is close to the woods" |

**CLIP 引导的 loss 流**：

1. Prompt → CLIP Tokenizer → 文本特征 (F_c, F_d1, F_d2, F_d3)
2. 文本特征 → Transformer × e 层 → F'_c, F'_d_i（世界维度表示）
3. 视觉特征 F_fus → FC → F'_fus（投影到同一世界维度）
4. 余弦相似度计算 + CrossEntropy loss 迫使图文对齐
5. 梯度反向传播 → 同时优化扩散生成和分类网络

### 5.3 性能数据

| 数据集 | OA | AA | Kappa | 训练样本/类 | Training Time | Test Time |
|--------|-----|------|-------|------------|--------------|-----------|
| Trento | 96.36 | 95.12 | 93.81 | 13-18 | 53.6s | 16.4s |
| Houston 2013 | 89.24 | 88.35 | 90.15 | ~20 | 75.3s | 30.1s |
| Houston 2018 | 77.36 | 71.56 | 82.39 | 50 | 200.1s | 66.9s |
| MUUFL | 75.96 | 71.22 | 73.73 | 15 | 61.4s | 18.3s |

**关键特性**：这是五篇论文中**唯一在少样本条件下**测试的方法。

### 5.4 这个范式能否扩展为 "Agent prompt → 融合策略"？

**答案是肯定的**，且改造路径非常直接。

**改造路径1：Agent 生成/替换 Prompt**

```python
# 原始：人工设计的 hard prompt
prompts = {
    "Apple trees": [
        "A HSI-LiDAR data of apple trees",
        "The apple trees appear khaki and green",
        ...
    ]
}

# 改造：Agent 生成实例级 prompt
agent_prompts = agent.generate_prompts(
    context={
        "phenology_stage": "leaf_expansion",     # 物候阶段
        "region": "north_slope",                   # 地形上下文
        "observed_species": "Pinus_sylvestris",    # 树种
        "uncertainty_map": uncertainty_map,         # 不确定性
        "temporal_info": "May_2024"                 # 时间信息
    }
)
# agent_prompts 注入 OK-CP 模块替代/增强人工 prompt
```

**改造路径2：Agent 直接生成 SSM 条件参数**

将 FusDreamer 的扩散 U-Net 与 MSFMamba 的 Fus-Mamba 结合：

```python
# Agent 编码 → 控制扩散过程的噪声调度和生成方向
agent_emb = Encoder(agent_signal)  # Agent 的高级决策

# 修改 LaMG 的去噪步骤
# 原始：x_{t-1} = 1/√α_t (x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t, t))
# 改造：
ε_θ(x_t, t, agent_emb) = UNet(x_t, t) + AgentConditionBranch(x_t, agent_emb)
```

**改造路径3（最优方案）：Agent 作为 "融合策略调度器"**

```python
class AgentControlledFusion:
    def forward(self, hsi, lidar, agent_context):
        # Step 1: Agent 分析上下文，决定融合策略
        fusion_plan = self.agent_planner(agent_context)
        # fusion_plan 包含：
        #   - which_modules: ['diffusion', 'mamba', 'kan']
        #   - diffusion_timesteps: 50 (少样本) or 200 (全量)
        #   - prompt_mode: 'phenology_focused' or 'general'
        
        # Step 2: 动态生成 CLIP prompts
        prompts = self.prompt_generator(fusion_plan, agent_context)
        
        # Step 3: 执行选定的融合路径
        features = self.fusion_router(hsi, lidar, fusion_plan, prompts)
        
        return features
```

**难度评估**：★★★☆☆（中等）
- Prompt 替换直接可用，零模型改动
- 扩散 + Agent 联合优化需要大量显存和训练时间
- 显著优势：Prompt 是可解释的（Agent 说出为什么选择这个策略）

**推理速度**：当前 FusDreamer 推理 16-67s（依赖扩散步数），Agent 逻辑增加 <1s。

---

## 6 综合对比：Agent 可控改造总览

### 6.1 改造难度与效果矩阵

| 方案 | 改造代码量 | 需重训练 | Agent影响精度上限 | 推理速度影响 | 可解释性 |
|------|----------|---------|------------------|------------|---------|
| A: DCMNet路由注入 | ~20行 | 是 | 中 | <+5% | 中（路由路径可视化）|
| B: DFFNet频域注入 | ~15行 | 是 | 低-中 | <+5% | 低（频域不直观）|
| B: IFGNet KAN注入 | ~10行 | 是 | 中 | <+5% | 中（KAN可视化）|
| C: MSFMamba三输入 | ~30行 | 是 | 高 | <+3% | 中（SSM状态可分析）|
| D: FusDreamer Prompt | ~50行 | 否 | 高 | +1-5s | 高（自然语言）|

### 6.2 性能基准速览表

| 方案 | H2013 OA | H2013 Kappa | Trento OA | H2018 OA | Berlin OA | Params | Inference |
|------|----------|-------------|-----------|----------|-----------|--------|-----------|
| DCMNet | 95.11 | 94.69 | 98.96 | 93.27 | — | 3.83M | 0.0097s |
| DFFNet | 92.35 | 91.70 | — | — | 75.42 | 1.28M | 0.2387s |
| IFGNet | **99.37** | **99.32** | — | — | — | (轻量) | (快) |
| MSFMamba | 92.86 | 92.25 | — | 92.38 | 76.92 | 1.53M | 0.1747s |
| FusDreamer | 89.24* | 90.15* | 96.36* | 77.36* | — | (大) | 16-67s* |

> *FusDreamer 为少样本设定（~20/类），其他为全监督（~200/类）；IFGNet 未报告参数量和推理时间。

### 6.3 融合策略自由度对比

```
        Agent可控的融合自由度
        ↑
   High │  ● FusDreamer  (Prompt → 全局融合策略，图文对齐)
        │
        │  ● IFGNet      (KAN 连续隐式聚合 → 类别条件)
Medium  │  ● DCMNet      (动态路由 → 路径选择)
        │  ● MSFMamba    (Fus-SSM 双输入 → 参数生成)
        │
   Low  │  ● DFFNet      (动态滤波核 → 频域选择)
        │
        └──────────────────────────────────────────→ Agent复杂上下文处理能力
           Simple                            Complex
```

### 6.4 推荐工程路线

**阶段1（快速验证，1-2周）**：
- 基于 **FusDreamer** 的 Prompt 注入机制
- Agent 生成物候感知的文本 prompt，零改动接入
- 验证"Agent prompt → 分类精度提升"的因果关系

**阶段2（深度优化，2-4周）**：
- 采用 **MSFMamba** 的三输入 Fus-SSM 改造
- Agent 编码为 SSM 的 B/C/Δ 参数的条件源
- 实现 O(n) 效率的 Agent 可控融合
- 更换 FusDreamer 为更轻量的 CLIP-native 分类

**阶段3（系统集成，4-8周）**：
- 构建 Agent 策略路由器（DCMNet 路由 + 外部控制联合）
- 混合动态路由和 SSM 两大范式
- KAN（IFGNet）作为物候-光谱连续建模的激活层

---

## 7 补充：新论文方向（2024-2026 需搜索的关键词）

### 7.1 高优先级搜索方向

1. **Mixture of Experts (MoE) for multimodal fusion**
   - 搜索：`mixture of experts multimodal fusion remote sensing`
   - 关注：MoE 的 router 本身可接受外部指令
   - 代表性工作可能来自 CV 领域然后在 RS 中迁移

2. **Conditional computation for remote sensing**
   - 搜索：`adaptive modality fusion deep learning remote sensing`
   - 关注：是否已有工作探索"外部条件"控制融合

3. **Uncertainty-aware multimodal fusion**
   - 搜索：`uncertainty-aware multimodal fusion remote sensing`
   - 关注：Bayesian 融合 → Agent 读取不确定性来决定是否采集更多数据

4. **Agent-guided / LLM-guided multimodal fusion**
   - 搜索：`LLM agent multimodal fusion remote sensing`
   - 关注：2025-2026 新趋势：LLM/Agent 直接作为融合调度器

### 7.2 理论支撑方向

- **Gating Networks with context injection** (FiLM, HyperNetworks, AdaIN)
- **Cross-modal prompt tuning** (CoOp, CoCoOp variants applied to RS)
- **Neural Architecture Search (NAS) for fusion** — Agent 等价于动态 NAS
- **Hypernetworks** — 一个小网络生成融合网络的权重，Agent 可以控制这个 Hypernetwork

---

## 8 关键实现细节速查

### 8.1 DCMNet 路由门 PyTorch 伪代码

```python
class RoutingGate(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_next_blocks=3):
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_next_blocks)
    
    def forward(self, F_h, F_l, X):
        # F_h, F_l: encoder features (same shape)
        # X: input from previous layer's blocks
        combined = F_h + F_l + X
        combined_pooled = combined.mean(dim=[-2, -1])  # spatial average
        logits = self.fc2(F.relu(self.fc1(combined_pooled)))
        weights = F.relu(torch.tanh(logits))  # restricted Tanh
        return weights  # [B, 3]
```

### 8.2 DFFNet 动态滤波核生成

```python
class DynamicFilterBlock(nn.Module):
    def __init__(self, num_basis_filters=4, C=64):
        self.basis = nn.Parameter(torch.randn(num_basis_filters, C, H, W))
        self.mlp = nn.Sequential(nn.Linear(C, C//4), nn.ReLU(), nn.Linear(C//4, num_basis_filters))
    
    def forward(self, X):
        X_freq = torch.fft.fft2(X)
        weights = F.softmax(self.mlp(X.mean(dim=[-2,-1])), dim=-1)  # [B, N]
        K = torch.einsum('bn,nchw->bchw', weights, self.basis)  # dynamic kernel
        filtered = X_freq * K
        return torch.fft.ifft2(filtered).real
```

### 8.3 MSFMamba Fus-SSM 的关键修改点

```python
class AgentAwareFusSSM(nn.Module):
    def forward(self, F_h, F_x, agent_ctx=None):
        # 原始：B, C, Δ 仅来自 F_x
        B = self.linear_B(F_x)
        C = self.linear_C(F_x)
        Delta = F.softplus(self.linear_Delta(F_x) + self.Delta_param)
        
        # Agent 注入（新增）
        if agent_ctx is not None:
            agent_emb = self.agent_proj(agent_ctx)
            B = B + self.agent_B(agent_emb)
            C = C + self.agent_C(agent_emb)
            Delta = Delta + F.softplus(self.agent_Delta(agent_emb))
        
        # 标准 selective scan
        A_bar = torch.exp(Delta.unsqueeze(-1) * self.A)
        B_bar = Delta.unsqueeze(-1) * B
        return self.selective_scan(F_h, A_bar, B_bar, C)
```

### 8.4 FusDreamer Agent Prompt 替换

```python
# 原始 prompt 字典硬编码
# 改造为 Agent 动态生成：

class AgentPromptGenerator:
    def generate(self, class_name, phenology_stage, region_info, uncertainty):
        base = f"A HSI-LiDAR multimodal data of {class_name}"
        
        if phenology_stage == "leaf_expansion":
            spectral_hint = "The spectral signature shows increased red-edge reflectance"
        elif phenology_stage == "senescence":
            spectral_hint = "The spectral signature shows decreased NIR and increased red"
        
        if uncertainty > 0.5:
            strategy_hint = "Focus on LiDAR structural features over spectral ambiguity"
        
        return [base, spectral_hint, strategy_hint, region_info]
```

---

## 9 总结

| 维度 | 最佳选择 | 理由 |
|------|---------|------|
| 改动最小、验证最快 | FusDreamer prompt注入 | 零模型改动，Agent输出文本即可 |
| 效率最高 | MSFMamba三输入Fus-SSM | O(n)线性复杂度，参数最少 |
| 精度上限最高 | IFGNet KAN注入 | Houston2013 OA 99.37% |
| 少样本场景 | FusDreamer | 唯一验证少样本的方法 |
| 可解释性最强 | FusDreamer | 自然语言prompt可直接审计 |
| 物候建模 | IFGNet KAN + DFFNet频域 | B-spline连续建模 + 频域物候分析 |
| 综合推荐 | **MSFMamba 骨干 + FusDreamer Prompt 机制** | 效率+可控性的最佳折中 |
