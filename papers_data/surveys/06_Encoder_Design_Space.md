# 06 — ForestPheno Encoder Design Space Survey

> Based on thorough reading of DUNIA (ICML 2025), ESFM Survey (arXiv 2026.05), 5 domain surveys, and systematic comparison of SatMAE / CROMA / AnySat / DOFA / Scale-MAE / DeCUR.

---

## 1. ForestPheno Encoder Design Space Definition

The encoder must satisfy the following conditions (priority-ordered):

| # | Requirement | Description | Hard/Soft |
|---|------------|-------------|-----------|
| 1 | **Pixel-level spatial granularity** | Individual tree crown-level segmentation, at least 10m/pixel | Hard |
| 2 | **Cross-modal fusion** | Must support MSI+SAR+LiDAR alignment across at least 3 modalities | Hard |
| 3 | **Zero-shot / few-shot capability** | Forest ground truth is scarce; performance must hold with <20% labels | Hard |
| 4 | **Temporal phenology awareness** | Leaf-on/off dynamics are core information for species identification | Hard* |
| 5 | **Long-tail / rare class balance** | Natural forests have 18-64 species with severe long-tail distribution | Soft |
| 6 | **Low computational cost** | Single-GPU (A6000/A100) pretraining feasible; supports region-specific retraining | Soft |
| 7 | **Open-source availability** | Code + weights publicly available | Soft |
| 8 | **Interpretability** | Semantic interpretability of embedding space (for Agent quality assessment) | Bonus |

> * Temporal phenology is the only dimension DUNIA does not satisfy, and is the core self-research direction for ForestPheno.

---

## 2. Candidate Methods: 8-Dimension Comparison

### 2.1 Core Metrics Comparison

| Dimension | DUNIA | AnySat | CROMA | SatMAE | DOFA | Scale-MAE | DeCUR |
|-----------|-------|--------|-------|--------|------|----------|-------|
| **Venue** | ICML 2025 | CVPR 2025 | NeurIPS 2023 | NeurIPS 2022 | arXiv 2024 | ICCV 2023 | AAAI 2024 |
| **Pretraining Paradigm** | Contrastive + Recons | Multi-modal fusion encoder | Contrastive + MAE | Masked Reconstruction (MAE) | MAE + Dynamic Weights | MAE + Scale Embed | Contrastive (disentangled) |
| **Modality Support** | S-1+S-2+GEDI LiDAR | S-1+S-2+VHR+Temporal | S-1+S-2 | S-2 | Any optical sensor | Multi-res optical | S-1+S-2 |
| **Output Granularity** | Pixel-level (10m) | Patch-level (adaptable resolution) | Patch-level (8x8) | Patch-level (8x8) | Patch-level | Patch-level | Patch-level |
| **Zero-Shot Capability** | Height RMSE 2.0m, r=0.93; Species wF1=76.0% | — | — | — | — | — | — |
| **Fine-tuned Height** | RMSE 1.3m r=0.95 | RMSE 2.8m r=0.89 | RMSE 3.5m r=0.78 | RMSE 10.5m r=0.52 | RMSE 11.0m r=0.51 | — | RMSE 11.0m r=0.55 |
| **Fine-tuned LCC** | wF1 90.3 | wF1 90.1 | wF1 86.4 | wF1 75.0 | wF1 72.0 | — | wF1 75.1 |
| **Fine-tuned Species** | wF1 82.2 (PF) | wF1 82.3 (PF) | wF1 80.5 (PF) | wF1 78.8 (PF) | wF1 79.8 (PF) | — | wF1 78.9 (PF) |
| **Fine-tuned Crops** | wF1 77.0 (PASTIS) | wF1 81.1 (PASTIS) | wF1 73.3 (PASTIS) | wF1 55.2 (PASTIS) | wF1 54.5 (PASTIS) | — | wF1 57.3 (PASTIS) |
| **20% Label Degradation** | Height unchanged (2.0->2.1); Species improves (76.0->80.1) | — | — | — | — | — | — |
| **Temporal Support** | Limited: single median composite (3-phase AE during pretrain) | Native multi-temporal | Static | Temporal masking | Static | Multi-scale spatial only | Static |
| **Compute Cost** | Single A6000 48GB, 250K steps | Multi-GPU distributed | Medium | ViT-L (~304M params) | ViT-B/ViT-L | ViT-B/ViT-L | Medium |
| **Pretraining Data Volume** | 836K patches, 19M GEDI waveforms | Millions of multi-source images | 100K-level | fMoW-Sentinel ~500K | Millions | Millions | Millions |
| **Code Open-Source** | github.com/AI4Forest/DUNIA | Open source | Open source | Open source | Open source | Open source | Open source |
| **Weights Available** | Pretrained weights | Yes | Yes | Yes | Yes | Yes | Yes |
| **Key Limitation** | Static input, no phenology | Patch output, weak vertical structure | No vertical structure awareness | S-2 only, needs heavy fine-tuning | Terrible vertical structure perf | No cross-modal support | Poor vertical structure |

> All fine-tuning comparison data comes from DUNIA paper Table 2 under unified experimental settings: same pretraining dataset (836K S-1&2 patches + GEDI), 250K steps, same downstream tasks. **This is the fairest apples-to-apples comparison to date.**

### 2.2 Zero-Shot Performance Details (DUNIA only; other methods do not provide zero-shot results)

| Task | Metric | Supervised SOTA | DUNIA Zero-Shot (KNN=50) | Improvement |
|------|--------|-----------------|--------------------------|-------------|
| Forest height (Wrh) | RMSE (r) | 5.2m (r=0.77) | **2.0m (r=0.93)** | +3.2m (+0.16 r) |
| Canopy cover (Wc) | RMSE (r) | 22.1% (r=0.54) | **11.7% (r=0.89)** | +10.4pp (+0.35 r) |
| Plant area index (Wpai) | RMSE (r) | 1.5 (r=0.35) | **0.71 (r=0.75)** | +0.79 (+0.40 r) |
| Land cover (CLC+) | wF1 | — | 80.1% | — |
| Tree species (PF) | wF1 | 74.6% | **76.0% (KNN=5)** | +1.4pp |
| Crop type (PASTIS) | OA | 84.2% | 56.2% | -28pp (phenology gap) |
| GEDI waveform retrieval | r | — | **0.70** | — |

> Zero-shot retrieval database needs only 50K labeled pixels (~31 km^2), approximately **0.25%** of data required by supervised methods.

### 2.3 Label Efficiency Comparison (low-data fine-tuning)

| Model | 20% Labels Wrh | 20% Labels CLC+ | 20% Labels PF |
|-------|----------------|-----------------|---------------|
| DUNIA | RMSE 2.1 (r=0.92) | wF1 89.4 | wF1 80.1 |
| AnySat | RMSE 2.8 (r=0.89) | wF1 89.5 | wF1 80.0 |
| CROMA | RMSE 3.6 (r=0.76) | wF1 85.9 | wF1 80.2 |
| DOFA | RMSE 11.2 (r=0.50) | wF1 71.1 | wF1 79.8 |
| DeCUR | RMSE 11.1 (r=0.52) | wF1 74.6 | wF1 78.5 |
| SatMAE | RMSE 10.5 (r=0.52) | wF1 73.8 | wF1 78.7 |

> DUNIA's unique advantage comes from Zero-CL loss + GEDI waveform alignment — vertical structure information is encoded into pixel embeddings during pretraining.

---

## 3. Detailed Architecture Per Method

### 3.1 DUNIA (ICML 2025, Fayad et al.)

**Encoder Architecture**:
- Backbone: 16-layer ViT Transformer (4 blocks x 4 layers), 8 attention heads, embedding dim=512, FFN hidden=2048, GeGLU activation
- Patch embedding: Conv layer, patch size=8, input 64x64x14 images
- Dual decoders (vertical OV + horizontal OH): hierarchical upsampling (4 decoder blocks), top-resolution layer with Neighborhood Attention (w=19)
- Output projection: 64-dim pixel embeddings (Dp=64)

**Auxiliary Modules**:
- Multi-temporal image AE: UNet + ConvLSTM, 3 time steps (4-month intervals) -> temporal average pooling -> aligned with pretrained model OH
- Waveform AE: ResNet encoder / mirror decoder + RVQ (8 quantizers x 512 codebook) -> channel-wise average pooling -> 64-dim embedding

**Loss Functions**:
1. **Zero-CL** (pixel-waveform alignment): ZCA whitening + instance/feature-dim contrast
   - Reason: ~26 GEDI waveforms per batch; VICReg variance term fails at this scale (positive pair cosine similarity: VICReg 0.56 vs Zero-CL 0.86)
2. **Hierarchical VICReg** (pixel-pixel alignment): variance + invariance + covariance across 4 decoder levels
3. **MSE Reconstruction**: waveform reconstruction + multi-temporal image reconstruction + single-image reconstruction

**Input Specifications**:
- Pretraining: single leaf-on season median composite, 64x64 px, 14 channels (S-2 10 bands + S-1 VV/VH + additional)
- Inference: arbitrary input size (validated at 128/256/512 px), output unaffected

**Training Configuration**:
- GPU: Single A6000 48GB
- Batch size: 60
- Optimizer: Lion, lr=5e-5, weight decay=0.4, 250K steps
- 5K warmup + cosine annealing
- Switch EMA regularization (decay=0.9, sync every 5 steps)

**Inference Speed**: 4.22s @ 20.48x20.48 km (4.2M pixels), KNN=100, database 256K

### 3.2 AnySat (CVPR 2025, Astruc et al.)

**Encoder Architecture**:
- Resolution-adaptive patch embedding: dynamically adjusts patch size and positional encoding based on input GSD
- Cross-modal fusion encoder: multi-source data unified via cross-attention
- Supports S-1/S-2 time series + Spot VHR (1.5m) and other resolutions

**Loss Function**: Self-supervised multi-modal fusion + semantic learning

**Key Strengths**:
- Native multi-resolution support (1.5m VHR to 60m coarse)
- Native multi-temporal support
- Best classification/segmentation overall (PASTIS wF1=81.1, PF wF1=82.3)

**Key Limitations**:
- Patch-level output (not pixel-level); cannot do pixel-level parameter mapping
- Vertical structure estimation significantly weaker than DUNIA (Wrh=2.8 vs 1.3)
- Multi-GPU distributed pretraining; higher reproduction barrier
- Inference ~40x slower than DUNIA (177s vs 4.22s for 20km^2)

### 3.3 CROMA (NeurIPS 2023, Fuller et al.)

**Encoder Architecture**:
- Dual encoders (SAR + optical) + cross-modal contrastive loss
- Auxiliary MAE reconstruction head
- Patch-level embeddings

**Loss Function**: Cross-modal contrast (S-1 <-> S-2) + per-modality MAE reconstruction

**Key Limitations**: Pure 2D image modalities, no vertical structure awareness; patch output is overly smooth, losing spatial detail; no temporal capability.

### 3.4 SatMAE (NeurIPS 2022, Cong et al.)

**Encoder Architecture**:
- ViT-Large with spectral-temporal-aware masking strategy
- Input: S-2 time series (spectral + temporal positional encoding)
- Grouped masking: masks applied separately in temporal and spectral dimensions

**Key Strengths**: First successful MAE adaption for RS temporal data; joint spectral-temporal masking design derived from RS data characteristics.

**Key Limitations**: S-2 optical only; no SAR/LiDAR — forest structure performance is poor (Wrh=10.5m); MAE pretraining requires substantial fine-tuning.

### 3.5 DOFA (arXiv 2024, Xiong et al.)

**Encoder Architecture**:
- ViT-B/ViT-L + dynamic wavelength embedding (maps arbitrary bands to shared encoder via spectral response function)
- Masked reconstruction training

**Key Innovation**: One encoder handles any optical sensor configuration (different band setups)

**Key Limitations**: Optical only; no SAR/LiDAR; in DUNIA's fair comparison, vertical structure performance is extremely poor (Wrh=11.0, r=0.51).

### 3.6 Scale-MAE (ICCV 2023, Reed et al.)

**Encoder Architecture**:
- ViT-B + GSD-encoded positional embedding for resolution
- Mixed multi-resolution training data, differentiated by GSD embedding
- Low-frequency / high-frequency masking (Laplacian pyramid band selection)

**Key Limitations**: Optical single-modal only; no cross-modal/temporal/vertical structure capability.

### 3.7 DeCUR (AAAI 2024, Wang et al.)

**Encoder Architecture**:
- Barlow Twins-based disentangled cross-modal contrastive learning
- Explicit separation of intra-modal and inter-modal contrastive signals
- Curriculum learning strategy

**Key Limitations**: Vertical structure performance not materially different from MAE methods.

---

## 4. Direct Recommendations for ForestPheno

### 4.1 Recommended Choice: DUNIA as Base Encoder

**Justification** (mapped to design space requirements):

| Requirement | DUNIA Status | Explanation |
|-------------|-------------|-------------|
| Pixel-level granularity | Perfect | Only method achieving pixel-level cross-modal embeddings |
| Cross-modal fusion | Perfect | S-1+S-2 <-> GEDI waveform alignment validated |
| Zero/few-shot | Exceptional | Zero-shot beats supervised SOTA; near lossless at 20% labels |
| Temporal phenology | Missing | **Needs self-research**; most critical extension direction |
| Long-tail balance | Needs improvement | Species wF1=76-82% but no special tail-class handling |
| Low compute cost | Satisfied | Single A6000 pretraining; transferable to A100/H100 |
| Open-source | Satisfied | Code + weights at github.com/AI4Forest/DUNIA |
| Interpretability | Moderate | Dual decoders naturally separate vertical/horizontal semantics |

### 4.2 Required Modifications (by priority)

**P0 — Temporal Phenology Alignment** (DUNIA's biggest gap):
- Problem: DUNIA uses single median composite; PASTIS zero-shot OA only 56.2% (28pp below temporal AnySat)
- Option A (low cost): Modify ConvLSTM multi-temporal AE from 3 four-month composites to 6-12 bimonthly composites; add cross-temporal pixel consistency loss before temporal pooling
- Option B (high performance): Replace ViT encoder with temporal Transformer (tube patch embedding like VideoMAE), preserving dual-decoder architecture
- Expected gain: +20pp for crop/species classification; phenology stages become identifiable

**P1 — Long-Tail Balanced Embeddings** (integrate TaxoNet):
- Problem: ForestPheno species show severe long-tail; CE loss head-class dominance
- Solution: Introduce TaxoNet's dual-margin loss in DUNIA's OH embedding space; larger intra-class compactness margin for tail classes
- Key: margin values should dynamically adapt based on taxonomic distance (same genus vs different family)

**P1 — Data-Quality-Aware Dynamic Fusion Routing** (integrate DCMNet):
- Problem: Current DUNIA fusion is static (S-1+S-2 channel stacking), not adaptive to data quality
- Solution: Add lightweight routing module before encoder; dynamically adjust S-1/S-2 channel weights based on cloud probability / saturation / SAR coherence of input patch

**P2 — GEDI Waveform Generation + New Region Generalization**:
- Problem: DUNIA pretrained on France 2020 data; retraining needed for new regions
- Solution: Leverage DUNIA's diffusion waveform generation (r=0.75-0.78); generate synthetic training samples for few-shot domain adaptation

### 4.3 Why Not Other Methods as Base Encoder

| Method | Rejection Reason |
|--------|-----------------|
| **AnySat** | Patch-level output cannot do individual tree crown pixel mapping; weak vertical structure (Wrh=2.8 vs 1.3); inference 40x slower |
| **SatMAE/DOFA** | No vertical structure awareness (Wrh >10m); MAE paradigm needs heavy fine-tuning labels; zero-shot capability = 0 |
| **CROMA/DeCUR** | Poor vertical structure (Wrh >3.5m); no temporal capability; patch-level |
| **Scale-MAE** | Optical single-modal only; no cross-modal/vertical/temporal capability |

**AnySat's Strategic Value**: Use as **complement** to DUNIA, not replacement. AnySat excels at multi-temporal classification (PASTIS 81.1 vs 77.0). Its temporal patch embeddings can serve as temporal enhancement source for DUNIA's vertical structure information.

---

## 5. Required Validation Experiments

### 5.1 If Choosing DUNIA: Must Validate on Following Data

**A. Domain Transfer Experiments on Chinese Forest Data** (highest priority):

| Data Source | Task | Metric | Expected Risk |
|------------|------|--------|---------------|
| China GEDI data (Hainan/Yunnan/Northeast) | Height/canopy cover zero-shot | RMSE / r | Tropical canopy denser; GEDI waveform quality differs |
| China GF series + Sentinel-2 | Species classification fine-tuning | wF1 / tail-class recall | Chinese species not in French training set |
| GF-7 LiDAR + Sentinel | Canopy height calibration | RMSE | Systematic bias across LiDAR systems |

**B. Temporal Enhancement Validation**:

| Experiment | Setting | Baseline |
|-----------|---------|----------|
| DUNIA + 6-phase input vs single-phase | Species classification (PureForest) | wF1 gain |
| Cross-temporal consistency loss vs none | Known-phenology species (deciduous vs evergreen) | Discriminability |
| Temporal DUNIA vs AnySat | PASTIS crop classification | OA |

**C. Long-Tail Enhancement Validation**:

| Experiment | Setting | Baseline |
|-----------|---------|----------|
| DUNIA OH + dual-margin head vs CE head | PureForest 13 classes | Tail-class recall (<=5 samples/class) |
| Norm-guided sampling vs random | PlantD 64 classes | macro-F1 |

**D. LiDAR Modality Coverage Testing**:

| Experiment | Setting | Baseline |
|-----------|---------|----------|
| DUNIA zero-shot height | GEDI density 100% -> 10% -> 1% | RMSE degradation curve |
| Waveform diffusion generation + domain adaptation | Old->new region transfer | r (generated vs measured waveforms) |

### 5.2 Rapid Prototype Validation Route (2 Weeks)

1. **Day 1-2**: Deploy DUNIA pretrained weights -> run zero-shot height on China GEDI region -> establish baseline RMSE
2. **Day 3-5**: Replace single-phase composite with 3 bimonthly Sentinel-2 composites -> run species fine-tuning -> confirm temporal gain
3. **Day 6-7**: Implement dual-margin head -> compare on PureForest tail classes
4. **Day 8-10**: Implement data-quality-aware routing -> A/B test on regions with varying cloud cover
5. **Day 11-14**: Aggregate all results -> produce final design decision document

---

## 6. Appendix: Key References

| Paper | Venue | DOI / ID | Core Contribution |
|-------|-------|---------|-------------------|
| DUNIA | ICML 2025 | arXiv:2502.17066 | Pixel-level cross-modal contrastive alignment |
| AnySat | CVPR 2025 | arXiv:2412.14123 | Multi-resolution multi-modal fusion encoder |
| CROMA | NeurIPS 2023 | arXiv:2311.00566 | Cross-modal radar-optical contrastive learning |
| SatMAE | NeurIPS 2022 | arXiv:2207.08051 | Spectral-temporal masked pretraining |
| DOFA | arXiv 2024 | — | Dynamic wavelength-aware foundation model |
| Scale-MAE | ICCV 2023 | — | Multi-scale masked autoencoder |
| DeCUR | AAAI 2024 | — | Disentangled cross-modal contrastive learning |
| TaxoNet | arXiv 2025 | — | Long-tail fine-grained plant classification |
| ESFM Survey | arXiv 2026 | arXiv:2605.12542 | Earth science foundation model comprehensive review |

> *This report is based on thorough reading of the DUNIA paper (32 pages), ESFM survey (30 pages), 5 domain surveys, and OpenAlex literature search. All performance numbers come from the original papers' experimental results.*
