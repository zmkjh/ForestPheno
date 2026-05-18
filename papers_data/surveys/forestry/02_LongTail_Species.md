# 02 — Long-Tail Tree Species Classification

> **Survey Date**: 2026-05-18
> **Focus**: Long-tail / class-imbalanced tree species classification; cross-modal + long-tail balancing methods
> **Method**: OpenAlex API (3 queries) + arXiv API (4 queries), 2022–2026

---

## 1. Problem Statement

Forest tree species classification presents a canonical **long-tail distribution** challenge:
- **High diversity**: 64–21,000+ species per dataset
- **Extreme skew**: Top-3 species frequently account for 40%+ of all samples
- **Rare species**: Often single-digit samples per class (few-shot / zero-shot regime)
- **High inter-class visual similarity**: Fine-grained discrimination needed between closely related species

Compounding factors unique to forestry:
- Multi-modal data (RGB, multispectral, hyperspectral, LiDAR/ALS, SAR)
- Spatiotemporal domain shift (season, weather, geography)
- Hierarchical taxonomy (family → genus → species) with open-world deployment
- Costly expert annotation at large scales

---

## 2. Datasets

| Dataset | #Species | #Samples | Modality | Long-Tail? | Year | Source |
|---------|----------|----------|----------|-----------|------|--------|
| **StreetTree** | 8,300+ | 12M images | RGB (street-level) | ✅ Explicitly | 2026 | arXiv 2602.19123 |
| **GlobalGeoTree** | 21,001 | 6.3M occurrences | Sentinel-2 + bioclimatic | ✅ (few-shot eval) | 2025 | arXiv 2505.12513 |
| **CU-Tree102** | 102 | Large-scale | RGB | ✅ Explicitly | 2026 | arXiv 2601.16498 |
| **PureForest** | 18 (13 semantic) | 339 km² | ALS point cloud + VHR aerial | Moderate | 2024 | arXiv 2404.12064 |
| **ALS Finland** | 9 | 6,326 segments | Multispectral ALS (3λ) | ✅ (rare species) | 2025 | ISPRS 2026 |
| **Auto-Arborist** | ~300 | Urban trees | RGB (Google Street View) | ✅ | 2022 | Google → TaxoNet |
| **CAPE Floristic** | 639 | 662 sites | Presence-absence | ✅ High skew | 2017 | stat.ME |
| **NAFlora-Mini** | many | Herbarium | Preserved specimen images | ✅ | — | TaxoNet eval |

---

## 3. Methods Taxonomy

### 3.1 Loss Function Alternatives (CE替代)

| Method | Paper | Key Idea | Tail Recall? |
|--------|-------|----------|-------------|
| **Dual-Margin Embedding (TaxoNet)** | arXiv 2512.18994 (2025) | Reshapes class decision boundaries under imbalance via dual-margin objective; strengthens rare-class representation geometry | ✅ Explicit, beats multimodal foundation models |
| **Focal Loss / Class-Balanced Loss** | Various, including Sentinel-2 paper (2408.08887) | DL naturally robust to imbalance vs RF (RF F1=60% → DL F1=80% on 10 species) | Partially |
| **EKDC-Net (Uncertainty-Guided Calibration)** | arXiv 2601.16498 (2026) | Plug-and-play decision calibration: local prior + expert knowledge extraction via CAM, uncertainty-guided correction | ✅ +6.42% accuracy, +11.46% precision with 0.08M params |

### 3.2 Two-Phase / Dual Sampling Strategies

| Method | Paper | Key Idea |
|--------|-------|----------|
| **Knowledge-Driven Pseudo-Labelling + LLM** | arXiv 2604.16115 (2026) | Biologically informed semi-supervised: canopy graph + LLM-derived species cohabitation priors → pseudo-labelling at low training cost. +5.6% over best ref. |
| **Precomputed Deep Embeddings + RF** | arXiv 2508.18829 (2025) | Pre-trained remote sensing model embeddings (Presto, Alpha Earth, Tessera) + few annotated data + RF classifier; outperforms handcrafted features by 2-9pp |
| **Sentinel-2 Time Series DL** | arXiv 2408.08887 (2024) | 10 species, France; MLP/CNN/Transformer on multispectral TS; DL naturally robust to imbalance |

### 3.3 Contrastive / Metric Learning

| Method | Paper | Key Idea |
|--------|-------|----------|
| **GeoTreeCLIP (Vision-Language)** | arXiv 2505.12513 (2025) | Remote sensing + taxonomic text labels in VL framework; zero/few-shot on 10k eval species |
| **Siamese + XAI for Few-Shot** | arXiv 2411.00684 (2024) | MobileNet Siamese, 3-shot → F1=0.86 on invasive trees, UAV imagery, with visual explanations |
| **TaxoNet (Dual-Margin)** | arXiv 2512.18994 (2025) | Embedding learning with dual-margin; outperforms multimodal foundation models on urban + natural + herbarium domains |

### 3.4 Multi-Modal + Long-Tail (Critical Frontier)

| Method | Paper | Cross-Modal? | Long-Tail? | Both? |
|--------|-------|-------------|-----------|-------|
| **GlobalGeoTree (GeoTreeCLIP)** | 2505.12513 | ✅ Vision + Text + Bioclimatic | ✅ Zero/few-shot | ✅ |
| **LLM Pseudo-Labelling** | 2604.16115 | ✅ HSI + ALS + LLM knowledge | ✅ Imbalanced labels handled | ✅ |
| **TaxoNet** | 2512.18994 | ✅ Beats multimodal FMs | ✅ Dual-margin for tail | ✅ |
| **PureForest** | 2404.12064 | ✅ ALS + VHR | ❌ (relatively balanced) | ❌ |
| **ALS Point Transformer** | 2504.14337 | ✅ Multispectral ALS (3λ) | ✅ Validated on rare species | ✅ |
| **MonoCrown (MCrown)** | OpenAlex W7157682284 | ✅ RGB + Monocular Depth | ❌ (focus on crown seg) | ❌ |

---

## 4. Key Finding: "Cross-Modal + Long-Tail" Joined Methods

Three papers explicitly address **both** cross-modal fusion **and** long-tail balancing:

### 4.1 GlobalGeoTree / GeoTreeCLIP (arXiv 2505.12513)
- **Modalities**: Sentinel-2 time series + 27 environmental variables (bioclimatic, geographic, soil) + taxonomic text labels
- **Long-tail**: 21,001 species, zero/few-shot evaluation on 10k species subset
- **Method**: Vision-Language pretraining (CLIP-style) with remote sensing + text, then zero/few-shot transfer
- **Key insight**: VL paradigm inherently addresses open-world / long-tail via text-driven classification

### 4.2 TaxoNet — Dual-Margin Embedding (arXiv 2512.18994)
- **Modalities**: Evaluated against multimodal foundation models across 3 plant datasets (urban trees, iNaturalist Plantae, herbarium)
- **Long-tail**: Theoretically grounded dual-margin objective reshapes decision boundaries for rare classes
- **Key insight**: Single method that works across heterogeneous domains (urban/natural/herbarium) while handling long-tail

### 4.3 LLM-Driven Pseudo-Labelling (arXiv 2604.16115)
- **Modalities**: Hyperspectral Imaging (HSI) + Airborne Laser Scanning (ALS) + LLM-derived ecological priors
- **Long-tail**: Semi-supervised pseudo-labelling over canopy graph handles limited labels; ecological cohabitation priors from LLMs
- **Key insight**: LLM as external knowledge source for ecological plausibility constraints → addresses both multimodality and label scarcity

---

## 5. Summary of Results

| Paper | Dataset Scale | Best Metric | Tail Improvement |
|-------|--------------|-------------|-----------------|
| EKDC-Net (2601.16498) | CU-Tree102 (102 spp) | +6.42% acc, +11.46% precision | Explicit long-tail correction |
| TaxoNet (2512.18994) | 3 plant datasets | Consistent outperformance vs multimodal FMs | Rare-class representation geometry |
| Sentinel-2 DL (2408.08887) | 10 spp, France | OA 95%, F1-macro 80% vs RF 60% | DL naturally robust |
| LLM Pseudo-Lab (2604.16115) | Real forest | +5.6% over best reference | Semi-supervised handles limited labels |
| ALS PT (2504.14337) | 9 spp, Finland | OA 92.0%, macro 85.1% | Point transformer best for rare spp |
| Few-Shot Invasive (2411.00684) | 3-shot | F1=0.86 | Siamese + XAI |
| StreetTree (2602.19123) | 8,300 spp | Baselines established | Long-tailed benchmark |

---

## 6. Research Gap Analysis

### What Exists
1. **Pure long-tail methods**: TaxoNet, EKDC-Net — strong but mostly single-modal RGB
2. **Pure cross-modal methods**: PureForest, ALS benchmarks — good fusion but ignore imbalance
3. **Vision-Language approaches**: GeoTreeCLIP — inherently handles open-world but not optimized for forestry skew

### What's Missing
1. **True "Cross-modal + Long-tail" joint optimization**: No method simultaneously optimizes (a) multi-modal fusion loss, (b) long-tail rebalancing loss, and (c) forestry-specific hierarchical loss
2. **Remote sensing modality fusion for tail species**: Most multi-modal work (HSI+ALS, Sentinel-2+SAR) targets balanced datasets
3. **Self-supervised pretraining with long-tail awareness for forestry**: No large-scale self-supervised pretraining specifically designed for imbalanced tree species distributions
4. **Benchmarks that stress-test both axes simultaneously**: StreetTree is closest but focuses on single-modal RGB

---

## 7. Downloaded Papers

| File | Title | Year |
|------|-------|------|
| `2601.16498_EKDC_TreeSpecies.pdf` | Expert Knowledge-Guided Decision Calibration (EKDC-Net) | 2026 |
| `2602.19123_StreetTree.pdf` | StreetTree: Global Benchmark for Fine-Grained Tree Species | 2026 |
| `2512.18994_TaxoNet_DualMargin.pdf` | Dual-Margin Embedding (TaxoNet) | 2025 |
| `2408.08887_Sentinel2_Imbalanced.pdf` | Tree species classification with DL in imbalanced context | 2024 |
| `2604.16115_LLM_TreePseudoLabel.pdf` | LLM Knowledge-Driven Pseudo-Labelling | 2026 |
| `2505.12513_GlobalGeoTree.pdf` | GlobalGeoTree: Vision-Language Dataset (GeoTreeCLIP) | 2025 |
| `2504.14337_ALS_Benchmark.pdf` | Multispectral ALS Benchmark (Point Transformer) | 2025 |
| `2411.00684_FewShot_InvasiveTrees.pdf` | Explainable Few-Shot for Invasive Trees | 2024 |
| `2404.12064_PureForest.pdf` | PureForest: ALS + VHR Aerial Dataset | 2024 |
| `2508.18829_DutchNFI_Embeddings.pdf` | Deep Embeddings for Dutch NFI Classification | 2025 |

---

## 8. Recommended Next Steps

1. **Read TaxoNet (2512.18994)** in detail — the dual-margin framework is directly applicable to multi-modal extension
2. **Read EKDC-Net (2601.16498)** — plug-and-play architecture can be ported to multi-modal backbones
3. **Design experiment**: Take GeoTreeCLIP-style VL pretraining + add TaxoNet-style dual-margin loss for tail classes + fuse ALS/LiDAR data as second modality
4. **Benchmark**: Evaluate on StreetTree (long-tail) and PureForest (multi-modal) simultaneously

---

*Generated via OpenAlex API + arXiv API search, 2026-05-18*
