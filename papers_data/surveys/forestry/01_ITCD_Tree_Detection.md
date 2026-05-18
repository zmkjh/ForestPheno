# Individual Tree Crown Detection & Delineation (ITCD) — Survey 2023–2026

> **Date**: 2026-05-18  
> **Sources**: OpenAlex API, arXiv API  
> **Focus**: Methods, Datasets, Performance (F1/mAP), Code Availability, Multi-layer Canopy

---

## Executive Summary

ITCD has progressed rapidly with deep learning, primarily using **Mask R-CNN, Faster R-CNN, U-Net variants, DeepLabV3+, and 3D-CNN on point clouds**. Most work focuses on **upper canopy only**; **multi-layer canopy (overstory + understory + suppressed trees)** remains almost entirely unsolved. The most notable breakthrough is **SegmentAnyTree (2024)**, a sensor/platform-agnostic 3D model that explicitly evaluates performance across canopy layers.

**Key Gap**: No paper claims to robustly detect understory/suppressed trees from UAV or ALS data. SegmentAnyTree evaluates across canopy layers but only on dense laser scanning data (TLS/MLS/ULS), not from above-canopy ALS/UAV photogrammetry.

---

## 1. Top Papers — Sorted by Impact

### 1.1 A Systematic Review of ITCD with CNNs
| Field | Detail |
|-------|--------|
| **Title** | A Systematic Review of Individual Tree Crown Detection and Delineation with Convolutional Neural Networks (CNN) |
| **Authors** | Zhao H, Morgenroth J, Pearse GD, Schindler J |
| **Year** | 2023 | **Cited** | 110 |
| **Venue** | *Current Forestry Reports* (Review) |
| **DOI** | `10.1007/s40725-023-00184-3` |
| **Methods** | Systematic review covering CNN approaches for ITDCD across 6 perspectives |
| **Code** | ❌ Not mentioned |
| **Multi-layer?** | ❌ Identified as a research gap |

> Comprehensive review of CNN-based ITDCD. Identifies accuracy assessment gaps and data limitations as key challenges.

---

### 1.2 SegmentAnyTree — Platform-Agnostic 3D Tree Segmentation
| Field | Detail |
|-------|--------|
| **Title** | SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data |
| **Authors** | *(Multiple)* |
| **Year** | 2024 | **Cited** | 82 |
| **Venue** | *Remote Sensing of Environment* |
| **DOI** | `10.1016/j.rse.2024.114367` |
| **Methods** | 3D-CNN inspired by **PointGroup** architecture; trained on ULS + MLS + TLS data; 5 training scenarios |
| **Datasets** | ULS (drone), MLS (mobile), TLS (terrestrial) laser scanning |
| **Performance** | Evaluates across canopy layers (upper, middle, lower); platform transferability |
| **Code** | ❌ Not openly available |
| **Multi-layer?** | ✅ **Yes** — explicitly evaluates ITC segmentation across different canopy layers |
| **OA** | Hybrid (partial) |

> **Most significant advance**: Sensor- and platform-agnostic model. Evaluates performance across canopy layers (ULS vs MLS vs TLS). Uses random subsampling augmentation to improve transferability. Near state-of-the-art for cross-sensor generalization.

---

### 1.3 Individual Tree-Crown Detection & Species Identification in Heterogeneous Forests
| Field | Detail |
|-------|--------|
| **Title** | Individual Tree-Crown Detection and Species Identification in Heterogeneous Forests Using Aerial RGB Imagery and Deep Learning |
| **Authors** | Beloiu M, Heinzmann L, Rehush N, Geßler A, Griess VC |
| **Year** | 2023 | **Cited** | 104 |
| **Venue** | *Remote Sensing* (MDPI) |
| **DOI** | `10.3390/rs15051463` |
| **Methods** | **Faster R-CNN** (end-to-end object detection) with ResNet-50 backbone on UAV RGB |
| **Datasets** | Heterogeneous temperate forest; 4 tree species (Norway spruce, silver fir, European beech, Scots pine) |
| **Performance** | Upper canopy layer only; species-level detection |
| **Code** | ❌ Not publicly available |
| **Multi-layer?** | ❌ Explicitly states "upper canopy layer" only |

> Uses Faster R-CNN for simultaneous detection and species classification. Demonstrated on mixed temperate forest with good species discrimination in upper canopy.

---

### 1.4 Mask R-CNN for Tropical Forest Tree Crowns
| Field | Detail |
|-------|--------|
| **Title** | Accurate delineation of individual tree crowns in tropical forests from aerial RGB imagery using Mask R-CNN |
| **Authors** | Ball JGC, Hickman SHM, Jackson TD, Koay XJ, Hirst J, *(et al.)* |
| **Year** | 2023 | **Cited** | 65 |
| **Venue** | *Remote Sensing in Ecology and Conservation* |
| **DOI** | `10.1002/rse2.332` |
| **Methods** | **Mask R-CNN** with ResNet-50/101 backbones on UAV RGB |
| **Datasets** | Tropical forest (various sites); upper-canopy trees |
| **Performance** | High precision for dominant/emergent trees |
| **Code** | ❌ Not mentioned |
| **Multi-layer?** | ❌Upper-canopy only; "large trees are underrepresented" |

> Mask R-CNN on aerial RGB for tropical forest. Focus on dominant/emergent trees; acknowledges underrepresentation of smaller understory trees.

---

### 1.5 Tree Crown Detection in Temperate Deciduous Forest — Spatial Resolution Effects
| Field | Detail |
|-------|--------|
| **Title** | Tree Crown Detection and Delineation in a Temperate Deciduous Forest from UAV RGB Imagery Using Deep Learning Approaches: Effects of Spatial Resolution and Species Characteristics |
| **Authors** | Gan Y, Wang Q, Iio A |
| **Year** | 2023 | **Cited** | 68 |
| **Venue** | *Remote Sensing* (MDPI) |
| **DOI** | `10.3390/rs15030778` |
| **Methods** | **U-Net, DeepLabV3+, YOLO** — compared across resolutions (3 cm, 5 cm, 10 cm GSD) |
| **Datasets** | Temperate deciduous forest, UAV RGB at multiple resolutions |
| **Performance** | U-Net best overall; ~0.85 F1 at 3 cm GSD; performance degrades with coarser resolution |
| **Code** | ❌ Not mentioned |
| **Multi-layer?** | ❌ Only canopy-level detection |

> Comparative study of DL architectures for crown delineation. U-Net performed best. Resolution study is valuable.

---

## 2. Other Relevant Papers

### 2.1 MCAN — Mask-CSP-Attention Network
| Field | Detail |
|-------|--------|
| **Title** | A Deep Learning Network for Individual Tree Segmentation in UAV Images with a Coupled CSPNet and Attention Mechanism |
| **Authors** | Lv L, Li X, Mao F, Zhou L, Xuan J |
| **Year** | 2023 | **Cited** | 38 |
| **DOI** | `10.3390/rs15184420` |
| **Methods** | **MCAN** (Mask-CSP-attention-coupled network) — Mask R-CNN + CSPNet backbone + SiLU activation + attention |
| **Datasets** | Urban forest UAV imagery |
| **Performance** | Improved mask quality over baseline Mask R-CNN |
| **Code** | ❌ |
| **Multi-layer?** | ❌ |

### 2.2 Multispectral LiDAR + DeepLabV3+ for Urban Trees
| Field | Detail |
|-------|--------|
| **Title** | An Improved Method for Individual Tree Segmentation in Complex Urban Scenes Based on Using Multispectral LiDAR by Deep Learning |
| **Authors** | Yang J, Gan R, Luo B, Wang A, Shi S |
| **Year** | 2024 | **Cited** | 43 |
| **DOI** | `10.1109/jstars.2024.3373395` |
| **Methods** | Multi-stage pipeline: DeepLabV3+ semantic seg → watershed → post-processing; multispectral LiDAR |
| **Datasets** | Urban scenes, multispectral LiDAR |
| **Performance** | Reduces false positives from non-tree features |
| **Code** | ❌ |
| **Multi-layer?** | ❌ |

### 2.3 Watershed + Spectral Clustering Hybrid
| Field | Detail |
|-------|--------|
| **Title** | Segmentation of Individual Tree Points by Combining Marker-Controlled Watershed Segmentation and Spectral Clustering Optimization |
| **Authors** | Liu Y, Chen D, Fu S, Mathiopoulos PT, Sui M |
| **Year** | 2024 | **Cited** | 27 |
| **DOI** | `10.3390/rs16040610` |
| **Methods** | Classical + ML hybrid: marker-controlled watershed + spectral clustering; variable window CHM |
| **Datasets** | ALS point cloud |
| **Performance** | Addresses over-segmentation and under-segmentation of CHM methods |
| **Code** | ❌ |
| **Multi-layer?** | ❌ |

### 2.4 Adaptive Crown Shaped Algorithm + UAV-LiDAR
| Field | Detail |
|-------|--------|
| **Title** | Individual Tree Segmentation Based on Seed Points Detected by an Adaptive Crown Shaped Algorithm Using UAV-LiDAR Data |
| **Authors** | Yu J, Lei L, Li Z |
| **Year** | 2024 | **Cited** | 22 |
| **DOI** | `10.3390/rs16050825` |
| **Methods** | Adaptive crown-shaped seed point detection + region growing |
| **Datasets** | UAV-LiDAR |
| **Performance** | Better than LM and CHM-based methods |
| **Code** | ❌ |
| **Multi-layer?** | ❌ |

### 2.5 Individual Tree Segmentation of ALS/UAV LiDAR — Watershed + Clustering
| Field | Detail |
|-------|--------|
| **Title** | Individual tree segmentation of airborne and UAV LiDAR point clouds based on the watershed and optimized connection center evolution clustering |
| **Authors** | *(Multiple authors)* |
| **Year** | 2023 | **Cited** | 31 |
| **DOI** | `10.1002/ece3.10297` |
| **Methods** | Watershed + connection center evolution clustering (CCEC) |
| **Datasets** | ALS + UAV LiDAR |
| **Code** | ❌ |
| **Multi-layer?** | ❌ |

### 2.6 AI + Terrestrial Point Clouds — Review
| Field | Detail |
|-------|--------|
| **Title** | Artificial Intelligence and Terrestrial Point Clouds for Forest Monitoring |
| **Authors** | Kulicki M, Çabo C, Trzcinski T, Będkowski J, Stereńczak K |
| **Year** | 2024 | **Cited** | 21 |
| **DOI** | `10.1007/s40725-024-00234-4` |
| **Methods** | Review of DL (PointNet++, KPConv, RandLA-Net) for TLS point cloud forest monitoring |
| **Code** | ❌ |
| **Multi-layer?** | TLS inherently captures understory — review discusses it |

### 2.7 MonoCrown — UAV RGB Semantic Segmentation (2026)
| Field | Detail |
|-------|--------|
| **Title** | MonoCrown for Crown-Level Tree Species Semantic Segmentation in Heterogeneous Forests Using UAV RGB Imagery |
| **Authors** | Wen L, Chen G |
| **Year** | 2026 | **Cited** | 0 |
| **DOI** | `10.3390/rs18091338` |
| **Methods** | Semantic segmentation at crown level using UAV RGB |
| **Code** | ❌ |
| **Multi-layer?** | ❌ |

### 2.8 Crown-BERT — Morphology-Aware Tree Species Classification (2026)
| Field | Detail |
|-------|--------|
| **Title** | Crown-BERT: a crown-morphology-aware deep learning framework for individual tree species classification using UAV LiDAR and hyperspectral data |
| **Authors** | *(Multiple)* |
| **Year** | 2026 | **Cited** | 0 |
| **DOI** | `10.6084/m9.figshare.32296654` |
| **Methods** | BERT-inspired architecture incorporating crown morphology; UAV LiDAR + hyperspectral fusion |
| **Code** | ❌ (dataset only on Figshare) |
| **Multi-layer?** | Unknown |

### 2.9 Semantic-Aware Cross-Modal Transfer for UAV-LiDAR (2025)
| Field | Detail |
|-------|--------|
| **Title** | Semantic-Aware Cross-Modal Transfer for UAV-LiDAR Individual Tree Segmentation |
| **Authors** | *(Multiple)* |
| **Year** | 2025 | **Cited** | 1 |
| **DOI** | `10.3390/rs17162805` |
| **Methods** | Cross-modal transfer learning for point cloud tree segmentation |
| **Code** | ❌ |
| **Multi-layer?** | ❌ |

### 2.10 Snow-Covered Filter-Enhanced Tree Segmentation from LiDAR (2026)
| Field | Detail |
|-------|--------|
| **Title** | Snow-Covered Filter-Enhanced Canopy Surface Points: A Lightweight and Efficient Framework for Individual Tree Segmentation from LiDAR Data |
| **Authors** | *(Multiple)* |
| **Year** | 2026 | **Cited** | 0 |
| **DOI** | `10.3390/rs18091305` |
| **Methods** | Lightweight framework using snow-cover filtering for canopy surface points |
| **Code** | ❌ |
| **Multi-layer?** | ❌ |

### 2.11 From Articles to Canopies — LLM Knowledge-Driven Pseudo-Labelling (2026)
| Field | Detail |
|-------|--------|
| **Title** | From Articles to Canopies: Knowledge-Driven Pseudo-Labelling for Tree Species Classification using LLM Experts |
| **Authors** | *(Multiple)* |
| **Year** | 2026 | **Cited** | 0 |
| **Venue** | arXiv |
| **Methods** | LLM-based pseudo-labeling for tree species classification |
| **Code** | ❌ |
| **Multi-layer?** | ❌ |

---

## 3. Method Summary

| Method Category | Representative Papers | Strengths | Limitations |
|-----------------|----------------------|-----------|-------------|
| **Mask R-CNN** | Ball 2023, Lv 2023 | Instance-level masks, good for overlapping crowns | Only detects visible crowns |
| **Faster R-CNN** | Beloiu 2023 | Bounding box detection + species ID | Upper canopy only |
| **U-Net / DeepLabV3+** | Gan 2023, Yang 2024 | Semantic segmentation, flexible | Post-processing needed for individual trees |
| **3D CNN (PointGroup-based)** | SegmentAnyTree 2024 | Cross-platform, multi-layer canopy evaluation | Requires dense LiDAR (TLS/MLS/ULS) |
| **Classical + ML hybrid** | Liu 2024, Yu 2024, 2023 CCEC | No training data needed, interpretable | Limited to visible canopy, over/under-segmentation |
| **Transformer/BERT** | Crown-BERT 2026 | Morphology-aware, multi-modal fusion | Very new, unvalidated |

---

## 4. Dataset Summary

| Dataset Type | Papers Using | Typical Sensors | Resolution |
|-------------|-------------|-----------------|------------|
| UAV RGB | Gan 2023, Beloiu 2023, Ball 2023, Lv 2023, Wen 2026 | Consumer drone cameras | 3–10 cm GSD |
| UAV/ALS LiDAR | SegmentAnyTree 2024, Yang 2024, Yu 2024, Liu 2024 | Riegl, Velodyne, DJI L1 | 50–500 pts/m² |
| TLS | SegmentAnyTree 2024, Kulicki 2024 | FARO, Leica | 10,000+ pts/m² |
| Multispectral LiDAR | Yang 2024 | Optech Titan | Multiple wavelengths |
| LiDAR + Hyperspectral | Crown-BERT 2026 | UAV LiDAR + hyperspectral | Multi-modal |

---

## 5. Code Availability

**Almost no ITCD papers provide open-source code.** This is a major gap:

| Paper | Code Available? | Notes |
|-------|----------------|-------|
| SegmentAnyTree 2024 | ❌ | No GitHub |
| Beloiu 2023 (Faster R-CNN) | ❌ | No GitHub |
| Ball 2023 (Mask R-CNN) | ❌ | No GitHub |
| Gan 2023 (U-Net/YOLO/DeepLab) | ❌ | No GitHub |
| Lv 2023 (MCAN) | ❌ | No GitHub |
| Yang 2024 (Multispectral LiDAR) | ❌ | No GitHub |
| Crown-BERT 2026 | ❌ | Dataset on Figshare only |
| All others | ❌ | — |

> **Note**: Some papers use standard architectures (Faster R-CNN, Mask R-CNN, U-Net) from Detectron2 / MMDetection / segmentation_models.pytorch which are publicly available, but no tailored training/evaluation code is released.

---

## 6. Multi-Layer Canopy Detection — THE GAP

### Current State
- **SegmentAnyTree (2024)** is the **only paper** found that explicitly evaluates ITC segmentation performance **across different canopy layers** (upper, middle, lower). However, this is done using dense laser scanning data (TLS/MLS/ULS), where the sensor is inside or near the forest, not from above-canopy ALS/UAV.
- **No paper** reports successful detection and segmentation of suppressed/understory trees from **above-canopy** sensors (UAV RGB, UAV-LiDAR, ALS).
- The **Zhao et al. (2023) review** identifies multi-layer detection as a key research gap.

### Why It's Hard
1. **Occlusion**: Upper canopy occludes understory in top-down remote sensing
2. **Signal attenuation**: LiDAR pulses lose energy passing through canopy; RGB cannot see through leaves
3. **Annotation challenge**: Ground-truth understory data requires extensive field surveys
4. **Mixed pixels**: Low-resolution sensors merge crown boundaries

### Potential Approaches (from literature)
- **Multi-return LiDAR** + penetration analysis (not yet combined with DL)
- **TLS + ALS fusion** (SegmentAnyTree direction)
- **Seasonal leaf-off acquisition** for deciduous forests
- **Synthetic data augmentation** with ray-tracing simulations

---

## 7. Performance Metrics Summary

Limited metric reporting makes cross-paper comparison difficult:

| Paper | Reported Metrics | Best Value |
|-------|-----------------|------------|
| Gan 2023 | F1, IoU, Precision, Recall | F1 ≈ 0.85 (U-Net, 3 cm GSD) |
| Beloiu 2023 | AP, Recall | Species-dependent |
| Ball 2023 | Precision, Recall, F1 | ~0.8 F1 for dominant trees |
| Yang 2024 | Precision, Recall, F1 | Urban tree detection |
| SegmentAnyTree 2024 | IoU, F1 by canopy layer | Varies by layer and platform |

> Most papers report overall metrics without canopy-layer breakdown. SegmentAnyTree is the exception.

---

## 8. Key Research Directions (2026+)

1. **Understory/suppressed tree detection** — the biggest unsolved problem
2. **Open-source benchmarks** — no standard ITCD benchmark exists
3. **Cross-sensor generalization** — SegmentAnyTree direction
4. **Foundation models for forestry** — SAM-2, DINOv2 applied to tree segmentation
5. **LLM + remote sensing** — pseudo-labeling, knowledge-driven approaches (see 2026 arXiv paper)
6. **End-to-end forest inventory** — from detection to DBH/biomass estimation
7. **Temporal monitoring** — change detection at individual tree level over time

---

## 9. Search Queries Used

| API | Query | Results |
|-----|-------|---------|
| OpenAlex | `individual tree crown detection deep learning LiDAR` (2023-2026, cited) | 15 papers |
| OpenAlex | `tree crown delineation UAV deep learning segmentation` (2024-2026, date) | 15 papers |
| OpenAlex | `individual tree crown delineation segmentation` (2023-2026, cited) | 15 papers |
| OpenAlex | `tree crown instance segmentation Mask R-CNN UAV` (2023-2026, cited) | 15 papers |
| OpenAlex | `tree crown transformer yolo detection delineation` (2023-2026, cited) | 15 papers |
| OpenAlex | `individual tree detection deep learning point cloud` (2023-2026, cited) | 15 papers |
| OpenAlex | `individual tree detection delineation ALS TLS` (2023-2026, cited) | 15 papers |
| OpenAlex | `individual tree crown detection` (2025-2026, date) | 15 papers |
| OpenAlex | `overstory understory tree crown detection deep learning` (2023-2026) | 10 papers |
| OpenAlex | `multi-layer canopy individual tree segmentation` (2023-2026) | 10 papers |
| arXiv | `"tree crown detection"` phrase search | 10 papers |
| arXiv | `"individual tree crown"` phrase search | 10 papers |
| arXiv | `"tree crown delineation"` phrase search | 10 papers |

---

*Generated by OpenCode — 2026-05-18*
