# 03 — Forest 3D Structure Phenotyping

> **Survey Date**: 2026-05-18
> **Focus**: Extraction of 3D structural parameters (tree height, DBH, biomass, volume, crown cover) from point clouds; joint species + structure prediction; cross-scale upscaling
> **Method**: OpenAlex API (6 queries), 2022–2026, sorted by citation count
> **Note**: These tasks have **no direct agricultural counterpart** — agricultural phenotyping measures canopy cover/plant height at plot level, not individual-tree geometry with woody biomass allometry.

---

## 1. Problem Statement

Forest 3D structure phenotyping aims to extract **individual-tree-level geometric and biophysical parameters** from remote sensing point clouds (ALS, TLS, UAV-LiDAR, spaceborne GEDI). Unlike agricultural phenotyping (which targets homogeneous crop plots), forestry faces:

- **Individual-tree geometry** in dense, overlapping canopies
- **Woody biomass allometry** requiring DBH → volume → biomass chains
- **Multi-species heterogeneity** in mixed natural forests
- **3D structural complexity**: understory layers, branches, dead trees
- **Scale gap**: single tree → plot (0.1 ha) → stand (1–100 ha) → landscape → regional

Key parameters extractable from point clouds:

| Parameter | Abbrev. | Direct from PC? | Allometry Needed? | Typical Sensor |
|-----------|---------|-----------------|-------------------|----------------|
| **Tree height** | H | Yes (canopy top − ground) | No | ALS, UAV-LiDAR, TLS, GEDI |
| **Diameter at breast height** | DBH | Partially (TLS only) | Yes (H-D allometry for ALS) | TLS, close-range photogrammetry |
| **Crown area / diameter** | CA / CD | Yes | No | ALS, UAV RGB, UAV-LiDAR |
| **Crown base height** | CBH | Yes | No | ALS, TLS |
| **Above-ground biomass** | AGB | No (indirect) | Yes (H+DBH → volume → biomass) | ALS, TLS, GEDI |
| **Stem volume** | V | Partially (TLS-QSM) | Yes (taper equations) | TLS |
| **Canopy cover / LAI** | CC / LAI | Yes | No | ALS, GEDI |
| **Vertical profile / stratification** | — | Yes | No | ALS (waveform), GEDI (RH metrics) |
| **Bark roughness / stem curve** | — | TLS only | No | TLS |

---

## 2. Key Review Papers

### 2.1 Foundational Reviews

| Paper | Year | Venue | Cites | Focus |
|-------|------|-------|-------|-------|
| **Fassnacht et al.** — "Remote sensing in forestry: current challenges, considerations and directions" | 2023 | *Forestry* | 295 | Comprehensive RS overview; ALS standard, TLS emerging, GEDI promising for global scale |
| **Tian et al.** — "Review of Remote Sensing-Based Methods for Forest Aboveground Biomass Estimation: Progress, Challenges, and Prospects" | 2023 | *Forests* | 177 | Systematic AGB estimation review; allometric equation uncertainty = dominant error source |
| **Balestra et al.** — "LiDAR Data Fusion to Improve Forest Attribute Estimates: A Review" | 2024 | *Current Forestry Reports* | 90 | Multi-source LiDAR fusion (ALS+TLS, ALS+GEDI); gains in AGB accuracy via vertical profile enrichment |
| **Ecke et al.** — "UAV-Based Forest Health Monitoring: A Systematic Review" | 2022 | *Remote Sensing* | 245 | UAV platforms for forest health, stress detection, structural parameter extraction |
| **Zhao et al.** — "A Systematic Review of Individual Tree Crown Detection and Delineation with Convolutional Neural Networks (CNN)" | 2023 | *Current Forestry Reports* | 110 | CNN-based ITCD; Mask R-CNN and U-Net dominate; key challenge = dense overlapping crowns |
| **Murtiyoso et al.** — "Virtual forests: a review on emerging questions in the use and application of 3D data in forestry" | 2023 | *Int. J. Forest Engineering* | 30 | 3D reconstruction, visualization, digital twins; full pipeline from acquisition to application |
| **Borsah et al.** — "LIDAR-Based Forest Biomass Remote Sensing: A Review of Metrics, Methods, and Assessment Criteria for the Selection of Allometric Equations" | 2023 | *Forests* | 41 | Critical review of LiDAR metrics + allometric equations for AGB; allometry as bottleneck |
| **Estrada et al.** — "Machine learning assisted remote forestry health assessment: a comprehensive state of the art review" | 2023 | *Frontiers in Plant Science* | 41 | ML for moisture, chlorophyll, nitrogen, canopy degradation estimation from RS |

### 2.2 Key Distinction from Agricultural Phenotyping

Agricultural phenotyping extracts:
- Plot-level canopy cover, plant height, leaf area index
- Organ-level traits (leaf count, panicle detection)
- Single-species, homogeneous stands, known planting dates

Forest 3D phenotyping extracts:
- Individual-tree geometry with **woody biomass allometry linking DBH → volume → biomass**
- Multi-species discrimination in heterogeneous canopies
- **Vertical stratification** (overstory, midstory, understory)
- Dead tree / snag detection for mortality monitoring

---

## 3. Parameter-by-Parameter: State of the Art

### 3.1 Tree Height (H)

**Direct point cloud → height**: The most mature task. ALS provides RMSE = 0.5–2.0 m at individual-tree level; GEDI provides RH100 (relative height 100th percentile) at 25 m footprint.

| Method | Sensor | Scale | R² | RMSE (m) | Key Paper |
|--------|--------|-------|-----|----------|-----------|
| **Global canopy height model (ETH Zurich)** | Sentinel-2 + GEDI training | Global (1 m res) | — | 6–8 m MAE at 1 m | Lang et al. 2023, *Nature Eco Evo* (545 cites) |
| **Hy-TeC (Hybrid Vision Transformer)** | GEDI + Sentinel-1/2 | Continental | R²=0.67 | 7.3 m | Fayad et al. 2023, *RSE* (79 cites) |
| **UAV-LiDAR individual tree** | UAV-LiDAR (Riegl miniVUX) | Plot (< 1 ha) | R²>0.95 | < 1.0 m | Various |
| **TLS individual tree** | TLS (FARO/Leica) | Plot (0.01 ha) | R²>0.99 | < 0.1 m | Wilkes et al. 2023, TLS2trees |
| **GEDI L2A RH100** | Spaceborne GEDI | Global sample | R²=0.85–0.90 | 3–4 m vs ALS | Dubayah et al. 2022 |

**Gap**: UAV-LiDAR achieves < 1 m RMSE for individual tree heights, but the 3D point cloud → CHM → ITD → height extraction pipeline is **error-prone in dense stands with overlapping crowns**.

### 3.2 Diameter at Breast Height (DBH)

**The hardest parameter to extract from above-canopy LiDAR.** DBH at 1.3 m above ground is typically **occluded** in ALS/UAV-LiDAR unless the understory is sparse.

| Approach | Sensor | R² | RMSE (cm) | Key Paper |
|----------|--------|-----|-----------|-----------|
| **Direct circle fitting (TLS)** | TLS | R²>0.98 | < 1.0 cm | Laino et al. 2024, 3DFin |
| **H-D allometry from ALS** | ALS → H → species-specific equation → DBH | R²=0.70–0.85 | 3–8 cm | Standard in NFI workflows |
| **BSTDF: BLS trunk detection + DL** | Backpack LiDAR (BLS) | R²=0.95 | 2.01 cm | Zhang et al. 2023, *Remote Sensing* |
| **Drone LiDAR trunk section** | UAV-LiDAR (under-canopy) | R²=0.89 | 1.8 cm | Zhang et al. 2023, *Forest Ecosystems* |
| **Multi-view photogrammetry** | UAV RGB (SfM) | R²=0.75–0.85 | 3–5 cm | Gan et al. 2023, *Remote Sensing* |

**Key insight**: DBH can ONLY be measured directly when the trunk at 1.3 m height is laser-scanned (TLS, BLS, or low-altitude drone LiDAR under canopy). For standard ALS/UAV-LiDAR (top-down), **species-specific H-D allometry is mandatory**, introducing the dominant error source.

### 3.3 Above-Ground Biomass (AGB)

**AGB is the ultimate target parameter but is NEVER direct from point clouds.** The chain is:

```
Point cloud → H, CD, crown volume → DBH (allometry) → Stem volume (taper) → AGB (wood density × BEF)
```

Error propagation: each step adds uncertainty. Tian et al. (2023) estimate that **allometric equation selection alone contributes 10–30% relative RMSE**.

| Method | Sensor/Data | Scale | R² | rRMSE | Key Paper |
|--------|------------|-------|-----|-------|-----------|
| **Random Forest + ALS metrics** | ALS | Plot (0.05–0.1 ha) | 0.80–0.92 | 15–25% | Many NFI studies |
| **DL (DNN/CNN) + multisource** | Sentinel-1/2 + GEDI + ALOS-2 | Regional | 0.79–0.87 | 18–28% | Wang et al. 2024; Tian et al. 2024 |
| **Ensemble ML (stacking)** | ALS + multispectral | Stand | R²=0.89 | 16.2% | Luo et al. 2024 |
| **GEDI L4A** | Spaceborne GEDI | Global sample | R²=0.55–0.70 | 40–80% | Dubayah et al. 2022 |
| **TLS QSM (Quantitative Structure Model)** | TLS | Individual tree | R²>0.95 | < 10% | Wilkes et al. 2023 (TLS2trees) |
| **UAV-LiDAR + Sentinel-2 fusion** | UAV-LiDAR + S2 | Bamboo forest | R²=0.82 | 20.1% | Zhang et al. 2024 |
| **ALS parametric + non-parametric** | ALS | Arid shrub | R²=0.79 | 22.4% | Xie et al. 2023 |

**Dominant error sources in AGB estimation** (Tian et al., 2023):
1. Allometric equation mismatch (species, site, diameter range) — **largest contributor**
2. LiDAR point density variation → missed trees (omission errors)
3. Saturation in high-biomass forests (ALS metrics saturate at ~300–500 Mg/ha)
4. Co-registration errors between field plots and LiDAR data

### 3.4 Stem Volume

| Method | Approach | Key Paper |
|--------|----------|-----------|
| **3DFin software** | TLS → automated DBH + taper → volume per tree | Laino et al. 2024, *Forestry* |
| **TLS2trees** | TLS segmentation + QSM fitting → volume | Wilkes et al. 2023, *MEE* |
| **Sentinel-2 VI + RF** | Optical-only → plot volume (no LiDAR) | Ma et al. 2023, *Remote Sensing* (R²=0.60–0.75) |

### 3.5 Canopy Vertical Stratification

Forest-specific: **overstory / midstory / understory** layers. ALS waveform and GEDI relative height (RH) metrics can stratify the canopy profile.

| Method | Key Metric | Paper |
|--------|-----------|-------|
| **GEDI RH metrics** | RH25, RH50, RH75, RH98 → canopy structural complexity | Torresani et al. 2023, *Ecol. Informatics* |
| **ALS point density by height stratum** | % returns in 0–2m, 2–5m, 5–10m, >10m | Standard in NFI |
| **Fuel parameter mapping** | Canopy height, base height, bulk density at European scale | Aragoneses et al. 2024, *RSE* |

### 3.6 Crown Delineation

| Method | Sensor | Metric | Key Paper |
|--------|--------|--------|-----------|
| **Faster R-CNN** | UAV RGB | F1=0.72–0.85 for detection | Beloiu et al. 2023; Gan et al. 2023 |
| **Mask R-CNN** | Aerial RGB (tropical) | IoU=0.65–0.78 | Ball et al. 2023, *RSE&C* |
| **PointNet++ (canopy + trunk + branch)** | UAV-LiDAR PC | mIoU=0.70–0.85 | Kim et al. 2023 |
| **Marker-controlled watershed + spectral clustering** | ALS CHM | F1=0.82–0.91 | Liu et al. 2024 |

---

## 4. Joint Species + Structure Prediction

### 4.1 The Critical Question

> **Can a single model jointly predict tree species AND structural parameters (H, DBH, biomass) from the same point cloud?**

**Current answer**: No. These are treated as **separate tasks** in almost all published work.

Reasons:
1. **Different optimal feature spaces**: Species classification relies on spectral + textural features + crown shape; structure estimation relies on 3D geometry + point statistics
2. **Different label granularity**: Species = per-crown categorical; structure = per-tree continuous
3. **Annotation asymmetry**: Species labels require expert botanists; structure requires field measurements (H, DBH) from the same trees

### 4.2 Evidence for Separate Pipelines

| Paper | Task | Modality | Species? | Structure? | Both? |
|-------|------|----------|----------|------------|-------|
| **Zhong et al. 2024** | Species ID | UAV-LiDAR + RGB | Yes (OA=85.0%) | No | No |
| **Beloiu et al. 2023** | Species + localization | Aerial RGB | Yes (mAP=0.75) | Crown area only | Partial |
| **Ball et al. 2023** | Crown delineation | Aerial RGB (Mask R-CNN) | No | Crown area | No |
| **Kim et al. 2023** | Tree structure segmentation | UAV-LiDAR (PointNet++) | No | Canopy/trunk/branch | No |
| **Zhang et al. 2023** | DBH estimation | BLS (YOLO + optimization) | No | DBH | No |
| **Wilkes et al. 2023** | Tree segmentation + AGB | TLS (graph-based) | No | Volume/AGB | No |
| **Liu et al. 2024** | Crown segmentation | ALS CHM (watershed) | No | Crown area | No |

### 4.3 Possible Path to Joint Prediction (Speculative)

1. **Multi-task learning with shared encoder** (e.g., PointNet++/Transformer backbone → species head + structure head)
2. **Hierarchical approach**: First classify species → then apply **species-specific allometric equations** for DBH/biomass (practical workaround)
3. **Geometric + spectral fusion** in a single architecture: UAV-LiDAR → point features + RGB texture → multi-head output
4. **Pretext task strategy**: Self-supervised pretraining on point cloud geometry → fine-tune on species + structure jointly

**Status**: An open research gap. No published method does this at scale.

---

## 5. Cross-Scale Problem: Individual Tree → Stand → Region

### 5.1 The Scale Gap

| Scale | Resolution | Parameters | Typical Method | Key Challenge |
|-------|-----------|------------|---------------|---------------|
| **Individual tree** | 0.01–0.1 m | H, DBH, CD, volume, AGB | TLS, UAV-LiDAR (high density) | Occlusion, segmentation in dense stands |
| **Plot** (0.01–0.1 ha) | 0.1–1 m | Plot-level H, AGB, LAI | ALS, field inventory | Allometric equation uncertainty |
| **Stand** (1–100 ha) | 1–10 m | Mean H, stand volume, basal area | ALS wall-to-wall, NFI | Aggregation error, edge effects |
| **Landscape / Regional** | 10–30 m | AGB map, height map | ALS + satellite (Sentinel-2, GEDI) fusion | Upscaling bias, saturation in high biomass |
| **Continental / Global** | 30–100 m | Canopy height, AGB | GEDI, ICESat-2, satellite-based models | Sparse sampling (GEDI), spatial extrapolation error |

### 5.2 Methods Addressing Cross-Scale

| Paper | Scaling Approach | From Scale | To Scale | Key Result |
|-------|-----------------|------------|----------|------------|
| **Lang et al. 2023** | GEDI (sample) + Sentinel-2 (wall-to-wall) as DNN training labels | 25 m GEDI footprints | Global 1 m CHM | First global canopy height model; MAE ~6–8 m |
| **Fayad et al. 2023 (Hy-TeC)** | ViT trained on GEDI + Sentinel-1/2 | GEDI footprint → pixel | Continental CHM | R²=0.67 vs ALS validation |
| **Aragoneses et al. 2024** | GEDI + Sentinel-1/2 + ALS calibration | 25 m → European wall-to-wall | European fuel maps | Canopy height R²=0.70, canopy base height R²=0.45 |
| **Dubayah et al. 2022 (GEDI)** | Explicit multi-footprint sampling design | 25 m footprints → 1 km grid aggregation | Global AGB | ±20 Mg/ha at 1 km aggregation |
| **Wang et al. 2024** | GEDI L4A + Sentinel-1/2 + ALOS-2 → RF/SVR → regional AGB map | Plot → regional | Regional forest AGB | R²=0.87 (conifer), R²=0.79 (broadleaf) |
| **Ball et al. 2023** | Individual trees → per-hectare extrapolation | Single tree → plot | Tropical forest plot | Per-tree mask → crown area per ha |
| **Liu et al. 2023 (Science Advances)** | 3 m nanosatellite → country-level | Tree outside forest → country | Europe-wide tree carbon | Systematic bias = 7.6% (R²=0.98 vs NFI) |

### 5.3 Key Findings on Cross-Scale

1. **Individual tree → plot upscaling** is NOT a bottleneck. If individual tree detection is accurate, summing per-tree estimates gives unbiased plot-level AGB (TLS evidence: Wilkes et al. 2023, R²>0.95).

2. **Plot → regional upscaling** is the main bottleneck. Methods:
   - **Regression kriging**: RF/SVR on satellite data trained on ALS-calibrated plots (standard in NFI)
   - **GEDI as bridge**: GEDI footprints provide 25 m AGB estimates → train Sentinel-2 wall-to-wall model → regional map
   - **Error**: Upscaling from sparse plots to wall-to-wall adds 20–40% relative RMSE

3. **Saturation** at high biomass (>300–500 Mg/ha) affects all optical/SAR methods. Full-waveform LiDAR and GEDI partially mitigate but do not fully solve.

4. **"Trees outside forests"** problem: Liu et al. (2023) showed that urban and agricultural trees contribute **significant uncounted biomass**, missed by traditional NFI that only samples forest-area plots.

---

## 6. Key Datasets

| Dataset | Modality | Scale | Key Content | Year | Source |
|---------|----------|-------|-------------|------|--------|
| **TomoSense** | P/L/C-band SAR + TLS + ALS + UAV-LiDAR | 4.5 ha temperate forest, Sweden | Multi-frequency tomographic SAR + coincident LiDAR + field inventory | 2023 | Tebaldini et al., *RSE* |
| **TreeSatAI** | Aerial RGB + Sentinel-1/2 | 50,000 image patches, Germany | Multi-sensor multi-label tree species | 2023 | Ahlswede et al., *ESSD* |
| **GEDI L4A** | Spaceborne LiDAR (1064 nm, full waveform) | Global (51.6 N–51.6 S) | AGB density at 25 m footprints, ~10 billion shots | 2019– | Dubayah et al. 2022 |
| **PureForest** | ALS + VHR aerial imagery | 339 km², France | 18-class tree species + semantic segmentation | 2024 | arXiv 2404.12064 |
| **ALS Finland benchmark** | Multispectral ALS (3 wavelengths) | 6,326 segments, Finland | 9 tree species + structure | 2025 | ISPRS 2026 |

---

## 7. Methods Comparison: Traditional vs Deep Learning

### 7.1 Individual Tree Detection (ITD)

| Category | Method | Strengths | Limitations | Papers |
|----------|--------|-----------|-------------|--------|
| **Traditional** | CHM-based local maxima + region growing | Fast, interpretable, well-established | Misses suppressed trees; over-segmentation in dense canopies | Standard in NFI software (LAStools, lidR) |
| **Traditional** | Point cloud segmentation (normal segmentation, watershed, spectral clustering) | Works on raw point cloud, no rasterization loss | Parameter-sensitive; slow on full-density ALS | Liu et al. 2024 |
| **Deep Learning** | Mask R-CNN / Faster R-CNN on RGB/CHM | End-to-end; learns shape features; best for heterogeneous forest | Requires annotated bounding boxes; struggles with overlapping crowns in high-density stands | Beloiu et al. 2023; Ball et al. 2023; Gan et al. 2023 |
| **Deep Learning** | PointNet++ and variants on raw point cloud | Full 3D geometry; keeps point density info | Computationally heavy; needs dense annotation; open-world species generalization weak | Kim et al. 2023 |

### 7.2 AGB Estimation

| Category | Method | Typical R² | Typical rRMSE | Notes |
|----------|--------|-----------|---------------|-------|
| **Field + allometry** | DBH → volume → biomass | — | 10–30% | Ground truth reference |
| **ALS metrics + parametric regression** | Height percentiles, density metrics → linear/multiplicative regression | 0.75–0.90 | 18–30% | Saturation in high biomass; species-specific models better |
| **ALS metrics + ML (RF, SVR, XGBoost)** | Same ALS metrics → non-parametric models | 0.80–0.92 | 15–25% | RF is the operational standard; prone to extrapolation error |
| **DL + ALS point cloud** | PointNet++ / KPConv directly on point cloud → AGB | 0.82–0.91 | 16–24% | Less saturation than metrics-based; harder to deploy at scale |
| **Satellite + GEDI + DL** | Sentinel-1/2 + GEDI calibration → pixel-level AGB | 0.55–0.80 | 25–50% | Scale-dependent; degrades rapidly away from GEDI tracks |
| **TLS + QSM** | Tree-level volume from quantitative structure models | > 0.95 | < 10% | Best accuracy but labor-intensive; plot-scale only |

### 7.3 DBH Estimation

| Category | Method | Direct or Indirect | R² | RMSE (cm) |
|----------|--------|-------------------|-----|-----------|
| **TLS circle fitting** | RANSAC / Hough transform on TLS points at 1.3 m height | Direct | > 0.98 | < 1.0 |
| **BLS + deep learning** | YOLO detection + optimization-based fitting | Direct | 0.95 | 2.0 |
| **Drone LiDAR trunk slice** | Point cloud section at 1.3 m → circle fitting | Direct | 0.89 | 1.8 |
| **UAV SfM photogrammetry** | 3D reconstruction → cylinder fitting | Direct | 0.75–0.85 | 3.0–5.0 |
| **ALS + H-D allometry** | Height → species-specific DBH equation | Indirect | 0.70–0.85 | 3–8 |
| **ALS + ML** | ALS metrics (not just height) → DBH | Indirect | 0.75–0.88 | 3–6 |

---

## 8. Research Gaps and Future Directions

### 8.1 Gap 1: Unified Species + Structure Model

**No published method jointly outputs species identity AND full structural parameters (H, DBH, AGB, crown volume) from a single architecture.**

Potential approach: Multi-task PointNet++/Transformer with:
- Branch 1: Species classification head (cross-entropy)
- Branch 2: Structural regression head (MSE for H, DBH, AGB)
- Shared 3D backbone pretrained on self-supervised point cloud tasks

### 8.2 Gap 2: DBH from Above-Canopy ALS

DBH is critical for allometric biomass estimation but is almost never directly measured from ALS. The fundamental limitation is occlusion at 1.3 m height. Potential solutions:
- **Height-DBH transfer learning**: Learn H→DBH mapping from TLS data, apply to ALS-visible trees
- **Structural proxy**: ALS metrics (crown volume, height percentiles) as indirect DBH predictors
- **Low-density airborne bathymetric LiDAR** with wider scan angle could potentially reach lower trunk

### 8.3 Gap 3: Allometric Equation Uncertainty Propagation

Tian et al. (2023) identify **allometric equation mismatch as the single largest error source** in AGB estimation. Needed:
- Species-specific, site-specific allometric equations
- Uncertainty quantification that propagates H error → DBH error → volume error → AGB error
- Bayesian frameworks for multi-level uncertainty

### 8.4 Gap 4: Temporal Dynamics (3D Structure Change Over Time)

Almost all work is **static**: single-time point cloud → structure. Needed:
- Tree growth (delta H, delta DBH) from multi-temporal LiDAR
- Mortality detection from repeated ALS/GEDI
- Thinning effect quantification from time-series point clouds

### 8.5 Gap 5: Open-World Generalization

Current models are trained and tested on **single-site, single-forest-type data**. Generalization to new forest types, new LiDAR sensors, and new geographic regions is unproven. Borsah et al. (2023) highlight that LiDAR feature importance varies drastically across forest types.

### 8.6 Gap 6: Operational Deployment

While research shows high accuracy (R²>0.90 at plot level), **operational NFI workflows still rely heavily on field plots + allometry**, with LiDAR/RS used as an auxiliary data source, not a replacement. Barriers:
- Cost of high-density ALS collection (~$5–20/ha)
- Regulatory acceptance of RS-based carbon reporting
- Lack of standardized, validated workflows

---

## 9. Key Take-Home Messages

1. **Tree height is the most reliable parameter from point clouds** (RMSE < 1 m for UAV-LiDAR individual trees). All other parameters depend on it directly or indirectly.

2. **DBH is the bottleneck.** It can only be measured directly with TLS or under-canopy drone LiDAR. For standard ALS, H-D allometry introduces 15–30% DBH error.

3. **AGB is NEVER directly from point clouds.** The chain H → DBH → volume → biomass accumulates errors at each step, with allometric equation selection as the dominant error source.

4. **No method jointly predicts species + structure in one model.** The tasks remain decoupled in all published work. This is a major open research gap.

5. **Cross-scale (individual→stand→regional) upscaling is partially solved** via GEDI-as-bridge approaches (GEDI 25 m footprints → train Sentinel-2 model → regional wall-to-wall), but saturation in high-biomass forests (>300 Mg/ha) and sparse GEDI sampling remain challenges.

6. **TLS represents the accuracy ceiling**: tree-level DBH RMSE < 1 cm, AGB RMSE < 10%. But it is plot-scale only. The frontier is bridging TLS-scale accuracy to ALS/UAV/satellite scale.

---

## 10. Downloaded Papers

All papers saved to `papers_data/pdfs_downloaded/`:

| File | Paper | OA Status |
|------|-------|-----------|
| `Fassnacht2023_Remote_Sensing_Forestry_Challenges.pdf` | Fassnacht et al. 2023 | OA |
| `Tian2023_Review_AGB_Estimation.pdf` | Tian et al. 2023 (AGB review) | OA |
| `Balestra2024_LiDAR_Fusion_Forest_Attributes.pdf` | Balestra et al. 2024 (LiDAR fusion review) | OA |
| `Zhao2023_TreeCrownDetection_CNN_Review.pdf` | Zhao et al. 2023 (ITCD CNN review) | OA |
| `Ecke2022_UAV_Forest_Health_Review.pdf` | Ecke et al. 2022 (UAV forest health) | OA |
| `Borsah2023_LiDAR_Biomass_Metrics_Review.pdf` | Borsah et al. 2023 (LiDAR metrics review) | OA |
| `Lang2023_Canopy_Height_Model.pdf` | Lang et al. 2023 (global CHM) | OA (Nature) |
| `Dubayah2022_GEDI_Biomass.pdf` | Dubayah et al. 2022 (GEDI biomass) | OA |
| `Beloiu2023_TreeCrown_Species_Identification.pdf` | Beloiu et al. 2023 (species + detection) | OA |
| `Kim2023_TreeSegmentation_DeepLearning.pdf` | Kim et al. 2023 (PointNet++ tree seg) | OA |
| `Zhang2023_BSTDF_DBH_Estimation.pdf` | Zhang et al. 2023 (BSTDF DBH) | OA |
| `Zhong2024_Species_UAV_LiDAR_RGB.pdf` | Zhong et al. 2024 (species UAV LiDAR) | OA |
| `Wilkes2023_TLS2trees.pdf` | Wilkes et al. 2023 (TLS2trees) | OA |
| `Xie2023_AGB_Arid_Shrub_LiDAR.pdf` | Xie et al. 2023 (AGB arid shrub) | OA |
| `Zhang2024_UAV_LiDAR_AGB_Bamboo.pdf` | Zhang et al. 2024 (AGB bamboo UAV) | OA |
| `Aragoneses2024_Canopy_Fuel_Europe.pdf` | Aragoneses et al. 2024 (canopy fuel Europe) | OA |
| `Ball_2023_MaskRCNN_Tropical_RSE.pdf` | Ball et al. 2023 (Mask R-CNN tropical) | OA |
| `TreeSatAI_Benchmark_2023.pdf` | Ahlswede et al. 2023 (TreeSatAI dataset) | OA |

---

## 11. References

1. Fassnacht, F. E., White, J. C., Wulder, M. A., & Naesset, E. (2023). Remote sensing in forestry: current challenges, considerations and directions. *Forestry*. DOI: 10.1093/forestry/cpad024
2. Tian, L., Wu, X., Yu, T., & Li, M. (2023). Review of Remote Sensing-Based Methods for Forest Aboveground Biomass Estimation: Progress, Challenges, and Prospects. *Forests*, 14(6), 1086.
3. Balestra, M., Marselis, S., Sankey, T. T., & Cabo, C. (2024). LiDAR Data Fusion to Improve Forest Attribute Estimates: A Review. *Current Forestry Reports*.
4. Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2023). A high-resolution canopy height model of the Earth. *Nature Ecology & Evolution*.
5. Dubayah, R., et al. (2022). GEDI launches a new era of biomass inference from space. *Environmental Research Letters*, 17, 095001.
6. Zhao, H., Morgenroth, J., Pearse, G. D., & Schindler, J. (2023). A Systematic Review of Individual Tree Crown Detection and Delineation with Convolutional Neural Networks. *Current Forestry Reports*.
7. Murtiyoso, A., Holm, S., Riihimaki, H., & Krucher, A. (2023). Virtual forests: a review on emerging questions in the use and application of 3D data in forestry. *Int. J. Forest Engineering*.
8. Borsah, A. A., Nazeer, M., & Wong, M. S. (2023). LIDAR-Based Forest Biomass Remote Sensing: A Review of Metrics, Methods, and Assessment Criteria. *Forests*, 14(10), 2095.
9. Ecke, S., Dempewolf, J., Frey, J., & Schwaller, A. (2022). UAV-Based Forest Health Monitoring: A Systematic Review. *Remote Sensing*, 14(13), 3205.
10. Wilkes, P., Disney, M., Armston, J., & Bartholomeus, H. (2023). TLS2trees: A scalable tree segmentation pipeline for TLS data. *Methods in Ecology and Evolution*.
11. Laino, D., Cabo, C., Prendes, C., & Janvier, R. (2024). 3DFin: a software for automated 3D forest inventories from terrestrial point clouds. *Forestry*.
12. Zhang, H., et al. (2023). A Novel Framework for Stratified-Coupled BLS Tree Trunk Detection and DBH Estimation in Forests (BSTDF). *Remote Sensing*, 15(14), 3480.
13. Kim, D., Ko, C. U., Kim, D. G., & Kang, J. T. (2023). Automated Segmentation of Individual Tree Structures Using Deep Learning over LiDAR Point Cloud Data. *Forests*, 14(6), 1159.
14. Beloiu, M., Heinzmann, L., Rehush, N., & Gessler, A. (2023). Individual Tree-Crown Detection and Species Identification in Heterogeneous Forests. *Remote Sensing*, 15(5), 1463.
15. Zhong, H., Zhang, Z., Liu, H., & Wu, J. (2024). Individual Tree Species Identification for Complex Coniferous and Broad-Leaved Mixed Forests Based on Deep Learning Combined with UAV LiDAR Data and RGB Images. *Forests*, 15(2), 293.
16. Zhang, Y., Tan, Y., Onda, Y., & Hashimoto, A. (2023). A tree detection method based on trunk point cloud section in dense plantation forest using drone LiDAR data. *Forest Ecosystems*, 100088.
17. Fayad, I., Ciais, P., Schwartz, M. A., & Wigneron, J. P. (2023). Hy-TeC: a hybrid vision transformer model for high-resolution and large-scale mapping of canopy height. *Remote Sensing of Environment*.
18. Aragoneses, E., Garcia, M., Ruiz-Benito, P., & Chuvieco, E. (2024). Mapping forest canopy fuel parameters at European scale using spaceborne LiDAR and satellite data. *Remote Sensing of Environment*.
19. Tebaldini, S., et al. (2023). TomoSense: A unique 3D dataset over temperate forest. *Remote Sensing of Environment*.
20. Ahlswede, S., Schulz, C., Gava, C., & Helber, P. (2023). TreeSatAI Benchmark Archive: a multi-sensor, multi-label dataset for tree species classification. *Earth System Science Data*, 15, 681.
21. Zhang, L., Zhao, Y., Chen, C., & Li, X. (2024). UAV-LiDAR Integration with Sentinel-2 Enhances Precision in AGB Estimation for Bamboo Forests. *Remote Sensing*, 16(4), 705.
22. Tian, X., Li, J., Zhang, F., & Zhang, H. (2024). Forest Aboveground Biomass Estimation Using Multisource Remote Sensing Data and Deep Learning Algorithms. *Remote Sensing*, 16(6), 1074.
23. Luo, M., Anees, S. A., Huang, Q., & Xin, Q. (2024). Improving Forest Above-Ground Biomass Estimation by Integrating Individual Machine Learning Models. *Forests*, 15(6), 975.
24. Wang, C., Zhang, W., Ji, Y., & Marino, A. (2024). Estimation of Aboveground Biomass for Different Forest Types Using Data from Sentinel-1, Sentinel-2, ALOS PALSAR-2, and GEDI. *Forests*, 15(1), 215.
25. Ball, J., Hickman, S., Jackson, T., & Koay, X. J. (2023). Accurate delineation of individual tree crowns in tropical forests from aerial RGB imagery using Mask R-CNN. *Remote Sensing in Ecology and Conservation*.
26. Xie, D., Huang, H., Feng, L., & Sharma, R. P. (2023). Aboveground Biomass Prediction of Arid Shrub-Dominated Community Based on Airborne LiDAR. *Remote Sensing*, 15(13), 3344.
27. Liu, S., Brandt, M., Nord-Larsen, T., & Chave, J. (2023). The overlooked contribution of trees outside forests to tree cover and woody biomass across Europe. *Science Advances*, adh4097.
28. Liu, Y., Chen, D., Fu, S., & Mathiopoulos, P. T. (2024). Segmentation of Individual Tree Points by Combining Marker-Controlled Watershed Segmentation and Spectral Clustering Optimization. *Remote Sensing*, 16(4), 610.
29. Gan, Y., Wang, Q., & Iio, A. (2023). Tree Crown Detection and Delineation in a Temperate Deciduous Forest from UAV RGB Imagery Using Deep Learning Approaches. *Remote Sensing*, 15(3), 778.
30. Estrada, J. S., Fuentes, A., Reszka, P., & Cheein, F. A. (2023). Machine learning assisted remote forestry health assessment: a comprehensive state of the art review. *Frontiers in Plant Science*, 14, 1139232.
