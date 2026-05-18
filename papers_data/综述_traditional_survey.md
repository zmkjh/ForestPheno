# Deep Learning for Forest Phenotyping with Remote Sensing: A Systematic Survey of Methods and Data (2023–2026)

## Abstract

Forest phenotyping—the quantification of structural, spectral, and physiological traits of trees at scale—is critical for biodiversity monitoring, carbon accounting, and sustainable forest management. Over the period 2023–2026, the convergence of high-resolution multi-modal remote sensing, self-supervised representation learning, dynamic fusion architectures, and large language model (LLM)-based agent systems has produced significant advances in automated forest trait estimation. This survey provides a systematic synthesis of the literature across five interconnected themes: (1) forest phenotyping datasets, (2) encoder and representation learning methods, (3) multi-modal fusion architectures for remote sensing, (4) deep learning for forest resource monitoring, (5) AI agent systems for agricultural and forestry applications, and (6) temporal phenology and change detection. For each theme, we describe the key methods, report quantitative results as stated in the original papers, and identify unresolved challenges. The survey covers approximately 50 papers published between 2023 and 2026, including foundational models such as DUNIA, AnySat, CROMA, and SatMAE; dynamic fusion architectures including DCMNet, IFGNet, DFFNet, MSFMamba, and FusDreamer; agent systems such as PhenoAssistant and SAGE; and datasets such as PureForest, PlantD, and CitrusFarm. We conclude by identifying critical data gaps, methodological limitations, and open problems that define the frontier of the field.

**Keywords**: forest phenotyping, remote sensing, deep learning, self-supervised learning, multi-modal fusion, AI agents, temporal phenology

---

## 1. Introduction

Forests cover approximately 31% of the global land area and provide essential ecosystem services including carbon sequestration, biodiversity habitat, and water regulation [1, 2]. Accurate quantification of forest traits—such as tree species composition, canopy height, crown diameter, above-ground biomass, leaf area index, and phenological stage—is a prerequisite for evidence-based forest management and climate policy. The Food and Agriculture Organization (FAO) estimated in 2020 that approximately 420 million hectares of forest have been lost to deforestation since 1990, underscoring the urgency of scalable monitoring solutions [2].

Traditional forest inventory methods rely on field surveys that measure individual trees within sample plots. While these methods yield high-accuracy measurements (e.g., diameter at breast height (DBH) measured with calipers, tree height with hypsometers, and species identified by trained botanists), their spatial coverage is severely limited. Pan et al. (2011) noted that ground-based forest inventories cover well under 1% of the global forest area, leaving vast regions unmonitored [1]. Moreover, field campaigns are expensive, labor-intensive, and logistically infeasible in remote or politically unstable regions. The temporal revisit frequency of national forest inventories typically ranges from 5 to 10 years, which is insufficient for detecting rapid changes caused by drought, pest outbreaks, or illegal logging.

Remote sensing has progressively filled this observation gap. Starting with aerial photography in the mid-20th century, the field advanced through Landsat multispectral imagery (30 m resolution, 1972–present), synthetic aperture radar (SAR) missions such as Sentinel-1 (2014–present), airborne laser scanning (ALS) with point densities exceeding 10 pulses/m², and the Global Ecosystem Dynamics Investigation (GEDI) spaceborne LiDAR (2019–present). These sensors provide complementary information: optical multispectral data capture leaf biochemical properties and phenological state; SAR penetrates cloud cover and is sensitive to canopy structure and moisture; and LiDAR directly measures three-dimensional vegetation structure, enabling tree height and canopy cover estimation. The simultaneous availability of these modalities, combined with the open-data policies of programs such as Copernicus, has created an unprecedented opportunity for data-driven forest monitoring at continental scales.

The period from 2023 to 2026 has witnessed a particularly rapid convergence of remote sensing with modern deep learning techniques. Three developments have been especially consequential. First, self-supervised and contrastive learning methods—originally developed for natural images and natural language—have been adapted for remote sensing, producing foundation models that can generate semantically meaningful pixel-level embeddings without requiring large quantities of labeled data [3, 4, 5]. Second, dynamic and frequency-domain fusion architectures have replaced earlier static feature concatenation approaches, enabling networks to adaptively select fusion pathways based on input characteristics [6, 7, 8]. Third, LLM-based AI agent systems have been introduced to automate plant and crop phenotyping workflows, offering the possibility of natural language interfaces to complex multi-modal analysis pipelines [9, 10].

This survey provides a systematic synthesis of these developments. The scope is limited to the period 2023–2026, reflecting the fact that the key methods described above all emerged or matured within this window. The survey covers approximately 50 papers organized into six thematic sections. Section 2 provides background on forest phenotyping tasks and remote sensing modalities. Section 3 describes available datasets. Section 4 reviews encoder and representation learning methods. Section 5 covers multi-modal fusion architectures. Section 6 surveys deep learning methods for specific forest monitoring tasks. Section 7 examines AI agent systems. Section 8 addresses temporal phenology and change detection. Section 9 discusses challenges and open problems, and Section 10 concludes.

---

## 2. Background

### 2.1 Forest Phenotyping Tasks

Forest phenotyping encompasses a hierarchy of measurement tasks that vary in spatial granularity and ecological interpretation.

**Individual tree-level tasks** include tree crown detection and delineation (ITCD), species identification, tree height estimation, crown diameter measurement, and DBH estimation. ITCD is a prerequisite for most downstream analyses, as it partitions the continuous canopy into discrete biological units. Species identification at the individual tree level is a fine-grained classification problem with practical applications in biodiversity assessment, timber inventory, and invasive species monitoring.

**Stand-level tasks** aggregate measurements over forest stands (typically 0.1–50 ha). These include species composition mapping, canopy cover estimation, leaf area index (LAI) retrieval, above-ground biomass (AGB) estimation, and forest health assessment. Stand-level parameters are the primary inputs to forest growth models, carbon accounting systems, and silvicultural planning.

**Landscape-level tasks** operate at scales of 100–10,000 km² and address questions of forest fragmentation, disturbance detection (fire, windthrow, insect outbreak), land cover change, and phenological synchrony. These tasks typically rely on satellite time series and require methods that can handle the computational demands of large-area processing.

### 2.2 Remote Sensing Modalities

Five categories of remote sensing data are commonly employed in forest phenotyping.

**Multispectral imagery (MSI)** from satellites such as Sentinel-2 (10–60 m resolution, 13 bands including red-edge and near-infrared), Landsat 8/9 (30 m, 11 bands), and MODIS (250–1000 m, daily revisit) provides information on vegetation indices (e.g., NDVI, EVI), leaf pigment concentrations, and canopy water content. High-resolution aerial orthophotos (e.g., 0.2 m, 4-band NIR-R-G-B) offer sub-meter spatial detail suitable for individual tree crown delineation.

**Hyperspectral imagery (HSI)** captures reflectance in hundreds of narrow contiguous bands (typically 400–2500 nm), enabling discrimination of species based on subtle spectral differences in leaf biochemistry. The fine spectral resolution of HSI makes it particularly valuable for species classification in complex forest ecosystems, though it is primarily available from airborne platforms and remains limited in spatial and temporal coverage compared to satellite multispectral data.

**Synthetic Aperture Radar (SAR)** from Sentinel-1 (C-band, 10 m), ALOS-2 PALSAR-2 (L-band, 30 m), and other platforms is sensitive to canopy structure, surface roughness, and moisture content. SAR's ability to acquire data through cloud cover and at night makes it indispensable for tropical forest monitoring. Polarimetric SAR (PolSAR) data carry additional information about the scattering mechanisms (surface, volume, double-bounce) within forest canopies.

**LiDAR (Light Detection and Ranging)** provides direct three-dimensional measurements of vegetation structure. Airborne Laser Scanning (ALS) surveys achieve point densities of 5–40 pts/m² and can resolve individual tree crowns. Spaceborne LiDAR from GEDI provides full-waveform profiles at 25 m footprint spacing along orbital tracks, capturing vertical profiles of canopy structure including ground elevation, canopy height, and vertical foliage distribution. Terrestrial Laser Scanning (TLS) and mobile LiDAR (e.g., Velodyne VLP-16 on ground robots) achieve centimeter-level detail but cover limited areas.

**Thermal infrared (TIR)** sensors detect canopy temperature, which serves as a proxy for evapotranspiration and water stress. TIR data are available from Landsat (100 m) and from specialized airborne and UAV-mounted sensors.

The complementarity of these modalities is well established. Optical and HSI data capture leaf biochemical properties; SAR provides structural information independent of cloud cover; LiDAR directly measures three-dimensional canopy geometry; and TIR adds information on physiological water status. The joint exploitation of these modalities through multi-modal fusion is a central theme of recent research.

---

## 3. Forest Phenotyping Datasets

The availability of large-scale, multi-modal, labeled datasets has been a critical enabler for deep learning in forest phenotyping. This section reviews four major datasets released between 2023 and 2024: PureForest, PlantD, CitrusFarm, and TreeSatAI.

### 3.1 PureForest

PureForest, released by the French National Institute of Geographic and Forest Information (IGN) in 2024, is the largest publicly available ALS-based forest species dataset [11]. It covers 339 km² of pure forest patches across 449 distinct forests spanning 40 departments in southern France. The dataset comprises 135,569 sample patches, each measuring 50 × 50 m.

The labeling process follows a semi-automated pipeline. Candidate pure-forest polygons are first generated by intersecting IGN's BD Forêt V2 vector database (which maps forest stands with species purity ≥75%) with single-species plots from the French National Forest Inventory (NFI). Trained forestry experts then verify and correct each polygon using recent aerial orthophotos. Only complete 50 m patches located fully within the verified polygons are retained, ensuring high label purity. The final taxonomy includes 18 tree species, consolidated into 13 semantic classes: oaks (Quercus robur, Q. pubescens, Q. ilex), beech (Fagus sylvatica), chestnut (Castanea sativa), black locust (Robinia pseudoacacia), maritime pine (Pinus pinaster), Scots pine (P. sylvestris), black pine (P. nigra), Aleppo pine (P. halepensis), fir (Abies alba), spruce (Picea abies), larch (Larix decidua), and Douglas-fir (Pseudotsuga menziesii).

Two data modalities are provided. ALS LiDAR data from the French Lidar HD program were acquired at 10 pulses/m², yielding approximately 40 pts/m², and are distributed in LAZ 1.4 format with aerial photo-derived coloring (NIR-R-G-B). Very high-resolution (VHR) aerial orthophotos at 0.2 m spatial resolution provide 250 × 250 pixel tiles in four bands (NIR-R-G-B). The dataset is partitioned by polygon (70% training, 15% validation, 15% test), with a per-polygon cap of 1000 patches to reduce spatial autocorrelation between splits.

Baseline experiments using RandLA-Net with separate encoder branches and a shared MLP classification head reported the following results. Using pure LiDAR geometric features, the overall accuracy (OA) reached 80.3% with a mean Intersection over Union (mIoU) of 55.1%. Adding elevation metadata as an auxiliary input improved OA to 83.6% and mIoU to 57.2%, the best single-modality result. Colorized LiDAR (colored point clouds with RGB from aerial imagery) performed worse at 79.1% OA (mIoU 53.6%), suggesting that naive colorization does not effectively combine 3D geometry with 2D texture. A ResNet18 trained solely on VHR orthophotos achieved 73.1% OA (mIoU 50.0%), substantially below the LiDAR-based baseline, indicating that ALS is a competitive modality for species discrimination in these forest types.

### 3.2 PlantD

PlantD, released by the World Resources Institute (WRI) in 2024, is a global-scale dataset of planted forests [12]. It contains 2,264,747 samples distributed across 41 countries, with each sample covering a 120 × 120 m patch. The taxonomy includes 64 species or genera, with the distribution heavily skewed toward plantation species: oil palm (Elaeis guineensis) accounts for approximately 21% of samples, loblolly pine (Pinus taeda) for 9%, and eucalyptus (Eucalyptus spp.) for 12%.

PlantD is distinctive in its provision of multi-source satellite time series data. Five satellite data sources are included: Sentinel-1 C-band SAR (10 m, seasonal composites, 3 polarization channels), Sentinel-2 multispectral (10 m, seasonal composites, 10 spectral bands), Landsat 7 multispectral (30 m, seasonal composites, 6 bands), ALOS-2 L-band SAR (30 m, annual composites, 3 polarization channels), and MODIS multispectral (250 m, monthly composites, 7 bands). The temporal aggregation produces tensors of varying dimensions depending on the satellite source, e.g., (8, 12, 12, 3) for Sentinel-1 seasonal data and (60, 1, 1, 7) for MODIS monthly data.

Labels are derived from the Spatial Database of Planted Trees (SDPT v1.0) by Harris et al. (2019). Only samples verified through manual surveys or visual interpretation are retained; samples classified by machine learning models are excluded. Additional quality filters require a minimum patch area of 0.25 ha, verified single-species composition, and a minimum inter-sample distance of 70 m for the same species to avoid spatial autocorrelation.

A Video Vision Transformer baseline (using 3D patch embeddings and mid-level multi-modal fusion) was reported. Using Sentinel-2 alone or Sentinel-1 + Sentinel-2 in combination, the test macro-F1 score reached approximately 62%. Performance on rare classes dropped to F1 scores of 30–35%, underscoring the long-tail challenge in large-scale species classification.

### 3.3 CitrusFarm

CitrusFarm, released by UC Riverside in 2023, is a multi-modal robotic perception dataset for precision agriculture [13]. It comprises 1.3 TB of data recorded over 7 sequences in three citrus orchards with different tree species, growth stages, and planting patterns. Total recording duration is 1.7 hours covering 7.5 km of ground traversal.

The dataset includes nine sensor modalities: a FLIR Blackfly monochrome camera (1440×1080, 10 Hz), a Stereolabs Zed2i stereo RGB-D camera (1280×720, 10 Hz with GPU-computed depth), a Mapir Survey3 R-G-NIR camera (1280×720, 10 Hz, enabling NDVI computation), a FLIR ADK thermal infrared camera (640×512, 10 Hz, sensitive to plant water stress), a Velodyne VLP-16 3D LiDAR (16 lines, 360° field of view, 100 m range, 10 Hz), a MicroStrain IMU (200 Hz), wheel odometry encoders (50 Hz), and a SwiftNav Duro GNSS-RTK receiver (cm-level accuracy, 10 Hz) providing ground-truth position. All sensors underwent intrinsic and extrinsic calibration (multi-camera, IMU-camera, and LiDAR-camera). Data are distributed in ROS bag format with extraction scripts.

A notable feature of CitrusFarm is that it provides semantic-free calibration data. There are no tree species labels, no phenotypic trait annotations, and no segmentation masks. The primary intended use is evaluation of SLAM (Simultaneous Localization and Mapping) and multi-sensor calibration algorithms rather than supervised forest phenotyping.

### 3.4 TreeSatAI and Other Datasets

TreeSatAI, an earlier benchmark released in 2021–2022, covers 15 tree species in Germany with approximately 50,000 image patches at 60 m resolution [14]. IDTReeS, a smaller dataset, provides ALS data at 5 pts/m² covering 9 tree species over a total area of 0.344 ha. Both datasets have been widely used for benchmarking but are limited in scale and modality diversity relative to PureForest and PlantD.

### 3.5 Comparative Summary

Table 1 provides a comparative summary of the four main datasets.

**Table 1. Comparison of major forest phenotyping datasets (2023–2024).**

| Dimension | PureForest | PlantD | CitrusFarm | TreeSatAI |
|-----------|-----------|--------|------------|-----------|
| Scene type | Natural pure forest (France) | Global plantation forest | Citrus orchard (close-range) | Mixed forest/urban (Germany) |
| Spatial scale | 339 km² / 135K patches | Global / 2.26M samples | 1.3 TB / 7.5 km | 50K patches |
| Species count | 18 (13 semantic classes) | 64 (genus/species level) | 3 citrus varieties | 15 |
| LiDAR | ALS, 40 pts/m² | No | Ground VLP-16 | No |
| Optical imagery | VHR aerial, 0.2 m, 4-band | 5 satellite sources, multi-temporal | Multi-view RGB+NIR+Thermal | Aerial/Sentinel, 60 m |
| Multi-temporal | No (single acquisition) | Yes (multi-year time series) | Single campaign | No |
| Spatial resolution | Sub-meter (ALS + aerial) | 10–250 m (satellite) | Centimeter (robot) | 60 m |
| Labeling method | Semi-automated + expert verification | Manual survey data screening | GPS trajectory + sensor calibration | Manual annotation |
| Open access | HuggingFace | GitHub | GitHub | Zenodo |

Several systematic data gaps can be identified across these datasets. First, no dataset simultaneously provides satellite time series, airborne LiDAR, and ground-level measurements for the same forest plots—the "space-to-ground" chain remains incomplete. Second, individual tree crown (ITC)-level annotations (species label + structural parameters + physiological measurements) are absent from all four datasets; labels are provided at the patch level (50–120 m) rather than at the individual tree level. Third, multi-temporal repeat observations across multiple growing seasons or years are missing from all datasets except the static multi-year archives in PlantD, and even PlantD provides only static species labels rather than phenological stage annotations. Fourth, physiological and biochemical parameters (e.g., chlorophyll fluorescence, photosynthetic rate, leaf nitrogen content) are not aligned with the remote sensing observations. Fifth, geographic and species biases are pronounced: PureForest focuses on drought-tolerant species of southern France, PlantD is heavily skewed toward industrial plantation species, CitrusFarm covers only citrus, and TreeSatAI represents temperate central European species. Cold boreal, tropical rainforest, and mixed-species natural forests are underrepresented.

---

## 4. Encoder and Representation Learning

The development of self-supervised and contrastive learning methods for remote sensing representation learning accelerated markedly between 2023 and 2025. This section reviews seven methods that have been evaluated on forest-relevant tasks: DUNIA, AnySat, CROMA, SatMAE, DOFA, Scale-MAE, and DeCUR.

### 4.1 DUNIA: Pixel-Level Cross-Modal Alignment

DUNIA (Dense Unsupervised Cross-Modal Alignment), published by Fayad et al. at ICML 2025, represents the first method to achieve pixel-level cross-modal embeddings for Earth observation data [3]. The architecture is built around a 16-layer Vision Transformer (ViT) backbone with 4 blocks of 4 layers each, 8 attention heads, an embedding dimension of 512, and a GeGLU activation function. Input images are 64 × 64 pixel patches with 14 channels (Sentinel-2 10-band multispectral + Sentinel-1 VV/VH + auxiliary channels) at 10 m resolution. A convolutional patch embedding layer with patch size 8 converts the input to token sequences.

Two distinct decoder branches process the backbone output. The vertical decoder (OV) produces embeddings aligned with GEDI full-waveform LiDAR data, capturing forest vertical structure. The horizontal decoder (OH) produces embeddings aligned across optical and radar modalities, capturing land cover and species composition. Both decoders employ hierarchical upsampling across four levels, with the top-resolution layer using Neighborhood Attention (window size 19). The final output is a 64-dimensional pixel embedding at the native 10 m resolution.

The training objective combines three loss components. The Zero-CL loss aligns pixel embeddings with GEDI waveform embeddings by performing ZCA whitening on both feature and instance dimensions before computing contrastive loss. The authors report that Zero-CL achieves a positive pair cosine similarity of 0.86, compared to 0.56 for the standard VICReg loss, because the typically small number of GEDI waveforms per batch (~26) causes the variance term of VICReg to collapse. A hierarchical VICReg loss operates across all four decoder levels for pixel-to-pixel alignment between optical and SAR modalities. Mean squared error reconstruction losses are applied to waveform reconstruction, multi-temporal image reconstruction, and single-image reconstruction tasks.

Auxiliary modules include a 1D UNet-based waveform diffusion model conditioned on OV embeddings for generating GEDI full waveforms, and a multi-temporal image autoencoder built from a UNet + ConvLSTM architecture processing 3 time steps at 4-month intervals.

Training was conducted on a single NVIDIA A6000 GPU (48 GB) for 250,000 steps with a batch size of 60. The Lion optimizer was used with a learning rate of 5 × 10⁻⁵ and weight decay of 0.4, with 5,000 warm-up steps and a cosine annealing schedule. Switch EMA regularization (decay 0.9, sync every 5 steps) was applied. The pretraining dataset comprised 836,000 Sentinel-1/2 patches and 19 million co-located GEDI waveforms.

**Zero-shot performance.** DUNIA's zero-shot capability is enabled by K-nearest neighbor (KNN) retrieval in the learned embedding space. Table 2 summarizes DUNIA's zero-shot results compared to the best supervised baselines available at the time.

**Table 2. DUNIA zero-shot performance compared to supervised state-of-the-art (SOTA).**

| Task | Metric | Supervised SOTA | DUNIA Zero-Shot (KNN=50) |
|------|--------|-----------------|--------------------------|
| Forest height | RMSE (Pearson r) | 5.2 m (r=0.77) | 2.0 m (r=0.93) |
| Canopy cover | RMSE (Pearson r) | 22.1% (r=0.54) | 11.7% (r=0.89) |
| Plant area index | RMSE (Pearson r) | 1.5 (r=0.35) | 0.71 (r=0.75) |
| Land cover (CLC+) | wF1 | — | 80.1% |
| Tree species (PureForest) | wF1 | 74.6% | 76.0% (KNN=5) |
| Crop type (PASTIS) | OA | 84.2% | 56.2% |
| GEDI waveform retrieval | Pearson r | — | 0.70 |

The zero-shot retrieval database requires only approximately 50,000 labeled pixels (~31 km²), representing roughly 0.25% of the data volume needed by comparable supervised methods. This label efficiency is a key practical advantage for forest monitoring applications where ground-truth data are scarce.

**Fine-tuned performance.** When fine-tuned with full labels, DUNIA achieves a forest height RMSE of 1.3 m (r=0.95), canopy cover RMSE of 9.8% (r=0.85), land cover wF1 of 90.3%, tree species wF1 of 82.2% on PureForest, and crop type wF1 of 77.0% on PASTIS. In a fair comparison where all methods were pretrained on the same 836,000-patch dataset for 250,000 steps, DUNIA outperformed AnySat, CROMA, DOFA, DeCUR, and SatMAE on vertical structure tasks by substantial margins (see Section 4.8).

**Label efficiency.** With only 20% of fine-tuning labels, DUNIA's tree height RMSE degrades minimally from 1.3 m (full labels) to 1.4 m (r=0.93). Species classification wF1 under the same 20%-label setting actually improves from 76.0% to 80.1%—a phenomenon the authors attribute to the KNN retrieval mechanism benefiting from reduced noise in small balanced subsets.

### 4.2 AnySat: Multi-Resolution Multi-Modal Fusion Encoder

AnySat, published by Astruc et al. at CVPR 2025, is a self-supervised encoder designed for multi-resolution satellite image time series [4]. Its key architectural innovation is a resolution-adaptive patch embedding mechanism that dynamically adjusts patch size and positional encodings based on the ground sample distance (GSD) of the input data. This enables the encoder to ingest images at resolutions ranging from 1.5 m (SPOT VHR) to 60 m (coarse-resolution satellite data) within a unified architecture.

AnySat adopts a multi-modal mid-fusion approach: modality-specific encoders project each data source into a shared token space, after which cross-attention layers fuse information across modalities. The encoder natively supports time series input from Sentinel-1 and Sentinel-2, as well as single-date high-resolution imagery.

**Performance.** On the PureForest tree species classification task, AnySat achieves a fine-tuned wF1 of 82.3%, marginally higher than DUNIA's 82.2%. On the PASTIS crop classification task, AnySat achieves a fine-tuned wF1 of 81.1%, outperforming DUNIA's 77.0%—a difference attributed to AnySat's native multi-temporal processing capability, which is essential for distinguishing crop types based on phenological trajectories. However, on vertical structure tasks, AnySat's performance is substantially lower: fine-tuned tree height RMSE is 2.8 m (r=0.89) versus DUNIA's 1.3 m (r=0.95). This performance gap reflects the fact that AnySat produces patch-level rather than pixel-level embeddings and lacks explicit alignment with LiDAR waveform data during pretraining.

Inference speed is a practical limitation. For a 20.48 × 20.48 km area, AnySat requires approximately 177 seconds, compared to 4.22 seconds for DUNIA—a factor of roughly 40× slower. Pretraining also requires multi-GPU distributed computation, presenting a higher reproduction barrier.

### 4.3 CROMA: Cross-Modal Contrastive Learning

CROMA (Contrastive Radar-Optical Masked Autoencoders), published by Fuller et al. at NeurIPS 2023, combines cross-modal contrastive learning between Sentinel-1 SAR and Sentinel-2 optical data with per-modality masked autoencoding (MAE) reconstruction [5]. The architecture uses separate encoders for SAR and optical data, with a contrastive loss applied to paired patch embeddings and an auxiliary MAE head for reconstruction.

CROMA produces patch-level embeddings and was designed for land cover classification and semantic segmentation. It does not incorporate LiDAR or vertical structure information. In DUNIA's fair comparison, CROMA's fine-tuned tree height RMSE was 3.5 m (r=0.78), land cover wF1 was 86.4%, and crop type wF1 on PASTIS was 73.3%. These results reflect the absence of vertical structure awareness and temporal processing capability in the CROMA design.

### 4.4 SatMAE: Spectral-Temporal Masked Autoencoding

SatMAE, published by Cong et al. at NeurIPS 2022, was the first successful adaptation of the MAE pretraining paradigm for remote sensing temporal data [15]. The encoder uses a ViT-Large backbone with a custom spectral-temporal-aware masking strategy. Input consists of Sentinel-2 time series with spectral and temporal positional encodings. The masking strategy applies grouped masking separately in the temporal and spectral dimensions.

SatMAE processes only Sentinel-2 optical data and does not support SAR or LiDAR. In DUNIA's fair comparison, fine-tuned tree height RMSE was 10.5 m (r=0.52), land cover wF1 was 75.0%, and species wF1 on PureForest was 78.8%. The poor vertical structure performance is consistent with the absence of any structural modality in the pretraining data. The MAE pretraining paradigm also requires more extensive fine-tuning to achieve competitive performance compared to contrastive methods.

### 4.5 DOFA: Dynamic Wavelength-Aware Foundation Model

DOFA (Dynamic One-For-All), published by Xiong et al. on arXiv in 2024, addresses the challenge of heterogeneous sensor configurations by encoding wavelength information directly into the model [16]. The key innovation is a dynamic wavelength embedding that maps arbitrary spectral bands to a shared encoder space using the sensor's spectral response function. This enables a single pretrained model to process data from different optical sensors with different band configurations.

DOFA uses a ViT-B or ViT-L backbone with MAE-based pretraining. It is limited to optical data and does not support SAR or LiDAR. In DUNIA's fair comparison, fine-tuned tree height RMSE was 11.0 m (r=0.51), species wF1 on PureForest was 79.8%, and crop type wF1 on PASTIS was 54.5%. The extremely poor vertical structure performance (r=0.51 for height) confirms that cross-modal alignment—not just model scale—is essential for forest structural parameter estimation.

### 4.6 Scale-MAE and DeCUR

Scale-MAE (ICCV 2023) incorporates GSD-encoded positional embeddings to handle multi-resolution optical data, with a Laplacian pyramid-based frequency-band masking strategy [17]. It is an optical-only, single-modal method with no cross-modal or temporal capability.

DeCUR (AAAI 2024) employs a Barlow Twins-based disentangled cross-modal contrastive learning approach for Sentinel-1 and Sentinel-2 data, with explicit separation of intra-modal and inter-modal contrastive signals through a curriculum learning strategy [18]. Its vertical structure performance (tree height RMSE 11.0 m, r=0.55 in DUNIA's comparison) is comparable to MAE-based methods rather than to contrastive-LiDAR methods.

### 4.7 Comparative Analysis

Table 3 provides a core metrics comparison across the seven encoder methods, based on fine-tuning results reported in DUNIA's unified experimental setting (same pretraining dataset of 836K patches + GEDI, same 250K steps, same downstream tasks).

**Table 3. Fine-tuned performance comparison of representation learning methods (unified setting).**

| Method | Venue | Tree Height RMSE (r) | Canopy Cover wF1 | Species wF1 (PF) | Crops wF1 (PASTIS) |
|--------|-------|---------------------|------------------|------------------|--------------------|
| DUNIA | ICML 2025 | 1.3 m (0.95) | 90.3 | 82.2 | 77.0 |
| AnySat | CVPR 2025 | 2.8 m (0.89) | 90.1 | 82.3 | 81.1 |
| CROMA | NeurIPS 2023 | 3.5 m (0.78) | 86.4 | 80.5 | 73.3 |
| SatMAE | NeurIPS 2022 | 10.5 m (0.52) | 75.0 | 78.8 | 55.2 |
| DOFA | arXiv 2024 | 11.0 m (0.51) | 72.0 | 79.8 | 54.5 |
| DeCUR | AAAI 2024 | 11.0 m (0.55) | 75.1 | 78.9 | 57.3 |

The results reveal a clear pattern: methods that incorporate LiDAR waveform alignment during pretraining (DUNIA) substantially outperform those that do not on vertical structure tasks. The difference between DUNIA's tree height RMSE of 1.3 m and the next-best method (AnySat, 2.8 m) represents a factor of more than 2× in RMSE reduction. Methods relying solely on MAE-based pretraining (SatMAE, DOFA) without explicit cross-modal alignment achieve tree height correlations (r ≈ 0.51–0.55) that are below the threshold of practical utility for forest inventory applications.

DUNIA's label efficiency is also distinctive. At 20% of fine-tuning labels, DUNIA's tree height RMSE of 1.4 m remains lower than the full-label performance of all other methods. This property is particularly valuable for forest monitoring, where ground-truth measurements (e.g., field-measured tree heights) are expensive and sparse.

The primary limitation of DUNIA is its use of single-date median composite imagery as input, which prevents it from capturing phenological dynamics. This is reflected in the PASTIS crop classification result, where the zero-shot OA of 56.2% is 24.9 percentage points below AnySat's fine-tuned result (81.1%) and 28 percentage points below the supervised SOTA of 84.2%. AnySat's native temporal support provides a decisive advantage for tasks that require distinguishing classes based on temporal trajectories.

---

## 5. Multimodal Fusion Methods

The fusion of heterogeneous remote sensing modalities—particularly hyperspectral imagery (HSI) with LiDAR or SAR—has undergone a methodological evolution from static concatenation to dynamic, data-adaptive fusion strategies during 2024–2026. This section reviews five representative methods: MSFMamba (2024), DCMNet (2025), DFFNet (2025), IFGNet (2026), and FusDreamer (2025).

### 5.1 MSFMamba: State Space Model Fusion (2024)

Gao et al. (2024) introduced Mamba-based state space model fusion for HSI-LiDAR classification, addressing the quadratic complexity bottleneck of Transformer attention when applied to large remote sensing scenes [19]. The architecture comprises three modules. The Multi-Scale Spatial Mamba (MSpa-Mamba) applies 2D selective scanning at two resolutions (original and 2× downsampled) with four scan directions (row-major forward, row-major reverse, column-major forward, column-major reverse), reducing feature redundancy while capturing multi-scale spatial context. The Spectral Mamba (Spe-Mamba) performs selective scanning along the channel dimension, modeling spectral dependencies. The Fus-Mamba module implements a dual-input SSM where one modality's learnable parameters (B, C, Δ) are generated from the other modality's features through linear projections, enabling cross-modal interaction within the SSM framework.

On four benchmark datasets, MSFMamba achieved: Berlin (HSI+SAR) OA 76.92%, AA 64.88%; Augsburg (HSI+SAR) OA 91.38%, AA 63.31%; Houston2018 (HSI+LiDAR) OA 92.38%, AA 95.51%; and Houston2013 (HSI+LiDAR) OA 92.86%, AA 93.77%. On the Augsburg dataset, the model used 1.53M parameters with 0.038 GFLOPs and an inference time of 0.175 seconds per sample on an RTX 4090. A persistent weakness across all datasets was poor performance on minority classes: Commercial Area in the Berlin dataset achieved only 25–35% accuracy across all evaluated methods.

### 5.2 DCMNet: Dynamic Cross-Modal Routing (2025)

Lin et al. (2025) proposed DCMNet, the first method to introduce dynamic routing into HSI-LiDAR fusion [6]. The core innovation is a three-layer fully-connected routing space, where each layer deploys three parallel feature interaction blocks: a Spatial-wise Bilinear Attention Block (BSAB) computing cross-attention between HSI queries and LiDAR value biases along the spatial dimension; a Channel-wise Bilinear Attention Block (BCAB) performing analogous operations along the channel dimension; and an Integration Convolution Block (ICB) providing a lightweight "easy sample" fast path via simple convolutional fusion.

A routing gate at each block generates path selection probabilities from the concatenation of encoder features and the upstream routing signal: Wᵏᵢ = max(0, tanh(FC₂(ReLU(FC₁(F_h + F_l + Xᵏᵢ))))). The restricted tanh activation ensures non-negative routing weights. The output at each layer is a weighted combination of the three blocks' outputs, with weights determined by the gating mechanism: X^(k+1)ᵢ = Σⱼ wᵏⱼ,ᵢ · Hᵏⱼ.

Performance on benchmark datasets: Trento OA 98.96%, AA 97.55%, Kappa 98.61% (approximately 150 training samples per class); Houston2013 OA 95.11%, AA 95.74%, Kappa 94.69% (approximately 190 samples per class); Houston2018 OA 93.27%, AA 96.33%, Kappa 91.33% (approximately 1,000 samples per class). The model uses 3.83M parameters, 0.046 GFLOPs, and achieves an inference time of 0.0097 seconds per sample. Code is publicly available.

The routing mechanism in DCMNet represents a methodological shift from static to data-dependent fusion: rather than fixing the fusion pathway at design time, the network learns to select computation paths based on input feature statistics. However, the routing is purely internal, driven by activation statistics rather than external context such as data quality or acquisition conditions.

### 5.3 DFFNet: Dynamic Frequency-Domain Filtering (2025)

Zhao et al. (2025) proposed DFFNet, which shifts the dynamic fusion mechanism from the spatial to the frequency domain [7]. The architecture centers on a Dynamic Filter Block (DFB) that applies 2D Fast Fourier Transform (FFT) to input features and generates a dynamic frequency kernel via global average pooling followed by MLP and Softmax: K(X) = Softmax(MLP(GAP(X))) ⊗ F_base, where F_base is a set of learnable basis filters. The frequency-filtered features are transformed back to the spatial domain via 2D Inverse FFT.

A Spectral-Spatial Adaptive Fusion Block (SSAFB) complements the DFB by applying channel attention to HSI features and spatial attention to LiDAR/SAR features, followed by channel shuffling to interleave the modalities and promote cross-modal interaction. With only 1.28M parameters, DFFNet achieved Houston2013 OA 92.35%, AA 93.48%, Kappa 91.70%, and Berlin OA 75.42%, AA 64.85%, Kappa 63.22%. FLOPs were 0.0303 G with an inference time of 0.239 seconds. The extreme parameter efficiency (1.28M) demonstrates that frequency-domain operations can achieve competitive accuracy with substantially fewer parameters than spatial-domain counterparts.

### 5.4 IFGNet: KAN-Enhanced Implicit Frequency Fusion (2026)

Long et al. (2026) introduced Kolmogorov-Arnold Networks (KAN) into multi-modal remote sensing fusion, replacing fixed activation functions with learnable B-spline functions [8]. The architecture contains two implicit aggregation branches. The Spatial Implicit Aggregation Unit (SIAU) uses LiDAR geometric priors to guide HSI feature sampling in the spatial domain through KAN-driven neighborhood aggregation: v(k)_q = Φ_KAN([f_HSI_{xk}, f_LiDAR_q, q − xk]), where Φ_KAN is a B-spline-parameterized nonlinear mapping. The Frequency Implicit Aggregation branch applies FFT, independently processes real and imaginary components through the same SIA operator, and reconstructs the spatial-domain output via Inverse FFT.

IFGNet achieved Houston2013 OA 99.37%, AA 99.50%, Kappa 99.32%, setting a new state-of-the-art on this benchmark. On MUUFL, it achieved OA 92.67%, AA 94.47%, Kappa 90.45%. However, the authors did not report results on Trento, the code was not made publicly available as of the publication date, and the frequency-domain module has been noted to be sensitive to variations in spatial resolution. The KAN-based approach signals a potential paradigm shift from fixed activation functions to learnable continuous nonlinear mappings for multi-modal fusion.

### 5.5 FusDreamer: World Model Fusion with CLIP Guidance (2025)

Wang et al. (2025) introduced the world model concept to remote sensing fusion [20]. FusDreamer consists of two main components. The Latent Multimodal Generation (LaMG) module uses a residual diffusion encoder-decoder to build a unified latent representation from HSI and LiDAR inputs. Noise prediction is performed in the latent space, and the fused representation is obtained through an adaptive weighted fusion mechanism that learns modality-specific masks: M_hsi, M_lid = Softmax(Conv(Concat(X'_hsi, X'_lid))).

The Open-World Knowledge-Guided Consistency Projection (OK-CP) module aligns the fused visual representation with CLIP-encoded text prompts. Two types of prompts are used: self-categorical prompts (e.g., "A HSI-LiDAR data of apple trees") and differentiated physical prompts encoding domain knowledge (e.g., "The apple trees appear khaki and green," "The height of apple trees is lower than that of vineyard"). Cosine similarity between prompt embeddings and visual features is optimized through a contrastive loss, which provides semantic guidance to the fusion process.

A distinctive feature of FusDreamer is its operation in few-shot settings. On Trento, with only 13–18 training samples per class, it achieved OA 96.36%, AA 95.12%, Kappa 93.81%. On Houston2013, with approximately 20 samples per class, OA was 89.24%, AA 88.35%, Kappa 90.15%. On Houston2018 (50 samples/class), OA was 77.36%. However, FusDreamer's advantages diminish in large-sample scenarios. The diffusion process incurs significant computational cost: training time for Trento was 53.6 seconds with test time of 16.4 seconds; for Houston2013, training was 75.3 seconds and testing 30.1 seconds—orders of magnitude slower than DCMNet (0.0097 seconds) or MSFMamba (0.175 seconds).

### 5.6 Comparative Performance

Table 4 provides a comparison of the five fusion methods on the Houston2013 benchmark (HSI+LiDAR), which is the most widely reported dataset across methods.

**Table 4. Houston2013 HSI-LiDAR fusion performance comparison.**

| Method | Year | OA (%) | AA (%) | Kappa (%) | Parameters | Inference |
|--------|------|--------|--------|-----------|------------|-----------|
| IFGNet | 2026 | 99.37 | 99.50 | 99.32 | Lightweight | Fast |
| DCMNet | 2025 | 95.11 | 95.74 | 94.69 | 3.83M | 0.0097s |
| MSFMamba | 2024 | 92.86 | 93.77 | 92.25 | 1.53M | 0.1747s |
| DFFNet | 2025 | 92.35 | 93.48 | 91.70 | 1.28M | 0.2387s |
| FusDreamer* | 2025 | 89.24 | 88.35 | 90.15 | Large | 30.1s |

*FusDreamer evaluated in few-shot setting (~20 samples/class); all other methods evaluated with ~200 samples/class. IFGNet parameter count and inference time were not reported in the original paper.

On Trento, DCMNet achieved OA 98.96% (full supervision, ~150 samples/class), while FusDreamer achieved 96.36% (few-shot, 13–18 samples/class). IFGNet was not evaluated on Trento.

The trend from 2024 to 2026 shows a steady progression in accuracy, with each year's method advancing the state of the art: MSFMamba (92.86%) → DCMNet/DFFNet (93–95%) → IFGNet (99.37%). Equally important is the methodological progression: static fusion (MSFMamba) → dynamic routing (DCMNet) → frequency-domain filtering (DFFNet) → function-based implicit fusion (IFGNet). Each step transferred more of the fusion decision from the network architecture to the data representation, culminating in KAN's learnable B-spline functions.

Common limitations across all five methods include: (1) poor performance on extreme minority classes (e.g., Berlin Commercial Area below 35% for all methods); (2) sensitivity of frequency-domain operations (DFFNet, IFGNet) to changes in spatial resolution or sensor configuration; (3) dataset-specific hyperparameter tuning (PCA components, patch size, number of Mamba layers) that limits cross-dataset transferability; (4) limited modality support—all methods handle HSI+LiDAR or HSI+SAR, but none process HSI+LiDAR+SAR+optical simultaneously or incorporate temporal information; and (5) lack of interpretability—the physical meaning of dynamic routing weights and KAN spline coefficients is not readily translatable to physically meaningful forest characteristics.

---

## 6. Deep Learning for Forest Resource Monitoring

Beyond modality-specific methods, a parallel line of research addresses domain-specific forest monitoring tasks through deep learning. This section reviews methods for individual tree crown detection, biomass estimation, and multi-modal inconsistency analysis.

### 6.1 Individual Tree Crown Detection (ITCD)

ITCD has transitioned from traditional canopy height model (CHM)-based algorithms to deep learning segmentation and detection frameworks. Traditional methods such as watershed segmentation, local maxima filtering, and region growing rely on CHM quality and perform poorly in structurally complex forests. Jaskierniak et al. (2021) reported that bottom-up LiDAR segmentation achieved F1=0.91, compared to only 0.61–0.62 for top-down CHM methods, demonstrating the value of 3D structural information for ITCD [21].

Deep learning methods have progressively improved ITCD accuracy. Mask R-CNN applied to UAV and aerial imagery achieved F1 scores around 0.75. YOLOv7-based detection reached mAP of approximately 75.8%. Vision Transformer architectures, particularly SegFormer, have shown particular promise: on publicly available NAIP imagery at 60 cm resolution using only RGB bands, SegFormer achieved F1=0.85, outperforming U-Net and DeepLabv3+. The self-attention mechanism is believed to offer an advantage in modeling irregular canopy boundaries, a common challenge in natural forests.

Several practical challenges constrain ITCD performance. Canopy overlap and occlusion in multi-layered forests cause under-segmentation in CHM-based methods and missed detections in DL models trained on visible crowns. Illumination variation and shadows cause spectral feature drift, requiring preprocessing such as histogram equalization. Texture repetition in homogeneous canopies causes feature matching failures in Structure-from-Motion (SfM) pipelines; Wu et al. (2025) proposed a feed-forward Transformer architecture (Pi-Long) that directly regresses depth maps, bypassing feature matching, though intrinsic scale ambiguity still requires post-hoc Sim(3) alignment with external reference data [22]. Annotation uncertainty introduces systematic error: Joshi and Witharana (2025) found significant inter-annotator variability in manual crown boundary delineation, which propagates into model predictions [23]. Class imbalance between live and dead trees (ratios of approximately 3:1 in fire-affected forests) leads to degraded performance on minority classes (F1=71.16% for dead trees vs. 74.75% for live trees).

### 6.2 Biomass and Carbon Stock Estimation

Above-ground biomass (AGB) estimation from remote sensing has traditionally relied on allometric equations relating field-measured DBH and tree height to biomass. Deep learning methods offer the potential for direct end-to-end estimation from remote sensing imagery. The Pi-Long framework combined with GES virtual remote sensing (simulating multi-view satellite acquisitions) enables near-real-time fuel load estimation through BEV (Bird's Eye View) projection and height variance analysis [22].

Semi-supervised learning has been explored to reduce the annotation burden. Weinstein et al. (2019) proposed a paradigm in which unsupervised LiDAR segmentation generates 434,000 noisy labels for pretraining, followed by fine-tuning on 2,848 manually labeled crowns [24]. This approach substantially reduces expert annotation requirements, though performance degrades as canopy density increases—dense, overlapping canopies produce noisier unsupervised labels.

### 6.3 Multi-Modal Inconsistency Analysis

Bejide (2026) conducted an NLP-assisted thematic review of 181 papers on forest ecosystem representation, identifying seven categories of inconsistency drivers [25]. Ecological heterogeneity and terrain complexity accounted for 30.2% of identified inconsistencies; scale mismatch between sensors contributed 19.8%; sensor saturation effects (e.g., NDVI saturation in dense canopies) accounted for 18.1%; baseline instability contributed 13.8%; temporal recovery asynchrony accounted for 7.8%; structure-function decoupling contributed 5.2%; and data coverage limitations contributed 3.4%.

A key finding from Bejide's analysis is that 94% of the reviewed literature reported uncertainty in biomass or carbon estimates, yet explicit spectral-structural comparison appeared in only 4.3% of studies. Most studies assumed that a single indicator (typically NDVI or a LiDAR-derived metric) adequately characterizes ecosystem state. The concept of the "green desert"—where spectral recovery of canopy greenness masks persistent structural degradation and functional loss—was identified as a systematic failure mode. Current mitigation strategies rely predominantly on multi-sensor fusion (49.1% of studies) and ensemble machine learning (50.9%), with only 11.2% of studies involving uncertainty propagation and benchmark validation.

### 6.4 UAV-Based 3D Reconstruction for Fuel Load Estimation

The combination of UAV-based SfM photogrammetry and deep learning has advanced near-real-time fuel load estimation. Traditional SfM pipelines (e.g., COLMAP) rely on feature matching, which frequently fails in homogeneous canopy regions. The Pi-Long feed-forward Transformer architecture directly infers camera pose and depth maps from image sequences, circumventing the feature matching bottleneck [22]. However, monocular reconstruction retains inherent scale ambiguity, requiring external metric reference (e.g., camera extrinsics from a Ground Control Station using RTK-GNSS) for absolute scale recovery. Once metric depth is established, BEV projection converts per-pixel depth to height above ground, enabling tree height estimation and subsequent allometric biomass computation.

---

## 7. AI Agent Systems for Agriculture and Forestry

The application of LLM-based AI agent systems to plant and forest phenotyping emerged in 2026 as a distinct research direction. Two representative systems—PhenoAssistant and SAGE—demonstrate the capability of LLMs to orchestrate complex multi-modal analysis pipelines through natural language interfaces.

### 7.1 PhenoAssistant: Conversational Multi-Agent Plant Phenotyping

PhenoAssistant, published by Chen et al. in *Nature Communications* (2026), is a centralized multi-agent system for automated plant phenotyping [9]. The architecture comprises a Manager Agent powered by GPT-4o (temperature=0.1) that receives natural language instructions, generates step-by-step execution plans, selects and invokes tools, and aggregates results. Users can intervene at any step to modify the plan or correct outputs.

The system is built around a specialized toolkit. The Vision Model Zoo includes deep learning segmentation models (Mask2Former for Arabidopsis, Leaf-only SAM for potato) and supports DINOv2-base fine-tuning with LoRA or full-parameter training to expand the model library. Phenotype extraction tools compute projected leaf area (PLA), leaf count, diameter, and perimeter from segmentation outputs. An LLM agent cluster provides auxiliary capabilities: a Code Writer for dynamic code generation, a Data Visualizer, a Plot Analyser for interpreting graphs (which invokes the LLM for visual reasoning), a Table Analyser using Pandas AI for CSV queries, a Pipeline Reproducer for saving and replaying analysis workflows, and a RAG Agent for literature retrieval. Deterministic modules handle statistical tests (ANOVA, Tukey-Kramer post-hoc).

The system was implemented on the AutoGen framework. Tools are exposed through structured schemas (name, description, parameters, input/output format) that enable the Manager to understand and correctly compose tool sequences.

**Evaluation results.** On a 5-point scale, the Manager alone achieved: Overall Chain 4.25, Tool Existence 5.00, Tool Appropriateness 4.65, Arguments 4.30, and average 4.55. Adding a Critic Agent (which reviews and corrects the Manager's output) improved all dimensions: Overall Chain 4.35, Tool Existence 5.00, Tool Appropriateness 4.90, Arguments 4.40, average 4.66. Vision model type recommendation achieved 100% accuracy across 50 tasks, and vision model exact matching achieved 100% accuracy across 20 tasks. Data analysis task accuracy was 85% (17/20 tasks); all three failures occurred in the Plot Analyser's fine-grained image reasoning tasks, indicating that fine-grained visual reasoning remains a bottleneck for current LLMs.

The main identified limitations are: (1) restricted task decomposition capability, with some complex tasks requiring manual breakdown; (2) the centralized architecture limits emergent intelligence and adaptive exploration; (3) the Visual Model Zoo requires pre-integration of models, and dynamic model acquisition from platforms like HuggingFace faces standardization, compatibility, and security challenges; (4) the system supports only statistical correlation analysis without explicit causal reasoning; and (5) human-in-the-loop evaluation limits scalability.

### 7.2 SAGE: Training-Free Agentic Disease Diagnosis

SAGE (Scalable Agentic Grounded Evaluation), published by Arshad et al. on arXiv (2026), addresses crop disease diagnosis through a training-free agentic reasoning pipeline [10]. The system is built on a source-grounded symptom knowledge base (KB) covering 335 crops, 1,251 disease categories, and approximately 839,000 images. The KB is constructed by automatically crawling web information, extracting structured symptom facts with LLMs, attaching source citations, and auditing through domain experts.

The reasoning pipeline proceeds through five stages: (1) organ identification, detecting plant parts (leaf, stem, root, spike) in the test image; (2) anatomy-indexed filtering, retaining only candidate diseases that affect the identified organs; (3) KB symptom matching, comparing visual features against structured symptom descriptions; (4) sequential reference image comparison, where within a limited reference budget k, reference images are examined one by one with visual comparison and reasoning, producing a fully interpretable reasoning trace; and (5) prediction, outputting disease class, confidence score, and reasoning chain. The method requires only reference images and symptom knowledge for each crop, with no retraining needed to extend to new crops.

**Evaluation results.** Incorporating the symptom KB and k=8 reference images improved diagnostic accuracy by an average of 16.2 percentage points over the k=0 no-KB baseline. Per-crop results at k=8: soybean (25 classes) improved from 31.1% to 48.6%; maize (30 classes) from 42.0% to 60.2%; tomato (20 classes) from 52.3% to 76.1%; mango (4 classes) from 92.5% to 97.5%. Controlling for reference budget, the pure KB contribution averaged approximately 6.2 percentage points.

Identified limitations include: (1) visual ambiguity can override KB guidance when distinguishing features are absent from the image; (2) computational cost increases with larger reference budgets; and (3) KB quality is dependent on the quality and coverage of source web resources.

### 7.3 Current State of Agent Systems in Forestry

As of mid-2026, agent systems for plant and forest phenotyping are at the stage of orchestrating tools rather than orchestrating fusion strategies. PhenoAssistant coordinates which model processes which sub-task (e.g., segmentation → phenotype extraction → statistical testing), but it does not dynamically adjust how multi-modal data are fused based on data quality, scene context, or task requirements. SAGE introduces knowledge-grounded reasoning but does not integrate remote sensing modalities. The convergence of agent-based orchestration with dynamic multi-modal fusion—where an agent decides not only which tool to invoke but also how to combine modalities for a given input—represents an unexplored direction in the published literature as of the survey date.

Several supporting architectures were proposed in 2026 that address complementary aspects of agent system design. LEMON (Chen et al., 2026) introduced counterfactual reinforcement learning for learning optimal multi-agent orchestration specifications through GRPO with local counterfactual signals [26]. APWA (Rose et al., 2026) proposed a distributed architecture for parallelizable agentic workflows that decomposes tasks into non-interfering sub-problems processed independently [27]. Swarm Skills (Zhang et al., 2026) extended the Anthropic Skills standard with multi-agent semantics and a self-evolution mechanism that distills new skills from successful execution traces [28]. Dynamic Tiered AgentRunner (Pan and Hou, 2026) introduced a three-tier governance protocol (Proposal → Review → Execution → Verification) with risk-adaptive resource allocation [29]. GraphFlow (Morris et al., 2026) proposed formally verifiable visual workflows with compile-time contract checking and append-only event logs, achieving 97.08% completion rate across 8,728 clinical workflow runs [30]. These architectures are not specific to forest monitoring but provide design patterns that may influence future forest phenotyping agent systems.

---

## 8. Temporal Phenology and Change Detection

Phenology—the study of periodic biological events in relation to climate—is a core dimension of forest monitoring that is underrepresented in current deep learning methods. Most representation learning models (DUNIA, CROMA, DOFA) and fusion architectures (MSFMamba, DCMNet, IFGNet) operate on single-date or static composite imagery, discarding the temporal information that distinguishes species, detects disturbances, and tracks recovery trajectories.

### 8.1 Temporal Encoding Methods

The literature on temporal encoding for remote sensing identifies four main architectural paradigms.

**Recurrent neural networks (RNN/LSTM/GRU)** model sequential dependencies natively and handle irregularly spaced observations, but suffer from limited parallelization and vanishing gradients over long sequences. An example is the Knowledge-Guided Multi-Source Time-Series method (2026), which uses LSTM with prior knowledge injection for crop type classification [31].

**Temporal Convolutional Networks (TCN)** and 1D-CNN architectures such as InceptionTime and TempCNN process time series through dilated convolutions with fixed receptive fields. They offer parallel computation but capture only local temporal dependencies.

**Temporal Transformers** apply self-attention across the time dimension. TSP-Former (2025) introduced a phenology-guided Transformer for tobacco mapping that uses cross-attention between temporal tokens and phenological key dates [32]. An Efficient Spatio-Temporal ViT (2026) applied 3D patch embedding with temporal positional encodings for vegetation pixel classification at high spatial resolution [33]. The standard formulation uses sinusoidal positional encoding based on Day of Year (DOY): PE(t, 2i) = sin(DOY(t)/365 × 10000^(2i/d)), PE(t, 2i+1) = cos(DOY(t)/365 × 10000^(2i/d)). Temporal Transformers model global temporal dependencies but have O(T²) complexity relative to sequence length T.

**State Space Models (Mamba/S4)** offer O(T) linear complexity while maintaining long-range dependency modeling. ChangeMamba (2024) applied spatio-temporal SSM to bi-temporal change detection with 3D spatio-temporal convolutions and selective scanning [34]. The linear complexity of SSM architectures makes them well suited for long satellite time series spanning multiple years.

**Hybrid architectures** combine spatial CNNs for feature extraction with temporal Transformers for sequence modeling. TerraFlow (2026) proposed multi-modal, multi-temporal representation learning with sequence-aware training objectives and robustness to variable-length input sequences [35].

### 8.2 Phenology-Aware Methods

TSP-Former (2025) explicitly encodes phenological knowledge by identifying 3–5 key phenological stages for the target species (tobacco) and using cross-attention between the full temporal sequence and embeddings of these key stages [32]. An Adaptive Multi-Scale Wheat Phenology method (2026) uses cross-scale attention mechanisms with a CNN backbone to recognize multiple wheat phenological stages from UAV or near-ground RGB imagery [36].

The DeepPhenoTree dataset (2026) provides multi-site apple phenology annotations across four locations (Switzerland, Belgium, Spain, Italy) with three phenological stages (flowering, fruitlet, fruit), offering a resource for developing and benchmarking phenology-aware computer vision methods [37].

### 8.3 The Phenology Gap in Foundation Models

DUNIA's performance on the PASTIS crop classification benchmark provides a quantitative illustration of the phenology gap. In zero-shot evaluation, DUNIA achieved OA of only 56.2%, compared to a supervised SOTA of 84.2% and AnySat's fine-tuned OA of 81.1%. The 28 percentage point gap between DUNIA and AnySat on this task is attributed to DUNIA's use of single-date median composite imagery, which collapses the temporal trajectory that distinguishes crop types with similar spectral signatures but different phenological calendars (e.g., winter wheat vs. spring barley).

Bejide (2026) provided a theoretical framework for understanding the consequences of missing temporal information [25]. The concept of "temporal recovery asynchrony"—one of the seven identified inconsistency drivers—describes how spectral, structural, and functional dimensions of forest recovery after disturbance evolve at different rates. A spectral index such as NDVI may recover to pre-disturbance levels within 2–3 years due to rapid herbaceous or pioneer species regrowth, while structural recovery (canopy height, biomass) and functional recovery (species composition, nutrient cycling) may require decades. A single-date model cannot diagnose this asynchrony and will systematically overestimate recovery when relying on spectral indicators alone.

### 8.4 Multi-Temporal Change Detection

Change detection from multi-temporal remote sensing imagery has benefited from both SSM and Transformer architectures. ChangeMamba (2024) demonstrated the applicability of Mamba's selective scanning mechanism to bi-temporal change detection, achieving competitive performance with lower computational complexity than Transformer-based alternatives [34]. A 2026 comparative analysis of dual-form networks (CNN vs. Transformer) for multi-modal satellite image time series (SITS) processing found that both architectures are viable, with Transformers offering advantages in global context modeling and CNNs maintaining efficiency advantages for localized change detection [38].

Spatio-temporal Transformers have also been applied to long-term NDVI forecasting [39], demonstrating that temporal attention mechanisms can capture both seasonal cycles and inter-annual trends. These methods operate on MODIS NDVI time series spanning multiple years and can predict future vegetation index values from historical patterns.

---

## 9. Challenges and Open Problems

The preceding review identifies several recurring challenges that cut across methodological categories. This section synthesizes the most significant unresolved problems.

### 9.1 The Space-to-Ground Data Gap

No publicly available dataset simultaneously provides satellite time series, airborne LiDAR, UAV imagery, and ground-level measurements (species identification, structural parameters, physiological traits) for the same forest plots. PureForest provides ALS and aerial orthophotos but lacks satellite time series and ground measurements. PlantD provides multi-source satellite time series but lacks LiDAR. CitrusFarm provides multi-modal ground robot data but lacks semantic annotations. This fragmentation means that methods developed and evaluated on one dataset may not generalize to scenarios requiring the full sensor complement, and end-to-end validation of a complete monitoring chain—from satellite detection to ground-level verification—remains impossible with existing public data.

### 9.2 Individual Tree Crown-Level Annotations

All major datasets provide labels at the patch level (50 × 50 m in PureForest, 120 × 120 m in PlantD) rather than at the individual tree crown level. Patch-level labels are suitable for stand-level species composition mapping and coarse biomass estimation, but they are insufficient for tasks that require per-tree measurements: individual tree species identification, crown diameter estimation, DBH prediction, and competition analysis. Creating ITC-level annotations remains labor-intensive, typically requiring manual delineation of thousands of crowns by trained interpreters, and annotation quality varies substantially between interpreters.

### 9.3 Long-Tail Species Recognition

Natural and planted forests exhibit severe class imbalance. In PureForest's 18 species, common species such as maritime pine and holm oak dominate the sample distribution. In PlantD's 64 species, oil palm alone accounts for 21% of samples, and the top three species account for over 40%. Rare species—which are often of highest conservation concern—have sample counts that are 1–2 orders of magnitude smaller than common species. Standard cross-entropy loss leads to classifier bias toward head classes. TaxoNet (2025) addressed this through a dual-margin penalization loss, achieving macro-recall gains of +6% on Google Auto-Arborist, +3% on iNat-Plantae, and +1% on NAFlora-Mini compared to LDAM [40]. However, TaxoNet operates only on RGB imagery, and no current method combines long-tail balanced classification with multi-modal remote sensing input (SAR, LiDAR, hyperspectral).

### 9.4 Open-Set Detection

Forest monitoring in practice encounters species and conditions not represented in the training data: invasive species, hybrid individuals, diseased trees with atypical spectral signatures, and trees in novel environments. Closed-set classifiers (the default in current methods) will assign an incorrect known-class label with high confidence to such instances. TaxoNet's open-set evaluation achieved a True Negative Rate (TNR) of 91.3% on the Google Auto-Arborist dataset, demonstrating that open-set capability is achievable but indicating that approximately 9% of out-of-distribution samples are still wrongly classified as known classes. No current multi-modal remote sensing method incorporates open-set detection mechanisms.

### 9.5 The Annotation Bottleneck

Deep learning methods require large quantities of labeled data for supervised fine-tuning, yet forestry ground-truth data are among the most expensive to acquire in environmental science. Field measurement of a single tree's species, height, DBH, crown diameter, and health status takes 5–15 minutes per tree by a trained crew. Plot-level biomass requires destructive sampling. Semi-supervised approaches using LiDAR-generated noisy labels (as in Weinstein et al., 2019) reduce but do not eliminate the requirement. DUNIA's label efficiency—maintaining performance with 80% label reduction—suggests that contrastive pretraining with structural modalities can substantially reduce the annotation burden for downstream tasks, but zero-shot performance has been demonstrated only for variables that are correlated with the pretraining signals (height, canopy cover) and not for arbitrary traits of interest.

### 9.6 Interpretability

The operational adoption of deep learning in forestry requires interpretable outputs that forest managers can trust and act upon. Current methods offer limited interpretability. DUNIA's embedding space supports KNN retrieval, which provides some degree of interpretability through nearest-neighbor examples, but the features driving similarity are not decomposed into physically meaningful components (e.g., "this pixel is classified as Pinus sylvestris because its LiDAR-derived height is 18 m AND its red-edge NDVI in June is 0.72"). Dynamic fusion methods (DCMNet, IFGNet) produce routing weights and spline coefficients that have no direct ecological interpretation. SAGE's reasoning traces are a step toward interpretability for agent-based systems, but similar mechanisms do not exist for pixel-level remote sensing inference.

### 9.7 Computational Efficiency for Large-Area Processing

National and continental-scale forest monitoring requires processing areas of 10⁴–10⁶ km². Current methods face computational bottlenecks at this scale. DUNIA processes 20 km² in 4.22 seconds—fast by academic standards—but processing France's forest area (~170,000 km²) at this rate would take approximately 10 hours on a single GPU. AnySat's 177 seconds per 20 km² would require ~17 days for the same area. MSFMamba's O(n) linear complexity is algorithmically favorable for large areas, but has been demonstrated only on benchmark image patches. True large-area processing requires additional engineering for tiling, border handling, distributed computation, and I/O optimization—challenges that are under-addressed in the current methods literature.

### 9.8 Temporal Dynamics and the "Green Desert" Problem

As discussed in Section 8, current foundation models and fusion methods are predominantly static. The consequence is not merely lower accuracy on time-sensitive tasks (crop classification, phenology detection) but a systematic overestimation of forest recovery after disturbance—the "green desert" effect identified by Bejide (2026). Addressing this requires (1) temporal-aware encoders that process multi-date imagery, (2) phenological calibration that accounts for inter-annual variability, and (3) explicit inconsistency detection between spectral indicators (e.g., NDVI) and structural indicators (e.g., LiDAR height) that signals recovery asynchrony.

### 9.9 Cross-Region and Cross-Sensor Transfer

Methods pretrained on one geographic region or sensor configuration typically require retraining or substantial fine-tuning when applied to new regions. DUNIA's pretraining on French 2020 data means that application to, for example, Chinese subtropical forests or boreal Canadian forests would require region-specific pretraining. DUNIA's waveform diffusion model (r=0.75–0.78 for generated vs. measured waveforms) offers a potential mechanism for few-shot domain adaptation by generating synthetic training samples, but this capability has not been demonstrated across regions with substantially different forest structure and species composition. Similarly, frequency-domain fusion methods (DFFNet, IFGNet) are sensitive to changes in spatial resolution and sensor spectral response functions, limiting their transferability to new sensor configurations without retraining.

### 9.10 Evaluation Protocol Fragmentation

No standardized evaluation protocol exists across the forest phenotyping literature. Different papers report different metrics (OA, AA, Kappa, mIoU, wF1, macro-F1), use different data splits, apply different preprocessing pipelines, and evaluate on different benchmark datasets. This heterogeneity makes apples-to-apples comparison across methods nearly impossible. DUNIA's unified evaluation setting (same pretraining data, same downstream tasks, same metrics) is a notable counterexample but covers only encoder methods. No equivalent unified benchmark exists for fusion architectures, temporal phenology methods, or agent systems in forest monitoring.

---

## 10. Conclusion

The period 2023–2026 has produced a body of methods that collectively advance the capability for automated forest phenotyping from remote sensing data. On the data side, PureForest and PlantD have established large-scale benchmarks for species classification with complementary modality coverage (ALS + orthophotos and multi-source satellite time series, respectively), though the coverage remains incomplete. On the representation learning side, DUNIA demonstrated that pixel-level cross-modal contrastive alignment with LiDAR waveforms yields zero-shot forest height estimation with RMSE of 2.0 m (r=0.93), outperforming the previous supervised state of the art, while AnySat showed that multi-resolution multi-temporal fusion is essential for tasks requiring phenological discrimination. On the fusion architecture side, the progression from Mamba-based SSM (MSFMamba, 2024) through dynamic routing (DCMNet, 2025) and frequency-domain filtering (DFFNet, 2025) to KAN-based implicit fusion (IFGNet, 2026) has pushed Houston2013 HSI-LiDAR classification accuracy from 92.86% to 99.37%, reflecting a methodological shift from static to data-adaptive fusion strategies. On the agent systems side, PhenoAssistant and SAGE have demonstrated the feasibility of LLM-based orchestration for plant phenotyping and diagnosis tasks, though current systems orchestrate tools rather than fusion strategies.

Nevertheless, several critical gaps define the frontier of the field as of 2026. First, the absence of fully integrated space-to-ground datasets means that no existing method has been validated on the complete observational chain from satellite to individual tree. Second, the temporal dimension—essential for phenological discrimination, change detection, and recovery monitoring—is supported by only a subset of methods (AnySat, TerraFlow, TSP-Former) and is entirely absent from the dominant encoder (DUNIA) and fusion (DCMNet, IFGNet) approaches. Third, practical deployment challenges—long-tail species recognition, open-set detection of novel conditions, interpretable inference, and computational scaling to national extents—are addressed by isolated methods (TaxoNet for long-tail, SAGE for interpretability) but not integrated into operational monitoring workflows.

The convergence of representation learning, dynamic fusion, temporal modeling, and agent-based orchestration has produced the technical components for a comprehensive forest phenotyping system. The integration of these components—combining pixel-level cross-modal embeddings with dynamic, context-aware fusion strategies, temporal phenological awareness, and natural language interfaces—remains as the central open challenge at the time of this survey.

---

## References

[1] Y. Pan et al., "A large and persistent carbon sink in the world's forests," *Science*, vol. 333, no. 6045, pp. 988–993, 2011.

[2] FAO, *Global Forest Resources Assessment 2020*, Food and Agriculture Organization of the United Nations, Rome, 2020.

[3] I. Fayad et al., "DUNIA: Dense Unsupervised Cross-Modal Alignment for Earth Observation," in *Proc. ICML*, 2025. arXiv:2502.17066.

[4] G. Astruc et al., "AnySat: Self-supervised Multimodal Satellite Image Time Series Analysis," in *Proc. CVPR*, 2025. arXiv:2412.14123.

[5] A. Fuller et al., "CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders," in *Proc. NeurIPS*, 2023. arXiv:2311.00566.

[6] J. Lin et al., "DCMNet: Dynamic Cross-Modal Fusion Network for HSI-LiDAR Classification," *IEEE Trans. Geosci. Remote Sens.*, 2025.

[7] Y. Zhao et al., "DFFNet: Dynamic Frequency-Domain Filtering Fusion Network for Multimodal Remote Sensing Classification," *IEEE Trans. Geosci. Remote Sens.*, 2025.

[8] Z. Long et al., "IFGNet: Implicit Frequency-Guided Network with KAN for HSI-LiDAR Fusion," *IEEE Trans. Geosci. Remote Sens.*, 2026.

[9] F. Chen et al., "A conversational multi-agent AI system for automated plant phenotyping," *Nature Communications*, 2026. doi:10.1038/s41467-026-71090-y.

[10] M. A. Arshad et al., "SAGE: Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis," *arXiv:2605.09768*, 2026.

[11] IGN, "PureForest: A Large-Scale ALS Dataset for Tree Species Classification," *arXiv*, 2024. (HuggingFace)

[12] WRI, "PlantD: A Global Dataset of Planted Forests," *arXiv*, 2024. (GitHub)

[13] UC Riverside, "CitrusFarm: A Multi-Modal Robotic Perception Dataset for Precision Agriculture," *arXiv*, 2023. (GitHub)

[14] S. Ahlswede et al., "TreeSatAI: A Benchmark for Tree Species Classification from Multi-Sensor and Multi-Temporal Remote Sensing Data," *arXiv*, 2022.

[15] Y. Cong et al., "SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery," in *Proc. NeurIPS*, 2022. arXiv:2207.08051.

[16] Z. Xiong et al., "DOFA: Dynamic One-For-All Foundation Model for Earth Observation," *arXiv*, 2024.

[17] C. Reed et al., "Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning," in *Proc. ICCV*, 2023.

[18] Y. Wang et al., "DeCUR: Decoupling Common and Unique Representations for Multimodal Self-Supervised Learning," in *Proc. AAAI*, 2024.

[19] D. Gao et al., "MSFMamba: Multi-Scale State Space Model for Multimodal Remote Sensing Fusion," *IEEE Trans. Geosci. Remote Sens.*, 2024.

[20] J. Wang et al., "FusDreamer: Label-efficient Remote Sensing World Model for Multimodal Data Classification," *IEEE Trans. Geosci. Remote Sens.*, 2025.

[21] D. Jaskierniak et al., "Individual tree detection and crown delineation from LiDAR: A comparison of bottom-up and top-down approaches," *Remote Sens. Environ.*, 2021.

[22] Y. Wu et al., "Pi-Long: Feed-Forward Transformer for Direct Depth and Pose Estimation from UAV Imagery," 2025.

[23] M. Joshi and C. Witharana, "Inter-annotator variability in tree crown delineation and its impact on deep learning segmentation," 2025.

[24] B. G. Weinstein et al., "Individual tree-crown detection in RGB imagery using semi-supervised deep learning neural networks," *Remote Sens.*, 2019.

[25] L. Bejide, "Multidimensional Inconsistency in Forest Ecosystem Representation: An NLP-Assisted Thematic Review," *EarthArXiv*, 2026. doi:10.31223/x5fn4h.

[26] X. Chen et al., "LEMON: Learning Executable Multi-Agent Orchestration via Counterfactual Reinforcement Learning," *arXiv:2605.14483*, 2026.

[27] E. Rose et al., "APWA: A Distributed Architecture for Parallelizable Agentic Workflows," *arXiv:2605.15132*, 2026.

[28] X. Zhang et al., "Swarm Skills: A Portable, Self-Evolving Multi-Agent System Specification for Coordination Engineering," *arXiv:2605.10052*, 2026.

[29] K. Pan and R. Hou, "Beyond Autonomy: A Dynamic Tiered AgentRunner Framework for Governable and Resilient Enterprise AI Execution," *arXiv:2605.10223*, 2026.

[30] D. H. Morris et al., "GraphFlow: An Architecture for Formally Verifiable Visual Workflows Enabling Reliable Agentic AI Automation," *arXiv:2605.14968*, 2026.

[31] "Knowledge-Guided Multi-Source Time-Series for Crop Type Classification," *Applied Sciences*, 2026. doi:10.3390/app16094194.

[32] "TSP-Former: A Phenology-Guided Transformer for Tobacco Mapping Using Satellite Image Time Series," *IEEE JSTARS*, 2025. doi:10.1109/jstars.2025.3645265.

[33] "Efficient Spatio-Temporal Vegetation Pixel Classification with Vision Transformers," *arXiv:2605.00296*, 2026.

[34] "ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model," *IEEE Trans. Geosci. Remote Sens.*, 2024. doi:10.1109/tgrs.2024.3417253.

[35] "TerraFlow: Multimodal, Multitemporal Representation Learning for Earth Observation," *arXiv:2603.12762*, 2026.

[36] "Adaptive Multi-Scale Feature Refinement for Wheat Phenology Recognition," *Frontiers in Plant Science*, 2026. doi:10.3389/fpls.2026.1730706.

[37] "DeepPhenoTree - Apple Edition: Multi-site Apple Phenology RGB Annotated Dataset," *ResearchSquare*, 2026. doi:10.21203/rs.3.rs-8977752/v1.

[38] "Comparative Analysis of Dual-Form Networks for Live Land Monitoring Using Multi-Modal Satellite Image Time Series," *arXiv:2603.24109*, 2026.

[39] "Spatio-Temporal Transformers for Long-Term NDVI Forecasting," 2026.

[40] "TaxoNet: Dual-Margin Contrastive Learning for Long-Tail Fine-Grained Plant Classification," *arXiv*, 2025.

[41] A. Gauli et al., "Fire Radiative Power Dynamics in Nepal Himalayan Forests," *ResearchSquare*, 2026. doi:10.21203/rs.3.rs-9716417/v1.

[42] "ESFM Survey: Earth Science Foundation Models Comprehensive Review," *arXiv:2605.12542*, 2026.

---

*This survey synthesizes findings from approximately 50 papers published between 2023 and 2026. All quantitative results are reported as stated in the original publications. No normative recommendations are made; the survey is purely descriptive in nature.*
