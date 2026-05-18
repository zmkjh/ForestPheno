# ForestPheno — 林木表型多Agent系统

南京林业大学研究生项目。

## 目录结构

```
├── .gitignore
├── crawl_targeted_papers.py    # OpenAlex论文爬虫（4子方向分类）
├── crawl_latest_papers.py      # 最新论文爬虫（按时间倒序）
├── crawl_papers.py              # 元数据爬虫
├── filter_papers.py             # 关键词过滤筛选
├── batch_download.py            # 批量PDF下载（curl + 校验）
├── download_pdfs.py             # PDF下载脚本
├── 林业AI_Agent研究综述.md       # 9章综述初稿
└── papers_data/
    ├── targeted_forestry_ai_papers.csv    # 122篇定向论文元数据
    ├── latest_forestry_ai_papers.csv      # 98篇最新论文元数据
    ├── selected_forestry_ai_papers.csv    # 5篇精选论文
    ├── forestry_papers_filtered.csv       # 14篇高引用经典论文
    ├── MDPI_manual_download.md            # MDPI手动下载清单
    └── pdfs_downloaded/
        ├── 01_KeyReferences/      # PhenoAssistant, SAGE, DUNIA, DCMNet等
        ├── 02_Frontiers/
        ├── 03_Nature/
        ├── 04_Springer/
        ├── 05_arXiv/
        ├── 06_ResearchSquare/
        ├── 07_Copernicus/
        ├── 08_Other/
        └── 09_MDPI_Manual/        # 占位，手动下载后放入
```

## 核心参考文献（01_KeyReferences/）

| 文件 | 内容 | 重要性 |
|------|------|--------|
| `Key_PhenoAssistant_NatureComms2026.pdf` | 多Agent植物表型系统 | ★★★ 架构基线 |
| `Key_SAGE_CropDiseaseAgent_arXiv2026.pdf` | Agentic作物病害评估 | ★★★ Agent参考 |
| `Ref_DUNIA_CrossModal_arXiv2025.pdf` | 对比学习跨模态对齐 | ★★★ 融合方法 |
| `Ref_DCMNet_DynamicFusion_arXiv2025.pdf` | 动态路由HSI+LiDAR融合 | ★★★ 路由机制 |
| `Ref_PureForest_LiDAR_arXiv2024.pdf` | LiDAR+航片树种分类数据集 | ★★★ 数据参考 |
| `Ref_IFGNet_KANFusion_arXiv2026.pdf` | HSI+LiDAR频域融合 | ★★ 融合借鉴 |
| `Ref_MSFMamba_MambaFusion_arXiv2024.pdf` | Mamba架构多源融合 | ★★ 融合借鉴 |
| `Ref_TaxoNet_PlantTaxonomy_arXiv2025.pdf` | 植物分类Embedding学习 | ★★ 分类参考 |
| `Ref_PlantD_ForestDataset_arXiv2024.pdf` | 多卫星人工林数据集 | ★★ 数据参考 |
| `Ref_FusDreamer_WorldModel_arXiv2025.pdf` | World Model遥感融合 | ★★ 前沿参考 |

## 论文元数据

- **122篇** 定向分类论文 + **98篇** 最新论文 + **14篇** 经典论文
- 来源: OpenAlex API
- 覆盖4大方向: 碳/生物量、树种/遥感、病虫害/火灾、智慧林业/数字孪生

## 研究方向

Agent-Orchestrated Cross-Modal Contrastive Forest Phenotyping

基于LLM编排的多模态（RGB+LiDAR+高光谱+时序）林木表型分析系统，核心创新：
1. 对比学习预训练跨模态编码器
2. Agent动态模态路由融合
3. 野外复杂环境自适应
