# ForestPheno — 林木表型多智能体系统

## 项目概述

基于 LLM 编排的多模态林木表型分析系统的文献调研与设计空间分析。核心输出为一篇设计空间导向的学术综述，提出了 ForestPheno 系统的五维设计框架。

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

## 论文状态

- 英文版：`papers_data/综述_design_space_oriented.md`（34 篇引用，~10,600 词）
- 中文版：`papers_data/综述_design_space_oriented_CN.md`
- 设计空间调研：`papers_data/surveys/0*.md`（15 篇）
- 事实核查报告：`papers_data/surveys/1*_FactCheck*.md`（3 篇）
