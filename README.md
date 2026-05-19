# ForestPheno — 林木表型多智能体系统

基于 LLM 编排的多模态林木表型分析系统的文献调研与设计空间分析。

## 核心输出

| 文件 | 说明 |
|------|------|
| `papers_data/综述_design_space_oriented.md` | 英文学术综述（设计空间导向，34篇引用） |
| `papers_data/综述_design_space_oriented_CN.md` | 中文版 |
| `papers_data/综述_traditional_survey.md` | 英文学术综述（传统时间线/方法论，42篇引用） |
| `papers_data/surveys/` | 15篇设计空间调研 + 1篇交叉验证 + 3篇事实核查 |
| `papers_data/missing_papers.md` | 缺失论文下载清单 |

## 目录结构

```
├── README.md
├── .gitignore
├── batch_download.py          # PDF批量下载 + curl校验
├── crawl_targeted_papers.py   # OpenAlex论文爬虫
├── crawl_latest_papers.py     # 最新论文爬虫
├── crawl_papers.py            # 元数据爬虫
├── download_pdfs.py           # 下载脚本
├── filter_papers.py           # 关键词筛选
└── papers_data/
    ├── targeted_forestry_ai_papers.csv    # 122篇定向论文
    ├── latest_forestry_ai_papers.csv      # 98篇最新论文
    ├── selected_forestry_ai_papers.csv    # 5篇精选论文
    ├── forestry_papers_filtered.csv       # 14篇经典论文
    ├── MDPI_manual_download.md            # MDPI手动下载清单
    ├── missing_papers.md                  # 缺失论文清单
    ├── surveys/                           # 设计空间调研 + 事实核查
    └── pdfs_downloaded/                   # 分类PDF（00_Legacy ~ 10_QualityAwareness）
```

## 论文状态

- **设计空间综述**：英文+中文双版本，48条引用连续无空洞，~10,300词
  - 五维方法论全景（对比学习/动态融合/Agent编排/物候/质量感知）
  - 六个森林传输壁垒（冠层遮挡、混交林、生物量饱和、地形效应、泛化退化、长尾分布）
  - 零 prescriptive 语言，纯分析性综述
- **传统综述**：英文，10章纯描述性综述，42条引用，覆盖2023-2026森林AI表型全领域
- **数据质量**：53项原文逐条核对 + 180项交叉验证，修复5处错误
- **PDF**：100+篇下载，分11个类别目录
- **调研**：15篇设计空间 + 8篇林业聚焦 + 4篇事实核查
