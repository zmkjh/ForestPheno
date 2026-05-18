# ForestPheno — 林木表型多智能体系统

基于 LLM 编排的多模态林木表型分析系统的文献调研与设计空间分析。

## 核心输出

| 文件 | 说明 |
|------|------|
| `papers_data/综述_design_space_oriented.md` | 英文学术综述（34篇引用，~10,600词） |
| `papers_data/综述_design_space_oriented_CN.md` | 中文学术综述 |
| `papers_data/surveys/` | 15篇设计空间调研 + 3篇事实核查报告 |
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

- **综述**：英文+中文双版本，34条引用，经过53项原文对照核查
- **PDF**：68篇下载，分10个类别目录
- **调研**：15篇模块级设计空间分析覆盖编码器、融合、Agent、时序、评估
- **事实核查**：3篇核查报告逐条对照原文验证数字和方法
