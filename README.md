# ForestPheno — 林木表型多智能体系统

基于 LLM 编排的多模态林木表型分析系统的文献调研与设计空间分析。

## 综述论文（3 篇）

| 论文 | 语言 | 类型 | 说明 |
|------|------|------|------|
| `papers_data/综述_design_space_oriented.md` | EN | 设计空间综述 | 五维方法论空间分析（对比学习/动态融合/Agent编排/物候/质量感知），六类森林传输壁垒 |
| `papers_data/综述_design_space_oriented_CN.md` | CN | 设计空间综述 | 同上，中文版 |
| `papers_data/综述_traditional_survey.md` | EN | 描述性综述 | 按时间线与方法论展开，覆盖 2023–2026 森林 AI 表型全领域 |

**两类综述的区别**：设计空间综述以方法论维度为轴，识别设计选择与组合空间，不含 prescriptive 建议；描述性综述以领域发展脉络为轴，系统归纳已有工作。

## 专题调研（14 篇）

`papers_data/surveys/forestry/` — 面向林业场景的深度专题，涵盖单木检测、长尾物种、三维结构、野外鲁棒性、混交林缺口、中文林业文献等方向。

`papers_data/surveys/`（根级 16 篇）— 设计空间调研（编码器、融合、Agent 框架、时序物候、质量感知、评估长尾等）+ 事实核查 + 交叉验证。

## 实验方案（6 组）

`docs/baseline_plan.md` / `outputs/baseline_plan.md` — 混交林跨域迁移诊断实验。

## 目录结构

```
├── README.md
├── LICENSE                     # MIT
├── outputs/                    # 干净展示层（论文终稿 + 实验方案）
│   ├── README.md
│   ├── baseline_plan.md
│   └── papers/
├── papers_data/                # 文献数据、综述、调研
│   ├── 综述_design_space_oriented.md
│   ├── 综述_design_space_oriented_CN.md
│   ├── 综述_traditional_survey.md
│   ├── surveys/
│   └── pdfs_downloaded/
├── docs/
│   └── baseline_plan.md
├── batch_download.py
├── crawl_*.py
├── download_pdfs.py
├── filter_papers.py
└── .gitignore
```

## 许可

MIT
