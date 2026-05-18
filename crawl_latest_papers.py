import requests
import time
import os
import pandas as pd
import re

# 配置：针对前沿、低引用、新方向的关键词
KEYWORDS = [
    "Forestry Large Language Model",
    "Forest Management AI Agent",
    "Multi-agent System Smart Forestry",
    "Tree Species Classification Transformer 2024",
    "Plant Phenotyping AI Agent",
    "Forest Pest Detection Large Model",
    "Smart Forestry Digital Twin",
    "Agriculture LLM Agent"
]
OUTPUT_DIR = "papers_data"
CSV_FILE = os.path.join(OUTPUT_DIR, "latest_forestry_ai_papers.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# OpenAlex API
OPENALEX_API_URL = "https://api.openalex.org/works"

papers = []

def fetch_openalex(keyword, per_page=20):
    params = {
        "search": keyword,
        # 核心修改：只看最近 3 年
        "filter": "publication_year:2024-2026",
        # 核心修改：按发表时间倒序，不看引用数
        "sort": "publication_date:desc",
        "per_page": per_page
    }
    try:
        response = requests.get(OPENALEX_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        print(f"Error fetching {keyword}: {e}")
        return []

print("开始爬取最新前沿论文 (2024-2026, 按时间倒序)...")
for kw in KEYWORDS:
    print(f"搜索关键词: {kw}")
    results = fetch_openalex(kw, per_page=20)
    for paper in results:
        title = paper.get("title", "No Title")
        
        # 处理摘要
        abstract = paper.get("abstract_inverted_index")
        if abstract:
            abstract_words = sorted(abstract.items(), key=lambda x: list(x[1])[0])
            abstract_text = " ".join([w for w, pos in abstract_words])
        else:
            abstract_text = "No abstract available"
            
        authors = ", ".join([a.get("display_name", "") for a in paper.get("authorships", [])])
        doi = paper.get("doi", "")
        citations = paper.get("cited_by_count", 0)
        year = paper.get("publication_year", "Unknown")
        pub_date = paper.get("publication_date", "Unknown")
        
        # 查找 OA PDF 链接
        pdf_url = None
        best_oa = paper.get("best_oa_location")
        if best_oa and best_oa.get("pdf_url"):
            pdf_url = best_oa["pdf_url"]
        
        # 简单去重
        if doi and any(p['doi'] == doi for p in papers):
            continue
            
        paper_info = {
            "title": title,
            "year": year,
            "pub_date": pub_date,
            "authors": authors,
            "citations": citations,
            "abstract": abstract_text,
            "doi": doi,
            "pdf_url": pdf_url,
            "keyword": kw
        }
        papers.append(paper_info)
        
        time.sleep(0.1) # 快速请求

# 保存 CSV
df = pd.DataFrame(papers)
df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
print(f"\n完成！共获取 {len(papers)} 篇最新论文。")
print(f"数据已保存至: {CSV_FILE}")

# 生成 Markdown 报告
md = '# 🌲 林业 + AI Agent 前沿论文清单 (2024-2026)\n\n'
md += f'**筛选策略：按发表时间倒序，不看引用数，专注最新技术方向**\n\n'
md += f'**共收录 {len(df)} 篇论文**\n\n'

# 按年份分组展示
for year in sorted(df['year'].unique(), reverse=True):
    year_df = df[df['year'] == year]
    md += f"## {year} 年 ({len(year_df)} 篇)\n\n"
    for i, row in year_df.iterrows():
        md += f'### {row["title"]}\n'
        md += f'- **发表时间**: {row["pub_date"]}\n'
        md += f'- **引用数**: {row["citations"]} (新发表，引用少属正常)\n'
        md += f'- **作者**: {row["authors"]}\n'
        if pd.notna(row['doi']):
            md += f'- **DOI**: [{row["doi"]}]({row["doi"]})\n'
        if pd.notna(row['pdf_url']):
            md += f'- **PDF**: [下载链接]({row["pdf_url"]})\n'
        
        # 提取摘要前 200 字
        if pd.notna(row['abstract']) and row['abstract'] != 'No abstract available':
            abstract = row['abstract'][:200] + '...' if len(row['abstract']) > 200 else row['abstract']
            md += f'- **摘要**: {abstract}\n'
        md += '\n---\n\n'

with open('papers_data/latest_forestry_ai_papers.md', 'w', encoding='utf-8') as f:
    f.write(md)

print("报告已生成: papers_data/latest_forestry_ai_papers.md")
