import requests
import time
import os
import pandas as pd

# 配置：针对 4 个具体细分方向
SEARCH_QUERIES = [
    # 1. 碳汇与生物量估算
    "Forest Carbon Stock Estimation Machine Learning 2024",
    "Biomass Prediction Deep Learning Forest",
    # 2. 树种分类与遥感
    "Tree Species Classification Transformer Remote Sensing",
    "Forest Mapping Satellite Imagery Deep Learning",
    # 3. 灾害监测（火灾/病虫害）
    "Forest Fire Detection Computer Vision Deep Learning",
    "Plant Disease Detection YOLO Transformer Agriculture",
    # 4. 智慧林业与数字孪生
    "Smart Forestry Digital Twin IoT",
    "Precision Forestry AI Agent"
]

OUTPUT_DIR = "papers_data"
CSV_FILE = os.path.join(OUTPUT_DIR, "targeted_forestry_ai_papers.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENALEX_API_URL = "https://api.openalex.org/works"

papers = []

def fetch_openalex(query, per_page=20):
    params = {
        "search": query,
        "filter": "publication_year:2024-2026",
        "sort": "publication_date:desc",
        "per_page": per_page
    }
    try:
        response = requests.get(OPENALEX_API_URL, params=params)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        print(f"Error fetching {query}: {e}")
        return []

print("开始定向爬取细分领域前沿论文...")
for query in SEARCH_QUERIES:
    print(f"搜索: {query}")
    results = fetch_openalex(query, per_page=20)
    for paper in results:
        title = paper.get("title", "")
        doi = paper.get("doi", "")
        
        # 去重
        if doi and any(p['doi'] == doi for p in papers):
            continue
            
        # 提取摘要
        abstract = paper.get("abstract_inverted_index")
        if abstract:
            abstract_words = sorted(abstract.items(), key=lambda x: list(x[1])[0])
            abstract_text = " ".join([w for w, pos in abstract_words])
        else:
            abstract_text = "No abstract available"
            
        authors = ", ".join([a.get("display_name", "") for a in paper.get("authorships", [])])
        citations = paper.get("cited_by_count", 0)
        year = paper.get("publication_year", "Unknown")
        pub_date = paper.get("publication_date", "Unknown")
        
        # 查找 PDF
        pdf_url = None
        best_oa = paper.get("best_oa_location")
        if best_oa and best_oa.get("pdf_url"):
            pdf_url = best_oa["pdf_url"]
            
        paper_info = {
            "title": title,
            "year": year,
            "pub_date": pub_date,
            "authors": authors,
            "citations": citations,
            "abstract": abstract_text,
            "doi": doi,
            "pdf_url": pdf_url,
            "category": query
        }
        papers.append(paper_info)
        
        time.sleep(0.1)

df = pd.DataFrame(papers)
df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
print(f"\n完成！共获取 {len(df)} 篇定向论文。")

# 生成 Markdown
md = '# 🌲 林业 AI 细分领域前沿论文 (2024-2026)\n\n'
md += f'**按 4 大核心方向分类，按时间倒序排列**\n\n'

# 按类别分组
categories = df['category'].unique()
for cat in categories:
    cat_df = df[df['category'] == cat]
    md += f"## 📂 方向: {cat}\n\n"
    for i, row in cat_df.iterrows():
        md += f'### {row["title"]}\n'
        md += f'- **时间**: {row["pub_date"]} | **引用**: {row["citations"]}\n'
        if pd.notna(row['pdf_url']):
            md += f'- **PDF**: [链接]({row["pdf_url"]})\n'
        # 摘要前 150 字
        if pd.notna(row['abstract']) and row['abstract'] != 'No abstract available':
            abs_text = row['abstract'][:150] + '...' if len(row['abstract']) > 150 else row['abstract']
            md += f'- **摘要**: {abs_text}\n'
        md += '\n---\n\n'

with open('papers_data/targeted_forestry_ai_papers.md', 'w', encoding='utf-8') as f:
    f.write(md)

print("报告已生成: papers_data/targeted_forestry_ai_papers.md")
