import requests
import time
import os
import pandas as pd
import re

# 配置
KEYWORDS = [
    "Forestry AI Agent",
    "Smart Forest Management Large Language Model",
    "Multi-agent System Plant Phenotyping",
    "Forest Pest Detection Deep Learning",
    "Tree Species Classification Remote Sensing AI",
    "Forest Carbon Sequestration Machine Learning"
]
OUTPUT_DIR = "papers_data"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")
CSV_FILE = os.path.join(OUTPUT_DIR, "papers_list.csv")

os.makedirs(PDF_DIR, exist_ok=True)

# OpenAlex API
OPENALEX_API_URL = "https://api.openalex.org/works"

papers = []

def fetch_openalex(keyword, per_page=15):
    params = {
        "search": keyword,
        "filter": "publication_year:2020-2026",
        "sort": "cited_by_count:desc",
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

def download_pdf(url, filename):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

print("开始爬取论文 (OpenAlex API)...")
for kw in KEYWORDS:
    print(f"搜索关键词: {kw}")
    results = fetch_openalex(kw, per_page=15)
    for paper in results:
        title = paper.get("title", "No Title")
        abstract = paper.get("abstract_inverted_index")
        # 处理倒排索引摘要
        if abstract:
            abstract_words = sorted(abstract.items(), key=lambda x: list(x[1])[0])
            abstract_text = " ".join([w for w, pos in abstract_words])
        else:
            abstract_text = "No abstract available"
            
        authors = ", ".join([a.get("display_name", "") for a in paper.get("authorships", [])])
        doi = paper.get("doi", "")
        citations = paper.get("cited_by_count", 0)
        year = paper.get("publication_year", "Unknown")
        
        # 查找 OA PDF 链接
        pdf_url = None
        best_oa = paper.get("best_oa_location")
        if best_oa and best_oa.get("pdf_url"):
            pdf_url = best_oa["pdf_url"]
        else:
            # 检查其他位置
            for loc in paper.get("locations", []):
                if loc.get("pdf_url"):
                    pdf_url = loc["pdf_url"]
                    break
        
        # 去重
        if doi and any(p['doi'] == doi for p in papers):
            continue
            
        paper_info = {
            "title": title,
            "year": year,
            "authors": authors,
            "citations": citations,
            "abstract": abstract_text,
            "doi": doi,
            "pdf_url": pdf_url,
            "keyword": kw
        }
        papers.append(paper_info)
        
        # 下载 PDF (暂时禁用，避免超时)
        # if pdf_url:
        #     safe_title = re.sub(r'[^\w\s-]', '', title).strip()[:50]
        #     filename = f"{safe_title}_{year}.pdf"
        #     filepath = os.path.join(PDF_DIR, filename)
        #     if not os.path.exists(filepath):
        #         print(f"  下载: {title[:50]}...")
        #         download_pdf(pdf_url, filepath)
        #     else:
        #         print(f"  已存在: {title[:50]}...")
        
        time.sleep(0.2) # OpenAlex 限制较松，但保持礼貌

# 保存 CSV
df = pd.DataFrame(papers)
df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
print(f"\n完成！共获取 {len(papers)} 篇论文。")
print(f"数据已保存至: {CSV_FILE}")
print(f"PDF 已保存至: {PDF_DIR}")
