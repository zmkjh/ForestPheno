import pandas as pd
import re

# 读取最新爬取的论文
df = pd.read_csv('papers_data/latest_forestry_ai_papers.csv')

# 定义核心关键词（必须包含至少一个）
MUST_HAVE = [
    'forest', 'tree', 'plant', 'wood', 'timber', 'crop', 'agriculture', 'phenotyp',
    'ecology', 'biodiversity', 'carbon sink', 'vegetation'
]

# 定义技术关键词（必须包含至少一个）
TECH_KEYS = [
    'agent', 'llm', 'large language model', 'transformer', 'deep learning', 
    'machine learning', 'neural network', 'cnn', 'vision', 'remote sensing',
    'multimodal', 'ai ', 'artificial intelligence', 'digital twin', 'iot'
]

def is_relevant(title):
    if pd.isna(title): return False
    title_lower = title.lower()
    has_topic = any(kw in title_lower for kw in MUST_HAVE)
    has_tech = any(kw in title_lower for kw in TECH_KEYS)
    return has_topic and has_tech

# 过滤
filtered_df = df[df['title'].apply(is_relevant)]

print(f"原始论文数：{len(df)}")
print(f"过滤后高相关论文数：{len(filtered_df)}")

# 保存
filtered_df.to_csv('papers_data/selected_forestry_ai_papers.csv', index=False, encoding='utf-8-sig')

# 生成报告
md = '# 🌲 林业 + AI Agent 精选前沿论文清单 (2024-2026)\n\n'
md += f'**经过严格筛选，保留同时包含“林业主题”与“AI技术”关键词的 {len(filtered_df)} 篇论文**\n\n'

for i, row in filtered_df.iterrows():
    md += f'## {row["title"]}\n'
    md += f'- **发表时间**: {row["pub_date"]}\n'
    md += f'- **引用数**: {row["citations"]}\n'
    md += f'- **作者**: {row["authors"]}\n'
    if pd.notna(row['doi']):
        md += f'- **DOI**: [{row["doi"]}]({row["doi"]})\n'
    if pd.notna(row['pdf_url']):
        md += f'- **PDF**: [下载链接]({row["pdf_url"]})\n'
    
    if pd.notna(row['abstract']) and row['abstract'] != 'No abstract available':
        abstract = row['abstract'][:300] + '...' if len(row['abstract']) > 300 else row['abstract']
        md += f'- **摘要**: {abstract}\n'
    md += '\n---\n\n'

with open('papers_data/selected_forestry_ai_papers.md', 'w', encoding='utf-8') as f:
    f.write(md)

print("精选报告已生成: papers_data/selected_forestry_ai_papers.md")
