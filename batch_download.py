"""Batch download PDFs using Python (CSV parsing) + curl (robust SSL)."""

import csv
import os
import subprocess
import time
from pathlib import Path

CSV_PATH = 'papers_data/targeted_forestry_ai_papers.csv'
OUT_DIR = Path('papers_data/pdfs_downloaded')
OUT_DIR.mkdir(exist_ok=True)

CURL_CMD = ['curl', '-L', '-k', '-s', '-o']
HEADERS = ['-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)']
TIMEOUT = ['--max-time', '60']

WORKING_SOURCES = {
    'nature.com': 'Nature',
    'arxiv.org': 'arXiv',
    'frontiersin': 'Frontiers',
    'researchsquare': 'ResearchSquare',
    'link.springer': 'Springer',
    'springer.com': 'Springer',
    'eartharxiv': 'EartharXiv',
    'copernicus.org': 'Copernicus',
}

def classify_source(url: str) -> str | None:
    for domain, label in WORKING_SOURCES.items():
        if domain in url:
            return label
    return None

def safe_filename(title: str, max_len=60) -> str:
    safe = ''.join(c for c in title if c.isalnum() or c in ' _-.').strip() or 'untitled'
    return safe[:max_len]

def is_valid_pdf(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF' and path.stat().st_size > 10000
    except Exception:
        return False

def main():
    with open(CSV_PATH, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        papers = list(reader)

    print(f'Total papers: {len(papers)}')
    
    # Filter to working sources
    candidates = []
    for p in papers:
        url = p.get('pdf_url', '') or ''
        src = classify_source(url)
        if src:
            candidates.append((p, src, url))
    
    print(f'Downloadable (known good sources): {len(candidates)}')
    for src in sorted(set(s for _, s, _ in candidates)):
        cnt = sum(1 for _, s, _ in candidates if s == src)
        print(f'  {src}: {cnt}')
    
    results = []
    for i, (paper, src, url) in enumerate(candidates):
        title = paper.get('title', '')
        doi = paper.get('doi', '')
        
        fname = f'{src}_{safe_filename(title)}.pdf'
        fpath = OUT_DIR / fname
        
        # Skip if already valid
        if fpath.exists() and is_valid_pdf(fpath):
            print(f'[{i+1}/{len(candidates)}] SKIP (exists) [{src}] {title[:50]}')
            results.append((True, title[:60], 'exists', src))
            continue
        
        print(f'[{i+1}/{len(candidates)}] DOWNLOAD [{src}] {title[:50]}')
        
        # Use curl
        cmd = CURL_CMD + [str(fpath), url] + HEADERS + TIMEOUT
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(f'  FAIL: timeout')
            results.append((False, title[:60], 'timeout', src))
            continue
        
        if fpath.exists() and is_valid_pdf(fpath):
            size = fpath.stat().st_size
            print(f'  OK: {size//1024} KB')
            results.append((True, title[:60], f'{size//1024}KB', src))
        else:
            if fpath.exists():
                sz = fpath.stat().st_size
                print(f'  FAIL: invalid PDF (size={sz})')
                fpath.unlink()
            else:
                print(f'  FAIL: no file')
            results.append((False, title[:60], 'invalid', src))
        
        time.sleep(0.5)
    
    # Summary
    ok = sum(1 for r in results if r[0])
    print(f'\n===== Summary =====')
    print(f'Downloaded: {ok}/{len(candidates)}')
    
    by_src = {}
    for ok_flag, _, _, src in results:
        by_src.setdefault(src, {'ok': 0, 'total': 0})
        by_src[src]['total'] += 1
        if ok_flag:
            by_src[src]['ok'] += 1
    for src, v in sorted(by_src.items()):
        print(f'  {src}: {v["ok"]}/{v["total"]}')

if __name__ == '__main__':
    main()
