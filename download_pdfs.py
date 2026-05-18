"""Batch download PDFs from paper metadata with validation."""

import csv
import os
import time
import requests
from pathlib import Path

PDF_DIR = Path('papers_data/pdfs_downloaded')
PDF_DIR.mkdir(exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/pdf, */*',
}
TIMEOUT = 60
RETRIES = 2

def safe_filename(title: str, max_len=80) -> str:
    safe = ''.join(c for c in title if c.isalnum() or c in ' _-.').strip()
    return safe[:max_len] or 'untitled'

def is_valid_pdf(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            return f.read(4) == b'%PDF'
    except Exception:
        return False

def try_url(url: str) -> bytes | None:
    for attempt in range(RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code == 200:
                ct = r.headers.get('Content-Type', '')
                if 'text/html' in ct and 'pdf' not in url.lower():
                    return None  # HTML page, not PDF
                if len(r.content) > 10000:  # at least 10KB
                    return r.content
        except requests.exceptions.SSLError:
            try:
                r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False)
                if r.status_code == 200 and len(r.content) > 10000:
                    return r.content
            except Exception:
                pass
        except Exception:
            time.sleep(2)
    return None

def download_paper(title: str, doi: str, pdf_url: str):
    if not pdf_url:
        return False, 'no URL'

    fname = safe_filename(title) + '.pdf'
    fpath = PDF_DIR / fname

    if fpath.exists() and is_valid_pdf(fpath):
        return True, 'already exists'

    content = try_url(pdf_url)
    if content is None:
        return False, 'download failed'

    fpath.write_bytes(content)
    if is_valid_pdf(fpath):
        size = fpath.stat().st_size
        return True, f'OK ({size//1024}KB)'
    else:
        fpath.unlink(missing_ok=True)
        return False, 'invalid PDF header'

def main():
    csv_path = 'papers_data/targeted_forestry_ai_papers.csv'
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        papers = list(reader)

    print(f'Total papers: {len(papers)}')

    results = []
    for i, paper in enumerate(papers):
        title = paper.get('title', '')
        doi = paper.get('doi', '')
        pdf_url = paper.get('pdf_url', '')
        category = paper.get('category', '')

        ok, msg = download_paper(title, doi, pdf_url)
        results.append((ok, title[:60], msg, category))
        status = 'OK' if ok else 'FAIL'
        print(f'[{i+1}/{len(papers)}] {status}: {msg} | {title[:50]}')

        time.sleep(1)

    # Summary
    ok_count = sum(1 for r in results if r[0])
    print(f'\n=== Summary ===')
    print(f'Downloaded: {ok_count}/{len(papers)}')
    by_cat = {}
    for ok, title, msg, cat in results:
        by_cat.setdefault(cat, {'ok': 0, 'total': 0})
        by_cat[cat]['total'] += 1
        if ok:
            by_cat[cat]['ok'] += 1
    for cat, v in sorted(by_cat.items()):
        print(f'  {cat}: {v["ok"]}/{v["total"]}')

if __name__ == '__main__':
    main()
