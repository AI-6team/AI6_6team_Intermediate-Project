"""Diagnose why V1 (html_basic) dropped on doc_A (text_only)."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import subprocess
from bs4 import BeautifulSoup

# doc_A: text_only document
DOC_A = "data/raw/files/수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp"

# 1. hwp5txt output
txt_result = subprocess.run(['hwp5txt', DOC_A], capture_output=True, timeout=60, encoding='utf-8')
txt_output = txt_result.stdout

# 2. hwp5html output
html_result = subprocess.run(['hwp5html', '--html', DOC_A], capture_output=True, timeout=60)
html_raw = html_result.stdout.decode('utf-8', errors='replace')
soup = BeautifulSoup(html_raw, 'html.parser')
html_text = soup.get_text(separator='\n')

# Clean
import re
txt_clean = re.sub(r'\s+', ' ', txt_output).strip()
html_clean = re.sub(r'\s+', ' ', html_text).strip()

print(f"hwp5txt: {len(txt_output)} raw chars, {len(txt_clean)} clean chars")
print(f"hwp5html: {len(html_text)} raw chars, {len(html_clean)} clean chars")
print(f"Ratio: html/txt = {len(html_clean)/max(len(txt_clean),1):.2f}x")

# 3. Key question keywords - check if they exist in both
# From golden testset, doc_A questions:
# "본 사업의 사업기간은?", "본 사업의 추정가격은?", etc.
keywords = [
    "ISMP", "수산물", "사이버", "직매장", "재구축",
    "수협중앙회", "사업기간", "추정가격", "제안서",
    "보안", "인증", "시스템",
]

print("\n--- Keyword presence ---")
for kw in keywords:
    in_txt = kw.lower() in txt_clean.lower()
    in_html = kw.lower() in html_clean.lower()
    status = "BOTH" if (in_txt and in_html) else ("TXT_ONLY" if in_txt else ("HTML_ONLY" if in_html else "NEITHER"))
    print(f"  {kw}: {status}")

# 4. Show text sample differences
print("\n--- hwp5txt first 500 chars ---")
print(txt_clean[:500])

print("\n--- hwp5html first 500 chars ---")
print(html_clean[:500])

# 5. Check for HTML artifacts in html_text
html_artifacts = []
for pattern in ['class=', 'style=', '<div', '<span', '<table', 'font-', 'text-decoration']:
    count = html_text.lower().count(pattern)
    if count > 0:
        html_artifacts.append(f"{pattern}: {count}")
print(f"\n--- HTML artifacts in text ---")
for a in html_artifacts:
    print(f"  {a}")

# 6. Check chunk count difference
from bidflow.parsing.hwp_parser import HWPParser
from bidflow.parsing.table_chunker import TableAwareChunker

parser = HWPParser()
chunks_v0 = parser.parse(DOC_A)
chunker = TableAwareChunker(chunk_size=500, chunk_overlap=50)
chunks_v1 = chunker.chunk_v1_basic(DOC_A)

print(f"\n--- Chunk comparison ---")
print(f"V0 (hwp5txt): {len(chunks_v0)} chunks")
print(f"V1 (html_basic): {len(chunks_v1)} chunks")

# Compare content overlap
v0_texts = set(c.text[:100] for c in chunks_v0)
v1_texts = set(d.page_content[:100] for d in chunks_v1)
overlap = len(v0_texts & v1_texts)
print(f"First-100-char overlap: {overlap}/{len(v0_texts)} V0 chunks found in V1")
