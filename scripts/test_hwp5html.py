"""Quick test: hwp5html table structure analysis"""
import subprocess
from bs4 import BeautifulSoup

DOC = "data/raw/files/한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp"

result = subprocess.run(
    ['hwp5html', '--html', DOC],
    capture_output=True, timeout=60
)
html = result.stdout.decode('utf-8', errors='replace')
soup = BeautifulSoup(html, 'html.parser')

tables = soup.find_all('table')
print(f'Total tables: {len(tables)}')

# Show first 3 tables
for i, table in enumerate(tables[:3]):
    rows = table.find_all('tr')
    print(f'\n--- Table {i} ({len(rows)} rows) ---')
    for j, row in enumerate(rows[:5]):
        cells = row.find_all(['td', 'th'])
        cell_texts = []
        for c in cells:
            txt = c.get_text(strip=True)[:40]
            rs = c.get('rowspan', '1')
            cs = c.get('colspan', '1')
            span_info = ''
            if rs != '1' or cs != '1':
                span_info = f'[{rs}x{cs}]'
            cell_texts.append(f'{span_info}{txt}')
        print(f'  Row {j}: {" | ".join(cell_texts)}')
    if len(rows) > 5:
        print(f'  ... ({len(rows) - 5} more rows)')

# Text extraction comparison
all_text = soup.get_text()
# Remove excessive whitespace
import re
clean_text = re.sub(r'\s+', ' ', all_text).strip()
print(f'\nTotal text length (raw): {len(all_text)}')
print(f'Total text length (clean): {len(clean_text)}')

# Table text vs non-table text
table_text_len = sum(len(t.get_text()) for t in tables)
non_table_text = len(all_text) - table_text_len
print(f'Table text: {table_text_len} chars')
print(f'Non-table text: {non_table_text} chars')
print(f'Table ratio: {table_text_len/len(all_text)*100:.1f}%')
