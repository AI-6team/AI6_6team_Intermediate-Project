"""Inspect remaining data tables in doc_A to assess filter aggressiveness."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bidflow.parsing.hwp_html_parser import HWPHtmlParser

DOC_A = "data/raw/files/수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp"
parser = HWPHtmlParser()
_, blocks = parser.extract_with_tables(DOC_A)

data_tables = [b for b in blocks if HWPHtmlParser.is_data_table(b)]
layout_tables = [b for b in blocks if not HWPHtmlParser.is_data_table(b)]

print(f"=== doc_A: {len(data_tables)} data tables (remaining) ===\n")

# Show stats for remaining data tables
for b in data_tables[:10]:
    all_cells = []
    for row in b.rows + b.headers:
        all_cells.extend(row)
    non_empty = [c for c in all_cells if c.strip()]
    total = len(all_cells)
    fill = len(non_empty) / total if total else 0
    avg_len = sum(len(c.strip()) for c in non_empty) / max(len(non_empty), 1)
    total_text = sum(len(c.strip()) for c in all_cells)

    flat_preview = b.to_text_flat()[:150].replace('\n', ' | ')
    print(f"table_{b.table_idx}: {b.n_rows}r x {b.n_cols}c, fill={fill:.2f}, avg_len={avg_len:.1f}, total_chars={total_text}")
    print(f"  preview: {flat_preview}")
    print()

# Distribution of total_text in remaining tables
print("=== Total text character distribution (data tables) ===")
text_lens = []
for b in data_tables:
    all_cells = []
    for row in b.rows + b.headers:
        all_cells.extend(row)
    text_lens.append(sum(len(c.strip()) for c in all_cells))

text_lens.sort()
print(f"  min={min(text_lens)}, p25={text_lens[len(text_lens)//4]}, median={text_lens[len(text_lens)//2]}, p75={text_lens[3*len(text_lens)//4]}, max={max(text_lens)}")
print(f"  <50 chars: {sum(1 for t in text_lens if t < 50)}")
print(f"  <100 chars: {sum(1 for t in text_lens if t < 100)}")
print(f"  >=100 chars: {sum(1 for t in text_lens if t >= 100)}")
