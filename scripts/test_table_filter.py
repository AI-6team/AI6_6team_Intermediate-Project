"""Test table quality filter effect on all 5 documents."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bidflow.parsing.hwp_html_parser import HWPHtmlParser

DOCS = {
    "doc_A (text_only)": "data/raw/files/수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp",
    "doc_B (table_simple)": "data/raw/files/한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
    "doc_C (table_complex)": "data/raw/files/국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp",
    "doc_D (mixed)": "data/raw/files/한국지능정보사회진흥원_2024년 AI기반 소프트웨어 개발 지원(AI Pair-Pr.hwp",
    "doc_E (representative)": "data/raw/files/한국도로공사_한국도로공사 CCTV 관제시스템(ITS) 공사(시흥방향) 입찰.hwp",
}

parser = HWPHtmlParser()

print("=== Table Quality Filter Test ===\n")
for label, path in DOCS.items():
    _, blocks = parser.extract_with_tables(path)
    data_tables = [b for b in blocks if HWPHtmlParser.is_data_table(b)]
    layout_tables = [b for b in blocks if not HWPHtmlParser.is_data_table(b)]

    print(f"{label}:")
    print(f"  Total tables: {len(blocks)}")
    print(f"  Data tables:  {len(data_tables)} (kept)")
    print(f"  Layout tables: {len(layout_tables)} (filtered)")

    if layout_tables:
        # Show why some were filtered
        reasons = []
        for b in layout_tables[:3]:
            r = []
            if b.n_rows < 2: r.append(f"rows={b.n_rows}<2")
            if b.n_cols < 2: r.append(f"cols={b.n_cols}<2")
            all_cells = []
            for row in b.rows + b.headers:
                all_cells.extend(row)
            if all_cells:
                non_empty = [c for c in all_cells if c.strip()]
                fill = len(non_empty) / len(all_cells)
                avg = sum(len(c.strip()) for c in non_empty) / max(len(non_empty), 1)
                if fill < 0.3: r.append(f"fill={fill:.2f}<0.3")
                if avg < 2.0: r.append(f"avg_len={avg:.1f}<2.0")
            reasons.append(f"    table_{b.table_idx}: {', '.join(r) if r else 'marginal'}")
        for line in reasons:
            print(line)
        if len(layout_tables) > 3:
            print(f"    ... and {len(layout_tables)-3} more")
    print()
