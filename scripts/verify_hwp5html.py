"""Verify hwp5html parsing works correctly with Korean text."""
import subprocess
from bs4 import BeautifulSoup
import sys, os

sys.stdout.reconfigure(encoding='utf-8')

DOCS = {
    "doc_A": "data/raw/files/수협중앙회_수협중앙회 수산물사이버직매장 시스템 재구축 ISMP 수립 입.hwp",
    "doc_B": "data/raw/files/한국교육과정평가원_국가교육과정정보센터(NCIC) 시스템 운영 및 개선.hwp",
    "doc_C": "data/raw/files/국립중앙의료원_(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축.hwp",
    "doc_D": "data/raw/files/한국철도공사 (용역)_예약발매시스템 개량 ISMP 용역.hwp",
    "doc_E": "data/raw/files/재단법인스포츠윤리센터_스포츠윤리센터 LMS(학습지원시스템) 기능개선.hwp",
}

for doc_key, path in DOCS.items():
    print(f"\n{'='*60}")
    print(f"[{doc_key}] {os.path.basename(path)}")

    try:
        result = subprocess.run(
            ['hwp5html', '--html', path],
            capture_output=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  ERROR: hwp5html failed (code {result.returncode})")
            print(f"  stderr: {result.stderr.decode('utf-8', errors='replace')[:200]}")
            continue

        html = result.stdout.decode('utf-8', errors='replace')
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all('table')
        all_text = soup.get_text()
        table_text = sum(len(t.get_text()) for t in tables)

        print(f"  HTML: {len(html):,} bytes")
        print(f"  Tables: {len(tables)}")
        print(f"  Total text: {len(all_text):,} chars")
        print(f"  Table text: {table_text:,} chars ({table_text/max(len(all_text),1)*100:.1f}%)")

        # Show first table sample
        if tables:
            t = tables[0]
            rows = t.find_all('tr')
            print(f"  First table: {len(rows)} rows")
            for j, row in enumerate(rows[:2]):
                cells = row.find_all(['td', 'th'])
                parts = [c.get_text(strip=True)[:30] for c in cells]
                print(f"    Row {j}: {' | '.join(parts)}")

        # Show text sample (non-table)
        for el in soup.find_all(['p', 'span']):
            txt = el.get_text(strip=True)
            if len(txt) > 10 and not el.find_parent('table'):
                print(f"  Text sample: {txt[:80]}")
                break

    except Exception as e:
        print(f"  EXCEPTION: {e}")

print(f"\n{'='*60}")
print("Verification complete.")
