"""
HWP HTML Parser — hwp5html CLI 기반 테이블 + 텍스트 통합 추출

Phase CE: hwp5txt 대체 파서
- hwp5html --html → BeautifulSoup → 텍스트/테이블 분리 추출
- 테이블: col_path 기반 구조화 (EXP07 approach 응용)
- 텍스트: 비테이블 영역 plain text 추출
"""
import os
import re
import subprocess
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Tag


@dataclass
class TableBlock:
    """구조화된 테이블 블록"""
    table_idx: int
    caption: str = ""
    headers: List[List[str]] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    col_paths: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_text_flat(self) -> str:
        """단순 직렬화: 모든 셀을 행 단위로 ' | ' 구분"""
        lines = []
        if self.caption:
            lines.append(f"[표] {self.caption}")
        for row in self.headers:
            lines.append(" | ".join(row))
        for row in self.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)

    def to_text_colpath(self) -> str:
        """col_path 기반 직렬화: 각 데이터 행을 'col_path: value' 형식으로"""
        lines = []
        if self.caption:
            lines.append(f"[표] {self.caption}")
        if not self.col_paths:
            return self.to_text_flat()
        for row in self.rows:
            parts = []
            for j, cell in enumerate(row):
                if j < len(self.col_paths) and cell.strip():
                    parts.append(f"{self.col_paths[j]}: {cell.strip()}")
            if parts:
                lines.append(" | ".join(parts))
        return "\n".join(lines)


class HWPHtmlParser:
    """
    hwp5html CLI → BeautifulSoup → 텍스트/테이블 분리 추출

    Modes:
        - "basic": 텍스트만 추출 (테이블 포함, flat 직렬화)
        - "table": 테이블 구조화 추출 + col_path 직렬화
        - "full": table + 텍스트/테이블 분리 chunking 지원
    """

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    @staticmethod
    def is_data_table(block: 'TableBlock',
                      min_rows: int = 2,
                      min_cols: int = 2,
                      min_fill_ratio: float = 0.3,
                      min_avg_cell_len: float = 2.0) -> bool:
        """
        레이아웃/서식 테이블 vs 데이터 테이블 판별

        데이터 테이블 조건 (모두 충족해야 함):
          1) 데이터 행 >= min_rows
          2) 컬럼 수 >= min_cols
          3) 비어있지 않은 셀 비율 >= min_fill_ratio
          4) 비어있지 않은 셀의 평균 길이 >= min_avg_cell_len
        """
        # 행/열 수 체크
        if block.n_rows < min_rows or block.n_cols < min_cols:
            return False

        # 셀 통계 계산
        all_cells = []
        for row in block.rows:
            all_cells.extend(row)
        for row in block.headers:
            all_cells.extend(row)

        total = len(all_cells)
        if total == 0:
            return False

        non_empty = [c for c in all_cells if c.strip()]
        fill_ratio = len(non_empty) / total
        if fill_ratio < min_fill_ratio:
            return False

        avg_len = sum(len(c.strip()) for c in non_empty) / max(len(non_empty), 1)
        if avg_len < min_avg_cell_len:
            return False

        return True

    def parse_html(self, file_path: str) -> Optional[BeautifulSoup]:
        """hwp5html CLI로 HTML 추출 후 BeautifulSoup 객체 반환"""
        try:
            result = subprocess.run(
                ['hwp5html', '--html', file_path],
                capture_output=True,
                timeout=self.timeout
            )
            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='replace')[:200]
                print(f"[HWPHtmlParser] hwp5html failed (code {result.returncode}): {stderr}")
                return None

            html = result.stdout.decode('utf-8', errors='replace')
            if not html.strip():
                print("[HWPHtmlParser] hwp5html returned empty output")
                return None

            return BeautifulSoup(html, 'html.parser')

        except FileNotFoundError:
            print("[HWPHtmlParser] hwp5html command not found")
            return None
        except subprocess.TimeoutExpired:
            print(f"[HWPHtmlParser] hwp5html timed out ({self.timeout}s)")
            return None
        except Exception as e:
            print(f"[HWPHtmlParser] Error: {e}")
            return None

    def extract_text_basic(self, file_path: str) -> str:
        """모드 'basic': 전체 텍스트 추출 (테이블 flat 포함)"""
        soup = self.parse_html(file_path)
        if not soup:
            return ""

        text = soup.get_text(separator='\n')
        text = self._normalize_whitespace(text)
        return text

    def extract_with_tables(self, file_path: str) -> Tuple[str, List[TableBlock]]:
        """
        모드 'table'/'full': 텍스트와 테이블을 분리 추출

        Returns:
            (non_table_text, list_of_TableBlock)
        """
        soup = self.parse_html(file_path)
        if not soup:
            return "", []

        # 1) 테이블 추출
        table_blocks = self._extract_tables(soup)

        # 2) 비테이블 텍스트 추출
        non_table_text = self._extract_non_table_text(soup)

        return non_table_text, table_blocks

    def _extract_tables(self, soup: BeautifulSoup) -> List[TableBlock]:
        """HTML에서 모든 <table> 추출 → TableBlock 리스트"""
        tables = soup.find_all('table')
        blocks = []

        for idx, table in enumerate(tables):
            # 중첩 테이블은 부모만 처리 (자식 테이블은 부모의 셀 텍스트에 포함)
            if table.find_parent('table'):
                continue

            block = self._parse_single_table(table, idx)
            if block and (block.n_rows > 0 or block.headers):
                blocks.append(block)

        return blocks

    def _parse_single_table(self, table: Tag, idx: int) -> Optional[TableBlock]:
        """단일 <table> 요소를 TableBlock으로 변환"""
        # caption 추출
        caption_el = table.find('caption')
        caption = caption_el.get_text(strip=True) if caption_el else ""

        # 이전 형제에서 caption 후보 찾기
        if not caption:
            caption = self._find_caption_before(table)

        # rowspan/colspan을 해제하여 정규 그리드로 변환
        grid = self._resolve_spans(table)
        if not grid:
            return None

        # 헤더/데이터 행 분리
        headers, data_rows = self._split_header_data(table, grid)

        # col_path 구축
        col_paths = self._build_col_paths(headers) if headers else []

        n_cols = max((len(r) for r in grid), default=0)

        return TableBlock(
            table_idx=idx,
            caption=caption,
            headers=headers,
            rows=data_rows,
            col_paths=col_paths,
            n_rows=len(data_rows),
            n_cols=n_cols,
        )

    def _resolve_spans(self, table: Tag) -> List[List[str]]:
        """rowspan/colspan을 해제하여 정규 2D 그리드 생성"""
        rows = table.find_all('tr')
        if not rows:
            return []

        # 최대 열 수 추정
        max_cols = 0
        for row in rows:
            cols_in_row = 0
            for cell in row.find_all(['td', 'th']):
                cols_in_row += int(cell.get('colspan', 1))
            max_cols = max(max_cols, cols_in_row)

        if max_cols == 0:
            return []

        # 빈 그리드 초기화
        grid = [[None] * max_cols for _ in range(len(rows))]

        for r_idx, row in enumerate(rows):
            col_cursor = 0
            for cell in row.find_all(['td', 'th']):
                # 이미 채워진 셀 건너뛰기
                while col_cursor < max_cols and grid[r_idx][col_cursor] is not None:
                    col_cursor += 1
                if col_cursor >= max_cols:
                    break

                text = cell.get_text(strip=True)
                rs = int(cell.get('rowspan', 1))
                cs = int(cell.get('colspan', 1))

                for dr in range(rs):
                    for dc in range(cs):
                        ri = r_idx + dr
                        ci = col_cursor + dc
                        if ri < len(grid) and ci < max_cols:
                            grid[ri][ci] = text

                col_cursor += cs

        # None을 빈 문자열로 채우기
        for r in grid:
            for i in range(len(r)):
                if r[i] is None:
                    r[i] = ""

        return grid

    def _split_header_data(self, table: Tag, grid: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """<thead>/<th> 기반으로 헤더 행과 데이터 행 분리"""
        rows = table.find_all('tr')

        # 방법 1: <thead> 존재 시
        thead = table.find('thead')
        if thead:
            thead_rows = thead.find_all('tr')
            n_header = len(thead_rows)
            return grid[:n_header], grid[n_header:]

        # 방법 2: <th> 기반 탐지
        header_row_indices = []
        for i, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            th_count = sum(1 for c in cells if c.name == 'th')
            if th_count > 0 and th_count >= len(cells) * 0.5:
                header_row_indices.append(i)
            else:
                break  # 연속된 헤더 행만 취급

        if header_row_indices:
            n_header = max(header_row_indices) + 1
            return grid[:n_header], grid[n_header:]

        # 방법 3: 첫 번째 행을 헤더로 추정 (2행 이상인 경우)
        if len(grid) >= 2:
            return grid[:1], grid[1:]

        return [], grid

    def _build_col_paths(self, headers: List[List[str]]) -> List[str]:
        """
        다단 헤더에서 col_path 구축 (EXP07 approach)

        예시: headers = [["사업개요", "사업개요", "세부사항"],
                         ["사업명", "총사업비", "기간"]]
        → col_paths = ["사업개요/사업명", "사업개요/총사업비", "세부사항/기간"]
        """
        if not headers:
            return []

        if len(headers) == 1:
            return [h.strip() if h.strip() else f"col_{i}" for i, h in enumerate(headers[0])]

        n_cols = max(len(row) for row in headers)
        col_paths = []

        for col_idx in range(n_cols):
            parts = []
            prev_part = None
            for row in headers:
                if col_idx < len(row):
                    val = row[col_idx].strip()
                    # 중복 제거 (상위 헤더와 동일하면 skip)
                    if val and val != prev_part:
                        parts.append(val)
                        prev_part = val
            path = "/".join(parts) if parts else f"col_{col_idx}"
            col_paths.append(path)

        return col_paths

    def _extract_non_table_text(self, soup: BeautifulSoup) -> str:
        """테이블 외부의 텍스트만 추출"""
        # soup 복사 후 테이블 제거
        soup_copy = BeautifulSoup(str(soup), 'html.parser')
        for table in soup_copy.find_all('table'):
            table.decompose()

        text = soup_copy.get_text(separator='\n')
        return self._normalize_whitespace(text)

    def _find_caption_before(self, table: Tag) -> str:
        """테이블 직전 요소에서 caption 후보 탐색"""
        prev = table.find_previous_sibling()
        if prev and prev.name in ('p', 'div', 'span'):
            text = prev.get_text(strip=True)
            # "표", "Table", "[표" 등으로 시작하면 caption으로 판단
            if text and len(text) < 100:
                if re.match(r'^(\[?표|Table|\[Table|<표)', text, re.IGNORECASE):
                    return text
                # 짧은 텍스트 (제목 가능성)
                if len(text) < 50:
                    return text
        return ""

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """공백 정규화"""
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
