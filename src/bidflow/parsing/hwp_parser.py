import os
import olefile
import zlib
import subprocess
from typing import List
from bidflow.domain.models import ParsedChunk, ParsedTable

class HWPParser:
    """
    HWP 텍스트 추출 파서
    Primary: pyhwp(hwp5txt)를 사용한 고품질 텍스트 추출
    Fallback: olefile을 사용한 기본 추출 (텍스트 품질 낮음)
    """
    def parse(self, file_path: str) -> List[ParsedChunk]:
        # 1차 시도: hwp5txt CLI 사용 (고품질)
        text_content = self._parse_with_hwp5txt(file_path)

        # 2차 시도: olefile fallback (저품질)
        if not text_content:
            print("[HWPParser] hwp5txt failed, falling back to olefile...")
            text_content = self._parse_with_olefile(file_path)

        if not text_content:
            print("[HWPParser] All parsing methods failed")
            return []

        # [Conservative Normalization] 분할 전 정규화
        from bidflow.parsing.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        normalized_text = preprocessor.normalize(text_content)

        # Text Splitter 적용 (Token Limit 방지)
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        split_texts = splitter.split_text(normalized_text)

        chunks = []
        for i, text in enumerate(split_texts):
            chunks.append(ParsedChunk(
                chunk_id=f"hwp_chunk_{i}",
                text=text,
                page_no=1,
                metadata={
                    "source": "hwp",
                    "filename": os.path.basename(file_path),
                    "file_path": file_path,
                    "chunk_index": i,
                }
            ))
        return chunks

    def _parse_with_hwp5txt(self, file_path: str) -> str:
        """pyhwp(hwp5txt) CLI를 사용한 고품질 텍스트 추출"""
        try:
            # hwp5txt 명령어 실행
            result = subprocess.run(
                ["hwp5txt", file_path],
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8'
            )

            if result.returncode == 0 and result.stdout.strip():
                print(f"[HWPParser] hwp5txt succeeded, extracted {len(result.stdout)} chars")
                return result.stdout
            else:
                print(f"[HWPParser] hwp5txt failed: {result.stderr}")
                return ""
        except FileNotFoundError:
            print("[HWPParser] hwp5txt command not found")
            return ""
        except subprocess.TimeoutExpired:
            print("[HWPParser] hwp5txt timed out")
            return ""
        except Exception as e:
            print(f"[HWPParser] hwp5txt error: {e}")
            return ""

    def _parse_with_olefile(self, file_path: str) -> str:
        """olefile을 사용한 기본 텍스트 추출 (fallback)"""
        text_content = ""
        try:
            with olefile.OleFileIO(file_path) as ole:
                dirs = ole.listdir()
                body_sections = [
                    d for d in dirs
                    if d[0] == "BodyText" and d[1].startswith("Section")
                ]

                body_sections.sort(key=lambda x: int(x[1].replace("Section", "")))

                for section in body_sections:
                    stream = ole.openstream(section)
                    data = stream.read()

                    try:
                        decompressed = zlib.decompress(data, -15)
                    except zlib.error:
                        decompressed = data

                    raw_text = decompressed.decode('utf-16-le', errors='ignore')
                    # Preprocessor에서 정규화하므로 여기서는 Raw 병합
                    text_content += raw_text + "\n\n"

            return text_content
        except Exception as e:
            print(f"[HWPParser] olefile error: {e}")
            return ""

    def deep_scan(self, file_path: str, patterns: list) -> bool:
        """
        [Security] 파일 내 모든 스트림을 압축 해제하여 악성 패턴을 정밀 검사합니다.
        BodyText 뿐만 아니라 스크립트, 메모 등 숨겨진 영역도 검사합니다.
        """
        try:
            with olefile.OleFileIO(file_path) as ole:
                for entry in ole.listdir():
                    # 모든 스트림 읽기 시도
                    try:
                        stream = ole.openstream(entry)
                        data = stream.read()
                        
                        # zlib 압축 해제 시도
                        try:
                            decoded = zlib.decompress(data, -15).decode('utf-16-le', errors='ignore')
                        except:
                            # 압축 아님 or 디코딩 실패 -> 원본 바이너리를 utf-16/utf-8로 시도
                            try:
                                decoded = data.decode('utf-16-le', errors='ignore')
                            except:
                                decoded = data.decode('utf-8', errors='ignore')

                        # 패턴 검사
                        for pattern in patterns:
                            # Regex or Simple String
                            if hasattr(pattern, "search"):
                                if pattern.search(decoded):
                                    print(f"[DeepScan] Detected in stream {entry}: {pattern.pattern}")
                                    return True
                            elif isinstance(pattern, str):
                                if pattern.lower() in decoded.lower():
                                    print(f"[DeepScan] Detected in stream {entry}: {pattern}")
                                    return True
                    except:
                        continue
            return False
        except Exception as e:
            print(f"Deep Scan Error: {e}")
            return False

    def extract_tables(self, file_path: str) -> List[ParsedTable]:
        # olefile 방식으로는 표 구조 추출이 매우 어려움 (바이너리 분석 필요)
        # Phase 1: 빈 리스트 반환 (Not Supported)
        return []
