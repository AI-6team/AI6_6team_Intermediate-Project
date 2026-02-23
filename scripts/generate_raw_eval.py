"""
exp09_phase2_raw_eval.csv 자동 생성 스크립트

사용법:
    # 테스트 (첫 3개 파일만)
    python scripts/generate_raw_eval.py --dry-run

    # 전체 실행
    python scripts/generate_raw_eval.py

    # 특정 모델 지정
    python scripts/generate_raw_eval.py --model gpt-5-mini

환경변수:
    OPENAI_API_KEY: OpenAI API 키 (필수)
"""

import os
import sys
import csv
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

# ── 프로젝트 루트 설정 ──────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "files"
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
RAW_EVAL_CSV = EXPERIMENTS_DIR / "exp09_phase2_raw_eval.csv"
OUTPUT_CSV = EXPERIMENTS_DIR / "exp09_phase2_raw_eval.csv"

# dotenv 로드 시도
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


# ══════════════════════════════════════════════════
# 1. 텍스트 추출
# ══════════════════════════════════════════════════

def extract_text_hwp(file_path: Path) -> str:
    """HWP 파일 텍스트 추출 (hwp5txt → olefile fallback)"""
    # 1차: hwp5txt CLI
    try:
        result = subprocess.run(
            ["hwp5txt", str(file_path)],
            capture_output=True, text=True,
            timeout=60, encoding="utf-8"
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        print(f"  ⏳ hwp5txt timeout: {file_path.name}")
    except Exception as e:
        print(f"  ⚠️ hwp5txt error: {e}")

    # 2차: olefile fallback
    try:
        import olefile
        import zlib
        text_parts = []
        with olefile.OleFileIO(str(file_path)) as ole:
            dirs = ole.listdir()
            body_sections = [
                d for d in dirs
                if d[0] == "BodyText" and d[1].startswith("Section")
            ]
            body_sections.sort(key=lambda x: int(x[1].replace("Section", "")))

            for section in body_sections:
                data = ole.openstream(section).read()
                try:
                    decompressed = zlib.decompress(data, -15)
                except zlib.error:
                    decompressed = data
                raw_text = decompressed.decode("utf-16-le", errors="ignore")
                text_parts.append(raw_text)

        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"  ⚠️ olefile error: {e}")
        return ""


def extract_text_pdf(file_path: Path) -> str:
    """PDF 파일 텍스트 추출 (pdfplumber)"""
    try:
        import pdfplumber
        texts = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
        return "\n\n".join(texts)
    except Exception as e:
        print(f"  ⚠️ pdfplumber error: {e}")
        return ""


def extract_text(file_path: Path) -> str:
    """파일 확장자에 따라 텍스트 추출"""
    ext = file_path.suffix.lower()
    if ext == ".hwp":
        return extract_text_hwp(file_path)
    elif ext == ".pdf":
        return extract_text_pdf(file_path)
    else:
        print(f"  ⚠️ 지원하지 않는 형식: {ext}")
        return ""


# ══════════════════════════════════════════════════
# 2. LLM 기반 Q/A 생성
# ══════════════════════════════════════════════════

GENERATE_PROMPT = """당신은 한국 공공조달 입찰 문서 전문가입니다.
아래 입찰 문서 내용을 읽고, 이 문서에 대한 **평가 질문 1개**와 **정답(ground_truth)**, 그리고 **모범 답변(answer)**을 JSON 형식으로 생성하세요.

## 질문 생성 규칙
1. 문서에서 **명확하게 답을 찾을 수 있는** 사실 기반 질문을 만드세요.
2. 질문 유형은 다음 중 하나를 선택하세요: 사업명, 사업기간, 사업예산, 입찰방식, 자격요건, 수행장소, 하자보수, 보안요구사항, 시스템구축범위, 주요기능
3. ground_truth는 문서 내용에 근거한 **정확한 정답**이어야 합니다.
4. answer는 ground_truth와 같은 의미이되, 좀 더 자연스러운 **문장형 답변**으로 작성하세요.
5. 한국어로 작성하세요.

## 출력 형식 (JSON만 출력, 마크다운 코드블록 없이)
{{"question": "질문 내용", "ground_truth": "정답", "answer": "문장형 답변"}}

## 문서 내용 (앞부분 발췌)
{context}
"""


def generate_qa(
    text: str,
    filename: str,
    model: str = "gpt-5-mini",
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """OpenAI API로 question/ground_truth/answer 생성"""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ openai 패키지가 설치되어 있지 않습니다. pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # 컨텍스트 3000자 제한 (토큰 절약)
    context = text[:3000] if len(text) > 3000 else text

    if not context.strip():
        # 텍스트가 비어있으면 파일명 기반으로 최소한의 Q/A 생성
        context = f"사업명: {filename}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": GENERATE_PROMPT.format(context=context)}
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)

        # 필수 키 검증
        for key in ("question", "ground_truth", "answer"):
            if key not in result:
                result[key] = ""

        return result

    except Exception as e:
        print(f"  ❌ LLM 호출 실패 ({filename}): {e}")
        return {
            "question": f"본 사업의 프로젝트 명은 무엇인가?",
            "ground_truth": filename.rsplit("_", 1)[-1].replace(".hwp", "").replace(".pdf", ""),
            "answer": f"파일명에 따르면 본 사업은 '{filename}'입니다."
        }


# ══════════════════════════════════════════════════
# 3. CSV 업데이트
# ══════════════════════════════════════════════════

def load_csv(path: Path) -> list:
    """CSV를 dict 리스트로 로드"""
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def save_csv(path: Path, rows: list, fieldnames: list):
    """dict 리스트를 CSV로 저장"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════
# 4. 메인 실행
# ══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="exp09 raw eval CSV 자동 생성")
    parser.add_argument("--dry-run", action="store_true", help="첫 3개 파일만 처리")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI 모델명 (기본: gpt-5-mini)")
    parser.add_argument("--api-key", default=None, help="OpenAI API 키 (미지정 시 환경변수 사용)")
    args = parser.parse_args()

    # API 키 확인
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   export OPENAI_API_KEY=sk-... 또는 --api-key 옵션을 사용하세요.")
        sys.exit(1)

    # CSV 로드
    if not RAW_EVAL_CSV.exists():
        print(f"❌ CSV 파일을 찾을 수 없습니다: {RAW_EVAL_CSV}")
        sys.exit(1)

    rows, fieldnames = load_csv(RAW_EVAL_CSV)
    print(f"📄 CSV 로드 완료: {len(rows)}행, 컬럼: {fieldnames}")

    # 고유 파일명 추출
    unique_files = list(dict.fromkeys(row["file"] for row in rows))
    print(f"📂 고유 문서 수: {len(unique_files)}")

    if args.dry_run:
        unique_files = unique_files[:3]
        print(f"🧪 Dry-run 모드: 첫 {len(unique_files)}개 파일만 처리")

    # tqdm 사용 시도
    try:
        from tqdm import tqdm
        file_iter = tqdm(unique_files, desc="문서 처리 중")
    except ImportError:
        file_iter = unique_files
        print("💡 tqdm 미설치 → 진행률 바 없이 실행합니다.")

    # 파일별 Q/A 생성
    qa_cache: Dict[str, Dict[str, str]] = {}
    success_count = 0
    fail_count = 0

    for filename in file_iter:
        # 파일 경로 찾기
        file_path = DATA_DIR / filename
        if not file_path.exists():
            print(f"  ⚠️ 파일 없음: {filename}")
            qa_cache[filename] = {
                "question": "",
                "ground_truth": "",
                "answer": ""
            }
            fail_count += 1
            continue

        # 텍스트 추출
        text = extract_text(file_path)
        if not text.strip():
            print(f"  ⚠️ 텍스트 추출 실패: {filename}")
            text = ""

        # Q/A 생성
        qa = generate_qa(text, filename, model=args.model, api_key=api_key)
        qa_cache[filename] = qa
        success_count += 1

        # API rate limit 방지
        time.sleep(0.5)

    print(f"\n✅ Q/A 생성 완료: 성공 {success_count} / 실패 {fail_count}")

    # CSV 업데이트
    updated_count = 0
    for row in rows:
        filename = row["file"]
        if filename in qa_cache:
            qa = qa_cache[filename]
            row["question"] = qa.get("question", "")
            row["ground_truth"] = qa.get("ground_truth", "")
            row["answer"] = qa.get("answer", "")
            if qa.get("question"):
                updated_count += 1

    # 저장
    if args.dry_run:
        output_path = EXPERIMENTS_DIR / "exp09_phase2_raw_eval_dryrun.csv"
    else:
        output_path = OUTPUT_CSV

    save_csv(output_path, rows, fieldnames)
    print(f"💾 저장 완료: {output_path}")
    print(f"   업데이트된 행: {updated_count} / {len(rows)}")

    # 빈 셀 리포트
    empty_q = sum(1 for r in rows if not r.get("question"))
    empty_gt = sum(1 for r in rows if not r.get("ground_truth"))
    empty_a = sum(1 for r in rows if not r.get("answer"))
    print(f"\n📊 빈 셀 현황:")
    print(f"   question:     {empty_q} / {len(rows)} 빈 셀")
    print(f"   ground_truth: {empty_gt} / {len(rows)} 빈 셀")
    print(f"   answer:       {empty_a} / {len(rows)} 빈 셀")


if __name__ == "__main__":
    main()
