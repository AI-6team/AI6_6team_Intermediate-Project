import sys
import os
import unicodedata

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from bidflow.parsing.preprocessor import TextPreprocessor

def run_test():
    preprocessor = TextPreprocessor()
    failed_count = 0
    
    print("="*60)
    print("🧹 전처리 모델 회귀 테스트 (Golden Case Regression Test)")
    print("="*60)

    # 1. 필수 정규화 케이스 (Must Pass: 검색 재현율 보장)
    # 반드시 기대값과 일치해야 함
    must_pass_cases = [
        ("Ligature 정규화", "Efﬁcient Workﬂow", "Efficient Workflow"),
        ("전각 문자 정규화", "ＲＦＰ Ｎｏ．１２３", "RFP No.123"),
        ("한글 자모 조합", "한글", "한글"),
        ("제어문자 제거 (Null)", "Line 1\x00Line 2", "Line 1Line 2"),
        ("공백 정규화", "Col A\tCol B", "Col A Col B"), # 탭 -> 공백
        ("문단 경계 보존", "A\n\n\n\nB", "A\n\nB"), # 3줄 이상 -> 2줄로 축소
    ]

    print("\n[PART 1] 필수 정규화 검증 (Recall Guarantee)")
    for desc, inp, expected in must_pass_cases:
        output = preprocessor.normalize(inp)
        if output != expected:
            print(f"❌ 실패 [{desc}]")
            print(f"   입력: {repr(inp)}")
            print(f"   기대: {repr(expected)}")
            print(f"   실제: {repr(output)}")
            failed_count += 1
        else:
            print(f"✅ 통과 [{desc}]")

    # 2. 정책 검증 케이스 (Policy Check: Risk Awareness)
    # 변환이 일어나는 것을 확인하되, "Raw Text 사용 필요성"을 인지하는지 검증
    risk_cases = [
        ("원문자(①)", "제1조 ①항", "제1조 1항"),
        ("단위기호(㎡)", "100㎡", "100m2"),
    ]

    print("\n[PART 2] 정책 검증 (Risk Policy Check)")
    for desc, inp, output_pattern in risk_cases:
        output = preprocessor.normalize(inp)
        # 정책: 변환이 '일어남'을 확인 (변환되지 않으면 NFKC가 안 먹힌 것 -> 정책 위반)
        if output == inp:
            print(f"❌ 실패 [{desc}]: 변환되지 않음 (NFKC 미적용?) - {repr(output)}")
            failed_count += 1
        elif output == output_pattern:
            print(f"✅ 확인 [{desc}]: 예상대로 변환됨 ({repr(inp)} -> {repr(output)}) -> UI에서는 Raw 사용 필수")
        else:
            print(f"❌ 실패 [{desc}]: 예상과 다른 변환 ({repr(output)}) -> 정책 변경 의심")
            failed_count += 1

    # 3. 삭제된 문자 로깅 (Deleted Char Logging)
    # Preprocessor가 내부적으로 수집한 리포트 사용
    print("\n[PART 3] 삭제된 문자 로그 (Deleted Char Analysis)")
    dirty_input = "Start\x00\x01\x02\x03\x04End\nClean"
    clean_output, report = preprocessor.normalize(dirty_input, return_report=True)
    
    deleted_chars = report["removed_chars"]
    
    if deleted_chars:
        print(f"ℹ️  감지된 삭제 문자 (Code Points): {deleted_chars}")
        print(f"   -> 원인: isprintable() == False")
    else:
        print("   삭제된 문자 없음.")

    print("-" * 60)
    
    if failed_count > 0:
        print(f"❌ 총 {failed_count}건의 필수/정책 테스트 실패.")
        sys.exit(1)
    else:
        print("🎉 모든 필수 테스트 통과. (전처리 정책 준수)")
        sys.exit(0)

if __name__ == "__main__":
    run_test()
