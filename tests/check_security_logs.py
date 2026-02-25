import os
import json
import sys
import argparse

def print_security_logs(n=20, filter_keyword=None, file_type="audit"):
    log_file = os.path.join("logs", f"{file_type}.log")
    
    if not os.path.exists(log_file):
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
        return

    print(f"🔍 [Security Logs] Last {n} lines from {log_file} (Filter: {filter_keyword or 'None'}):\n" + "="*60)
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            # 필터링이 있으면 전체에서 검색, 없으면 마지막 n줄
            if filter_keyword:
                target_lines = [line for line in lines if filter_keyword in line]
                # 필터링 된 것 중 마지막 n개
                display_lines = target_lines[-n:] if len(target_lines) > n else target_lines
            else:
                display_lines = lines[-n:] if len(lines) > n else lines
            
            for line in display_lines:
                try:
                    log_entry = json.loads(line)
                    
                    # 로그 레벨에 따른 이모지 표시
                    level = log_entry.get("level", "UNKNOWN")
                    event = log_entry.get("event", "")
                    msg = log_entry.get("message", "")
                    
                    if level == "INFO":
                        print(f"✅ [AUDIT] {event if event else msg}")
                    elif level == "WARNING":
                        print(f"⚠️ [BLOCK] {msg[:50]}...")
                    
                    print(json.dumps(log_entry, indent=2, ensure_ascii=False))
                except json.JSONDecodeError:
                    print(line.strip())
                print("-" * 40)
                
    except Exception as e:
        print(f"❌ 로그 파일 읽기 실패: {e}")
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="보안 로그 확인 도구")
    parser.add_argument("lines", nargs="?", type=int, default=20, help="출력할 로그 줄 수 (기본값: 20)")
    parser.add_argument("-f", "--filter", type=str, help="필터링할 키워드 (예: AUDIT, BLOCK, Resident)")
    parser.add_argument("-t", "--type", type=str, default="audit", choices=["audit", "security"], help="확인할 로그 파일 (audit 또는 security)")
    
    args = parser.parse_args()
    print_security_logs(args.lines, args.filter, args.type)
