import os
import json

def print_security_logs(n=20):
    log_file = os.path.join("logs", "security.log")
    
    if not os.path.exists(log_file):
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
        return

    print(f"🔍 [Security Logs] Last {n} lines from {log_file}:\n" + "="*60)
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # 마지막 n줄만 가져오기
            last_lines = lines[-n:] if len(lines) > n else lines
            
            for line in last_lines:
                try:
                    log_entry = json.loads(line)
                    print(json.dumps(log_entry, indent=2, ensure_ascii=False))
                except json.JSONDecodeError:
                    print(line.strip())
                print("-" * 40)
                
    except Exception as e:
        print(f"❌ 로그 파일 읽기 실패: {e}")
    
    print("="*60)

if __name__ == "__main__":
    print_security_logs()
