"""
기존 전역 공유 데이터를 data/accounts/{user}/ 사용자 공간으로 이전하는 스크립트.

사용법:
    python scripts/migrate_data.py [--user admin] [--dry-run]

이전 전: data/raw/, data/processed/, data/vectordb/, data/profile.json, data/session.json
이전 후: data/accounts/admin/raw/, data/accounts/admin/processed/, data/accounts/admin/vectordb/,
         data/accounts/admin/profile.json, data/accounts/admin/session.json
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def migrate(target_user: str = "admin", dry_run: bool = False):
    from bidflow.ingest.storage import StorageRegistry
    from bidflow.core.config import get_config

    config = get_config("dev")
    registry = StorageRegistry(config)
    base = registry.base  # "data"

    migrations = [
        (os.path.join(base, "raw"),       registry.user_space(target_user, "raw")),
        (os.path.join(base, "processed"), registry.user_space(target_user, "processed")),
        (os.path.join(base, "vectordb"),  registry.user_space(target_user, "vectordb")),
        (os.path.join(base, "profile.json"), os.path.join(registry.user_base(target_user), "profile.json")),
        (os.path.join(base, "session.json"), os.path.join(registry.user_base(target_user), "session.json")),
    ]

    accounts = registry.accounts_dir or ""
    dst_prefix = f"{accounts}/{target_user}" if accounts else target_user
    print(f"=== 데이터 이전: {base}/ → {base}/{dst_prefix}/ ===")
    if dry_run:
        print("[DRY RUN 모드] 실제 파일 이동 없이 계획만 출력합니다.")

    for src, dst in migrations:
        if not os.path.exists(src):
            print(f"  SKIP (존재하지 않음): {src}")
            continue

        print(f"  MOVE: {src}  →  {dst}")
        if not dry_run:
            dst_parent = os.path.dirname(dst)
            os.makedirs(dst_parent, exist_ok=True)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    # 대상 디렉토리가 이미 있으면 파일을 개별 복사
                    for item in os.listdir(src):
                        s = os.path.join(src, item)
                        d = os.path.join(dst, item)
                        if not os.path.exists(d):
                            shutil.move(s, d)
                            print(f"    moved: {item}")
                        else:
                            print(f"    SKIP (already exists): {item}")
                else:
                    shutil.move(src, dst)
            else:
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                else:
                    print(f"    SKIP (already exists): {dst}")

    if not dry_run:
        # 공유 공간 초기화
        for space in registry._shared_spaces:
            os.makedirs(registry.shared_space(space), exist_ok=True)
        print(f"\n공유 공간 초기화 완료: {base}/shared/")

    print("\n이전 완료!")
    dst_processed = registry.user_space(target_user, "processed")
    print(f"검증: python -c \"import os; print(os.listdir('{dst_processed}'))\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="기존 데이터를 사용자 공간으로 이전합니다.")
    parser.add_argument("--user", default="admin", help="대상 사용자 ID (기본: admin)")
    parser.add_argument("--dry-run", action="store_true", help="실제 이동 없이 계획만 출력")
    args = parser.parse_args()

    migrate(target_user=args.user, dry_run=args.dry_run)
