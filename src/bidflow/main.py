"""Unified FastAPI entrypoint.

`bidflow.main:app` is the single backend entrypoint used by launcher/README.
The actual app construction lives in `bidflow.api.main:create_app`.
"""

import os
from pathlib import Path

from bidflow.api.main import create_app


def _bootstrap_env_from_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    # .env 값에 따옴표/공백이 포함된 경우를 방지해 런타임에서 안정적으로 사용한다.
    for key in (
        "OPENAI_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_BASE_URL",
        "HF_TOKEN",
        "BIDFLOW_JWT_SECRET",
    ):
        value = os.getenv(key)
        if value:
            os.environ[key] = value.strip().strip('"').strip("'")


_bootstrap_env_from_dotenv()
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bidflow.main:app", host="0.0.0.0", port=8000, reload=True)
