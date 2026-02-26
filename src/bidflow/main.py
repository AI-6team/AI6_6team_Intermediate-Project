"""Unified FastAPI entrypoint.

`bidflow.main:app` is the single backend entrypoint used by launcher/README.
The actual app construction lives in `bidflow.api.main:create_app`.
"""

from bidflow.api.main import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bidflow.main:app", host="0.0.0.0", port=8000, reload=True)
