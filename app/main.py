"""HTTP entrypoint for the GridWise LLM service."""

from fastapi import FastAPI


app = FastAPI(
    title="GridWise LLM",
    description="Smart campus energy optimization microservice.",
    version="0.1.0",
)


@app.get("/", tags=["system"])
def read_root() -> dict[str, str]:
    """Return a small service descriptor."""
    return {"service": "gridwise-llm", "status": "ok"}


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Liveness check for local and container orchestration."""
    return {"status": "healthy"}

