"""HTTP server: REST API for the Mac app plus the MCP endpoint at /mcp."""

import contextlib
import shutil

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__, catalog, downloader, settings
from .engine import Engine, GenerateParams
from .mcp_server import build_mcp

engine = Engine()
downloads = downloader.Downloader()
mcp = build_mcp(engine)
mcp_app = mcp.streamable_http_app(streamable_http_path="/mcp", host=settings.HOST)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    async with mcp.session_manager.run():
        yield


api = FastAPI(title="ImageGen", version=__version__, lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    aspect_ratio: str = "1:1"
    resolution: int = 1024
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    seed: int | None = None
    cfg_scale: float = 1.0
    transparent: bool = False
    reference_images: list[str] = Field(default_factory=list)


def _model_info(spec: catalog.ModelSpec) -> dict:
    state = downloads.state(spec.id)
    return {
        "id": spec.id,
        "name": spec.name,
        "repo": spec.repo,
        "url": spec.hf_url,
        "description": spec.description,
        "license": spec.license,
        "size_gb": spec.size_gb,
        "recommended_memory_gb": spec.recommended_memory_gb,
        "capabilities": list(spec.capabilities),
        "default_steps": spec.default_steps,
        "fixed_steps": spec.fixed_steps,
        "max_reference_images": spec.max_reference_images,
        "aspect_ratios": list(spec.aspect_ratios),
        "installed": downloader.is_installed(spec),
        "local_path": str(spec.local_dir),
        "download": state.public() if state else None,
    }


def _spec(model_id: str) -> catalog.ModelSpec:
    try:
        return catalog.get(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


@api.get("/api/health")
def health():
    return {"ok": True, "version": __version__}


@api.get("/api/status")
def status():
    return {
        "engine": engine.public(),
        "jobs": engine.jobs()[:20],
        "mcp_url": f"http://{settings.HOST}:{settings.PORT}/mcp",
        "output_dir": str(settings.OUTPUT_DIR),
        "free_disk_gb": round(shutil.disk_usage(settings.MODELS_DIR).free / 1e9, 1),
    }


@api.get("/api/models")
def models():
    return [_model_info(spec) for spec in catalog.MODELS.values()]


@api.post("/api/models/{model_id}/download")
def download(model_id: str):
    spec = _spec(model_id)
    downloads.start(spec)
    return _model_info(spec)


@api.post("/api/models/{model_id}/cancel-download")
def cancel_download(model_id: str):
    downloads.cancel(_spec(model_id).id)
    return {"ok": True}


@api.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    spec = _spec(model_id)
    if engine.model_id == spec.id:
        engine.unload()
    downloads.cancel(spec.id)
    shutil.rmtree(spec.local_dir, ignore_errors=True)
    return _model_info(spec)


@api.post("/api/models/{model_id}/load")
def load(model_id: str):
    try:
        engine.load(_spec(model_id).id)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return engine.public()


@api.post("/api/unload")
def unload():
    engine.unload()
    return engine.public()


@api.post("/api/generate")
def generate(req: GenerateRequest):
    try:
        job = engine.submit(GenerateParams(**req.model_dump()), source="ui")
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return job.public()


@api.get("/api/jobs")
def jobs():
    return engine.jobs()


@api.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = engine.job(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j.public()


@api.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    return {"ok": engine.cancel(job_id)}


@api.get("/api/history")
def history(limit: int = 200):
    return engine.history(limit)


@api.get("/api/images/{name}")
def image(name: str):
    path = settings.OUTPUT_DIR / name
    if path.name != name or not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@api.delete("/api/images/{name}")
def delete_image(name: str):
    if not engine.delete_image(name):
        raise HTTPException(404, "Image not found")
    return {"ok": True}


# Everything not matched above (i.e. /mcp) goes to the MCP app.
api.mount("/", mcp_app)
