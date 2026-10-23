"""Loads a pipeline and runs generation jobs one at a time on the GPU."""

import gc
import json
import logging
import os
import queue
import random
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path

from PIL import Image

from . import catalog, downloader, settings

log = logging.getLogger(__name__)

RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."


class JobCancelled(Exception):
    pass


class MemoryPressure(Exception):
    pass


@dataclass
class GenerateParams:
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
    reference_images: list[str] = field(default_factory=list)


@dataclass
class Job:
    id: str
    params: GenerateParams
    model_id: str
    source: str = "ui"
    status: str = "queued"  # queued | running | done | error | cancelled
    step: int = 0
    total_steps: int = 0
    stage: str = "Waiting in queue"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    image: dict | None = None
    error: str | None = None
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        data = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        data["params"] = vars(self.params)
        return data

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)


def gpu_memory_gb() -> float | None:
    import torch

    if torch.backends.mps.is_available():
        return round(torch.mps.driver_allocated_memory() / 1e9, 1)
    if torch.cuda.is_available():
        return round(torch.cuda.memory_reserved() / 1e9, 1)
    return None


def total_memory_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def gpu_memory_limit_gb() -> float:
    """GPU memory the engine may use. The rest stays free for macOS and other apps."""
    if settings.GPU_MEMORY_LIMIT_GB:
        return settings.GPU_MEMORY_LIMIT_GB
    total = total_memory_gb()
    return round(total - max(12.0, total * 0.3), 1)


def max_resolution() -> int:
    """Largest quality preset that fits. Native 2048 px needs far more headroom than the weights alone."""
    return 2048 if total_memory_gb() >= 100 else 1536


def memory_pressure_critical() -> bool:
    """True when macOS reports critical memory pressure, the point where it starts swapping hard."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True, timeout=2
        ).stdout
        return int(out) >= 4
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def limit_gpu_memory(device: str) -> None:
    """Make PyTorch fail with an out-of-memory error instead of pushing the Mac into swap.

    The MPS default allows 1.7x the recommended working set, which on a 64 GB Mac is more than
    the machine has, so an oversized job swaps until macOS freezes.
    """
    import torch

    if device != "mps":
        return
    fraction = gpu_memory_limit_gb() * 1e9 / torch.mps.recommended_max_memory()
    torch.mps.set_per_process_memory_fraction(max(0.1, min(fraction, 1.0)))
    log.info("GPU memory limit %.1f GB", gpu_memory_limit_gb())


def empty_cache() -> None:
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def best_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Engine:
    def __init__(self):
        self.model_id: str | None = None
        self.status = "unloaded"  # unloaded | loading | ready | error
        self.message = "No model loaded"
        self.device: str | None = None
        self._pipe = None
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.RLock()
        threading.Thread(target=self._worker, daemon=True, name="engine-worker").start()

    # Model lifecycle -------------------------------------------------------

    def public(self) -> dict:
        return {
            "model_id": self.model_id,
            "status": self.status,
            "message": self.message,
            "device": self.device,
            "gpu_memory_gb": gpu_memory_gb(),
            "gpu_memory_limit_gb": gpu_memory_limit_gb(),
            "max_resolution": max_resolution(),
        }

    def load(self, model_id: str) -> None:
        spec = catalog.get(model_id)
        if not downloader.is_installed(spec):
            raise RuntimeError(f"{spec.name} is not downloaded yet")
        with self._lock:
            if self.model_id == model_id and self.status in ("loading", "ready"):
                return
            if self.status == "loading":
                raise RuntimeError("Another model is currently loading")
            self.model_id, self.status, self.message = model_id, "loading", f"Loading {spec.name}…"
        threading.Thread(target=self._load, args=(spec,), daemon=True, name="engine-load").start()

    def _load(self, spec: catalog.ModelSpec) -> None:
        try:
            import diffusers
            import torch

            self._release()
            device = best_device()
            limit_gpu_memory(device)
            pipeline_cls = getattr(diffusers, spec.pipeline)
            log.info("Loading %s from %s on %s", spec.repo, spec.local_dir, device)
            start = time.time()
            # No device_map: on MPS it deadlocks in diffusers' post-load empty_cache(). The weights are
            # memory-mapped, so the CPU side is only reclaimable file cache anyway.
            pipe = pipeline_cls.from_pretrained(str(spec.local_dir), dtype=torch.bfloat16)
            self.message = f"Moving {spec.name} to {device.upper()}…"
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            if hasattr(pipe.vae, "enable_tiling"):
                # Decoding big images in one piece is the biggest memory spike. Up to 1024 px stays untiled.
                pipe.vae.enable_tiling(1024, 1024, 768, 768)
            empty_cache()
            with self._lock:
                if self.model_id != spec.id:  # unloaded or switched while loading
                    del pipe
                    return
                self._pipe, self.device = pipe, device
                self.status, self.message = "ready", f"{spec.name} ready on {device.upper()}"
            log.info("Loaded %s in %.1fs", spec.repo, time.time() - start)
        except Exception as exc:
            log.exception("Failed to load %s", spec.repo)
            with self._lock:
                self.status, self.message = "error", f"Failed to load {spec.name}: {exc}"
                self._pipe = None

    def unload(self) -> None:
        with self._lock:
            self.model_id, self.status, self.message, self.device = None, "unloaded", "No model loaded", None
            self._release()

    def _release(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
        empty_cache()

    # Jobs ------------------------------------------------------------------

    def submit(self, params: GenerateParams, source: str = "ui") -> Job:
        if not params.prompt or not params.prompt.strip():
            raise ValueError("Prompt must not be empty")
        with self._lock:
            if self.status not in ("ready", "loading") or not self.model_id:
                raise RuntimeError("No model loaded. Pick a model in the ImageGen app first.")
            spec = catalog.get(self.model_id)
        if len(params.reference_images) > spec.max_reference_images:
            raise ValueError(f"{spec.name} accepts at most {spec.max_reference_images} reference images")
        for ref in params.reference_images:
            if not self.resolve_image_path(ref).exists():
                raise ValueError(f"Reference image not found: {ref}")
        limit = max_resolution()
        params.resolution = min(params.resolution, limit)
        if params.width is None or params.height is None:
            params.width, params.height = catalog.dimensions(spec, params.aspect_ratio, params.resolution)
        params.width, params.height = params.width // 32 * 32, params.height // 32 * 32
        if min(params.width, params.height) < 256 or max(params.width, params.height) > 4096:
            raise ValueError("Width and height must be between 256 and 4096 pixels")
        if params.width * params.height > limit * limit * 1.1:
            raise ValueError(
                f"{params.width}x{params.height} is too large for this Mac's memory. "
                f"Keep it under about {limit * limit / 1e6:.1f} megapixels."
            )
        if spec.fixed_steps and params.steps not in (None, spec.default_steps):
            raise ValueError(f"{spec.name} uses a fixed {spec.default_steps}-step sampling schedule")
        params.steps = max(1, min(params.steps or spec.default_steps, 100))
        if params.seed is None or params.seed < 0:
            params.seed = random.randint(0, 2**31 - 1)

        job = Job(id=uuid.uuid4().hex[:12], params=params, model_id=spec.id, source=source)
        job.total_steps = params.steps
        with self._lock:
            self._jobs[job.id] = job
            self._trim_jobs()
        self._queue.put(job)
        return job

    def job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[dict]:
        return [j.public() for j in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status not in ("queued", "running"):
            return False
        job._cancel.set()
        if job.status == "queued":
            self._finish(job, "cancelled", stage="Cancelled")
        return True

    def _trim_jobs(self, keep: int = 50) -> None:
        finished = [j for j in self._jobs.values() if j.status in ("done", "error", "cancelled")]
        for j in sorted(finished, key=lambda j: j.created_at)[: max(0, len(finished) - keep)]:
            self._jobs.pop(j.id, None)

    def _finish(self, job: Job, status: str, stage: str, error: str | None = None) -> None:
        job.status, job.stage, job.error, job.finished_at = status, stage, error, time.time()
        job._done.set()

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job.status != "queued":
                continue
            try:
                self._run(job)
            except JobCancelled:
                self._finish(job, "cancelled", stage="Cancelled")
            except MemoryPressure:
                log.warning("Job %s stopped: critical memory pressure", job.id)
                self._finish(
                    job,
                    "error",
                    stage="Stopped",
                    error="Stopped to protect your Mac: memory ran low. Try a lower quality or close other apps.",
                )
            except Exception as exc:
                log.exception("Job %s failed", job.id)
                error = str(exc) or type(exc).__name__
                if "out of memory" in error.lower():
                    error = (
                        f"Not enough memory for {job.params.width}x{job.params.height}. "
                        "Try a lower quality or close other apps."
                    )
                self._finish(job, "error", stage="Failed", error=error)
            finally:
                # Give activation memory back to the system between jobs.
                if self._queue.empty():
                    empty_cache()

    def _run(self, job: Job) -> None:
        import torch

        job.status, job.started_at = "running", time.time()
        job.stage = "Waiting for model to load"
        while self.status == "loading" and not job._cancel.is_set():
            time.sleep(0.5)
        if job._cancel.is_set():
            raise JobCancelled()
        if self.status != "ready" or self._pipe is None:
            raise RuntimeError(self.message)
        if self.model_id != job.model_id:
            raise RuntimeError("The model was switched before this job started")

        p = job.params
        prompt = p.prompt.strip()
        if p.transparent and not prompt.startswith("This is an RGBA image"):
            prompt = RGBA_PREFIX + prompt + RGBA_SUFFIX
        refs = [Image.open(self.resolve_image_path(r)) for r in p.reference_images]
        for img in refs:
            img.load()

        job.stage = "Encoding prompt"

        low_memory = threading.Event()

        def on_step(pipe, step, timestep, kwargs):
            if memory_pressure_critical():
                low_memory.set()
            if job._cancel.is_set() or low_memory.is_set():
                pipe._interrupt = True
            job.step = step + 1
            job.stage = f"Step {job.step} of {job.total_steps}"
            if job.step == job.total_steps:
                job.stage = "Decoding image"
            return kwargs

        generator = torch.Generator(device="cpu").manual_seed(p.seed)
        with self._lock:
            pipe = self._pipe
        result = pipe(
            prompt=prompt,
            image=refs or None,
            negative_prompt=p.negative_prompt or None,
            true_cfg_scale=p.cfg_scale,
            width=p.width,
            height=p.height,
            num_inference_steps=p.steps,
            generator=generator,
            callback_on_step_end=on_step,
        )
        if low_memory.is_set():
            raise MemoryPressure()
        if job._cancel.is_set():
            raise JobCancelled()
        image = result.images[0]
        job.image = self._save(job, image, prompt)
        self._finish(job, "done", stage="Done")
        log.info("Job %s done in %.1fs", job.id, job.finished_at - job.started_at)

    # Outputs ---------------------------------------------------------------

    def _save(self, job: Job, image: Image.Image, final_prompt: str) -> dict:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{job.id}.png"
        path = settings.OUTPUT_DIR / name
        image.save(path)
        meta = {
            "name": name,
            "path": str(path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "model_id": job.model_id,
            "source": job.source,
            "job_id": job.id,
            "created_at": time.time(),
            "duration_s": round(time.time() - (job.started_at or time.time()), 1),
            "final_prompt": final_prompt,
            **vars(job.params),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return meta

    @staticmethod
    def resolve_image_path(ref: str) -> Path:
        """Accept absolute paths, ~ paths, or names of images in the output folder."""
        path = Path(ref).expanduser()
        if not path.is_absolute():
            path = settings.OUTPUT_DIR / ref
        return path

    @staticmethod
    def history(limit: int = 200) -> list[dict]:
        items = []
        for meta_path in sorted(settings.OUTPUT_DIR.glob("*.json"), reverse=True)[:limit]:
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if Path(meta.get("path", "")).exists():
                items.append(meta)
        return items

    @staticmethod
    def delete_image(name: str) -> bool:
        path = settings.OUTPUT_DIR / Path(name).name
        if path.suffix != ".png" or not path.exists():
            return False
        path.unlink()
        path.with_suffix(".json").unlink(missing_ok=True)
        return True
