"""MCP tools that expose the running engine to agents like Claude Code."""

import asyncio
import io
import json
import shutil
from pathlib import Path

from mcp.server.mcpserver import Context, Image, MCPServer
from PIL import Image as PILImage

from . import catalog, downloader
from .engine import Engine, GenerateParams, max_resolution

INSTRUCTIONS = """\
ImageGen runs image models locally on this Mac. Use generate_image to create images from a prompt and
edit_image to change existing images. Generation happens on-device and can take a few minutes, so prefer
one image at a time. Results are saved as PNG files; the tool returns the file path and a small preview.
Pass output_path to save a copy straight into the user's project. Models are downloaded and picked in the
ImageGen app. If no model is loaded, ask the user to pick one there, or call load_model with a downloaded model.
"""


def _preview(path: str, max_side: int = 768) -> Image:
    with PILImage.open(path) as img:
        img.thumbnail((max_side, max_side))
        if img.mode == "RGBA":
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return Image(data=buf.getvalue(), format="png")
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return Image(data=buf.getvalue(), format="jpeg")


def _copy_to(src: str, output_path: str) -> str:
    dest = Path(output_path).expanduser()
    if dest.is_dir() or output_path.endswith("/"):
        dest = dest / Path(src).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() in ("", ".png"):
        dest = dest.with_suffix(".png")
        shutil.copyfile(src, dest)
    else:
        with PILImage.open(src) as img:
            if dest.suffix.lower() in (".jpg", ".jpeg"):
                img = img.convert("RGB")
            img.save(dest)
    return str(dest.resolve())


def build_mcp(engine: Engine) -> MCPServer:
    mcp = MCPServer(name="imagegen", title="ImageGen", instructions=INSTRUCTIONS, version="0.1.0")

    async def run_job(params: GenerateParams, ctx: Context, output_path: str | None, preview: bool):
        try:
            job = engine.submit(params, source="mcp")
        except (ValueError, RuntimeError, KeyError) as exc:
            return f"Error: {exc}"
        last = None
        while not job.wait(timeout=0):
            state = (job.step, job.stage)
            if state != last:
                last = state
                try:
                    await ctx.report_progress(job.step, job.total_steps, job.stage)
                except Exception:
                    pass
            await asyncio.sleep(1)
        if job.status != "done" or not job.image:
            return f"Generation {job.status}: {job.error or job.stage}"

        info = {
            "path": job.image["path"],
            "width": job.image["width"],
            "height": job.image["height"],
            "seed": params.seed,
            "steps": params.steps,
            "model": job.model_id,
            "seconds": job.image["duration_s"],
        }
        if output_path:
            try:
                info["copied_to"] = _copy_to(job.image["path"], output_path)
            except OSError as exc:
                info["copy_error"] = str(exc)
        result: list = [json.dumps(info, indent=2)]
        if preview:
            result.append(_preview(job.image["path"]))
        return result

    @mcp.tool()
    async def generate_image(
        prompt: str,
        ctx: Context,
        aspect_ratio: str = "1:1",
        quality: str = "standard",
        transparent: bool = False,
        negative_prompt: str | None = None,
        steps: int | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        output_path: str | None = None,
        return_preview: bool = True,
    ):
        """Generate an image from a text prompt with the locally loaded model.

        Args:
            prompt: Detailed description of the image. Put any text that should appear in the image in quotes.
            aspect_ratio: One of 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16.
            quality: "draft" (~512px, fastest), "standard" (~1024px) or "high" (up to ~2048px, slowest; capped to what this Mac's memory allows).
            transparent: Produce an RGBA image with a transparent background (stickers, icons, cutouts).
            negative_prompt: Things to avoid. Only used when combined with a cfg scale, usually leave empty.
            steps: Denoising steps. Defaults to the model's recommended value (40 for base, fixed 8 for Turbo).
            seed: Fix for reproducible results. Random when omitted.
            width: Exact width in pixels (multiple of 32). Overrides aspect_ratio and quality together with height.
            height: Exact height in pixels (multiple of 32).
            output_path: Optional file or folder path to also save the result to, e.g. "./assets/hero.png".
            return_preview: Include a downscaled preview image in the result.
        """
        params = GenerateParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            resolution=_quality(quality),
            width=width if width and height else None,
            height=height if width and height else None,
            steps=steps,
            seed=seed,
            transparent=transparent,
        )
        return await run_job(params, ctx, output_path, return_preview)

    @mcp.tool()
    async def edit_image(
        instruction: str,
        image_paths: list[str],
        ctx: Context,
        quality: str = "standard",
        steps: int | None = None,
        seed: int | None = None,
        output_path: str | None = None,
        return_preview: bool = True,
    ):
        """Edit or combine existing images following an instruction.

        Examples: "Change the background to a sunset beach", "Remove the person on the left",
        "Put the product from image 1 on the table in image 2", "Extract the subject onto a transparent background".
        The output keeps the aspect ratio of the last image.

        Args:
            instruction: What to change.
            image_paths: One or more absolute paths to input images (up to 10), or names of earlier results.
            quality: "draft", "standard" or "high".
            steps: Denoising steps, defaults to the model's recommended value; Turbo requires 8.
            seed: Fix for reproducible results.
            output_path: Optional file or folder path to also save the result to.
            return_preview: Include a downscaled preview image in the result.
        """
        if not image_paths:
            return "Error: pass at least one image path"
        resolution = _quality(quality)
        try:
            with PILImage.open(engine.resolve_image_path(image_paths[-1])) as img:
                w, h = img.size
        except OSError as exc:
            return f"Error: cannot open {image_paths[-1]}: {exc}"
        scale = (resolution * resolution / (w * h)) ** 0.5
        params = GenerateParams(
            prompt=instruction,
            width=max(256, round(w * scale / 32) * 32),
            height=max(256, round(h * scale / 32) * 32),
            resolution=resolution,
            steps=steps,
            seed=seed,
            reference_images=[str(engine.resolve_image_path(p)) for p in image_paths],
        )
        return await run_job(params, ctx, output_path, return_preview)

    @mcp.tool()
    def list_models() -> str:
        """List models the app supports, whether they are downloaded, and which one is loaded."""
        return json.dumps(
            [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "capabilities": spec.capabilities,
                    "default_steps": spec.default_steps,
                    "fixed_steps": spec.fixed_steps,
                    "downloaded": downloader.is_installed(spec),
                    "loaded": engine.model_id == spec.id and engine.status == "ready",
                    "aspect_ratios": list(spec.aspect_ratios),
                }
                for spec in catalog.MODELS.values()
            ],
            indent=2,
        )

    @mcp.tool()
    def get_status() -> str:
        """Show the engine state and the current generation queue."""
        active = [j for j in engine.jobs() if j["status"] in ("queued", "running")]
        return json.dumps({"engine": engine.public(), "active_jobs": active}, indent=2, default=str)

    @mcp.tool()
    async def load_model(model_id: str) -> str:
        """Load a downloaded model so it can generate. Downloads only happen through the ImageGen app."""
        try:
            engine.load(model_id)
        except (KeyError, RuntimeError) as exc:
            return f"Error: {exc}"
        while engine.status == "loading":
            await asyncio.sleep(1)
        return engine.message

    @mcp.tool()
    def list_recent_images(limit: int = 10) -> str:
        """List recently generated images with their prompts and file paths."""
        keep = ("path", "prompt", "width", "height", "seed", "created_at", "source")
        return json.dumps([{k: m.get(k) for k in keep} for m in engine.history(limit)], indent=2)

    return mcp


def _quality(quality: str) -> int:
    resolution = {"draft": 512, "standard": 1024, "high": 2048}.get(quality.lower(), 1024)
    return min(resolution, max_resolution())
