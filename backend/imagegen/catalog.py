"""Models the app knows how to download and run."""

from dataclasses import dataclass, field

from . import settings


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    repo: str
    description: str
    pipeline: str
    license: str
    size_gb: float
    recommended_memory_gb: int
    capabilities: tuple[str, ...]
    default_steps: int = 40
    max_reference_images: int = 0
    # Checkpoints with a saved sampling schedule do not support custom step counts.
    fixed_steps: bool = False
    # Native resolution presets per aspect ratio (width, height).
    aspect_ratios: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Files matching these prefixes are not needed to run the model.
    skip_prefixes: tuple[str, ...] = ()

    @property
    def local_dir(self):
        return settings.MODELS_DIR / self.repo.replace("/", "--")

    @property
    def hf_url(self) -> str:
        return f"{settings.HF_ENDPOINT}/{self.repo}"


MODELS: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        ModelSpec(
            id="qwen-image-2.1",
            name="Qwen-Image 2.1",
            repo="Qwen/Qwen-Image-2.1",
            description=(
                "Unified text-to-image and image editing model from the Qwen team. "
                "Strong typography, native transparent (RGBA) output and up to 10 reference images."
            ),
            pipeline="QwenImage21Pipeline",
            license="Qwen Research License Agreement (non-commercial)",
            size_gb=33.1,
            recommended_memory_gb=48,
            capabilities=("text-to-image", "image-editing", "rgba"),
            default_steps=40,
            max_reference_images=10,
            aspect_ratios={
                "1:1": (2048, 2048),
                "4:3": (2400, 1792),
                "3:4": (1792, 2400),
                "3:2": (2528, 1696),
                "2:3": (1696, 2528),
                "16:9": (2752, 1536),
                "9:16": (1536, 2752),
            },
            skip_prefixes=("assets/",),
        ),
        ModelSpec(
            id="qwen-image-2.1-turbo",
            name="Qwen-Image 2.1 Turbo",
            repo="Qwen/Qwen-Image-2.1-Turbo",
            description=(
                "Accelerated text-to-image and image editing with a fixed 8-step schedule. "
                "Strong typography, native transparent (RGBA) output and up to 10 reference images."
            ),
            pipeline="QwenImage21Pipeline",
            license="Qwen Research License Agreement (non-commercial)",
            size_gb=32.5,
            recommended_memory_gb=48,
            capabilities=("text-to-image", "image-editing", "rgba"),
            default_steps=8,
            fixed_steps=True,
            max_reference_images=10,
            aspect_ratios={
                "1:1": (2048, 2048),
                "4:3": (2400, 1792),
                "3:4": (1792, 2400),
                "3:2": (2528, 1696),
                "2:3": (1696, 2528),
                "16:9": (2752, 1536),
                "9:16": (1536, 2752),
            },
            skip_prefixes=("assets/",),
        ),
    ]
}


def get(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError:
        raise KeyError(f"Unknown model '{model_id}'. Available: {', '.join(MODELS)}") from None


def dimensions(spec: ModelSpec, aspect_ratio: str, resolution: int) -> tuple[int, int]:
    """Scale the native preset for an aspect ratio to roughly resolution x resolution pixels."""
    if aspect_ratio not in spec.aspect_ratios:
        raise ValueError(f"Unsupported aspect ratio '{aspect_ratio}'. Use one of: {', '.join(spec.aspect_ratios)}")
    w, h = spec.aspect_ratios[aspect_ratio]
    scale = resolution / 2048
    return max(256, round(w * scale / 32) * 32), max(256, round(h * scale / 32) * 32)
