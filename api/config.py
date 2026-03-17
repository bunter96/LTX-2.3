import os
from pathlib import Path

# Model paths — set via environment variables or edit defaults here
DISTILLED_CHECKPOINT_PATH = os.environ.get("DISTILLED_CHECKPOINT_PATH", "")
SPATIAL_UPSAMPLER_PATH = os.environ.get("SPATIAL_UPSAMPLER_PATH", "")
GEMMA_ROOT = os.environ.get("GEMMA_ROOT", "")

# Optional LoRA paths (comma-separated "path:strength" pairs, e.g. "lora.safetensors:0.8")
LORA_SPECS = os.environ.get("LORA_SPECS", "")

# Quantization: "fp8-cast", "fp8-scaled-mm", or "" for none
QUANTIZATION = os.environ.get("QUANTIZATION", "")

# Where generated videos are saved
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", Path(__file__).parent / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
