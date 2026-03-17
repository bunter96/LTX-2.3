# LTX-2 Distilled API

FastAPI server wrapping the [`DistilledPipeline`](../packages/ltx-pipelines/src/ltx_pipelines/distilled.py) for text-to-video and image-to-video generation. The pipeline loads once at startup and stays resident in GPU memory, making repeated generation significantly faster than the CLI.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU with sufficient VRAM (22B model — 24GB+ recommended, use FP8 quantization for lower VRAM)
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

---

## 1. Environment Setup

From the repository root:

```bash
git clone https://github.com/bunter96/LTX-2.3.git
cd LTX-2.3
uv sync --frozen
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

Install FastAPI dependencies:

```bash
pip install fastapi uvicorn python-multipart
```

---

## 2. Download Required Models

The `DistilledPipeline` requires three models. Make sure you have the `huggingface_hub` CLI installed:

```bash
pip install huggingface_hub
hf login   # only needed for gated models like Gemma
```

### LTX-2.3 distilled checkpoint

```bash
mkdir -p checkpoints
hf download Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled.safetensors \
    --local-dir checkpoints
```

### Spatial upsampler

```bash
hf download Lightricks/LTX-2.3 \
    ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --local-dir checkpoints
```

### Gemma text encoder

```bash
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir models/gemma-3-12b-it-qat-q4_0-unquantized
```

> Gemma is a gated model — you need to accept the license on [HuggingFace](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) and run `hf login` first.

After downloading, your directory should look like:

```
LTX-2.3/
  checkpoints/
    ltx-2.3-22b-distilled.safetensors
    ltx-2.3-spatial-upscaler-x2-1.0.safetensors
  models/
    gemma-3-12b-it-qat-q4_0-unquantized/
      ...
```

> The `DistilledPipeline` does **not** require a distilled LoRA — that is only needed by the other two-stage pipelines.

---

## 3. Configuration

Copy the example env file and fill in your model paths:

```bash
cp api/.env.example .env
```

```env
# api/.env.example
DISTILLED_CHECKPOINT_PATH=checkpoints/ltx-2.3-22b-distilled.safetensors
SPATIAL_UPSAMPLER_PATH=checkpoints/ltx-2.3-spatial-upscaler-x2-1.0.safetensors
GEMMA_ROOT=models/gemma-3-12b-it-qat-q4_0-unquantized

# Optional: comma-separated LoRA specs as "path:strength"
LORA_SPECS=

# Optional: fp8-cast | fp8-scaled-mm | (empty = no quantization)
QUANTIZATION=

# Optional: where generated .mp4 files are saved (default: api/outputs/)
OUTPUT_DIR=api/outputs
```

### FP8 Quantization (lower VRAM)

If you're running on a GPU with less than 24GB VRAM:

```env
QUANTIZATION=fp8-cast
```

Also set this environment variable when launching:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uvicorn api.main:app ...
```

`fp8-scaled-mm` is available for Hopper GPUs (H100) with `tensorrt_llm` installed and gives better performance.

---

## 4. Running the Server

**Linux / macOS:**

```bash
set -a && source .env && set +a
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**

```powershell
$env:DISTILLED_CHECKPOINT_PATH="checkpoints\ltx-2.3-22b-distilled.safetensors"
$env:SPATIAL_UPSAMPLER_PATH="checkpoints\ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
$env:GEMMA_ROOT="models\gemma-3-12b-it-qat-q4_0-unquantized"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The server logs `Pipeline ready.` once models are loaded. This takes a minute on first start.

Interactive API docs are available at `http://localhost:8000/docs` once running.

---

## 5. API Reference

### POST `/generate/text-to-video`

Generate a video from a text prompt only.

**Request body:**

```json
{
  "prompt": "A lone astronaut walks across a red desert at golden hour, dust swirling around their boots",
  "seed": 42,
  "height": 1024,
  "width": 1536,
  "num_frames": 121,
  "frame_rate": 24.0,
  "enhance_prompt": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | Text description of the video |
| `seed` | int | `10` | Random seed for reproducibility |
| `height` | int | `1024` | Output height in pixels — must be divisible by 64 |
| `width` | int | `1536` | Output width in pixels — must be divisible by 64 |
| `num_frames` | int | `121` | Number of frames — use 8k+1 format (e.g. 97, 121, 193) |
| `frame_rate` | float | `24.0` | Frames per second |
| `enhance_prompt` | bool | `false` | Auto-enhance the prompt using the text encoder |

---

### POST `/generate/image-to-video`

Generate a video conditioned on one or more input images. Images are passed as base64-encoded strings.

**Request body:**

```json
{
  "prompt": "The scene slowly comes to life, gentle waves forming on the water",
  "images": [
    {
      "image_b64": "<base64-encoded JPEG or PNG>",
      "frame_idx": 0,
      "strength": 1.0,
      "crf": 33
    }
  ],
  "seed": 42,
  "height": 1024,
  "width": 1536,
  "num_frames": 121,
  "frame_rate": 24.0,
  "enhance_prompt": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | Text description of the video |
| `images` | array | required | At least one image conditioning item (see below) |
| `seed` | int | `10` | Random seed |
| `height` | int | `1024` | Output height — divisible by 64 |
| `width` | int | `1536` | Output width — divisible by 64 |
| `num_frames` | int | `121` | Number of frames |
| `frame_rate` | float | `24.0` | Frames per second |
| `enhance_prompt` | bool | `false` | Auto-enhance the prompt |

**Image conditioning item:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_b64` | string | required | Base64-encoded image (JPEG or PNG) |
| `frame_idx` | int | `0` | Which frame to condition on (0 = first frame) |
| `strength` | float | `1.0` | Conditioning strength, 0.0–1.0 |
| `crf` | int | `33` | Compression quality for internal encoding |

**Encoding an image to base64 (Python):**

```python
import base64

with open("input.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")
```

---

### Both endpoints return immediately:

```json
{
  "job_id": "3f2a1b4c-...",
  "status": "pending"
}
```

---

### GET `/jobs/{job_id}`

Poll for job status or download the completed video.

**While pending or running:**

```json
{
  "job_id": "3f2a1b4c-...",
  "status": "running",
  "error": null
}
```

**On completion:** returns the `.mp4` file directly as a binary download (`video/mp4`).

**On failure:**

```json
{
  "job_id": "3f2a1b4c-...",
  "status": "failed",
  "error": "Resolution must be divisible by 64"
}
```

---

## 6. Full Example (Python)

```python
import base64
import time
import requests

BASE_URL = "http://localhost:8000"

# --- Text-to-video ---
response = requests.post(f"{BASE_URL}/generate/text-to-video", json={
    "prompt": "A timelapse of storm clouds rolling over a mountain range at dusk",
    "seed": 42,
    "height": 1024,
    "width": 1536,
    "num_frames": 121,
    "frame_rate": 24.0,
})
job_id = response.json()["job_id"]

# Poll until done
while True:
    r = requests.get(f"{BASE_URL}/jobs/{job_id}")
    if r.headers.get("content-type") == "video/mp4":
        with open("output.mp4", "wb") as f:
            f.write(r.content)
        print("Saved output.mp4")
        break
    status = r.json()
    if status["status"] == "failed":
        print("Failed:", status["error"])
        break
    print("Status:", status["status"])
    time.sleep(5)
```

```python
# --- Image-to-video ---
with open("input.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(f"{BASE_URL}/generate/image-to-video", json={
    "prompt": "The landscape slowly animates, trees swaying in a gentle breeze",
    "images": [{"image_b64": image_b64, "frame_idx": 0, "strength": 1.0}],
    "seed": 7,
    "height": 1024,
    "width": 1536,
    "num_frames": 121,
    "frame_rate": 24.0,
})
job_id = response.json()["job_id"]
# ... same polling loop as above
```

---

## Notes

- Jobs run sequentially through a single GPU worker — concurrent requests queue automatically.
- Output `.mp4` files are saved to `OUTPUT_DIR` (default `api/outputs/`) and served on job completion.
- For best prompt results, follow the [LTX-2 prompting guide](https://ltx.video/blog/how-to-prompt-for-ltx-2) — detailed, chronological, cinematographer-style descriptions work best.
- `num_frames` should follow the 8k+1 format: 97, 121, 145, 193, etc.
- `height` and `width` must both be divisible by 64. The pipeline internally generates at half resolution in stage 1 and upsamples 2x in stage 2.
