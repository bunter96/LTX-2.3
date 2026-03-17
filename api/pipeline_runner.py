import base64
import tempfile
import uuid
from pathlib import Path

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

from api.config import OUTPUT_DIR
from api.models import ImageConditioningItem, ImageToVideoRequest, TextToVideoRequest


def _save_b64_image(b64_str: str) -> str:
    """Decode a base64 image string to a temp file and return its path."""
    data = base64.b64decode(b64_str)
    suffix = ".jpg"
    # Detect PNG by magic bytes
    if data[:4] == b"\x89PNG":
        suffix = ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _build_image_conditionings(items: list[ImageConditioningItem]) -> list[ImageConditioningInput]:
    return [
        ImageConditioningInput(
            path=_save_b64_image(item.image_b64),
            frame_idx=item.frame_idx,
            strength=item.strength,
            crf=item.crf,
        )
        for item in items
    ]


def _run_pipeline(
    pipeline: DistilledPipeline,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    enhance_prompt: bool,
    images: list[ImageConditioningInput],
) -> str:
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    video, audio = pipeline(
        prompt=prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=images,
        tiling_config=tiling_config,
        enhance_prompt=enhance_prompt,
    )

    output_path = str(OUTPUT_DIR / f"{uuid.uuid4()}.mp4")
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )
    return output_path


def run_text_to_video(pipeline: DistilledPipeline, req: TextToVideoRequest) -> str:
    return _run_pipeline(
        pipeline=pipeline,
        prompt=req.prompt,
        seed=req.seed,
        height=req.height,
        width=req.width,
        num_frames=req.num_frames,
        frame_rate=req.frame_rate,
        enhance_prompt=req.enhance_prompt,
        images=[],
    )


def run_image_to_video(pipeline: DistilledPipeline, req: ImageToVideoRequest) -> str:
    images = _build_image_conditionings(req.images)
    return _run_pipeline(
        pipeline=pipeline,
        prompt=req.prompt,
        seed=req.seed,
        height=req.height,
        width=req.width,
        num_frames=req.num_frames,
        frame_rate=req.frame_rate,
        enhance_prompt=req.enhance_prompt,
        images=images,
    )
