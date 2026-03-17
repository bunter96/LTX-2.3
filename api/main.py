import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

import api.config as config
from api.jobs import JobStatus, get_job
from api.routes.image_to_video import router as i2v_router
from api.routes.text_to_video import router as t2v_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_pipeline():
    from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.distilled import DistilledPipeline

    loras = []
    if config.LORA_SPECS:
        for spec in config.LORA_SPECS.split(","):
            path, strength = spec.strip().rsplit(":", 1)
            loras.append(LoraPathStrengthAndSDOps(path.strip(), float(strength), LTXV_LORA_COMFY_RENAMING_MAP))

    quantization = None
    if config.QUANTIZATION == "fp8-cast":
        quantization = QuantizationPolicy.fp8_cast()
    elif config.QUANTIZATION == "fp8-scaled-mm":
        quantization = QuantizationPolicy.fp8_scaled_mm()

    logger.info("Loading DistilledPipeline...")
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=config.DISTILLED_CHECKPOINT_PATH,
        spatial_upsampler_path=config.SPATIAL_UPSAMPLER_PATH,
        gemma_root=config.GEMMA_ROOT,
        loras=loras,
        quantization=quantization,
    )
    logger.info("Pipeline ready.")
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = _build_pipeline()
    yield
    # cleanup on shutdown if needed


app = FastAPI(title="LTX-2 Distilled API", lifespan=lifespan)
app.include_router(t2v_router)
app.include_router(i2v_router)


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "Job not found"})

    if job.status == JobStatus.done:
        return FileResponse(job.output_path, media_type="video/mp4", filename=f"{job_id}.mp4")

    return JSONResponse(
        content={
            "job_id": job.id,
            "status": job.status,
            "error": job.error,
        }
    )
