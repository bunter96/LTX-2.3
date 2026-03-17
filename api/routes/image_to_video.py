from fastapi import APIRouter, Request

from api.jobs import JobResponse, create_job, submit_job
from api.models import ImageToVideoRequest
from api.pipeline_runner import run_image_to_video

router = APIRouter()


@router.post("/generate/image-to-video", response_model=JobResponse)
async def image_to_video(req: ImageToVideoRequest, request: Request):
    pipeline = request.app.state.pipeline
    job = create_job()

    submit_job(job, lambda: run_image_to_video(pipeline, req))

    return JobResponse(job_id=job.id, status=job.status)
