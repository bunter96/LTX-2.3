from fastapi import APIRouter, Request

from api.jobs import create_job, submit_job
from api.models import JobResponse, TextToVideoRequest
from api.pipeline_runner import run_text_to_video

router = APIRouter()


@router.post("/generate/text-to-video", response_model=JobResponse)
async def text_to_video(req: TextToVideoRequest, request: Request):
    pipeline = request.app.state.pipeline
    job = create_job()

    submit_job(job, lambda: run_text_to_video(pipeline, req))

    return JobResponse(job_id=job.id, status=job.status)
