import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.pending
    output_path: str | None = None
    error: str | None = None


_jobs: dict[str, Job] = {}
_executor = ThreadPoolExecutor(max_workers=1)  # GPU is single-threaded


def create_job() -> Job:
    job = Job(id=str(uuid.uuid4()))
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def submit_job(job: Job, fn: Callable[[], str]) -> None:
    """Run fn in background; fn should return the output file path."""

    def _run():
        job.status = JobStatus.running
        try:
            job.output_path = fn()
            job.status = JobStatus.done
        except Exception as exc:
            job.error = str(exc)
            job.status = JobStatus.failed

    _executor.submit(_run)
