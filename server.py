#!/usr/bin/env python3
"""
Beat Analyzer API Server

FastAPI-based HTTP API for audio analysis.
Provides background job processing with SQLite persistence.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile
from pydantic import BaseModel

# Import analysis function from main module
from analyze import ALLIN1_AVAILABLE, MSAF_AVAILABLE, analyze_audio
from db import JobDatabase

# Configure logging
logger = logging.getLogger(__name__)

# Configuration from environment variables
DATABASE_PATH = os.environ.get("DATABASE_PATH", "beat_analyzer.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "beat-analyzer"))

app = FastAPI(
    title="Beat Analyzer API",
    description="Audio analysis API for extracting BPM, beats, sections, and more",
    version="2.0.0"
)

# Supported audio extensions
SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

# Database instance (path configurable via DATABASE_PATH env var)
db = JobDatabase(DATABASE_PATH)

# Optional httpx for webhooks
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed - webhook callbacks disabled")


class JobStatus:
    """Job status constants."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    filename: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress_pct: Optional[int] = None
    progress_stage: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    allin1_available: bool
    msaf_available: bool
    webhooks_available: bool


def create_progress_callback(job_id: str):
    """Create a progress callback function for a job."""
    def update_progress(pct: int, stage: str):
        db.update_progress(job_id, pct, stage)
    return update_progress


async def fire_webhook(job_id: str, webhook_url: str):
    """
    Fire webhook notification for job completion.

    Args:
        job_id: Job identifier
        webhook_url: URL to POST notification to
    """
    if not HTTPX_AVAILABLE:
        logger.warning(f"Cannot fire webhook for {job_id}: httpx not installed")
        return

    job = db.get_job(job_id)
    if not job:
        return

    payload = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job.get("filename"),
        "completed_at": job.get("completed_at"),
        "result": job.get("result"),
        "error": job.get("error")
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(2):  # 1 retry
            try:
                response = await client.post(webhook_url, json=payload)
                response.raise_for_status()
                db.mark_webhook_fired(job_id, success=True)
                logger.info(f"Webhook fired successfully for {job_id}")
                return
            except Exception as e:
                logger.warning(f"Webhook attempt {attempt + 1} failed for {job_id}: {e}")
                if attempt == 0:
                    await asyncio.sleep(1)

    db.mark_webhook_fired(job_id, success=False)
    logger.error(f"Webhook failed after retries for {job_id}")


def process_analysis_job(
    job_id: str,
    file_path: str,
    fast: bool,
    trim: bool,
    stems: bool,
    waveform: bool,
    vocals: bool,
    webhook_url: Optional[str]
):
    """
    Background worker that runs audio analysis.

    Args:
        job_id: Unique job identifier
        file_path: Path to uploaded audio file
        fast: Skip allin1 ML model if True
        trim: Create trimmed versions if True
        stems: Enable stem separation if True
        waveform: Include waveform data if True
        vocals: Include vocal detection if True
        webhook_url: Optional URL for completion callback
    """
    # Update status to processing
    db.update_status(job_id, JobStatus.PROCESSING)

    try:
        # Create progress callback
        progress_callback = create_progress_callback(job_id)

        # Run analysis
        result = analyze_audio(
            file_path,
            use_allin1=not fast,
            trim=trim,
            trim_output_dir=str(Path(file_path).parent) if trim else None,
            extract_waveform=waveform,
            detect_vocals=vocals,
            separate_stems_flag=stems,
            stems_output_dir=str(Path(file_path).parent / "stems") if stems else None,
            progress_callback=progress_callback
        )

        # Update job with result
        db.update_status(job_id, JobStatus.COMPLETED, result=result)

        # Fire webhook if configured
        if webhook_url:
            asyncio.run(fire_webhook(job_id, webhook_url))

    except Exception as e:
        # Update job with error
        db.update_status(job_id, JobStatus.FAILED, error=str(e))

        # Fire webhook for failure too
        if webhook_url:
            asyncio.run(fire_webhook(job_id, webhook_url))

    finally:
        # Cleanup temp file (unless trim/stems created files we want to keep)
        if not trim and not stems:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass


@app.on_event("startup")
async def startup_event():
    """Run on server startup."""
    # Cleanup expired jobs (older than 7 days)
    deleted = db.cleanup_expired(days=7)
    if deleted:
        logger.info(f"Cleaned up {deleted} expired jobs")


@app.post("/analyze", response_model=dict, tags=["Analysis"])
async def analyze(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    fast: bool = Query(False, description="Skip ML model (faster but less accurate section labels)"),
    trim: bool = Query(False, description="Create trimmed audio versions (_mix, _drop, _body)"),
    stems: bool = Query(False, description="Separate stems (vocals, drums, bass, other) - slower"),
    waveform: bool = Query(True, description="Include waveform data for visualization"),
    vocals: bool = Query(True, description="Include vocal detection"),
    webhook_url: Optional[str] = Query(None, description="URL for completion callback notification")
):
    """
    Submit an audio file for analysis.

    Returns a job ID immediately. Use GET /jobs/{job_id} to check status and retrieve results.

    - **file**: Audio file (mp3, wav, flac, m4a, ogg)
    - **fast**: Skip allin1 ML model for faster processing
    - **trim**: Create trimmed versions of the audio
    - **stems**: Separate into stems (vocals, drums, bass, other) - requires demucs
    - **waveform**: Include waveform peaks/minmax for visualization (default: true)
    - **vocals**: Include vocal presence detection (default: true)
    - **webhook_url**: URL to POST when job completes or fails
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Generate job ID and create upload directory
    job_id = str(uuid.uuid4())
    job_dir = Path(UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / file.filename

    # Save uploaded file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Create job record in database
    db.create_job(
        job_id=job_id,
        filename=file.filename,
        file_path=str(file_path),
        webhook_url=webhook_url,
        fast=fast,
        trim=trim,
        stems=stems,
        waveform=waveform,
        vocals=vocals
    )

    # Queue background task
    background_tasks.add_task(
        process_analysis_job,
        job_id,
        str(file_path),
        fast,
        trim,
        stems,
        waveform,
        vocals,
        webhook_url
    )

    return {"job_id": job_id, "status": JobStatus.QUEUED}


@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job(job_id: str):
    """
    Get job status and result.

    - **job_id**: The job ID returned from POST /analyze

    Returns job status (queued, processing, completed, failed), progress, and result when complete.
    """
    job = db.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        filename=job.get("filename"),
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at"),
        progress_pct=job.get("progress_pct"),
        progress_stage=job.get("progress_stage"),
        result=job.get("result"),
        error=job.get("error")
    )


@app.get("/jobs", response_model=List[dict], tags=["Jobs"])
async def list_jobs(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of jobs to return"),
    status: Optional[str] = Query(None, description="Filter by status (queued, processing, completed, failed)")
):
    """
    List all jobs.

    Returns jobs sorted by creation time (newest first).

    - **limit**: Maximum number of jobs to return (default 20)
    - **status**: Filter by job status
    """
    # Validate status if provided
    if status:
        valid_statuses = [JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.COMPLETED, JobStatus.FAILED]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid values: {', '.join(valid_statuses)}"
            )

    return db.list_jobs(limit=limit, status=status)


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """
    Delete a job and cleanup associated files.

    - **job_id**: The job ID to delete
    """
    job = db.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Cleanup files
    file_path = job.get("file_path")
    if file_path:
        temp_dir = Path(file_path).parent
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    db.delete_job(job_id)

    return {"deleted": True, "job_id": job_id}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Health check endpoint.

    Returns server status and availability of optional dependencies.
    """
    return HealthResponse(
        status="healthy",
        allin1_available=ALLIN1_AVAILABLE,
        msaf_available=MSAF_AVAILABLE,
        webhooks_available=HTTPX_AVAILABLE
    )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Beat Analyzer API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }
