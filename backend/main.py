
import os
import shutil
import uuid
import asyncio
from typing import Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Core
import sys
from pathlib import Path
# Add current directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from truegrade_core.engine import GradingEngine
from truegrade_core.models import JobStatus

app = FastAPI(title="TrueGradeAI Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
JOBS: Dict[str, JobStatus] = {}
JOB_LOGS: Dict[str, list] = {} # In-memory logs for SSE
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Engine Check
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
engine = GradingEngine(openai_api_key=OPENAI_KEY if OPENAI_KEY else "dummy")

class StartJobRequest(BaseModel):
    faculty_file: str
    student_file: str
    textbook_file: str

@app.get("/")
def read_root():
    return {"message": "TrueGradeAI API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": file_path}

def status_updater(job_id: str, message: str, progress: int):
    """Callback for the engine to update global state"""
    if job_id in JOBS:
        JOBS[job_id].status = "processing"
        JOBS[job_id].progress = progress
        JOBS[job_id].message = message
        # Append to logs for SSE
        if job_id not in JOB_LOGS:
            JOB_LOGS[job_id] = []
        JOB_LOGS[job_id].append(message)
        print(f"[{job_id}] {progress}% - {message}")

async def run_grading_task(job_id: str, req: StartJobRequest):
    try:
        results = await engine.process_grading_job(
            kafka_job_id=job_id,
            faculty_csv_path=os.path.join(UPLOAD_DIR, req.faculty_file),
            student_csv_path=os.path.join(UPLOAD_DIR, req.student_file),
            textbook_pdf_path=os.path.join(UPLOAD_DIR, req.textbook_file),
            status_callback=lambda msg, pct: status_updater(job_id, msg, pct)
        )
        # Save Results
        import json
        with open(f"results_{job_id}.json", "w") as f:
            json.dump(results, f, default=str)
        JOBS[job_id].status = "completed"
        JOBS[job_id].progress = 100
        JOBS[job_id].message = "Done"
    except Exception as e:
        import traceback
        traceback.print_exc()
        JOBS[job_id].status = "failed"
        JOBS[job_id].message = str(e)

@app.post("/job/start")
async def start_job(req: StartJobRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobStatus(job_id=job_id, status="queued", progress=0, message="Queued")
    background_tasks.add_task(run_grading_task, job_id, req)
    return {"job_id": job_id}

@app.get("/job/{job_id}/stream")
async def stream_status(job_id: str):
    async def event_generator():
        last_idx = 0
        while True:
            if job_id not in JOBS:
                break
            
            # Stream logs
            logs = JOB_LOGS.get(job_id, [])
            if last_idx < len(logs):
                for log in logs[last_idx:]:
                    yield f"data: {log}\n\n"
                last_idx = len(logs)
            
            # Check completion
            status = JOBS[job_id]
            if status.status in ["completed", "failed"]:
                yield f"data: [DONE] {status.status}\n\n"
                break
            
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/results/{job_id}")
def get_results(job_id: str):
    path = f"results_{job_id}.json"
    if os.path.exists(path):
        import json
        with open(path, "r") as f:
            return json.load(f)
    return {"error": "Results not found"}
