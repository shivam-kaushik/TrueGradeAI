from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class GradeResult(BaseModel):
    score: float
    deductions: List[Dict[str, Any]]
    confidence: Dict[str, float]
    notes: Dict[str, Any]

class StudentResult(BaseModel):
    student_id: str
    question_no: str
    faculty_match_idx: int
    faculty_match_sim: float
    used_sources: Dict[str, bool]
    max_marks: float
    result: GradeResult

class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    progress: int
    message: str
