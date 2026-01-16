
import os
import shutil
import uuid
import asyncio
import sys
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

from typing import Dict, List, Optional
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Core
try:
    from truegrade_core.engine import GradingEngine
    from truegrade_core.models import JobStatus
    from truegrade_core.input_parsers import parse_faculty_csv, parse_student_csv
    from truegrade_core.input_parsers import parse_student_csv
except ImportError:
    from .truegrade_core.engine import GradingEngine
    from .truegrade_core.models import JobStatus
    from .truegrade_core.input_parsers import parse_faculty_csv, parse_student_csv

app = FastAPI(title="TrueGradeAI Backend v2")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("----------- REGISTERED ROUTES -----------")
    for route in app.routes:
        print(f"PATH: {route.path} | NAME: {route.name}")
    print("-----------------------------------------")

# ================= STATE MANAGEMENT =================
# In-memory "Database"
class QuestionStore:
    def __init__(self):
        self.questions = [
            {"id": 0, "question": "Define feudalism", "answer": "Feudalism is a social system existing in medieval Europe in which people worked and fought for nobles who gave them protection and land in return.", "marks": 3.0},
            {"id": 1, "question": "Explain the decline of Mughal authority.", "answer": "The decline was caused by weak successors, economic instability, and the rise of regional powers like the Marathas.", "marks": 5.0}
        ]
        self.next_id = 2

    def add(self, q: str, a: str, m: float):
        new_q = {"id": self.next_id, "question": q, "answer": a, "marks": m}
        self.questions.append(new_q)
        self.next_id += 1
        return new_q

    def bulk_add(self, items: list):
        added = []
        for item in items:
            added.append(self.add(item["question"], item["answer"], item["marks"]))
        return added

    def find_by_text(self, text: str):
        # fuzzy match or exact match
        text = text.lower().strip()
        for q in self.questions:
            if q["question"].lower().strip() == text:
                return q
            # Simple substring fallback
            if text in q["question"].lower() or q["question"].lower() in text:
                 # Ensure reasonable length match to avoid false positives
                 if len(text) > 10: return q
        return None

    def get(self, qid: int):
        for q in self.questions:
            if q["id"] == qid:
                return q
        return None

    def all(self):
        return self.questions

db = QuestionStore()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
engine = GradingEngine(openai_api_key=OPENAI_KEY if OPENAI_KEY else "dummy")

# ================= MODELS =================
class AddQuestionRequest(BaseModel):
    question: str
    answer: str
    marks: float

class GradeRequest(BaseModel):
    facultyQuestionId: int
    studentAnswerText: str
    allowRag2: bool = True
    useCache: bool = True

class GradeResponse(BaseModel):
    score: float
    maxMarks: float
    confidence: float
    usedRag2: bool
    novelty: float
    deductions: List[Dict]
    keyPointBreakdown: List[Dict]
    facultyMatch: Dict
    meta: Dict

# ================= ROUTES =================

@app.get("/api/questions")
def get_questions():
    return db.all()

@app.post("/api/questions")
def add_question(req: AddQuestionRequest):
    return db.add(req.question, req.answer, req.marks)

@app.post("/api/upload/questions")
async def upload_questions(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")
        contents = await file.read()
        questions = parse_faculty_csv(contents)
        print(f"Parsed {len(questions)} questions")
        added = db.bulk_add(questions)
        return {"message": f"Successfully added {len(added)} questions"}
    except Exception as e:
        print(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/batch-grade")
async def batch_grade(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    print(f"Batch Grade Request Received for file: {file.filename}")
    answers = []
    try:
        contents = await file.read()
        print(f"File read, size: {len(contents)} bytes")
        answers = parse_student_csv(contents)
        print(f"Parsed {len(answers)} answers")
    except Exception as e:
        print(f"Error processing batch grade: {e}")
        # Return 500 so frontend sees the error
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV: {str(e)}")
    
    results = []
    
    for ans in answers:
        # Match question
        q_data = db.find_by_text(ans["question_text"])
        if not q_data:
            results.append({
                "student_answer": ans["student_answer"][:50] + "...",
                "error": "Question not found in bank"
            })
            continue
            
        try:
            # Grade (Generic dummy path for textbook)
            res = await engine.grade_single_answer(
                question_text=q_data["question"],
                faculty_answer=q_data["answer"],
                max_marks=q_data["marks"],
                student_answer=ans["student_answer"],
                allow_rag2=True,
                textbook_path="uploads/textbook.pdf"
            )
            
            results.append({
                "question": q_data["question"],
                "score": res["score"],
                "max_marks": res["maxMarks"],
                "feedback": "; ".join([d["reason"] for d in res["deductions"]]) or "Perfect"
            })
        except Exception as e:
            print(f"Error grading answer '{ans['student_answer'][:20]}...': {e}")
            results.append({
                "student_answer": ans["student_answer"][:50] + "...",
                "error": f"Grading failed: {str(e)}"
            })
        
    return results

@app.post("/api/grade", response_model=GradeResponse)
async def grade_answer(req: GradeRequest):
    q_data = db.get(req.facultyQuestionId)
    if not q_data:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Check for textbook (optional, generic path for now)
    textbook_path = "uploads/textbook.pdf" 
    
    result = await engine.grade_single_answer(
        question_text=q_data["question"],
        faculty_answer=q_data["answer"],
        max_marks=q_data["marks"],
        student_answer=req.studentAnswerText,
        allow_rag2=req.allowRag2,
        textbook_path=textbook_path
    )
    
    return result

# Legacy Routes (Keep for compatibility if needed, but deprioritized)
# ... code omitted for brevity but existing upload/job routes would technically remain ...
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request, path_name: str):
    print(f"DEBUG: Unhandled request to path: {path_name} method: {request.method}")
    return {"status": "catch_all", "path": path_name, "method": request.method}
