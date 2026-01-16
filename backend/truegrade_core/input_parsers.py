import pandas as pd
import io
import logging

# Configure logger
logger = logging.getLogger("uvicorn.error")
import io

def parse_faculty_csv(file_content: bytes) -> list:
    """
    Parses faculty_key.csv content and returns list of dicts for QuestionStore.
    Expected columns: s_no, question_number_and_question, answer, marks (optional, defaults to 5)
    """
    try:
        df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(file_content), encoding='latin1')
        
    logger.info(f"Parsed Faculty CSV Columns: {list(df.columns)}")
    
        
    questions = []
    
    # Normalizing column names
    df.columns = [c.lower().strip() for c in df.columns]
    
    for _, row in df.iterrows():
        # Heuristic to find question column
        q_col = next((c for c in df.columns if 'question' in c), None)
        a_col = next((c for c in df.columns if 'answer' in c or 'key' in c), None)
        m_col = next((c for c in df.columns if 'score' in c or 'marks' in c), None)
        
        if q_col and a_col:
            questions.append({
                "question": str(row[q_col]).strip(),
                "answer": str(row[a_col]).strip(),
                "marks": float(row[m_col]) if m_col and pd.notna(row[m_col]) else 5.0
            })
            
    return questions

def parse_student_csv(file_content: bytes) -> list:
    """
    Parses student_answers.csv content.
    Expected columns: sno, qno, question, student_answer
    """
    try:
        df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(file_content), encoding='latin1')
        
    answers = []
    df.columns = [c.lower().strip() for c in df.columns]
    
    for _, row in df.iterrows():
        # We need to map student answer to a question.
        # Ideally, we match by Question Text if QID is not robust.
        q_col = next((c for c in df.columns if 'question' in c), None)
        a_col = next((c for c in df.columns if 'answer' in c), None)
        
        if q_col and a_col:
            answers.append({
                "question_text": str(row[q_col]).strip(),
                "student_answer": str(row[a_col]).strip()
            })
            
    return answers
