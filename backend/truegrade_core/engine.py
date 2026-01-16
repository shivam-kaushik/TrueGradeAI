
import os, json, re, math, hashlib, asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Callable, AsyncGenerator
from PyPDF2 import PdfReader
from openai import OpenAI
import faiss

# Import pydantic models
from .models import GradeResult, StudentResult

class GradingEngine:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.embed_model = "text-embedding-3-large"
        self.llm_model = "gpt-4o-mini"
        
        # State
        self.cold_cache_path = "cold_cache.json"
        self.cold_faiss_path = "cold_cache.faiss"
        self.cold_meta_path = "cold_meta.json"
        self.keypoints_cache_path = "keypoints_cache.json"
        self.verdict_cache_path = "verdict_cache.json"

        # Caches
        self.cold_sentences = []
        self.cold_index = None
        self.cold_meta = {"usage": {}, "version": 1}
        self.kp_cache = {}
        self.verdict_cache = {}
        
        # In-Memory Vectors
        self.fac_q_embs = None
        self.fac_a_embs = None
        self.fac_rows = []
        self.df_fac = None
        
        # Config
        self.match_threshold = 0.55
        self.kp_relevance_threshold = 0.63
        self.conf_threshold = 0.65
        self.novelty_trigger = 0.20
        self.rag2_topk_per_kp = 2
        self.cold_topk = 2
        
        self.stopwords = set("a an the and or of to in on for from by with into about over after before during is are was were be being been this that these those it its their his her our your as at if then than so such not no nor also too very more most can could should would may might must will shall do does did done have has had having which who whom whose where when why how".split())

    # ===================== HELPERS ===========================
    def clean_text_for_match(self, s: str) -> str:
        s = str(s).lower()
        s = re.sub(r'\bq\s*\d+\b', '', s)
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def tokenize_content(self, s: str) -> List[str]:
        s = self.clean_text_for_match(s)
        return [w for w in s.split() if w not in self.stopwords and len(w) > 2]

    def novelty_ratio(self, student: str, faculty: str) -> float:
        st = set(self.tokenize_content(student))
        ft = set(self.tokenize_content(faculty))
        if not st:
            return 0.0
        novel = st - ft
        return len(novel) / max(1, len(st))

    def split_sentences(self, text: str) -> List[str]:
        parts = re.split(r'[.!?]\s+', str(text).strip())
        return [p.strip() for p in parts if p and 5 <= len(p.split()) <= 40]

    def round_to_half(self, x: float) -> float:
        return round(float(x) * 2) / 2

    def semantic_hash_answer(self, s: str) -> str:
        norm = self.clean_text_for_match(s)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def safe_json_load(self, path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def safe_json_save(self, path: str, obj):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving {path}: {e}")

    # ================== EMBEDDINGS =====================
    def embed_texts(self, texts: List[str], batch_size=64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 3072), dtype=np.float32)
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(model=self.embed_model, input=batch)
            embs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            all_embs.extend(embs)
        mat = np.vstack(all_embs)
        mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        return mat.astype(np.float32)

    # ================== TEXTBOOK / RAG2 ======================
    def extract_text_chunks_from_pdf(self, pdf_path: str, max_words=180) -> List[str]:
        reader = PdfReader(pdf_path)
        full_text = ""
        for p in reader.pages:
            t = p.extract_text()
            if t:
                full_text += t + " "
        words = full_text.split()
        chunks = [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
        return chunks

    def build_faiss_index_from_chunks(self, chunks: List[str]):
        embs = self.embed_texts(chunks, batch_size=64)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        return index, chunks

    def rag2_retrieve_for_keypoints(self, kp_vecs, rag2_index, rag2_chunks, top_k=2):
        if kp_vecs.shape[0] == 0:
            return []
        D, I = rag2_index.search(kp_vecs, max(1, top_k))
        out = []
        for row_d, row_i in zip(D, I):
            hits = []
            for score, idx in zip(row_d, row_i):
                if idx >= 0:
                    hits.append((rag2_chunks[idx], float(score)))
            out.append(hits)
        return out

    # ================== COLD CACHE ======================
    def load_cold_cache(self):
        self.cold_sentences = []
        self.cold_index = None
        
        if os.path.exists(self.cold_cache_path) and os.path.exists(self.cold_faiss_path):
            try:
                with open(self.cold_cache_path, "r", encoding="utf-8") as f:
                    self.cold_sentences = json.load(f)
                self.cold_index = faiss.read_index(self.cold_faiss_path)
            except Exception:
                self.cold_sentences, self.cold_index = [], None

        self.cold_meta = self.safe_json_load(self.cold_meta_path, {"usage": {}, "version": 1})

    def save_cold_cache(self):
        with open(self.cold_cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cold_sentences, f, indent=2, ensure_ascii=False)
        if self.cold_index is not None:
            faiss.write_index(self.cold_index, self.cold_faiss_path)
        with open(self.cold_meta_path, "w", encoding="utf-8") as f:
            json.dump(self.cold_meta, f, indent=2, ensure_ascii=False)

    def build_cold_cache_from_corpus(self, corpus_texts: List[str], top_n=200):
        sent_freq = {}
        for doc in corpus_texts:
            for s in self.split_sentences(doc):
                key = self.clean_text_for_match(s)
                if len(key) < 15:
                    continue
                sent_freq[key] = sent_freq.get(key, 0) + 1
        sorted_sents = sorted(sent_freq.items(), key=lambda kv: kv[1], reverse=True)
        top_sentences = [s for s, _ in sorted_sents[:top_n]]
        if not top_sentences:
            return [], None
        embs = self.embed_texts(top_sentences)
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs)
        return top_sentences, idx

    # ================== KEYPOINTS ======================
    def extract_keypoints_once(self, faculty_answer: str, max_marks: float):
        prompt = f"""
Extract atomic, independently checkable key points from the FACULTY ANSWER.
Rules:
- Each key point must be short and specific.
- Assign "weight" in multiples of 0.5 only.
- Total weight MUST sum exactly to {max_marks}.
- Mark "core": true for must-have points.

FACULTY ANSWER:
{faculty_answer}

Return VALID JSON ONLY as a list:
[
  {{"id":"KP1","point":"...","weight":0.5,"core":true}},
  ...
]
"""
        resp = self.client.chat.completions.create(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
        text = resp.choices[0].message.content
        try:
             # Extract list
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                 raise ValueError("No JSON list found")
            data = json.loads(text[start:end])
        except Exception:
            return [{"id": "KP1", "point": faculty_answer[:200], "weight": self.round_to_half(max_marks), "core": True}]

        # Normalization and Safety logic similar to original script
        scaled = []
        for kp in data:
            kp["weight"] = self.round_to_half(float(kp.get("weight", 0.5)))
            kp["core"] = bool(kp.get("core", False))
            kp["id"] = str(kp.get("id", "KP"))
            kp["point"] = str(kp.get("point", "")).strip()
            if kp["point"] and kp["weight"] > 0:
                scaled.append(kp)
        
        # Ensure sum matches max_marks (Simple realignment)
        current_sum = sum(k["weight"] for k in scaled)
        if current_sum != max_marks and scaled:
            diff = max_marks - current_sum
            # Dump diff into first item for simplicity in this port
            scaled[0]["weight"] = max(0.5, self.round_to_half(scaled[0]["weight"] + diff))

        return scaled

    # ================== LLM EVALUATION ======================
    def llm_evaluate_keypoints_batch(self, question: str, keypoints: list, student_answer: str, rag2_evidence: dict):
        evid = {}
        for kpid, snippets in (rag2_evidence or {}).items():
            short_snips = [s[:320] for s in snippets[:2]]
            evid[kpid] = short_snips

        payload = {
            "question": question,
            "keypoints": keypoints,
            "student_answer": student_answer,
            "rag2_evidence": evid
        }

        prompt = f"""
You are a strict academic grader.
Grade ONLY by checking coverage of the provided KEYPOINTS.
Award marks in increments of 0.5 ONLY.

For each keypoint:
- awarded: 0, 0.5, or full weight.
- status: FULL / PARTIAL / MISSING / INCORRECT
- reason: Explain deduction.

Input JSON:
{json.dumps(payload, ensure_ascii=False)}

Return VALID JSON ONLY as a list:
[
  {{"id":"KP1","awarded":0.5,"status":"PARTIAL","reason":"..."}},
  ...
]
"""
        resp = self.client.chat.completions.create(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
        text = resp.choices[0].message.content
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            data = json.loads(text[start:end])
        except:
             return []
        return data

    # ================== MAIN JOB ======================
    async def process_grading_job(self, 
                                  kafka_job_id: str,
                                  faculty_csv_path: str, 
                                  student_csv_path: str, 
                                  textbook_pdf_path: str,
                                  status_callback: Callable[[str, int], None]):
        """
        Main async generator logic
        """
        await asyncio.sleep(0.1)
        status_callback("Loading CSVs...", 5)
        
        # Load Data
        df_fac = pd.read_csv(faculty_csv_path, encoding="utf-8") # Assume utf-8 for now
        df_stu = pd.read_csv(student_csv_path, encoding="utf-8")
        
        # Normalize columns (simple version)
        df_fac.columns = [c.strip().lower().replace(" ", "_") for c in df_fac.columns]
        df_stu.columns = [c.strip().lower().replace(" ", "_") for c in df_stu.columns]
        
        status_callback("Embedding Faculty Questions...", 10)
        self.fac_rows = df_fac.to_dict('records')
        fac_questions = [str(r.get("question_number_and_question", "")) for r in self.fac_rows]
        fac_answers = [str(r.get("answer", "")) for r in self.fac_rows]
        
        self.fac_q_embs = self.embed_texts([self.clean_text_for_match(q) for q in fac_questions])
        self.fac_a_embs = self.embed_texts(fac_answers)

        # Cold Cache Init
        status_callback("Initializing Cold Cache...", 15)
        self.load_cold_cache()
        if not self.cold_sentences:
            bootstrap = fac_answers + fac_questions
            if "student_answer" in df_stu:
                 bootstrap += df_stu["student_answer"].fillna("").astype(str).tolist()
            self.cold_sentences, self.cold_index = self.build_cold_cache_from_corpus(bootstrap)
            self.save_cold_cache()

        self.kp_cache = self.safe_json_load(self.keypoints_cache_path, {})
        self.verdict_cache = self.safe_json_load(self.verdict_cache_path, {})
        
        # RAG 2 Lazy Load
        rag2_built = False
        rag2_index, rag2_chunks = None, None
        
        results = []
        total_students = len(df_stu)
        
        for idx, row in df_stu.iterrows():
            pct = 20 + int((idx / total_students) * 70)
            sid = str(row.get("sno", f"S{idx}"))
            q_text = str(row.get("question", ""))
            s_ans = str(row.get("student_answer", ""))
            
            status_callback(f"Grading Student {sid}...", pct)

            if not s_ans.strip():
                continue

            # 1. Match Faculty Question
            q_clean = self.clean_text_for_match(q_text)
            q_emb = self.embed_texts([q_clean])[0]
            sims = self.fac_q_embs @ q_emb
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            
            if best_sim < self.match_threshold:
                continue # Skip mismatch

            f_row = self.fac_rows[best_idx]
            f_ans = str(f_row.get("answer", ""))
            max_marks = float(f_row.get("factual_score", 1.0))

            # 2. Keypoints
            kp_key = str(best_idx)
            if kp_key not in self.kp_cache:
                self.kp_cache[kp_key] = {
                    "max_marks": max_marks,
                    "keypoints": self.extract_keypoints_once(f_ans, max_marks)
                }
                self.safe_json_save(self.keypoints_cache_path, self.kp_cache)
                
            keypoints = self.kp_cache[kp_key]["keypoints"]
            
            # 3. Simple Evaluation Logic (Ported logic simplified for brevity)
            # Embed Keypoints needed for selection
            kp_texts = [k["point"] for k in keypoints]
            # Ideally cache embeddings too, but re-embedding 10 keypoints is fast enough for now
            kp_embs = self.embed_texts(kp_texts)
            s_emb = self.embed_texts([s_ans])[0]
            
            # Select relevant KPs
            kp_sims = kp_embs @ s_emb
            selected_kps = []
            for k, sim in zip(keypoints, kp_sims):
                if k.get("core") or sim > self.kp_relevance_threshold:
                    selected_kps.append(k)
            if not selected_kps: 
                selected_kps = keypoints[:5]

            # LLM Eval
            rag2_evidence = {}
            # Check Novelty/RAG2 need
            nov = self.novelty_ratio(s_ans, f_ans)
            if nov > self.novelty_trigger and not rag2_built and os.path.exists(textbook_pdf_path):
                 status_callback("Building RAG2 Index from Textbook...", pct)
                 chunks = self.extract_text_chunks_from_pdf(textbook_pdf_path)
                 rag2_index, rag2_chunks = self.build_faiss_index_from_chunks(chunks)
                 rag2_built = True

            # If RAG2 built, verify
            if rag2_built:
                 # Minimal RAG2 logic for this pass
                 pass 

            eval_res = self.llm_evaluate_keypoints_batch(
                str(f_row.get("question_number_and_question")),
                selected_kps,
                s_ans,
                rag2_evidence
            )
            
            # Score Calc
            score = 0.0
            deductions = []
            for res in eval_res:
                awarded = float(res.get("awarded", 0))
                score += awarded
                if awarded < 0.5: # Arbitrary threshold for deduction note
                     deductions.append({"reason": res.get("reason"), "key_point": res.get("id")})
            
            score = min(score, max_marks)
            
            results.append({
                "student_id": sid,
                "score": score,
                "deductions": deductions,
                "notes": {"novelty": nov}
            })
            
            # Yield back control potentially
            await asyncio.sleep(0.01)

        status_callback("Finalizing...", 100)
        return results
