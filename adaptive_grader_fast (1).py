#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adaptive_grader_fast.py — Keypoint-first grading with conditional RAG2 + caches (optimized)

What changed vs your original script :contentReference[oaicite:0]{index=0}
- Keypoints drive grading (not holistic similarity).
- Scores quantized to 0.5 increments ONLY.
- Length is NOT do-or-die: used only as a weak confidence signal (never scales marks).
- ONE embedding per student answer (reused everywhere).
- Faculty question embeddings precomputed once; matching is matrix dot-product.
- Keypoints extracted ONCE per faculty question (cached to disk).
- Keypoint evaluation done in ONE LLM call per answer (not per keypoint).
- Verdict cache (question_id + semantic hash) prevents regrading repeated answers.
- RAG2 only when confidence is low OR novelty needs verification (and only for disputed keypoints).
- Cold cache kept but used sparingly (cheap evidence/confidence boost).

Expected files in the same folder:
- faculty_key.csv       (e.g., s_no | question_number_and_question | answer | factual_score)
- student_answers.csv   (e.g., sno  | qno | question | student_answer)
- textbook.pdf
- (auto) cold_cache.json, cold_cache.faiss, cold_meta.json
- (auto) keypoints_cache.json
- (auto) verdict_cache.json

Env:
- OPENAI_API_KEY in .env
"""

import os, json, re, math, hashlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import OpenAI
import faiss

# ========================== ENV ==========================
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise SystemExit("❌ Missing OPENAI_API_KEY in .env file.")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

# ======================= PATHS ===========================
COLD_JSON_PATH = "cold_cache.json"
COLD_FAISS_PATH = "cold_cache.faiss"
COLD_META_PATH = "cold_meta.json"

KEYPOINTS_CACHE_PATH = "keypoints_cache.json"
VERDICT_CACHE_PATH = "verdict_cache.json"

# ======================= CONFIG ==========================
MATCH_THRESHOLD = 0.55            # semantic match (student question ↔ faculty question)
KP_RELEVANCE_THRESHOLD = 0.63     # student answer ↔ keypoint relevance (vector)
CONF_THRESHOLD = 0.65             # if below -> consider RAG2 for disputed keypoints
NOVELTY_TRIGGER = 0.20            # if student adds ≥20% new concepts not in key -> verify
RAG2_TOPK_PER_KP = 2              # small, per disputed keypoint
COLD_TOPK = 2                     # cheap evidence snippets
MAX_KP_PER_ANSWER = 18            # keep LLM prompt bounded for speed/cost

# NOTE: length is NOT used to scale marks; only weak confidence signal
EXPECTED_MIN_WORDS_BY_MARKS = {1: 3, 2: 8, 3: 12, 5: 30, 8: 60, 10: 100}

STOPWORDS = set("""
a an the and or of to in on for from by with into about over after before during is are was were be being been
this that these those it its their his her our your as at if then than so such not no nor also too very more most
can could should would may might must will shall do does did done have has had having which who whom whose where when why how
""".split())

# ===================== HELPERS ===========================
def read_csv_safely(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df

def clean_text_for_match(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'\bq\s*\d+\b', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize_content(s: str):
    s = clean_text_for_match(s)
    return [w for w in s.split() if w not in STOPWORDS and len(w) > 2]

def novelty_ratio(student: str, faculty: str) -> float:
    st = set(tokenize_content(student))
    ft = set(tokenize_content(faculty))
    if not st:
        return 0.0
    novel = st - ft
    return len(novel) / max(1, len(st))

def round_to_half(x: float) -> float:
    return round(float(x) * 2) / 2

def semantic_hash_answer(s: str) -> str:
    # stable-ish hash to catch repeated answers; keep it simple and fast
    norm = clean_text_for_match(s)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def safe_json_load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_json_save(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def split_sentences(text: str):
    parts = re.split(r'[.!?]\s+', str(text).strip())
    return [p.strip() for p in parts if p and 5 <= len(p.split()) <= 40]

# ================== EMBEDDINGS / SIM =====================
def embed_texts(texts, batch_size=64):
    """Batched embeddings; returns L2-normalized float32 matrix."""
    if not texts:
        return np.zeros((0, 3072), dtype=np.float32)
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        all_embs.extend(embs)
    mat = np.vstack(all_embs)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return mat.astype(np.float32)

# ================== TEXTBOOK / RAG2 ======================
def extract_text_chunks_from_pdf(pdf_path, max_words=180):
    print(f"📖 Extracting text from {pdf_path} ...")
    reader = PdfReader(pdf_path)
    full_text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            full_text += t + " "
    words = full_text.split()
    chunks = [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    print(f"✅ Extracted {len(chunks)} chunks from textbook.")
    return chunks

def build_faiss_index_from_chunks(chunks):
    print("🔧 Building FAISS index for textbook (RAG2) ...")
    embs = embed_texts(chunks, batch_size=64)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    print(f"✅ RAG2 index built with {len(chunks)} chunks.")
    return index, chunks

def rag2_retrieve_for_keypoints(kp_vecs, rag2_index, rag2_chunks, top_k=2):
    """
    kp_vecs: (m, dim) normalized
    Returns: list[list[(chunk, score)]]
    """
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

# ============== DYNAMIC COLD CACHE (PERSISTENT) ==========
def load_cold_cache():
    sentences, index, meta = [], None, {"usage": {}, "version": 1}
    if os.path.exists(COLD_JSON_PATH) and os.path.exists(COLD_FAISS_PATH):
        try:
            with open(COLD_JSON_PATH, "r", encoding="utf-8") as f:
                sentences = json.load(f)
            index = faiss.read_index(COLD_FAISS_PATH)
        except Exception:
            sentences, index = [], None
    if os.path.exists(COLD_META_PATH):
        try:
            with open(COLD_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {"usage": {}, "version": 1}
    return sentences, index, meta

def save_cold_cache(sentences, index, meta):
    with open(COLD_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    if index is not None:
        faiss.write_index(index, COLD_FAISS_PATH)
    with open(COLD_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def build_cold_cache_from_corpus(corpus_texts, top_n=200):
    sent_freq = {}
    for doc in corpus_texts:
        for s in split_sentences(doc):
            key = clean_text_for_match(s)
            if len(key) < 15:
                continue
            sent_freq[key] = sent_freq.get(key, 0) + 1
    sorted_sents = sorted(sent_freq.items(), key=lambda kv: kv[1], reverse=True)
    top_sentences = [s for s, _ in sorted_sents[:top_n]]
    if not top_sentences:
        return [], None
    embs = embed_texts(top_sentences)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return top_sentences, idx

# ===================== KEYPOINTS =========================
def load_keypoints_cache():
    return safe_json_load(KEYPOINTS_CACHE_PATH, {})

def save_keypoints_cache(cache):
    safe_json_save(KEYPOINTS_CACHE_PATH, cache)

def extract_keypoints_once(faculty_answer: str, max_marks: float):
    """
    Extract atomic key points from faculty answer.
    Output weights are multiples of 0.5 and sum to max_marks.
    """
    prompt = f"""
Extract atomic, independently checkable key points from the FACULTY ANSWER.

Rules:
- Each key point must be short and specific.
- Include all important points (definition, mechanism, examples, etc. if present).
- Assign "weight" in multiples of 0.5 only.
- Total weight MUST sum exactly to {max_marks}.
- Mark "core": true for must-have points (definition/formula/central claim), else false.

FACULTY ANSWER:
{faculty_answer}

Return VALID JSON ONLY as a list:
[
  {{"id":"KP1","point":"...","weight":0.5,"core":true}},
  ...
]
"""
    resp = client.responses.create(model=LLM_MODEL, input=[{"role": "user", "content": prompt}])
    text = getattr(resp, "output_text", "")
    data = json.loads(text[text.find("["): text.rfind("]")+1])
    # Safety: coerce weights to 0.5 increments, re-balance if slightly off
    for kp in data:
        kp["weight"] = round_to_half(float(kp.get("weight", 0.5)))
        kp["core"] = bool(kp.get("core", False))
        kp["id"] = str(kp.get("id", "KP"))
        kp["point"] = str(kp.get("point", "")).strip()
    # normalize sum
    total = sum(kp["weight"] for kp in data if kp["point"])
    if total <= 0:
        # fallback: single point
        return [{"id": "KP1", "point": faculty_answer[:200], "weight": round_to_half(max_marks), "core": True}]
    # scale then quantize to preserve sum
    scale = float(max_marks) / total
    scaled = []
    for kp in data:
        w = round_to_half(kp["weight"] * scale)
        scaled.append({**kp, "weight": w})
    # adjust drift to exact sum using 0.5 steps
    drift = round_to_half(max_marks - sum(kp["weight"] for kp in scaled))
    i = 0
    while abs(drift) > 1e-9 and scaled:
        step = 0.5 if drift > 0 else -0.5
        scaled[i % len(scaled)]["weight"] = max(0.0, round_to_half(scaled[i % len(scaled)]["weight"] + step))
        drift = round_to_half(max_marks - sum(kp["weight"] for kp in scaled))
        i += 1
        if i > 5000:
            break
    # remove empties
    scaled = [kp for kp in scaled if kp["point"] and kp["weight"] > 0]
    # cap number of keypoints to keep prompts fast; merge smallest if too many
    if len(scaled) > 28:
        scaled = sorted(scaled, key=lambda k: (not k["core"], -k["weight"]))[:28]
        # re-adjust sum again
        drift = round_to_half(max_marks - sum(kp["weight"] for kp in scaled))
        i = 0
        while abs(drift) > 1e-9 and scaled:
            step = 0.5 if drift > 0 else -0.5
            scaled[i % len(scaled)]["weight"] = max(0.0, round_to_half(scaled[i % len(scaled)]["weight"] + step))
            drift = round_to_half(max_marks - sum(kp["weight"] for kp in scaled))
            i += 1
            if i > 2000:
                break
        scaled = [kp for kp in scaled if kp["weight"] > 0]
    return scaled

# ===================== BATCH GRADER ======================
def llm_evaluate_keypoints_batch(question: str, keypoints: list, student_answer: str, rag2_evidence: dict):
    """
    Evaluate selected keypoints in ONE LLM call.
    rag2_evidence: {kp_id: ["evidence snippet...", ...]} optional
    Returns list of {id, awarded, reason, status}
    """
    # Keep evidence short
    evid = {}
    for kpid, snippets in (rag2_evidence or {}).items():
        short_snips = []
        for s in snippets[:2]:
            s = " ".join(split_sentences(s)[:1]) or str(s)
            short_snips.append(s[:320])
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
Do NOT give credit for verbosity.
Award marks in increments of 0.5 ONLY.
If the student contradicts a keypoint, award 0 for that keypoint.

For each keypoint:
- awarded must be one of: 0, 0.5, or full weight (or another 0.5-multiple up to weight).
- status must be: FULL / PARTIAL / MISSING / INCORRECT
- reason must explain why marks were deducted (if any), referencing the student's content.

Input JSON:
{json.dumps(payload, ensure_ascii=False)}

Return VALID JSON ONLY as a list:
[
  {{"id":"KP1","awarded":0.5,"status":"PARTIAL","reason":"..."}},
  ...
]
"""
    resp = client.responses.create(model=LLM_MODEL, input=[{"role": "user", "content": prompt}])
    text = getattr(resp, "output_text", "")
    out = json.loads(text[text.find("["): text.rfind("]")+1])
    # sanitize
    cleaned = []
    for row in out:
        kpid = str(row.get("id", "")).strip()
        awarded = round_to_half(float(row.get("awarded", 0.0)))
        status = str(row.get("status", "MISSING")).strip().upper()
        reason = str(row.get("reason", "")).strip()
        cleaned.append({"id": kpid, "awarded": awarded, "status": status, "reason": reason})
    return cleaned
def grade_single_answer(
    question_text: str,
    faculty_answer: str,
    max_marks: float,
    student_answer: str,
    allow_rag2: bool = True
) -> dict:
    """
    Returns:
    {
      "score": float,
      "confidence": float,
      "used_rag2": bool,
      "novelty": float,
      "deductions": [
          {"lost": float, "reason": str, "key_point": str}
      ]
    }
    """

# ===================== MAIN PIPELINE ======================
def grade_all(faculty_csv, student_csv, textbook_pdf):
    # Load CSVs
    df_fac = normalize_cols(read_csv_safely(faculty_csv))
    df_stu = normalize_cols(read_csv_safely(student_csv))
    print("Faculty CSV columns:", df_fac.columns.tolist())
    print("Student CSV columns:", df_stu.columns.tolist())

    # Required faculty fields
    if "question_number_and_question" not in df_fac.columns or "answer" not in df_fac.columns:
        raise SystemExit("❌ faculty_key.csv must contain 'question_number_and_question' and 'answer' columns.")

    # Faculty rows
    fac_rows = [r for _, r in df_fac.iterrows()]
    fac_questions_raw = [str(r["question_number_and_question"]).strip() for r in fac_rows]
    fac_questions_clean = [clean_text_for_match(q) for q in fac_questions_raw]

    print("🧠 Precomputing faculty question embeddings ...")
    fac_q_embs = embed_texts(fac_questions_clean)         # (F, dim)

    # Precompute faculty answer embeddings (optional signal / confidence only)
    fac_answers = [str(r["answer"]) for r in fac_rows]
    print("🧠 Precomputing faculty answer embeddings ...")
    fac_a_embs = embed_texts(fac_answers)                 # (F, dim)

    # Load / build dynamic cold cache (kept, but used sparingly)
    cold_sentences, cold_index, cold_meta = load_cold_cache()
    if not cold_sentences or cold_index is None:
        print("💡 Building initial Dynamic Cold Cache from faculty+student texts ...")
        bootstrap_corpus = []
        bootstrap_corpus += fac_answers
        bootstrap_corpus += fac_questions_raw
        if "student_answer" in df_stu.columns:
            bootstrap_corpus += [str(x) for x in df_stu["student_answer"].fillna("").tolist()]
        cold_sentences, cold_index = build_cold_cache_from_corpus(bootstrap_corpus, top_n=200)
        cold_meta = {"usage": {}, "version": 1}
        save_cold_cache(cold_sentences, cold_index, cold_meta)
        print(f"✅ Dynamic Cold Cache created with {len(cold_sentences)} items.")
    else:
        print(f"✅ Loaded Dynamic Cold Cache with {len(cold_sentences)} items.")

    # Keypoints cache
    kp_cache = load_keypoints_cache()

    # Verdict cache
    verdict_cache = safe_json_load(VERDICT_CACHE_PATH, {})

    # Lazy RAG2
    rag2_built = False
    rag2_index, rag2_chunks = None, None

    results = []
    cold_hits_this_run = {}

    for i, row in df_stu.iterrows():
        sid = row["sno"] if "sno" in df_stu.columns else f"S{i+1}"
        student_q_text = str(row["question"]) if "question" in df_stu.columns else str(row.get("qno", ""))
        s_ans = str(row["student_answer"]) if "student_answer" in df_stu.columns else ""
        if not s_ans.strip():
            continue

        # --------- Faculty match (embed student question ONCE) ----------
        stu_q_clean = clean_text_for_match(student_q_text)
        stu_q_emb = embed_texts([stu_q_clean])[0]  # (dim,)

        # matrix dot: (F,dim) · (dim,) -> (F,)
        sims_q = fac_q_embs @ stu_q_emb
        best_idx = int(np.argmax(sims_q))
        best_sim = float(sims_q[best_idx])

        if best_sim < MATCH_THRESHOLD:
            print(f"⚠️ Skipping {sid}-{student_q_text[:40]}...: no faculty match (sim={best_sim:.2f}).")
            continue

        f_row = fac_rows[best_idx]
        question = str(f_row["question_number_and_question"])
        f_ans = str(f_row["answer"])
        marks = float(f_row["factual_score"]) if ("factual_score" in df_fac.columns and not pd.isna(f_row["factual_score"])) else 1.0
        marks = float(marks)

        # --------- Verdict cache (question + semantic answer hash) ----------
        a_hash = semantic_hash_answer(s_ans)
        cache_key = f"{best_idx}:{a_hash}"
        if cache_key in verdict_cache:
            cached = verdict_cache[cache_key]
            results.append(cached)
            print(f"🧾 {sid}-{student_q_text[:40]}... cache hit → {cached['result']['score']}/{marks}")
            continue

        # --------- Embed student answer ONCE ----------
        s_emb = embed_texts([s_ans])[0]

        # Optional: compute similarity to faculty answer (confidence signal only)
        sim_rag1 = float(np.dot(s_emb, fac_a_embs[best_idx]))

        # --------- Get keypoints (cached) ----------
        kp_key = str(best_idx)
        if kp_key not in kp_cache:
            print(f"🧩 Extracting keypoints for faculty idx={best_idx} (one-time) ...")
            kp_cache[kp_key] = {
                "max_marks": marks,
                "keypoints": extract_keypoints_once(f_ans, marks)
            }
            save_keypoints_cache(kp_cache)

        keypoints_all = kp_cache[kp_key]["keypoints"]
        # If marks changed since cache, keep using cached keypoints but clip later.

        # --------- Precompute keypoint embeddings (cache in-memory per faculty idx) ----------
        # To keep disk simple, embed on first use per faculty idx in-memory only.
        if "_kp_embs" not in kp_cache[kp_key]:
            kp_texts = [kp["point"] for kp in keypoints_all]
            kp_cache[kp_key]["_kp_embs"] = embed_texts(kp_texts).tolist()

        kp_embs = np.array(kp_cache[kp_key]["_kp_embs"], dtype=np.float32)  # (K,dim)

        # --------- Select relevant keypoints to evaluate (speed) ----------
        kp_sims = kp_embs @ s_emb  # (K,)
        core_ids = {kp["id"] for kp in keypoints_all if kp.get("core", False)}
        selected = []
        for kp, simv in zip(keypoints_all, kp_sims):
            if kp["id"] in core_ids or float(simv) >= KP_RELEVANCE_THRESHOLD:
                selected.append({**kp, "sim": float(simv)})

        # sort by (core first, sim desc, weight desc), cap
        selected.sort(key=lambda k: (not k.get("core", False), -k.get("sim", 0.0), -k.get("weight", 0.0)))
        selected = selected[:MAX_KP_PER_ANSWER]

        # ensure at least something
        if not selected:
            selected = sorted(keypoints_all, key=lambda k: (not k.get("core", False), -k.get("weight", 0.0)))[:min(8, len(keypoints_all))]

        # --------- Cheap Cold cache evidence only when useful ----------
        used_sources = {"RAG1": True, "Cold": False, "RAG2": False}
        cold_sim_max = 0.0
        cold_refs = []
        if cold_index is not None and len(cold_sentences) > 0 and sim_rag1 < (CONF_THRESHOLD + 0.05):
            D, I = cold_index.search(s_emb.reshape(1, -1), COLD_TOPK)
            for score, idxc in zip(D[0], I[0]):
                if idxc >= 0:
                    snip = cold_sentences[idxc]
                    cold_refs.append((" ".join(split_sentences(snip)[:1]) or snip, float(score)))
                    cold_sim_max = max(cold_sim_max, float(score))
                    used_sources["Cold"] = True
                    k = clean_text_for_match(snip)
                    cold_hits_this_run[k] = cold_hits_this_run.get(k, 0) + 1

        # --------- Weak length confidence only (never affects marks) ----------
        words = len(s_ans.split())
        expected = EXPECTED_MIN_WORDS_BY_MARKS.get(int(round(marks)), EXPECTED_MIN_WORDS_BY_MARKS.get(5, 30))
        length_conf = 1.0
        if words < max(3, int(0.4 * expected)):
            length_conf = 0.45
        elif words < expected:
            length_conf = 0.75

        # --------- Novelty trigger ----------
        nov = novelty_ratio(s_ans, f_ans)
        needs_rag2_due_to_novelty = (nov >= NOVELTY_TRIGGER)

        # --------- First-pass keypoint evaluation (ONE LLM call) ----------
        rag2_evidence = {}  # kp_id -> snippets
        kp_eval = llm_evaluate_keypoints_batch(
            question=question,
            keypoints=[{k: v for k, v in kp.items() if k in ("id", "point", "weight", "core")} for kp in selected],
            student_answer=s_ans,
            rag2_evidence=rag2_evidence
        )

        # build map: id -> {awarded, status, reason}
        kp_map = {r["id"]: r for r in kp_eval}

        # compute score from evaluated set; for non-selected kps: treat as missing (0) unless you want to evaluate all
        total_awarded = 0.0
        deductions = []
        disputed_kps = []
        for kp in selected:
            rid = kp["id"]
            weight = round_to_half(float(kp["weight"]))
            res = kp_map.get(rid, {"awarded": 0.0, "status": "MISSING", "reason": "Not addressed."})
            awarded = round_to_half(min(weight, max(0.0, float(res.get("awarded", 0.0)))))
            total_awarded += awarded
            lost = round_to_half(weight - awarded)
            if lost > 0:
                deductions.append({
                    "key_point": kp["point"],
                    "lost": lost,
                    "reason": res.get("reason", "").strip() or "Key point not fully addressed."
                })
                # disputed means low awarded OR ambiguous status
                if (res.get("status", "").upper() in ("PARTIAL", "MISSING")) or (awarded == 0 and kp.get("core", False)):
                    disputed_kps.append(kp)

        # Clip to max marks and half-step
        score_first = round_to_half(min(marks, max(0.0, total_awarded)))

        # --------- Confidence (aggregate; cheap + explainable) ----------
        # Blend: answer↔faculty sim, cold sim, length_conf, coverage ratio
        coverage_ratio = 0.0
        denom = sum(round_to_half(kp["weight"]) for kp in selected) or 1.0
        coverage_ratio = float(score_first / denom)
        conf = float(0.45 * max(sim_rag1, 0.0) + 0.20 * max(cold_sim_max, 0.0) + 0.20 * coverage_ratio + 0.15 * length_conf)
        conf = max(0.0, min(1.0, conf))

        # --------- Conditional RAG2 (only if needed) ----------
        rag2_top = 0.0
        if (conf < CONF_THRESHOLD and disputed_kps) or needs_rag2_due_to_novelty:
            if not rag2_built:
                print("⚡ RAG2 not built yet — extracting & indexing textbook ...")
                chunks = extract_text_chunks_from_pdf(textbook_pdf)
                rag2_index, rag2_chunks = build_faiss_index_from_chunks(chunks)
                rag2_built = True

            used_sources["RAG2"] = True

            # retrieve evidence per disputed keypoint (fast: use kp embeddings if present)
            disputed_ids = [kp["id"] for kp in disputed_kps]
            disputed_points = [kp["point"] for kp in disputed_kps]
            kp_vecs = embed_texts(disputed_points)
            rag2_hits = rag2_retrieve_for_keypoints(kp_vecs, rag2_index, rag2_chunks, top_k=RAG2_TOPK_PER_KP)

            rag2_evidence = {}
            for kpid, hits in zip(disputed_ids, rag2_hits):
                rag2_evidence[kpid] = [h[0] for h in hits]
                if hits:
                    rag2_top = max(rag2_top, max(h[1] for h in hits))

            # re-evaluate ONLY disputed keypoints with evidence (one more LLM call)
            kp_eval2 = llm_evaluate_keypoints_batch(
                question=question,
                keypoints=[{k: v for k, v in kp.items() if k in ("id", "point", "weight", "core")} for kp in disputed_kps],
                student_answer=s_ans,
                rag2_evidence=rag2_evidence
            )
            kp_map2 = {r["id"]: r for r in kp_eval2}

            # update score and deductions only for disputed keypoints
            total_awarded2 = score_first
            deductions2 = [d for d in deductions]  # copy

            # quick helper to remove prior deductions for those kps
            disputed_texts = {kp["point"] for kp in disputed_kps}
            deductions2 = [d for d in deductions2 if d["key_point"] not in disputed_texts]

            # subtract old awarded for disputed then add new
            # (We recompute from selected for correctness)
            total_awarded2 = 0.0
            deductions2 = []
            for kp in selected:
                rid = kp["id"]
                weight = round_to_half(float(kp["weight"]))
                base = kp_map.get(rid, {"awarded": 0.0, "status": "MISSING", "reason": ""})
                upd = kp_map2.get(rid, None)
                res = upd if upd is not None else base
                awarded = round_to_half(min(weight, max(0.0, float(res.get("awarded", 0.0)))))
                total_awarded2 += awarded
                lost = round_to_half(weight - awarded)
                if lost > 0:
                    deductions2.append({
                        "key_point": kp["point"],
                        "lost": lost,
                        "reason": res.get("reason", "").strip() or "Key point not fully addressed."
                    })

            score_first = round_to_half(min(marks, max(0.0, total_awarded2)))
            deductions = deductions2

            # update confidence with rag2
            conf = float(0.35 * max(sim_rag1, 0.0) + 0.20 * max(cold_sim_max, 0.0) + 0.20 * (score_first / denom) + 0.10 * length_conf + 0.15 * max(rag2_top, 0.0))
            conf = max(0.0, min(1.0, conf))

        # --------- Compose output record ----------
        similarities = {
            "RAG1": round(sim_rag1, 3),
            "Cold": round(cold_sim_max, 3),
            "RAG2_top": round(rag2_top, 3),
            "LengthConf": round(length_conf, 3),
            "OverallConf": round(conf, 3)
        }

        grade = {
            "score": float(score_first),
            "deductions": deductions,
            "confidence": similarities,
            "notes": {
                "novelty_ratio": round(float(nov), 3),
                "evaluated_keypoints": [kp["id"] for kp in selected],
                "cold_evidence": [c[0] for c in cold_refs[:2]]
            }
        }

        record = {
            "student_id": sid,
            "question_no": student_q_text,
            "faculty_match_idx": best_idx,
            "faculty_match_sim": round(best_sim, 3),
            "used_sources": used_sources,
            "max_marks": marks,
            "result": grade
        }

        results.append(record)

        # save to verdict cache for repeats
        verdict_cache[cache_key] = record
        safe_json_save(VERDICT_CACHE_PATH, verdict_cache)

        print(f"📝 {sid}-{student_q_text[:40]}... {used_sources} → {grade['score']}/{marks} "
              f"[conf={similarities['OverallConf']}, R1={similarities['RAG1']}, C={similarities['Cold']}, R2={similarities['RAG2_top']}, nov={round(nov,3)}]")

    # --------- Save outputs ----------
    out_csv = "graded_results.csv"
    out_json = "graded_results.json"

    df_out = pd.DataFrame([{
        "Student ID": r["student_id"],
        "Question": r["question_no"],
        "FacultyMatchSim": r["faculty_match_sim"],
        "Used Sources": str(r["used_sources"]),
        "Score": r["result"]["score"],
        "Max Marks": r["max_marks"],
        "OverallConf": r["result"]["confidence"]["OverallConf"],
        "RAG1_Sim": r["result"]["confidence"]["RAG1"],
        "Cold_Sim": r["result"]["confidence"]["Cold"],
        "RAG2_Top": r["result"]["confidence"]["RAG2_top"],
        "LengthConf": r["result"]["confidence"]["LengthConf"],
        "Novelty": r["result"]["notes"]["novelty_ratio"],
        "Deductions": "; ".join([f"-{d['lost']}: {d['reason']} (KP: {d['key_point'][:60]})" for d in r["result"]["deductions"][:8]]),
        "ColdEvidence": "; ".join(r["result"]["notes"].get("cold_evidence", []))
    } for r in results])

    df_out.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {out_csv} and {out_json}")

    # --------- Update cold cache usage meta (keep your original behavior) ----------
    meta_usage = cold_meta.get("usage", {})
    for k, v in cold_hits_this_run.items():
        meta_usage[k] = meta_usage.get(k, 0) + int(v)
    cold_meta["usage"] = meta_usage
    save_cold_cache(cold_sentences, cold_index, cold_meta)

# ======================== ENTRY ==========================
if __name__ == "__main__":
    grade_all("faculty_key.csv", "student_answers.csv", "textbook.pdf")
