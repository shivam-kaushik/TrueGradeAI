# 🧠 Adaptive Grader — Dual-RAG AI Evaluation System
pip install pandas numpy openai faiss-cpu PyPDF2 python-dotenv tqdm matplotlib scikit-learn rich openpxyl
An **AI-powered automated answer grading pipeline** based on the research paper  
*“Automated Educational Assessment using Dual-RAG Grading (2025)”*.

This system intelligently evaluates student answers using:
- **RAG1 (Faculty Key):** Semantic comparison with the official answer key  
- **Dynamic Cold Cache:** Learns and stores frequently occurring factual sentences  
- **RAG2 (Textbook):** Deep retrieval from textbooks using FAISS + embeddings  
- **LLM Reasoning:** Uses OpenAI GPT models to assign marks and generate rationale  

The system adapts based on:
✅ Answer length  
✅ Similarity thresholds  
✅ Content sufficiency  
✅ Factual novelty  
✅ Missing concepts

---

## 📁 Project Structure

```
adaptive_grader/
│
├── adaptive_grader_real.py       # Main Python pipeline
├── faculty_key.csv               # Faculty questions + ideal answers + factual scores
├── student_answers.csv           # Student answers to be graded
├── textbook.pdf                  # Full textbook for RAG2 retrieval
│
├── cold_cache.json               # Dynamic Cold Cache (auto-generated)
├── cold_cache.faiss              # Vector index for cache
├── cold_meta.json                # Cache metadata
│
├── graded_results.csv            # Final numerical results
├── graded_results.json           # Full structured JSON output
│
├── .env                          # Stores your OpenAI API key
└── README.md                     # This documentation
```

---

## ⚙️ Installation Guide (Windows + VS Code)

### 1️⃣ Activate Virtual Environment
Open **VS Code Terminal**:
```bash
cd C:\Users\kaush\adaptive_grader
python -m venv .venv
.venv\Scripts\activate
```

### 2️⃣ Upgrade Essentials
```bash
pip install --upgrade pip setuptools wheel
```

### 3️⃣ Install Required Libraries
```bash
pip install pandas numpy openai faiss-cpu PyPDF2 python-dotenv tqdm matplotlib scikit-learn rich openpxyl
```

### 4️⃣ (Optional Enhancements)
```bash
pip install matplotlib scikit-learn rich
```

### 5️⃣ Verify Installation
```bash
python -c "import faiss, PyPDF2, pandas, numpy, openai; print('✅ All core libs OK')"
```

If you see `✅ All core libs OK`, you’re ready to go.

---

## 🔑 Environment Setup

Create a file named **`.env`** in your root directory with:

```
OPENAI_API_KEY=sk-your-secret-api-key
```

> ⚠️ Never share or push this key to GitHub.

---

## 🧩 Required Data Files

| File | Description |
|------|--------------|
| **faculty_key.csv** | Faculty answers + factual scores (columns: `s_no`, `question_number_and_question`, `answer`, `factual_score`) |
| **student_answers.csv** | Student answers (columns: `sno`, `qno`, `question`, `student_answer`) |
| **textbook.pdf** | The main textbook used to build RAG2 FAISS retrieval index |

---

## ▶️ Run the Grading Pipeline

```bash
python adaptive_grader_real.py
```

### 🖥️ Example Output:
```
Faculty CSV columns: [...]
Student CSV columns: [...]
💡 Building initial Dynamic Cold Cache from faculty+student texts ...
✅ Dynamic Cold Cache created with 200 items.
⚡ RAG2 not built yet — extracting & indexing textbook ...
📖 Extracting text from textbook.pdf ...
✅ RAG2 index built with 1982 chunks.
📝 12-Feudalism?... (M) {'RAG1': True, 'Cold': True, 'RAG2': True} → 2.5/3.0 [conf: R1=0.87, C=0.54, R2=0.48]
✅ Results saved to graded_results.csv and graded_results.json
🧊 Dynamic Cold Cache updated → 261 entries.
```

---

## 🧠 Output Breakdown

Each answer output includes:

| Field | Description |
|--------|--------------|
| `score` | Marks awarded (rounded to nearest 0.5) |
| `max_marks` | Maximum possible marks |
| `category` | VS / S / M / L (based on word length) |
| `used` | Which layers were triggered (RAG1, Cold, RAG2) |
| `confidence` | Confidence from each retrieval layer |
| `rationale` | Correct points, omissions, improvements |
| `deduction_reason` | Why marks were deducted |
| `missing_points` | List of omitted concepts |
| `added_irrelevant` | Unnecessary or unrelated content |

---

## 🧊 Dynamic Cold Cache

The **Cold Cache** layer is an *auto-learning memory* that:
- Extracts the most frequent factual sentences from faculty, student, and textbook content.  
- Stores and reuses them for faster grading in future runs.  
- Updates after each full execution (default = top 200 sentences).  

Cache files:
- `cold_cache.json` → list of top recurring factual sentences  
- `cold_cache.faiss` → vector embeddings for semantic search  
- `cold_meta.json` → usage statistics

---

## 🧪 Example Evaluation (JSON)

```json
{
  "student_id": "S12",
  "question": "Explain the decline of Mughal authority in the eighteenth century.",
  "category": "M",
  "score": 4.0,
  "max_marks": 5,
  "rationale": {
    "correct": ["Mentions regional powers like Marathas and Nawabs."],
    "omissions": ["Did not mention British expansion or weak successors."],
    "improvements": ["Add brief reference to economic decentralization."]
  },
  "deduction_reason": "Missed key causes such as British interference and weak successors.",
  "missing_points": ["British interference", "weak successors"],
  "added_irrelevant": [],
  "confidence": {"RAG1": 0.872, "Cold": 0.0, "RAG2": 0.627}
}
```

---

## 🩵 Troubleshooting

| Issue | Fix |
|--------|-----|
| `faiss` not found | Install CPU version → `pip install faiss-cpu` |
| UTF-8 decode error | CSV auto-fallbacks to Latin-1 encoding |
| API key not found | Ensure `.env` file exists with `OPENAI_API_KEY=` |
| PDF too large | Split textbook into multiple PDFs |
| Missing omissions | Ensure OpenAI model is GPT-4o-mini or GPT-4o |

---

## 📈 Future Roadmap
- Weighted rubric grading by concept importance  
- Real-time web dashboard  
- Auto-rubric generation from faculty CSV  
- Heatmap visualization of confidence vs score

---

## 🧾 License
MIT License © 2025  
Developed by **Shivam Kaushik** — Centennial College  
For educational and research use only.

---

## 💬 Contact
**Developer:** Shivam Kaushik  
📧 Email: shivamkaushik.ai@gmail.com  
🌐 LinkedIn: [linkedin.com/in/shivamkaushik](https://linkedin.com/in/shivamkaushik)
