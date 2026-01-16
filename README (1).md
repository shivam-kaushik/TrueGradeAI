# 🎓 TrueGradeAI — Role-Based AI Grading Platform

A full-stack local web application for **automated, explainable subjective grading**.  
Transforming the original CLI tool into a modern **React + FastAPI** platform with specific roles for **Teachers** and **Students**.

---

## 🚀 Quick Start Guide

You need two terminal windows to run the full application.

### 1️⃣ Prerequisites
*   **Python 3.10+** (for Backend)
*   **Node.js 18+** (for Frontend)
*   **OpenAI API Key** (in `.env`)

### 2️⃣ Start the Backend API
In your first terminal:
```powershell
# 1. Navigate to project root
cd C:\Users\shiva\Downloads\TrueGradeAI

# 2. Install Python dependencies (only first time)
pip install -r backend/requirements.txt

# 3. Process the backend
uvicorn backend.main:app --reload --port 8001
```
> ✅ You will see: `Uvicorn running on http://127.0.0.1:8001`

### 3️⃣ Start the Frontend UI
In your **second** terminal:
```powershell
# 1. Navigate to frontend folder
cd C:\Users\shiva\Downloads\TrueGradeAI\frontend

# 2. Install Node dependencies (only first time)
npm install

# 3. Start the Vite dev server
npm run dev
```
> ✅ You will see: `Local: http://localhost:5173/`

---

## 🖥️ How to Use

Open your browser to **[http://localhost:5173](http://localhost:5173)**.

### 👩‍🏫 Teacher Role
1.  Click **Teacher Dashboard**.
2.  **Add Questions**: Input Question, Max Marks, and the official Answer Key.
3.  **Manage Bank**: View all active questions in the system.

### 👨‍🎓 Student Role
1.  Click **Student Portal**.
2.  **Select a Question** from the list.
3.  **Write Answer**: Type your response in the text area.
4.  **Submit**: Click "Submit Answer" to get instant AI grading.
5.  **View Results**: See your Score, Key Point Breakdown, and specific Deductions.

---

## 🛠️ Project Structure

```
TrueGradeAI/
├── backend/
│   ├── main.py                 # FastAPI Entry Point (Routes)
│   ├── requirements.txt        # Python Dependencies
│   └── truegrade_core/         # Core Grading Logic (engine.py)
│
├── frontend/
│   ├── src/
│   │   ├── components/         # React Components (Dashboards, Results)
│   │   ├── api.ts              # API Client
│   │   └── App.tsx             # Main Router
│   ├── package.json            # Node Dependencies
│   └── text-gray-500postcss.config.js    # Tailwind Config
│
├── uploads/                    # Local storage for docs
└── README (1).md               # This guide
```

---

## 🔑 Environment Variables
Ensure you have a `.env` file in the root `TrueGradeAI/` folder:
```
OPENAI_API_KEY=sk-your-key-here
```
