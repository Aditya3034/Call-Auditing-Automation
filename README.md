# 🎧 Call Auditing Automation — Whisper + Gemini Integration

This FastAPI application enables automated call auditing by transcribing audio recordings, structuring transcripts, generating call summaries, and evaluating agent interactions using OpenAI's Whisper and Google Gemini LLMs.

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Call-Auditing-Automation.git
cd Call-Auditing-Automation
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
uvicorn main:app --reload
```

---

## 📍 API Endpoints

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Endpoint                            | Description                                             |
|--------|-------------------------------------|---------------------------------------------------------|
| POST   | `/`                                 | Health check — returns `"Application is started"`       |
| POST   | `/api/analyze-audio`               | Full workflow: transcribe, structure, summarize, score  |
| POST   | `/api/structure-transcript`        | Format raw transcript into structured format            |
| POST   | `/api/generate-summary`            | Create a title and summary from transcript              |
| POST   | `/api/evaluate-metrics`            | Score structured transcript using interaction metrics   |
| GET    | `/health`                          | Returns model status, Gemini initialization, timestamp  |

---

## 🧪 Sample Requests

### `/api/structure-transcript`

```json
{
  "transcript": "Customer: I'm trying to cancel my ticket because of an emergency..."
}
```

### `/api/generate-summary`

```json
{
  "transcript": "Structured transcript goes here..."
}
```

### `/api/evaluate-metrics`

```json
{
  "transcript": "Structured transcript goes here..."
}
```

---

## 🛠 Technologies Used

- 🎙️ Whisper Large V3 — audio transcription
- 🧠 Gemini 1.5 Flash — transcript analysis, summarization, scoring
- 🚀 FastAPI — backend framework for endpoints
- 🔁 Modular LLM dispatcher — supports Gemini, OpenAI, Ollama, Sarvam, DeepSeek
- 📦 Uvicorn, Pydantic — server runtime and schema validation

---

## ✅ Health Check Output

```http
GET /health
```

Returns:

```json
{
  "status": "healthy",
  "whisper_loaded": true,
  "gemini_initialized": true,
  "timestamp": "2025-07-12T06:10:32.821Z"
}
```

---

## 📂 Response Example (`/api/analyze-audio`)

```json
{
  "structured_transcript": "[00:00:00] Agent (calm): Hello, thank you for calling...",
  "title": "Flight Cancellation Due to Emergency",
  "summary": "Customer contacted support regarding...",
  "evaluation_metrics": {
    "Empathy": { "score": 9, "comment": "Agent demonstrated sincere concern..." },
    ...
  },
  "success": true
}
```

---

## 🔐 Notes

- Set your Gemini API key securely via environment variables or config file
- Whisper model must be loaded before handling requests
- Audio is processed securely via temporary file handling
- Transcript is saved locally with timestamped filename

---

## 📬 Contributions

Pull requests and feature suggestions welcome!  
Help improve evaluation logic, add more LLMs, or enhance output display.

---
