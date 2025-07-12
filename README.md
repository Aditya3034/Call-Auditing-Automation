# 🎧 Payone Call Analysis API — Whisper + Gemini Integration

This FastAPI application provides end-to-end call analysis by transcribing audio files, structuring transcripts, generating summaries, and evaluating interactions using OpenAI's Whisper and Google's Gemini LLMs.

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/payone-call-analysis-api.git
cd payone-call-analysis-api
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

## 🧪 API Endpoints

Access Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Route                            | Description                                       |
|--------|----------------------------------|---------------------------------------------------|
| POST   | `/`                              | Health check — returns `"Application is started"` |
| POST   | `/api/analyze-audio`            | Upload audio for transcription + full analysis    |
| POST   | `/api/structure-transcript`     | Convert raw transcript into structured format     |
| POST   | `/api/generate-summary`         | Generate title and summary from transcript        |
| POST   | `/api/evaluate-metrics`         | Score transcript using behavioral metrics         |
| GET    | `/health`                       | Returns model initialization status and timestamp |

---

## 📄 Request Payload Samples

### `/api/structure-transcript`

```json
{
  "transcript": "Customer: I'm trying to cancel my flight due to an emergency..."
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

## ⚙️ Key Features

- 🧠 Dynamic LLM integration: supports Gemini, OpenAI, Ollama, Sarvam, DeepSeek
- 🎙️ Whisper Large V3 for transcription
- 📄 Clean call structuring, emotional tagging, and segmentation
- 📝 Descriptive summaries and resolution analysis
- 📊 Agent performance scoring with clear metrics
- 🩺 Health monitoring via `/health` endpoint

---

## ✅ Health Endpoint

```http
GET /health
```

Sample output:

```json
{
  "status": "healthy",
  "whisper_loaded": true,
  "gemini_initialized": true,
  "timestamp": "2025-07-12T06:10:32.821Z"
}
```

---

## 📦 Output Overview

`/api/analyze-audio` returns:

- Structured transcript with timestamps
- Summary and title
- Evaluation metrics as JSON
- Transcript saved as `.txt` file locally

---

## 💡 Notes

- Whisper must be loaded before handling requests
- Gemini API key should be securely configured via environment variables or `config.py`
- Supports multiple LLM backends for flexibility

---

## 🤝 Contributions

Feel free to open issues or submit pull requests to improve accuracy, support new models, or enhance output formatting.

---
