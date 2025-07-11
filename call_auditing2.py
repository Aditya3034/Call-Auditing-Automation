import os
import whisper
import requests
from datetime import datetime, timedelta

# --- Configuration ---
AUDIO_FILE = "call_recording.wav"
WHISPER_MODEL = "large-v3"
LLM_PROVIDER = "sarvam"  # Fallbacks: deepseek, ollama
SARVAM_API_KEY = "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"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:70b"

EVALUATION_METRICS = [
    "Call Opening", "Empathy", "Logical Effective Probing",
    "Listening Skills Comprehension", "Hold", "Dead Air",
    "Tone Mannerism Positive Approach", "Objection Escalation Handling Ownership",
    "Further Assistance Call Closing", "Raise Complaint", "Resolution"
]

# --- Whisper Transcription ---
def transcribe_audio(file_path):
    print("🎙️ Transcribing with Whisper Large V3...")
    model = whisper.load_model(WHISPER_MODEL)
    return model.transcribe(file_path)

# --- Structure Transcript ---
def structure_transcript(segments):
    structured = []
    for seg in segments:
        text = seg["text"].strip()
        timestamp = str(timedelta(seconds=int(seg["start"])))
        speaker = "Agent" if "sir" in text.lower() or "thank you" in text.lower() else "Client"
        structured.append(f"[{timestamp}] {speaker}: {text}")
    return "\n".join(structured)

# --- Silence Evaluation ---
def detect_silence_events(segments):
    result = {"Hold": False, "Dead Air": False}
    for i in range(1, len(segments)):
        silence = segments[i]["start"] - segments[i - 1]["end"]
        if silence >= 120:
            result["Hold"] = True
        elif silence >= 19:
            result["Dead Air"] = True
    return result

# --- Unified LLM Call with Fallback ---
def call_llm(prompt, provider):
    try:
        if provider == "sarvam":
            headers = {"Authorization": f"Bearer {SARVAM_API_KEY}"}
            payload = {"input": prompt, "model": "sarvamai/sarvam-m:free"}
            r = requests.post("https://api.sarvam.ai/v1/chat/completions", json=payload, headers=headers)
        elif provider == "ollama":
            payload = {"model": OLLAMA_MODEL, "prompt": prompt}
            r = requests.post(OLLAMA_URL, json=payload)
        elif provider == "deepseek":
            raise NotImplementedError("Deepseek integration pending")
        else:
            raise ValueError("Unsupported provider")

        if r.status_code == 200:
            return r.json()
        else:
            raise Exception(f"{provider} error: {r.status_code}")
    except Exception as e:
        print(f"⚠️ LLM failed ({provider}):", str(e))
        fallback = "deepseek" if provider == "sarvam" else "ollama"
        print(f"🔁 Trying fallback: {fallback}")
        return call_llm(prompt, fallback)

# --- Generate Summary + Title ---
def summarize_call(transcript_text):
    prompt = f"""You are a call audit assistant.
Transcript:
{transcript_text}

1. Provide a short title summarizing the call.
2. Write a detailed summary of what happened.
Respond in structured JSON: {{ "title": "...", "summary": "..." }}
"""
    response = call_llm(prompt, LLM_PROVIDER)
    try:
        return eval(response["response"]) if isinstance(response["response"], str) else response
    except Exception:
        return {"title": "N/A", "summary": "Failed to parse LLM output."}

# --- Agent Evaluation ---
def evaluate_call(transcript_text):
    metrics_list = ", ".join(EVALUATION_METRICS)
    prompt = f"""You're evaluating a support call using the following metrics:
{metrics_list}

Call Transcript:
{transcript_text}

Return a JSON object rating each metric from 1–5 with brief feedback.
"""
    response = call_llm(prompt, LLM_PROVIDER)
    try:
        return eval(response["response"]) if isinstance(response["response"], str) else response
    except Exception:
        return {metric: "N/A" for metric in EVALUATION_METRICS}

# --- MAIN ---
if __name__ == "__main__":
    print("🚀 Starting AI Call Analysis for Payone")
    start_time = datetime.now()

    # Step 1: Transcribe
    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(f"{AUDIO_FILE} not found.")
    result = transcribe_audio(AUDIO_FILE)
    segments = result.get("segments", [])
    transcript_text = structure_transcript(segments)

    # Step 2: Silence Events
    silence_eval = detect_silence_events(segments)

    # Step 3: Summary & Title
    summary_block = summarize_call(transcript_text)

    # Step 4: Agent Evaluation
    evaluation_block = evaluate_call(transcript_text)

    # Combine Silence + LLM Evaluation
    final_evaluation = {**silence_eval, **evaluation_block}

    # Display Results
    print("\n📝 Transcript:\n", transcript_text)
    print("\n📌 Title:", summary_block.get("title"))
    print("\n📖 Summary:", summary_block.get("summary"))
    print("\n📊 Evaluation:")
    for k, v in final_evaluation.items():
        print(f"- {k}: {v}")

    print("\n✅ Done in:", datetime.now() - start_time)
