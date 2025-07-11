import os
import whisper
import requests
import json
import ffmpeg
import multiprocessing
from datetime import datetime, timedelta

# --- Configuration ---
OLLAMA_URL = "http://192.168.1.55:11434/api/generate"
OLLAMA_MODEL = "llama3:70b"
EVALUATION_METRICS = [
    "Call Opening", "Empathy", "Logical Effective Probing",
    "Listening Skills Comprehension", "Hold", "Dead Air",
    "Tone Mannerism Positive Approach", "Objection Escalation Handling Ownership",
    "Further Assistance Call Closing", "Raise Complaint", "Resolution"
]

def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))

def preprocess_audio(audio_path):
    temp_path = audio_path.replace('.wav', '_processed.wav')
    try:
        print("🔄 Preprocessing audio with FFmpeg...")
        ffmpeg.input(audio_path).output(
            temp_path, acodec='pcm_s16le', ar=16000, ac=1, af='loudnorm'
        ).overwrite_output().run(quiet=True)
        print("✅ Audio preprocessing completed")
        return temp_path
    except Exception as e:
        print(f"⚠️ FFmpeg preprocessing failed: {e}")
        return audio_path

def run_transcription(model, processed_audio, return_dict):
    try:
        result = model.transcribe(
            processed_audio,
            fp16=False,
            language="en",
            word_timestamps=True,
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0
        )
        return_dict["result"] = result
    except Exception as e:
        return_dict["error"] = str(e)

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found at {audio_path}")
        return None, None

    print(f"\n🎙️ Found audio file: {audio_path}")
    file_size = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"📊 Audio file size: {file_size:.2f} MB")

    processed_audio = preprocess_audio(audio_path)
    print("🔄 Loading Whisper model...")

    try:
        model = whisper.load_model("large-v3")
        print("✅ Whisper large-v3 model loaded successfully")
    except Exception:
        model = whisper.load_model("large-v2")
        print("✅ Whisper large-v2 model loaded successfully")

    print("🔄 Starting transcription process (timeout: 5 mins)...")
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=run_transcription, args=(model, processed_audio, return_dict))
    p.start()
    p.join(timeout=3000)

    if p.is_alive():
        print("❌ Transcription timed out. Trying fallback model 'medium'...")
        p.terminate()
        model = whisper.load_model("medium")
        try:
            result = model.transcribe(processed_audio, fp16=False, language="en")
            transcript = result["text"].strip()
            segments = result.get("segments", [])
        except Exception as e:
            print(f"❌ Fallback medium model also failed: {e}")
            return None, None
    else:
        if "error" in return_dict:
            print(f"❌ Error during transcription: {return_dict['error']}")
            return None, None
        result = return_dict["result"]
        transcript = result["text"].strip()
        segments = result.get("segments", [])

    print("✅ Transcription completed successfully")
    print(f"📄 Generated transcript length: {len(transcript)} characters")
    print(f"Transcript: {transcript}")
    timestamped_transcript = ""
    for segment in segments:
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        timestamped_transcript += f"[{start}-{end}] {segment['text'].strip()}\n"

    if processed_audio != audio_path and os.path.exists(processed_audio):
        os.remove(processed_audio)

    return transcript, timestamped_transcript

def analyze_call_with_ollama(transcript_text, timestamped_transcript, url, model):
    print("\n🔄 Analyzing call with Ollama...")
    
    # Truncate transcript to prevent overloading Ollama
    if len(transcript_text) > 4000:
        transcript_text = transcript_text[:4000] + "\n...[TRUNCATED]"

    if len(timestamped_transcript) > 6000:
        timestamped_transcript = timestamped_transcript[:6000] + "\n...[TRUNCATED]"

    prompt = f"""
Analyze this customer service call transcript and provide a comprehensive analysis.

Raw Transcript:
{transcript_text}

Timestamped Segments:
{timestamped_transcript}

Provide your analysis in the following JSON format:
{{
  "title": "Create a descriptive title for this call (8-15 words max)",
  "detailed_summary": "...",
  "conversation": "..."
}}
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"num_thread": 8, "num_ctx": 8192, "temperature": 0.1}
    }

    try:
        print("🔄 Sending data to Ollama for analysis...")
        response = requests.post(url, json=payload, timeout=3000)
        response.raise_for_status()
        response_json_str = response.json().get("response", "{}").strip()
        analysis_data = json.loads(response_json_str)
        print("✅ Analysis completed")
        return analysis_data
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        try:
            print("💥 Ollama response:", response.text[:500])
        except:
            pass
        return None

def evaluate_call_with_ollama(transcript_text, metrics, url, model):
    print(f"\n🔄 Evaluating call quality against {len(metrics)} metrics...")

    metrics_str = "\n".join(f'- "{metric}"' for metric in metrics)
    prompt = f"""
Evaluate this call transcript against the specified metrics.

Transcript:
{transcript_text[:4000]}

Metrics:
{metrics_str}

Provide evaluation in JSON format:
{{
  "evaluation": [
    {{
      "metric": "metric_name",
      "rating": "Excellent/Good/Fair/Poor/N/A",
      "score": "1-10",
      "justification": "detailed_explanation"
    }}
  ]
}}
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"num_thread": 8, "num_ctx": 4096}
    }

    try:
        response = requests.post(url, json=payload, timeout=3000)
        response.raise_for_status()
        response_json_str = response.json().get("response", "{}").strip()
        evaluation_data = json.loads(response_json_str)
        print("✅ Call evaluation completed")
        return evaluation_data
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    print("🚀 Starting Call Audio Analysis Pipeline")
    print("=" * 60)
    print(f"🕒 Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    audio_filename = "call_recording.wav"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.join(script_dir, audio_filename)

    print("\n📝 STEP 1: AUDIO TRANSCRIPTION")
    print("-" * 40)
    transcript, timestamped_transcript = transcribe_audio(audio_file_path)

    if not transcript:
        print("\n❌ Transcription failed. Aborting analysis.")
        exit(1)

    print("\n🔧 STEP 2: CALL ANALYSIS")
    print("-" * 40)
    analysis = analyze_call_with_ollama(transcript, timestamped_transcript, OLLAMA_URL, OLLAMA_MODEL)
    if not analysis:
        print("\n❌ Analysis failed. Aborting.")
        exit(1)

    print("\n📊 STEP 3: CALL QUALITY EVALUATION")
    print("-" * 40)
    evaluation = evaluate_call_with_ollama(transcript, EVALUATION_METRICS, OLLAMA_URL, OLLAMA_MODEL)

    print("\n" + "=" * 80)
    print("📋 CALL ANALYSIS RESULTS")
    print("=" * 80)

    print(f"\n📞 CALL TITLE:\n{analysis.get('title', 'N/A')}")
    print(f"\n📄 DETAILED SUMMARY:\n{analysis.get('detailed_summary', 'N/A')}")
    print(f"\n💬 CONVERSATION:\n{analysis.get('conversation', 'N/A')}")

    if evaluation and "evaluation" in evaluation:
        print(f"\n📊 QUALITY EVALUATION:")
        for item in evaluation["evaluation"]:
            print(f"\n{item.get('metric', 'N/A')}:")
            print(f"  Rating: {item.get('rating', 'N/A')} (Score: {item.get('score', 'N/A')}/10)")
            print(f"  Justification: {item.get('justification', 'N/A')}")

    print(f"\n✅ ANALYSIS COMPLETE")
    print("=" * 80)