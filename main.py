# import os
# import json
# import asyncio
# import whisper
# import requests
# import tempfile
# from typing import Optional, Dict, Any
# from pathlib import Path

# from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import uvicorn

# # Configuration - Update with your LAN IP
# LAN_IP = "192.168.1.55"  # Replace with your actual LAN IP
# OLLAMA_URL = f"http://{LAN_IP}:11434/api/generate"
# OLLAMA_MODEL = "llama3:70b"
# WHISPER_MODEL = "large-v3"
# UPLOAD_DIR = "./uploads"

# # Create upload directory if it doesn't exist
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Initialize FastAPI app
# app = FastAPI(
#     title="Whisper-Ollama API",
#     description="Audio transcription and summarization API using Whisper and Ollama",
#     version="1.0.0"
# )

# # Global variables for models
# whisper_model = None
# ollama_available = False

# # Pydantic models for API requests/responses
# class TranscriptionResponse(BaseModel):
#     transcript: str
#     file_path: str
#     success: bool
#     message: str

# class PromptRequest(BaseModel):
#     prompt: str
#     context: Optional[str] = None

# class PromptResponse(BaseModel):
#     response: str
#     success: bool
#     message: str

# class SummaryRequest(BaseModel):
#     transcript: str
#     summary_type: str = "structured"  # "structured" or "general"

# class SummaryResponse(BaseModel):
#     summary: str
#     customer_insights: Optional[Dict[str, Any]] = None
#     agent_insights: Optional[Dict[str, Any]] = None
#     success: bool
#     message: str

# class HealthResponse(BaseModel):
#     whisper_loaded: bool
#     ollama_available: bool
#     status: str

# # Utility functions
# def load_whisper_model():
#     """Load Whisper model on startup"""
#     global whisper_model
#     try:
#         print(f"Loading Whisper model: {WHISPER_MODEL}")
#         whisper_model = whisper.load_model(WHISPER_MODEL)
#         print("✅ Whisper model loaded successfully")
#         return True
#     except Exception as e:
#         print(f"❌ Failed to load Whisper model: {e}")
#         return False

# def check_ollama_availability():
#     """Check if Ollama server is available"""
#     global ollama_available
#     try:
#         tags_url = f"http://{LAN_IP}:11434/api/tags"
#         response = requests.get(tags_url, timeout=5)
#         if response.status_code == 200:
#             ollama_available = True
#             print(f"✅ Ollama server is available at {LAN_IP}:11434")
#             return True
#         else:
#             ollama_available = False
#             print(f"❌ Ollama server at {LAN_IP}:11434 is not responding properly")
#             return False
#     except Exception as e:
#         ollama_available = False
#         print(f"❌ Ollama server at {LAN_IP}:11434 is not available: {e}")
#         return False

# def send_to_ollama(prompt: str, context: str = None) -> Optional[str]:
#     """Send prompt to Ollama API"""
#     if not ollama_available:
#         return None
    
#     full_prompt = f"{context}\n\n{prompt}" if context else prompt
    
#     payload = {
#         "model": OLLAMA_MODEL,
#         "prompt": full_prompt,
#         "stream": False,
#         "options": {
#             "num_thread": 8,
#             "num_ctx": 4096,
#             "temperature": 0.7
#         }
#     }
    
#     try:
#         response = requests.post(OLLAMA_URL, json=payload, timeout=300)
#         response.raise_for_status()
        
#         response_data = response.json()
#         return response_data.get("response", "").strip()
    
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Error connecting to Ollama: {e}")
#         return None
#     except Exception as e:
#         print(f"❌ Unexpected error with Ollama: {e}")
#         return None

# def save_upload_file(upload_file: UploadFile) -> str:
#     """Save uploaded file to disk and return absolute path"""
#     try:
#         # Ensure upload directory exists
#         os.makedirs(UPLOAD_DIR, exist_ok=True)
        
#         # Create safe filename
#         safe_filename = upload_file.filename.replace(" ", "_")
#         file_path = os.path.join(UPLOAD_DIR, safe_filename)
#         abs_path = os.path.abspath(file_path)

#         print(f"📁 Saving uploaded file to: {abs_path}")

#         # Save the file
#         with open(abs_path, "wb") as buffer:
#             # Reset file pointer to beginning
#             upload_file.file.seek(0)
#             content = upload_file.file.read()
#             buffer.write(content)

#         # Verify file was saved
#         if not os.path.exists(abs_path): 
#             raise Exception(f"File was not saved properly: {abs_path}")
        
#         file_size = os.path.getsize(abs_path)
#         print(f"✅ File saved successfully. Size: {file_size} bytes")
        
#         return abs_path
        
#     except Exception as e:
#         print(f"❌ Error saving file: {e}")
#         raise
# import time

# def transcribe_audio_file(file_path: str) -> Optional[str]:
#     """Transcribe audio file using Whisper with improved error handling"""
#     global whisper_model

#     if whisper_model is None:
#         print("❌ Whisper model not loaded")
#         return None

#     if not os.path.exists(file_path):
#         print(f"❌ Audio file not found: {file_path}")
#         return None
#     start_time = time.time()  # Start timer
#     print(f"📂 File exists: {file_path}")
#     print(f"📏 File size: {os.path.getsize(file_path)} bytes")

#     temp_path = None
#     try:
#         # Check if we need to convert audio format
#         file_extension = os.path.splitext(file_path)[1].lower()
        
#         if file_extension == '.wav':
#             # If it's already a WAV file, try to use it directly first
#             print(f"🎯 Attempting direct transcription of WAV file: {file_path}")
#             try:
#                 result = whisper_model.transcribe(file_path, fp16=False, language="en")
#                 transcript = result["text"].strip()
                
#                 # Save transcript to file
#                 transcript_file = os.path.splitext(file_path)[0] + "_transcript.txt"
#                 with open(transcript_file, 'w', encoding='utf-8') as f:
#                     f.write(transcript)

#                 print("✅ Direct transcription completed successfully")
#                 return transcript
                
#             except Exception as e:
#                 print(f"⚠️ Direct transcription failed, trying conversion: {e}")
        
#         # If direct transcription failed or it's not a WAV file, convert it
#         try:
#             import librosa
#             import soundfile as sf
            
#             # Create temporary file with proper cleanup
#             temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
#             os.close(temp_fd)  # Close file descriptor, we'll open it again
            
#             print(f"🔄 Converting audio to: {temp_path}")
            
#             # Load and convert audio
#             audio, sr = librosa.load(file_path, sr=16000)
#             sf.write(temp_path, audio, 16000)
            
#             # Verify temp file was created
#             if not os.path.exists(temp_path):
#                 raise Exception(f"Temporary file was not created: {temp_path}")
            
#             print(f"✅ Audio converted successfully. Temp file size: {os.path.getsize(temp_path)} bytes")
            
#             # Transcribe the converted audio
#             print(f"🎯 Transcribing converted audio: {temp_path}")
#             result = whisper_model.transcribe(temp_path, fp16=False, language="en")
#             transcript = result["text"].strip()

#             # Save transcript to file
#             transcript_file = os.path.splitext(file_path)[0] + "_transcript.txt"
#             with open(transcript_file, 'w', encoding='utf-8') as f:
#                 f.write(transcript)

#             print("✅ Transcription completed successfully")
#             return {
#                 "transcript": transcript,
#                 "duration": time.time() - start_time  # End timer and calculate duration
#             }
            
#         except ImportError:
#             print("❌ librosa and soundfile are required for audio conversion")
#             print("Install them with: pip install librosa soundfile")
#             return None
            
#     except Exception as e:
#         print(f"❌ Error during transcription: {e}")
#         import traceback
#         traceback.print_exc()
#         return None
    
#     finally:
#         # Clean up temporary file if it was created
#         if temp_path and os.path.exists(temp_path):
#             try:
#                 os.unlink(temp_path)
#                 print(f"🧹 Cleaned up temporary file: {temp_path}")
#             except Exception as e:
#                 print(f"⚠️ Warning: Could not clean up temp file {temp_path}: {e}")

# def generate_structured_summary(transcript: str) -> Dict[str, Any]:
#     """Generate structured summary with customer and agent insights"""
    
#     structured_prompt = f"""
#     Analyze the following call transcript and provide a structured summary with the following format:

#     **CALL SUMMARY:**
#     [Provide a brief overview of the call]

#     **CUSTOMER INSIGHTS:**
#     - Name: [Customer name if mentioned]
#     - Issue/Request: [What the customer wanted]
#     - Sentiment: [Positive/Negative/Neutral]
#     - Satisfaction Level: [High/Medium/Low]
#     - Key Concerns: [List main concerns]

#     **AGENT INSIGHTS:**
#     - Name: [Agent name if mentioned]
#     - Performance: [How well did the agent handle the call]
#     - Solutions Provided: [What solutions were offered]
#     - Follow-up Actions: [Any follow-up mentioned]

#     **KEY POINTS:**
#     - [List important points from the call]

#     **RESOLUTION STATUS:**
#     [Resolved/Partially Resolved/Unresolved]

#     Transcript:
#     {transcript}
#     """
    
#     summary = send_to_ollama(structured_prompt)
    
#     if summary:
#         # Parse the structured response (basic parsing)
#         lines = summary.split('\n')
#         customer_insights = {}
#         agent_insights = {}
        
#         current_section = None
#         for line in lines:
#             line = line.strip()
#             if "CUSTOMER INSIGHTS:" in line:
#                 current_section = "customer"
#             elif "AGENT INSIGHTS:" in line:
#                 current_section = "agent"
#             elif line.startswith("- ") and current_section:
#                 parts = line[2:].split(": ", 1)
#                 if len(parts) == 2:
#                     key, value = parts
#                     if current_section == "customer":
#                         customer_insights[key.lower().replace(" ", "_")] = value
#                     elif current_section == "agent":
#                         agent_insights[key.lower().replace(" ", "_")] = value
        
#         return {
#             "summary": summary,
#             "customer_insights": customer_insights,
#             "agent_insights": agent_insights
#         }
    
#     return {"summary": "Failed to generate summary", "customer_insights": {}, "agent_insights": {}}

# # Startup event
# @app.on_event("startup")
# async def startup_event():
#     """Initialize models on startup"""
#     print("🚀 Starting up FastAPI application...")
    
#     # Load Whisper model
#     load_whisper_model()
    
#     # Check Ollama availability
#     check_ollama_availability()
    
#     print("✅ Startup completed")

# # API Endpoints

# @app.get("/", response_model=Dict[str, str])
# async def root():
#     """Root endpoint with basic info"""
#     return {
#         "message": "Whisper-Ollama API is running!",
#         "version": "1.0.0",
#         "endpoints": "/docs for Swagger UI"
#     }

# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint"""
#     global whisper_model, ollama_available
    
#     # Re-check Ollama availability
#     check_ollama_availability()
    
#     whisper_loaded = whisper_model is not None
#     status = "healthy" if whisper_loaded and ollama_available else "unhealthy"
    
#     return HealthResponse(
#         whisper_loaded=whisper_loaded,
#         ollama_available=ollama_available,
#         status=status
#     )

# @app.post("/api/transcribe", response_model=TranscriptionResponse)
# async def transcribe_audio(file: UploadFile = File(...)):
#     """
#     Transcribe audio file using Whisper
    
#     Supported formats: wav, mp3, m4a, flac, ogg
#     """
#     global whisper_model
    
#     if whisper_model is None:
#         raise HTTPException(status_code=500, detail="Whisper model not loaded")
    
#     # Validate file format
#     supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
#     file_extension = os.path.splitext(file.filename)[1].lower()
    
#     if file_extension not in supported_formats:
#         raise HTTPException(
#             status_code=400, 
#             detail=f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"
#         )
    
#     file_path = None
#     try:
#         # Save uploaded file
#         file_path = save_upload_file(file)
        
#         # Transcribe audio
#         transcript = transcribe_audio_file(file_path)
        
#         if transcript:
#             return TranscriptionResponse(
#                 transcript=transcript,
#                 file_path=file_path,
#                 success=True,
#                 message="Transcription completed successfully"
#             )
#         else:
#             raise HTTPException(status_code=500, detail="Failed to transcribe audio")
    
#     except Exception as e:
#         # Clean up uploaded file if transcription failed
#         if file_path and os.path.exists(file_path):
#             try:
#                 os.unlink(file_path)
#             except:
#                 pass
#         raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# @app.post("/api/prompt", response_model=PromptResponse)
# async def send_prompt(request: PromptRequest):
#     """
#     Send a prompt to Ollama Llama3.70b model
#     """
#     if not ollama_available:
#         raise HTTPException(status_code=503, detail="Ollama service not available")
    
#     try:
#         response = send_to_ollama(request.prompt, request.context)
        
#         if response:
#             return PromptResponse(
#                 response=response,
#                 success=True,
#                 message="Prompt processed successfully"
#             )
#         else:
#             raise HTTPException(status_code=500, detail="Failed to get response from Ollama")
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing prompt: {str(e)}")

# @app.post("/api/summary", response_model=SummaryResponse)
# async def generate_summary(request: SummaryRequest):
#     """
#     Generate structured summary of customer and agent conversation
#     """
#     if not ollama_available:
#         raise HTTPException(status_code=503, detail="Ollama service not available")
    
#     try:
#         if request.summary_type == "structured":
#             result = generate_structured_summary(request.transcript)
            
#             return SummaryResponse(
#                 summary=result["summary"],
#                 customer_insights=result["customer_insights"],
#                 agent_insights=result["agent_insights"],
#                 success=True,
#                 message="Structured summary generated successfully"
#             )
#         else:
#             # General summary
#             general_prompt = f"Please provide a concise summary of the following call transcript:\n\n{request.transcript}"
#             summary = send_to_ollama(general_prompt)
            
#             if summary:
#                 return SummaryResponse(
#                     summary=summary,
#                     success=True,
#                     message="General summary generated successfully"
#                 )
#             else:
#                 raise HTTPException(status_code=500, detail="Failed to generate summary")
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

# @app.post("/api/full-process")
# async def full_process(file: UploadFile = File(...), summary_type: str = "structured"):
#     """
#     Complete workflow: Upload audio -> Transcribe -> Generate summary
#     """
#     try:
#         # Step 1: Transcribe audio
#         transcription_result = await transcribe_audio(file)
        
#         if not transcription_result.success:
#             raise HTTPException(status_code=500, detail="Failed to transcribe audio")
        
#         # Step 2: Generate summary
#         summary_request = SummaryRequest(
#             transcript=transcription_result.transcript,
#             summary_type=summary_type
#         )
        
#         summary_result = await generate_summary(summary_request)
        
#         if not summary_result.success:
#             raise HTTPException(status_code=500, detail="Failed to generate summary")
        
#         # Return combined results
#         return {
#             "transcription": {
#                 "transcript": transcription_result.transcript,
#                 "file_path": transcription_result.file_path
#             },
#             "summary": {
#                 "summary": summary_result.summary,
#                 "customer_insights": summary_result.customer_insights,
#                 "agent_insights": summary_result.agent_insights
#             },
#             "success": True,
#             "message": "Full process completed successfully"
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error in full process: {str(e)}")

# # Run the application
# if __name__ == "__main__":
#     print(f"🚀 Starting FastAPI server on {LAN_IP}:8000")
#     print(f"📡 Ollama server expected at {LAN_IP}:11434")
#     print(f"🌐 Access API documentation at: http://{LAN_IP}:8000/docs")
    
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",  # Listen on all interfaces
#         port=8000,
#         reload=True,
#         log_level="info"
#     )

import os
import json
import tempfile
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import whisper
import google.generativeai as genai
from utils import dispatch_llm_request, parse_llm_response

# Configuration
GEMINI_API_KEY = "AIzaSyDh2gPu9X_96rpioBXcw7BQCDPZcFGMuO4"  # Replace with your actual API key
WHISPER_MODEL = "large-v3"
UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./outputs"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Call Analysis API with Gemini", version="1.0.0")

# Global variables
whisper_model = None
gemini_model = None

# Pydantic models
class TranscriptRequest(BaseModel):
    transcript: str

class EvaluationResponse(BaseModel):
    structured_transcript: str
    title: str
    summary: str
    evaluation_metrics: Dict[str, Any]
    success: bool

class StructuredTranscriptResponse(BaseModel):
    structured_transcript: str
    success: bool

class SummaryResponse(BaseModel):
    title: str
    summary: str
    success: bool

class MetricsResponse(BaseModel):
    evaluation_metrics: Dict[str, Any]
    success: bool

# Evaluation metrics
EVALUATION_METRICS = [
    "Call Opening", "Empathy", "Logical Effective Probing",
    "Listening Skills Comprehension", "Hold", "Dead Air",
    "Tone Mannerism Positive Approach", "Objection Escalation Handling Ownership",
    "Further Assistance Call Closing", "Raise Complaint", "Resolution"
]

# Utility functions
def initialize_gemini_model():
    """Initialize Gemini model"""
    global gemini_model
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Gemini model: {e}")
        return False


def call_llm(prompt: str) -> str:
    return dispatch_llm_request(
        prompt=prompt,
        llm_provider="gemini",
        api_key=GEMINI_API_KEY,
        model="gemini-1.5-flash"
    )

def load_whisper_model():
    """Load Whisper model"""
    global whisper_model
    try:
        whisper_model = whisper.load_model(WHISPER_MODEL)
        return True
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        return False

def transcribe_audio(file_path: str) -> str:
    """Transcribe audio file using Whisper"""
    if whisper_model is None:
        raise Exception("Whisper model not loaded")
    
    result = whisper_model.transcribe(file_path, fp16=False, language="en")
    return result["text"].strip()

def create_structured_transcript(transcript: str) -> str:
    """Create structured transcript with timestamps"""
    prompt = f"""
    You're a conversation analyst for Payone—a customer experience company. Convert the following raw transcript into a structured call log with the following format:

    Format:
    [00:00:00] Agent (confident, professional): Hello, thank you for calling Payone...
    [00:00:08] Customer (frustrated): I've been charged twice...

    Instructions:
    - Add realistic and progressive timestamps (start at 00:00:00)
    - Tag speakers clearly: "Agent" or "Customer"
    - Identify emotional tone where possible (e.g., calm, frustrated, apologetic)
    - Use professional, readable language; remove filler words and irrelevant chatter
    - Segment the conversation naturally based on dialogue turns
    - Preserve the true intent of each speaker
    - Highlight key action items or escalations where applicable

    Raw transcript:
    {transcript}
    """

    return call_llm(prompt)

def generate_title_summary(transcript: str) -> Dict[str, str]:
    """Generate title and detailed summary"""
    prompt = f"""
    You are a senior call analyst for Payone, a customer experience and transaction processing company. Your task is to generate a precise and informative title and summary for the following customer service call.

    Instructions:
    1. Create a descriptive, concise title that captures the central issue or nature of the call.
    2. Write a detailed, professional summary that includes:
    - Key points raised by the customer
    - Actions taken or promised by the agent
    - Emotional tone shifts (e.g., frustration resolved to satisfaction)
    - Outcome or resolution of the call
    - Any escalation, complaint, or follow-up mentioned

    Format your response strictly as JSON:
    {{
    "title": "Call title here",
    "summary": "Detailed summary here"
    }}

    Transcript:
    {transcript}
    """

    
    try:
        response = call_llm(prompt)
        # Clean the response to extract JSON
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:-3]  # Remove ```json and ```
        elif response.startswith('```'):
            response = response[3:-3]  # Remove ``` and ```
        
        return json.loads(response)
    except json.JSONDecodeError:
        return {"title": "Call Analysis", "summary": "Unable to generate structured summary"}

def evaluate_call_metrics(transcript: str) -> Dict[str, Any]:
    """Evaluate call based on specific metrics"""
    metrics_str = '", "'.join(EVALUATION_METRICS)
    
    prompt = f"""
    You are an expert call evaluator for Payone, a leading customer experience company. Your task is to analyze a service call transcript using professional interaction metrics. Each metric reflects a specific behavioral and procedural competency.

    Instructions:
    - Assign a score from 1 to 10 (1 = Poor, 10 = Excellent)
    - Provide a brief, actionable comment explaining the score
    - If any metric is not observed in the call, set score to 0 and note "Not applicable"
    - Base evaluation on the full context of conversation—tone, flow, issue resolution, and escalation if applicable
    - Highlight outstanding performance or critical failures explicitly

    Evaluation metrics:
    {metrics_str}

    Format your response as structured JSON:
    {{
    "Call Opening": {{ "score": 8, "comments": "Agent greeted with enthusiasm and confirmed customer identity" }},
    "Empathy": {{ "score": 9, "comments": "Agent acknowledged frustration and used reassuring tone" }},
    "Logical Effective Probing": {{ "score": 7, "comments": "Asked clarifying questions but missed a billing detail" }},
    ...
    }}

    Transcript:
    {transcript}
    """

    
    try:
        response = call_llm(prompt)
        # Clean the response to extract JSON
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:-3]
        elif response.startswith('```'):
            response = response[3:-3]
        
        return json.loads(response)
    except json.JSONDecodeError:
        return {metric: {"score": 0, "comments": "Unable to evaluate"} for metric in EVALUATION_METRICS}

def save_transcript_file(content: str, filename: str) -> str:
    """Save transcript to file"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("🚀 Starting up FastAPI application...")
    load_whisper_model()
    print("✅ Startup completed")

# API Endpoints
@app.post("/")
def start():
    return "Application is started"

@app.post("/api/analyze-audio", response_model=EvaluationResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """Complete audio analysis: transcribe, structure, summarize, and evaluate"""
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")

    temp_path = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name
        
        # Process
        raw_transcript = transcribe_audio(temp_path)
        structured_transcript = create_structured_transcript(raw_transcript)
        title_summary = generate_title_summary(structured_transcript)
        evaluation_metrics = evaluate_call_metrics(structured_transcript)
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}.txt"
        save_transcript_file(structured_transcript, filename)
        
        return EvaluationResponse(
            structured_transcript=structured_transcript,
            title=title_summary.get("title", "Call Analysis"),
            summary=title_summary.get("summary", ""),
            evaluation_metrics=evaluation_metrics,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/api/structure-transcript", response_model=StructuredTranscriptResponse)
async def structure_transcript(request: TranscriptRequest):
    """Convert raw transcript to structured format"""
    
    try:
        structured = create_structured_transcript(request.transcript)
        return StructuredTranscriptResponse(structured_transcript=structured, success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/generate-summary", response_model=SummaryResponse)
async def generate_summary(request: TranscriptRequest):
    """Generate title and detailed summary"""
    
    
    try:
        result = generate_title_summary(request.transcript)
        return SummaryResponse(
            title=result.get("title", "Call Analysis"),
            summary=result.get("summary", ""),
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/evaluate-metrics", response_model=MetricsResponse)
async def evaluate_metrics(request: TranscriptRequest):
    """Evaluate call based on specific metrics"""
    
    
    try:
        metrics = evaluate_call_metrics(request.transcript)
        return MetricsResponse(evaluation_metrics=metrics, success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "gemini_initialized": gemini_model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting FastAPI server with Gemini integration")
    print("🌐 Access API documentation at: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)