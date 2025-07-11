import os
import json
import requests
from typing import Dict, Any, Optional
from openai import OpenAI
import google.generativeai as genai

# -------------------------------------
# 🔁 Unified Dispatcher for LLMs
# -------------------------------------
def dispatch_llm_request(
    prompt: str,
    llm_provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    llm_url: Optional[str] = None
) -> str:
    """Send a prompt to selected LLM provider and return raw text response"""

    if llm_provider == "gemini":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model or "gemini-1.5-flash")
        return model.generate_content(prompt).text

    elif llm_provider == "openai":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model or "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    elif llm_provider == "ollama":
        payload = {
            "model": model or "llama3:70b",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(llm_url or "http://192.168.1.55:11434/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"]

    elif llm_provider == "sarvam":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model or "sarvamai/sarvam-m:free",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    elif llm_provider == "deepseek":
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        completion = client.chat.completions.create(
            model=model or "deepseek/deepseek-r1-0528:free",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

# -------------------------------------
# 🧹 Response Cleaner & JSON Loader
# -------------------------------------
def parse_llm_response(raw: str) -> Dict[str, Any]:
    """Parse LLM response and extract JSON if present"""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:-3]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:-3]
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format", "raw": raw}
    except Exception as e:
        return {"error": str(e)}
