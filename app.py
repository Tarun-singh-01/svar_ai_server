# app.py

import os
import httpx
from fastapi import FastAPI, UploadFile, HTTPException, Form
from openai import OpenAI # Import the OpenAI library

# --- Load Environment Variables ---
# Make sure your .env file now includes both keys:
# SARVAM_API_KEY="YOUR_SARVAM_AI_API_KEY"
# OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
from dotenv import load_dotenv
load_dotenv()

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SARVAM_API_URL = "https://api.sarvam.ai/v1/voice/stt" 

app = FastAPI()

# Initialize the OpenAI client
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None


@app.get("/")
def read_root():
    return {"status": "Svar AI server is running"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile, language: str = Form("en")):
    if not SARVAM_API_KEY or not openai_client:
        raise HTTPException(status_code=500, detail="API keys are not configured correctly.")

    audio_data = await file.read()

    # --- Step 1: Transcription with Sarvam AI ---
    headers = {"Authorization": f"Bearer {SARVAM_API_KEY}"}
    form_data = {'language': language}
    files = {'file': (file.filename, audio_data, file.content_type)}
    
    transcript_result = ""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            print(f"Sending audio to Sarvam AI for transcription...")
            response = await client.post(SARVAM_API_URL, headers=headers, data=form_data, files=files)
            response.raise_for_status() 
            response_data = response.json()
            transcript_result = response_data.get("text", "Transcription not found.")
            print("Transcription successful.")
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

    # --- Step 2: Summarization with GPT-4o ---
    summary_result = "Could not generate summary."
    try:
        print("Sending transcript to GPT-4o for summarization...")
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert assistant who creates concise, professional summaries of meeting transcripts."},
                {"role": "user", "content": f"Please summarize the following transcript:\n\n{transcript_result}"}
            ]
        )
        summary_result = completion.choices[0].message.content or "Summary was empty."
        print("Summarization successful.")
    except Exception as e:
        print(f"Error during summarization: {e}")
        # We don't raise an exception here, so the user still gets the transcript
        # even if summarization fails.
        summary_result = "Error: Could not generate summary."


    return {
        "transcript": transcript_result,
        "summary": summary_result
    }