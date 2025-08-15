# app.py

import os
import openai
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from sarvamai import SarvamAI
import glob
import json
import shutil

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# --- Initialize API Clients ---
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SARVAM_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError("API keys for Sarvam and OpenAI must be set.")

sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Helper Functions ---
def format_diarized_transcript(sarvam_result: dict) -> str:
    """Formats the diarized JSON from Sarvam AI into a readable string."""
    entries = sarvam_result.get("diarized_transcript", {}).get("entries", [])
    if not entries:
        return sarvam_result.get("transcript", "No content found.")

    formatted_lines = []
    for entry in entries:
        speaker = entry.get("speaker_id", "Unknown Speaker").replace('_', ' ').title()
        text = entry.get("transcript", "")
        formatted_lines.append(f"**{speaker}:** {text}")
    
    return "\n".join(formatted_lines).strip()

def generate_summary_prompt(template_type: str, transcript: str) -> str:
    """Generates a specific prompt for GPT-4o based on the selected template."""
    prompts = {
        "meeting_notes": f"Summarize the key decisions, action items, and discussion points from this transcript:\n\n{transcript}",
        "todo_list": f"Extract all actionable tasks and to-do items from this transcript into a checklist:\n\n{transcript}",
    }
    return prompts.get(template_type, f"Provide a concise summary of this transcript:\n\n{transcript}")

# --- API Endpoint ---
@app.get("/")
def read_root():
    return {"status": "Svar AI server is running"}

@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    template_type: str = Form("meeting_notes")
):
    temp_dir = "temp_processing"
    output_dir = os.path.join(temp_dir, "sarvam_output")
    os.makedirs(output_dir, exist_ok=True)
    
    temp_audio_path = os.path.join(temp_dir, file.filename if file.filename else "audio.tmp")
    
    try:
        with open(temp_audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        print("Starting Sarvam AI transcription job...")
        job = sarvam_client.speech_to_text_job.create_job(
            language_code="en-IN",
            model="saarika:v2.5",
            with_timestamps=True,
            with_diarization=True,
        )
        job.upload_files(file_paths=[temp_audio_path])
        job.start()
        job.wait_until_complete(timeout=300)

        if job.is_failed():
            raise HTTPException(status_code=502, detail=f"Sarvam job failed: {job.get_status().get('reason')}")
        
        # --- THE CORRECT METHOD ---
        # Use download_outputs() to save the result JSON to a file.
        job.download_outputs(output_dir=output_dir)
        print("Transcription job outputs downloaded.")

        # Find the resulting JSON file in the output directory.
        output_files = glob.glob(os.path.join(output_dir, "*.json"))
        if not output_files:
            raise HTTPException(status_code=404, detail="No transcript output file found from Sarvam.")

        # Read the result from the JSON file.
        with open(output_files[0]) as jf:
            sarvam_result = json.load(jf)
        
        diarized_transcript_string = format_diarized_transcript(sarvam_result)
        
        print("Generating summary with GPT-4o...")
        summary_prompt = generate_summary_prompt(template_type, diarized_transcript_string)
        
        summary_completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = summary_completion.choices[0].message.content
        print("Summary generated successfully.")

        # Return a simple string for the transcript to the app.
        return {
            "transcript": diarized_transcript_string,
            "summary": summary
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)