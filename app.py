# app.py (with extensive logging and combined AI call)

import os
import openai
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
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
    raise RuntimeError("API keys for Sarvam (SARVAM_API_KEY) and OpenAI (OPENAI_API_KEY) must be set.")

sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Helper Functions ---
def format_diarized_transcript(sarvam_result: dict) -> str:
    print("Step 4: Formatting diarized transcript...")
    entries = sarvam_result.get("diarized_transcript", {}).get("entries", [])
    if not entries:
        return sarvam_result.get("transcript", "No content found.")
    
    formatted_lines = []
    for entry in entries:
        speaker = entry.get("speaker_id", "Unknown Speaker").replace('_', ' ').title()
        text = entry.get("transcript", "")
        formatted_lines.append(f"**{speaker}:** {text}")
    
    formatted_string = "\n".join(formatted_lines).strip()
    print("Step 4b: Transcript formatted successfully.")
    return formatted_string

# --- API Endpoint ---
@app.get("/")
def read_root():
    return {"status": "Svar AI server is running"}

@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    template_type: str = Form("meeting_notes") # Note: template_type is not used in this version for simplicity
):
    temp_dir = "temp_processing"
    output_dir = os.path.join(temp_dir, "sarvam_output")
    
    try:
        print("\n--- NEW REQUEST RECEIVED ---")
        print("Step 1: Creating temporary directory...")
        os.makedirs(output_dir, exist_ok=True)
        
        temp_audio_path = os.path.join(temp_dir, file.filename if file.filename else "audio.tmp")
        
        print("Step 1b: Saving uploaded audio file to temp path...")
        with open(temp_audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print("Step 1c: Audio file saved.")
        
        print("Step 2: Starting Sarvam AI transcription job...")
        job = sarvam_client.speech_to_text_job.create_job(
            language_code="en-IN", model="saarika:v2.5", with_diarization=True
        )
        job.upload_files(file_paths=[temp_audio_path])
        job.start()
        print("Step 2b: Job started. Waiting for completion...")
        job.wait_until_complete(timeout=300)
        print("Step 2c: Job completed.")

        if job.is_failed():
            reason = job.get_status().get('reason')
            print(f"!!! Sarvam job failed: {reason}")
            raise HTTPException(status_code=502, detail=f"Sarvam job failed: {reason}")
        
        print("Step 3: Downloading Sarvam job outputs...")
        job.download_outputs(output_dir=output_dir)
        print("Step 3b: Outputs downloaded.")

        output_files = glob.glob(os.path.join(output_dir, "*.json"))
        if not output_files:
            print("!!! No transcript output file found from Sarvam.")
            raise HTTPException(status_code=404, detail="No transcript output file found from Sarvam.")

        print("Step 3c: Reading transcript from JSON file...")
        with open(output_files[0]) as jf:
            sarvam_result = json.load(jf)
        
        diarized_transcript_string = format_diarized_transcript(sarvam_result)
        
        print("Step 5: Starting SINGLE OpenAI call for summary and action items...")
        
        prompt = f"""
        Analyze the following transcript and provide two things in a JSON format:
        1. A concise "summary" of the conversation.
        2. A list of "action_items". If there are none, return an empty list [].

        The final output must be a single, valid JSON object with the keys "summary" and "action_items".

        Transcript:
        ---
        {diarized_transcript_string}
        ---
        """
        
        ai_response = openai_client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"}, # Force JSON output
            messages=[
                {"role": "system", "content": "You are an assistant that processes transcripts into structured JSON data."},
                {"role": "user", "content": prompt}
            ]
        )
        
        print("Step 5b: OpenAI call successful.")
        
        # The response content is a JSON string, so we parse it
        result_json = json.loads(ai_response.choices[0].message.content)
        summary = result_json.get("summary", "Summary could not be generated.")
        # Join the list of action items into a single string for the app
        action_items_list = result_json.get("action_items", [])
        action_items = "\n".join(f"- {item}" for item in action_items_list) if action_items_list else "No action items were identified."

        print("Step 6: Successfully processed all data. Returning response.")
        
        return JSONResponse(content={
            "transcript": diarized_transcript_string,
            "summary": summary,
            "action_items": action_items
        })

    except Exception as e:
        print(f"!!! AN UNEXPECTED ERROR OCCURRED: {type(e).__name__} - {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    finally:
        print("Step 7: Cleaning up temporary directory...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("--- REQUEST FINISHED ---")
