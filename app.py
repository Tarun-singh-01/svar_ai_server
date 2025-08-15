# app.py

import os
from fastapi import FastAPI, UploadFile, HTTPException
from dotenv import load_dotenv

# Import the official SYNCHRONOUS SDKs
from sarvamai import SarvamAI
from openai import OpenAI

# --- Load Environment Variables ---
load_dotenv()

# --- Initialize API Clients ---
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

sarvam_client = None
openai_client = None

if SARVAM_API_KEY:
    sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Svar AI server is running"}

@app.post("/transcribe")
def transcribe_audio(file: UploadFile):
    """
    This is a synchronous function. FastAPI will run it in a background
    thread pool to avoid blocking the server.
    """
    if not sarvam_client or not openai_client:
        raise HTTPException(status_code=500, detail="API keys are not configured on the server.")

    file_path = f"temp_{file.filename}"
    try:
        # Save the uploaded file temporarily
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # --- Step 1: Transcription with Sarvam AI (Synchronous) ---
        print("Starting Sarvam AI transcription job...")
        job = sarvam_client.speech_to_text_job.create_job(
            model="saarika:v2.5",
            with_diarization=True,
            with_timestamps=True,
            language_code="en-IN",
        )
        
        job.upload_files(file_paths=[file_path])
        job.start() 
        job.wait_until_complete(poll_interval=5, timeout=300)

        if job.is_failed():
            raise RuntimeError(f"Transcription failed: {job.get_status().reason}")

        result_list = job.get_outputs()
        if not result_list:
            raise RuntimeError("No output found from transcription job.")
        
        transcription_result = result_list[0]
        print("Transcription successful.")

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

    # --- Step 2: Format Transcript for Summarization ---
    plain_text_transcript = ""
    if "utterances" in transcription_result:
        for utterance in transcription_result["utterances"]:
            speaker = utterance.get("speaker", "Unknown Speaker")
            text = utterance.get("text", "")
            plain_text_transcript += f"{speaker}: {text}\n"
    else:
        plain_text_transcript = transcription_result.get("text", "No text found.")

    # --- Step 3: Summarization with GPT-4o ---
    summary_result = "Could not generate summary."
    try:
        print("Sending transcript to GPT-4o for summarization...")
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert assistant who creates concise, professional summaries of meeting transcripts, including key discussion points and action items."},
                {"role": "user", "content": f"Please summarize the following transcript:\n\n{plain_text_transcript}"}
            ]
        )
        summary_result = completion.choices[0].message.content or "Summary was empty."
        print("Summarization successful.")
    except Exception as e:
        print(f"Error during summarization: {e}")
        summary_result = "Error: Could not generate summary."

    return {
        "transcript": transcription_result,
        "summary": summary_result
    }