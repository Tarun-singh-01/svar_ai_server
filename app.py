# app.py

import os
from fastapi import FastAPI, UploadFile, HTTPException
from dotenv import load_dotenv

# Import the official SDKs
from sarvamai import SarvamAI
from openai import OpenAI

# --- Load Environment Variables ---
# Your .env file should contain both keys:
# SARVAM_API_KEY="YOUR_SARVAM_AI_API_KEY"
# OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
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
async def transcribe_audio(file: UploadFile):
    """
    This endpoint handles the full process:
    1. Transcribes audio with Sarvam AI (including diarization).
    2. Summarizes the transcript with OpenAI's GPT-4o.
    """
    if not sarvam_client or not openai_client:
        raise HTTPException(status_code=500, detail="API keys are not configured on the server.")

    # --- Step 1: Transcription with Sarvam AI ---
    try:
        # Save the uploaded file temporarily to pass its path to the SDK
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        print("Starting Sarvam AI transcription job with diarization...")
        
        job = sarvam_client.speech_to_text_job.create_job(
            model="saarika:v2.5",
            with_diarization=True,
            with_timestamps=True,
            language_code="en-IN",
        )
        
        await job.upload_files(file_paths=[file_path])
        await job.start()
        await job.wait_until_complete(poll_interval=5, timeout=300)

        if job.is_failed():
            raise RuntimeError(f"Transcription failed: {job.get_status().reason}")

        result_list = await job.get_outputs()
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
    # We need to convert the structured transcript into a simple string for GPT-4o
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

    # Return the full structured transcript and the new summary
    return {
        "transcript": transcription_result,
        "summary": summary_result
    }