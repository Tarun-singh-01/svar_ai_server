# app.py (FastAPI Server)

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import httpx
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get API keys from environment variables
SARVAM_AI_API_KEY = os.getenv("SARVAM_AI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- NEW FUNCTION TO GENERATE ACTION ITEMS ---
def generate_action_items(transcript: str) -> str:
    """
    Uses GPT-4o to extract action items from a transcript.
    """
    if not transcript:
        return "No transcript provided to generate action items."

    try:
        # This prompt is specifically designed to get a list of tasks or to-dos
        prompt = f"""
        Analyze the following transcript and extract a clear, concise list of action items or tasks.
        If no specific action items are mentioned, state 'No action items were identified.'.
        Format the output as a simple list.

        Transcript:
        ---
        {transcript}
        ---
        Action Items:
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert assistant skilled at identifying action items from meeting transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Lower temperature for more deterministic output
            max_tokens=250,
        )
        action_items = response.choices[0].message.content.strip()
        return action_items
    except Exception as e:
        print(f"Error generating action items: {e}")
        return "Could not generate action items due to an error."

def generate_summary(transcript: str, template_type: str) -> str:
    """
    Uses GPT-4o to generate a summary based on the transcript and a template.
    """
    if not transcript:
        return "No transcript provided to summarize."

    # Basic prompt engineering based on the template type
    if template_type == 'To-Do List':
        prompt_template = "Summarize the following transcript into a concise to-do list."
    else: # Default to 'Meeting Notes'
        prompt_template = "Summarize the key points and decisions from the following meeting transcript."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes meeting notes."},
                {"role": "user", "content": f"{prompt_template}\n\nTranscript:\n{transcript}"}
            ],
            temperature=0.5,
            max_tokens=300,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Could not generate summary due to an error."


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    template_type: str = Form("Meeting Notes")
):
    """
    Receives an audio file, transcribes it, and generates a summary and action items.
    """
    if not SARVAM_AI_API_KEY or not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="API keys are not configured on the server.")

    sarvam_url = "https://api.sarvam.ai/v1/voice/stt"
    headers = {
        "Authorization": f"Bearer {SARVAM_AI_API_KEY}",
        "language": "en",
        "diarize": "true",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # 1. Get Transcript from Sarvam AI
            files = {'file': (file.filename, await file.read(), file.content_type)}
            response = await client.post(sarvam_url, headers=headers, files=files)
            response.raise_for_status()
            transcript_data = response.json()
            transcript = transcript_data.get('text', 'Transcription failed.')

            # 2. Generate Summary from Transcript
            summary = generate_summary(transcript, template_type)

            # 3. GENERATE ACTION ITEMS from Transcript
            action_items = generate_action_items(transcript)

            # 4. Return all three pieces of data
            return JSONResponse(content={
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items # <-- New field in the response
            })

        except httpx.HTTPStatusError as e:
            print(f"Error during transcription API call: {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Transcription service error: {e.response.text}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)