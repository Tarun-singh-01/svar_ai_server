# app.py (Debug Version)

import os
print("✅ 1. Imports successful")

from fastapi import FastAPI, UploadFile, HTTPException
from dotenv import load_dotenv
from sarvamai import SarvamAI
from openai import OpenAI
print("✅ 2. All libraries imported")

# --- Load Environment Variables ---
try:
    print("- 3. Attempting to load .env file...")
    load_dotenv()
    print("✅ 3. .env file loaded")
except Exception as e:
    print(f"❌ ERROR loading .env file: {e}")

# --- Initialize API Clients ---
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
if SARVAM_API_KEY:
    print("✅ 4a. Sarvam API Key FOUND")
else:
    print("❌ 4a. Sarvam API Key IS MISSING!")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print("✅ 4b. OpenAI API Key FOUND")
else:
    print("❌ 4b. OpenAI API Key IS MISSING!")

sarvam_client = None
openai_client = None

try:
    print("- 5. Initializing Sarvam AI client...")
    if SARVAM_API_KEY:
        sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        print("✅ 5. Sarvam AI client initialized")
    else:
        print("- 5. Skipping Sarvam AI client initialization (no key)")
except Exception as e:
    print(f"❌ ERROR initializing Sarvam AI client: {e}")

try:
    print("- 6. Initializing OpenAI client...")
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ 6. OpenAI client initialized")
    else:
        print("- 6. Skipping OpenAI client initialization (no key)")
except Exception as e:
    print(f"❌ ERROR initializing OpenAI client: {e}")


print("- 7. Setting up FastAPI app...")
app = FastAPI()
print("✅ 7. FastAPI app set up")

@app.get("/")
def read_root():
    return {"status": "Svar AI server is running"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    # This function will only be called after the server starts,
    # so the error is not here.
    pass

print("✅ 8. Server setup complete and ready to run.")