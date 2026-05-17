"""
api.py — FastAPI wrapper for compare_chant.py
Run:  uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, shutil, os, traceback

# ── Import everything from your unchanged analysis script ──
import compare_chant   # this runs the module-level code (loads ref data, Whisper)

app = FastAPI(title="Svara Vaidya API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload user chant audio → returns full feedback JSON.
    Calls compare_chant.analyze_chant() directly — no code duplication.
    """
    allowed = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(status_code=400, detail=f"Audio file required: {allowed}")

    # Save upload to a temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = compare_chant.analyze_chant(tmp_path)   # ← calls YOUR function
        return JSONResponse(content=result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)