"""
api.py — FastAPI wrapper for compare_chant.py
Run:  uvicorn api:app --host 0.0.0.0 --port 10000
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, shutil, os, traceback

# ── compare_chant is imported lazily on first request ──
# This lets uvicorn bind the port immediately, before heavy libs load.
compare_chant = None

app = FastAPI(title="Svara Vaidya API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_compare_chant():
    """Import compare_chant once, on first use."""
    global compare_chant
    if compare_chant is None:
        print("Loading compare_chant module (first request)...")
        import compare_chant as _cc
        compare_chant = _cc
        print("compare_chant loaded.")
    return compare_chant


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/warmup")
def warmup():
    """
    Optional: call this endpoint once after deploy to trigger model loading.
    Returns immediately — loading happens in background via first /analyze call.
    """
    return {"status": "warming up — send an /analyze request to fully load models"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload user chant audio → returns full feedback JSON.
    """
    allowed = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(status_code=400, detail=f"Audio file required: {allowed}")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        cc = get_compare_chant()
        result = cc.analyze_chant(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, log_level="info")