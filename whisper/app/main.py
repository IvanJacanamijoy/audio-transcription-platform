from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile

app = FastAPI()

# Cargar modelo Whisper (puedes usar "base", "small", "medium", "large")
model = whisper.load_model("base")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Transcribir audio
    result = model.transcribe(tmp_path)

    return {"text": result["text"]}
