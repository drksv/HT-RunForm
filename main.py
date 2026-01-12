from fastapi import FastAPI, UploadFile, Form
from warmup import get_warmup_cooldown
from posture import analyze_posture_image
from video_analysis import analyze_running_video
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to healthtimeout.in
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze-running")
async def analyze_running(
    age: int = Form(...),
    gender: str = Form(...),
    experience: str = Form(...),
    file: UploadFile = Form(...)
):
    warmup = get_warmup_cooldown(age, gender, experience)

    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
    return {"error": "File too large. Max 20MB"}

    if file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        posture_feedback = analyze_posture_image(data)
    else:
        posture_feedback = analyze_running_video(data)

    return {
        "warmup_cooldown": warmup,
        "posture_feedback": posture_feedback
    }
