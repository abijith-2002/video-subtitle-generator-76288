import os
import uuid
import shutil
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import json
import subprocess
import datetime

# DB logic would connect to subtitle_database -- placeholder for now
import sqlite3

# PUBLIC_INTERFACE
def get_db():
    """Dependency that yields a database connection."""
    db_path = os.getenv("DB_PATH", "video_subtitles.db")
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in .env")

# App setup
app = FastAPI(
    title="Video Subtitle Generator API",
    description="Backend for generating subtitles (SRT/VTT) from video files using OpenAI transcription.",
    version="1.0.0",
    openapi_tags=[
        {"name": "video", "description": "Endpoints for video uploading and processing"},
        {"name": "subtitles", "description": "Endpoints for retrieving subtitle files and subtitles history"}
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploaded_videos"
SUBTITLE_FOLDER = "generated_subtitles"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUBTITLE_FOLDER, exist_ok=True)

# --- MODELS ---

class UploadResponse(BaseModel):
    video_id: str = Field(..., description="ID assigned to the uploaded video")
    filename: str = Field(..., description="Original filename of the uploaded video")
    status_text: str = Field("uploaded", description="Status of the upload")

class GenerateRequest(BaseModel):
    video_id: str = Field(..., description="ID of the uploaded video to generate subtitles for")
    format: str = Field("srt", description="Subtitle format (srt or vtt)")

class GenerateResponse(BaseModel):
    video_id: str = Field(..., description="ID of the processed video")
    format: str = Field(..., description="Subtitle format generated")
    status_text: str = Field(..., description="Status of generation: success, failed, or error message")

class SubtitleInfo(BaseModel):
    video_id: str = Field(..., description="ID of the video file")
    original_filename: str = Field(..., description="Uploaded filename")
    formats: List[str] = Field(..., description="Available subtitle formats")
    created_at: str = Field(..., description="When this was processed")

class SubtitleHistoryResponse(BaseModel):
    subtitles: List[SubtitleInfo]

# --- UTILS ---

# PUBLIC_INTERFACE
def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video file using ffmpeg (requires ffmpeg in env)."""
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    try:
        subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Audio extraction failed") from e

# PUBLIC_INTERFACE
def call_openai_transcription(audio_path: str):
    """Send audio file to OpenAI Whisper API, return transcript in segments."""
    import requests
    api_url = "https://api.openai.com/v1/audio/transcriptions"  # Whisper endpoint
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    with open(audio_path, "rb") as audio_file:
        files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
        data = {
            "model": "whisper-1",
            "response_format": "verbose_json"
        }
        resp = requests.post(api_url, headers=headers, files=files, data=data)
        if resp.status_code == 200:
            return resp.json().get("segments", [])
        else:
            # Might contain error message
            raise HTTPException(status_code=500, detail=f"OpenAI API Error: {resp.text}")

# PUBLIC_INTERFACE
def segments_to_srt(segments: list) -> str:
    """Convert segments (with start/stop and text) to SRT format content."""
    srt_lines = []
    for idx, seg in enumerate(segments, 1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = seg.get("text", "")
        srt_lines.append(str(idx))
        srt_lines.append(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        srt_lines.append(text.strip())
        srt_lines.append("")
    return "\n".join(srt_lines)

# PUBLIC_INTERFACE
def segments_to_vtt(segments: list) -> str:
    """Convert segments to VTT format content."""
    vtt_lines = ["WEBVTT", ""]
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = seg.get("text", "")
        vtt_lines.append(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        vtt_lines.append(text.strip())
        vtt_lines.append("")
    return "\n".join(vtt_lines)

# PUBLIC_INTERFACE
def seconds_to_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timecode."""
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    ms = int(td.microseconds / 1000)
    total_hours = td.days * 24 + hours
    return f"{total_hours:02}:{minutes:02}:{seconds:02},{ms:03}"

# --- DB INIT (SQLite for demo, replace with subtitle_database connection as required) ---

def init_db():
    """Initialize the demo DB if not present."""
    db_path = os.getenv("DB_PATH", "video_subtitles.db")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS subtitles (
                video_id TEXT PRIMARY KEY,
                original_filename TEXT,
                created_at TEXT,
                formats TEXT
            )
            """
        )
        conn.commit()
init_db()

def save_history(video_id, filename, formats):
    db_path = os.getenv("DB_PATH", "video_subtitles.db")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO subtitles (video_id, original_filename, created_at, formats) VALUES (?, ?, ?, ?)",
            (video_id, filename, datetime.datetime.utcnow().isoformat(), json.dumps(formats)),
        )
        conn.commit()

def get_history():
    db_path = os.getenv("DB_PATH", "video_subtitles.db")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT video_id, original_filename, created_at, formats FROM subtitles")
        result = []
        for video_id, original_filename, created_at, formats_json in cursor.fetchall():
            formats = json.loads(formats_json)
            result.append(SubtitleInfo(
                video_id=video_id,
                original_filename=original_filename,
                formats=formats,
                created_at=created_at
            ))
        return result

# --- ENDPOINTS ---

# PUBLIC_INTERFACE
@app.post("/upload", response_model=UploadResponse, tags=["video"], summary="Upload Video", description="Upload a video file for subtitle generation.")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file, store it, and return assigned video_id."""
    # Assign a UUID for video_id
    video_id = str(uuid.uuid4())
    uploaded_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_{file.filename}")
    with open(uploaded_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    # Store entry with empty formats
    save_history(video_id, file.filename, [])
    return UploadResponse(video_id=video_id, filename=file.filename, status_text="uploaded")

# PUBLIC_INTERFACE
@app.post("/generate_subtitles", response_model=GenerateResponse, tags=["video"], summary="Generate subtitles", description="Generate SRT/VTT subtitles for a video using OpenAI transcription.")
async def generate_subtitles(request: GenerateRequest):
    """Trigger subtitle generation for a previously uploaded video by video_id."""
    video_id = request.video_id
    sub_format = request.format.lower()
    if sub_format not in ("srt", "vtt"):
        raise HTTPException(status_code=400, detail="format must be 'srt' or 'vtt'")
    # Find uploaded video file
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.startswith(video_id+"_"):
            video_filename = fname
            break
    else:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    audio_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.wav")
    try:
        extract_audio(video_path, audio_path)
        segments = call_openai_transcription(audio_path)
        if sub_format == "srt":
            sub_content = segments_to_srt(segments)
            ext = "srt"
        else:
            sub_content = segments_to_vtt(segments)
            ext = "vtt"
        subfile_path = os.path.join(SUBTITLE_FOLDER, f"{video_id}.{ext}")
        with open(subfile_path, "w", encoding="utf-8") as f:
            f.write(sub_content)
        # Update formats in DB
        db_path = os.getenv("DB_PATH", "video_subtitles.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT formats FROM subtitles WHERE video_id = ?",
                (video_id,)
            )
            res = cursor.fetchone()
            formats = []
            if res:
                try:
                    formats = json.loads(res[0])
                except Exception:
                    formats = []
            if ext not in formats:
                formats.append(ext)
            cursor.execute(
                "UPDATE subtitles SET formats = ? WHERE video_id = ?",
                (json.dumps(formats), video_id)
            )
            conn.commit()
        return GenerateResponse(video_id=video_id, format=ext, status_text="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# PUBLIC_INTERFACE
@app.get(
    "/subtitles/{video_id}",
    response_class=Response,
    tags=["subtitles"],
    summary="Download subtitle",
    description="Download SRT or VTT subtitle file for a video."
)
async def download_subtitle(
    video_id: str,
    format: Optional[str] = "srt"
):
    """Download the subtitle file of requested format (srt or vtt) for a video."""
    ext = format.lower()
    if ext not in ("srt", "vtt"):
        raise HTTPException(status_code=400, detail="format must be srt or vtt")
    subfile_path = os.path.join(SUBTITLE_FOLDER, f"{video_id}.{ext}")
    if not os.path.exists(subfile_path):
        raise HTTPException(status_code=404, detail="Subtitle file not found")
    media_type = "text/vtt" if ext == "vtt" else "application/x-subrip"
    filename = f"{video_id}.{ext}"
    return FileResponse(subfile_path, media_type=media_type, filename=filename)

# PUBLIC_INTERFACE
@app.get(
    "/history",
    response_model=SubtitleHistoryResponse,
    tags=["subtitles"],
    summary="Subtitle history",
    description="Get history of uploaded videos and generated subtitles."
)
async def subtitles_history():
    """Return subtitle generation/download history."""
    subtitles = get_history()
    return SubtitleHistoryResponse(subtitles=subtitles)

# Swagger UI Usage notes for WebSocket/doc completeness
@app.get(
    "/docs/websocket_usage",
    tags=["subtitles"],
    include_in_schema=True,
    summary="WebSocket usage (N/A)",
    description="Real-time API not implemented; interaction is RESTful only."
)
def websocket_usage_note():
    """Inform clients that subtitles API supports only REST, not WebSocket."""
    return {"note": "This API only supports REST endpoints; no WebSocket endpoints."}
