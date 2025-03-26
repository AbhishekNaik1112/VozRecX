import time
import torch
import numpy as np
import io
import base64
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
import whisper
import uvicorn
from pydub import AudioSegment
import noisereduce as nr
import redis
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu" 
print(f"Using device: {device}")
model = whisper.load_model("tiny", device=device)

redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

default_candidates = [
    "find me a red dress",
    "find me a jacket",
    "find me shoes",
    "find me a bag",
    "find me a watch",
]


def get_candidates():
    """
    Retrieve candidate suggestions from Redis sorted set.
    If not present, initialize with default candidates at a popularity score of 1.
    """
    candidates = redis_client.zrange("autocomplete_candidates", 0, -1)
    if not candidates:
        for candidate in default_candidates:
            redis_client.zadd("autocomplete_candidates", {candidate: 1})
        candidates = default_candidates
    return candidates

@app.get("/")
async def root():
    return {"message": "Welcome to VozRecX!"}

@app.post("/api/voice-to-text-noisy")
async def voice_to_text_noisy(audio: UploadFile = File(...)):
    """
    Upload a noisy audio file, perform noise reduction, and convert speech to text.
    Saves the denoised transcription in Redis for use in autocomplete.
    Returns both the raw (noisy) and denoised transcriptions along with processing times.
    """
    allowed_extensions = {".wav", ".mp3", ".m4a"}
    file_ext = audio.filename.split(".")[-1].lower()

    if f".{file_ext}" not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail="Invalid file format. Use .wav, .mp3, or .m4a"
        )

    try:
        total_start = time.time()
        audio_bytes = await audio.read()
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_ext)
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        audio_np = (
            np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
        )

        start_noisy = time.time()
        result_noisy = model.transcribe(audio_np)
        time_noisy = round(time.time() - start_noisy, 3)

        start_denoise = time.time()
        denoised_audio_np = nr.reduce_noise(y=audio_np, sr=16000)
        time_denoise = round(time.time() - start_denoise, 3)

        start_denoised = time.time()
        result_denoised = model.transcribe(denoised_audio_np)
        time_denoised = round(time.time() - start_denoised, 3)

        total_time = round(time.time() - total_start, 3)

        denoised_text = result_denoised["text"].strip()
        if denoised_text:
            redis_client.zincrby("autocomplete_candidates", 1, denoised_text)

        return {
            "noisy_transcription": result_noisy["text"],
            "noisy_processing_time_sec": time_noisy,
            "denoised_transcription": denoised_text,
            "denoise_processing_time_sec": time_denoise,
            "denoised_transcription_time_sec": time_denoised,
            "total_processing_time_sec": total_time,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")


@app.get("/api/autocomplete")
async def autocomplete(q: str = Query(...)):
    """
    GET endpoint for autocomplete suggestions.
    Example: /api/autocomplete?q=find+me
    Combines AI embeddings with stored popularity in Redis to rank suggestions.
    """
    query_embedding = embedding_model.encode(q, convert_to_tensor=True)

    candidates = get_candidates()
    candidate_embeddings = embedding_model.encode(candidates, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

    candidate_popularity = []
    for candidate in candidates:
        score = redis_client.zscore("autocomplete_candidates", candidate) or 1.0
        candidate_popularity.append(score)
    candidate_popularity = np.array(candidate_popularity)

    max_pop = candidate_popularity.max() if candidate_popularity.max() > 0 else 1.0
    popularity_norm = candidate_popularity / max_pop

    final_scores = 0.7 * cosine_scores.cpu().numpy() + 0.3 * popularity_norm

    sorted_indices = np.argsort(-final_scores)
    sorted_candidates = [candidates[i] for i in sorted_indices]

    return sorted_candidates


@app.websocket("/ws/speech-to-search")
async def websocket_speech_to_search(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speech-to-text and autocomplete.
    Accepts audio chunks, processes them continuously, and returns transcription and autocomplete results.
    """
    await websocket.accept()
    
    audio_buffer = np.array([], dtype=np.float32)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if "audio_chunk" not in data:
                await websocket.send_json({"error": "Missing audio_chunk in request"})
                continue
                
            try:
                audio_bytes = base64.b64decode(data["audio_chunk"])
                
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                
                chunk_np = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
                
                audio_buffer = np.concatenate([audio_buffer, chunk_np])
                
                if len(audio_buffer) >= 16000:
                    denoised_audio = nr.reduce_noise(y=audio_buffer, sr=16000)
                    
                    result = model.transcribe(denoised_audio)
                    transcription = result["text"].strip()
                    
                    suggestions = []
                    if transcription:
                        redis_client.zincrby("autocomplete_candidates", 1, transcription)
                        
                        query_embedding = embedding_model.encode(transcription, convert_to_tensor=True)
                        candidates = get_candidates()
                        
                        if candidates:
                            candidate_embeddings = embedding_model.encode(candidates, convert_to_tensor=True)
                            
                            cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
                            
                            candidate_popularity = []
                            for candidate in candidates:
                                score = redis_client.zscore("autocomplete_candidates", candidate) or 1.0
                                candidate_popularity.append(score)
                            candidate_popularity = np.array(candidate_popularity)
                            
                            max_pop = candidate_popularity.max() if candidate_popularity.max() > 0 else 1.0
                            popularity_norm = candidate_popularity / max_pop
                            
                            final_scores = 0.7 * cosine_scores.cpu().numpy() + 0.3 * popularity_norm
                            
                            sorted_indices = np.argsort(-final_scores)
                            suggestions = [candidates[i] for i in sorted_indices]
                    
                    await websocket.send_json({
                        "transcription": transcription,
                        "suggestions": suggestions[:5] 
                    })
                    
                    if len(audio_buffer) > 32000:
                        audio_buffer = audio_buffer[-32000:]
                
            except Exception as e:
                await websocket.send_json({"error": f"Error processing audio: {str(e)}"})
    
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)