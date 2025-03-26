# 🎤 VozRecX: Speech-to-Text Search Assistant

> ⚠️ **Hosting Notice**: Currently not hosted on AWS Lambda due to payment verification issues during signup. Docker Deployment Available.

## 📌 Overview
VozRecX is a robust speech-to-text search assistant that processes voice input in real-time. It features noise reduction, smart autocomplete suggestions, and WebSocket support for live speech processing.

## ✨ Key Features
- Voice-to-text conversion using OpenAI Whisper
- Background noise reduction with RNNoise
- AI-powered search autocompletion
- Real-time speech processing via WebSockets
- Docker deployment support

## 🛠️ Technology Stack
- FastAPI: Modern, fast web framework for building APIs
- OpenAI Whisper: Robust speech recognition model
- RNNoise: Audio noise reduction
- Redis: In-memory data store
- WebSockets
- Docker
- SentenceTransformer: For semantic search
- PyDub: Audio processing

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- Redis server
- FFmpeg
- CUDA-capable GPU (optional)

## 📦 Deployment
```bash
docker-compose up -d
```

### Installation
```bash
# Clone repository
git clone https://github.com/AbhishekNaik1112/VozRecX.git

# Install dependencies
pip install -r requirements.txt

# Start Redis server
docker run -d -p 6379:6379 redis

# Start the API server
uvicorn src.main:app --reload
```

## 📚 API Documentation
Available endpoints:
- `POST /api/voice-to-text-noisy`: Convert audio to text
- `GET /api/autocomplete`: Get search suggestions
- `WS /ws/speech-to-search`: Real-time speech processing

Visit `http://localhost:8000/docs` for Swagger documentation.

## 🔧 Technical Implementation

### Design Choices
- **Whisper over DeepSpeech**: More accurate, smaller models, better noise handling, and actively maintained.  
- **Redis over Pinecone**: Simple setup, low latency, cost-effective, and great for AI-powered search.

### Challenges & Solutions
1. **Real-time Processing using websockets?**: Referred Github, docs and fastapi+ws example codes and some videos. 
2. **Choice in Noise Reduction Library?**: Used noisereduce library for easier implementation
3. **How to Rank text?**: Used hybrid scoring (70% similarity, 30% popularity)

### Performance Optimizations
- Better audio buffer size?
- Redis caching for repeating words

## 🔮 Future Improvements
1. Rate Limiting
2. Language Supports
