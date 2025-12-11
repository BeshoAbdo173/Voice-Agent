**Project: Voice Agent Monorepo**

Overview
- `rag_engine/` - FastAPI service providing a LangChain agentic RAG backend.
- `twilio/` - (existing) Twilio realtime voice agent server (WebSocket + Media Streams).
- `frontend/` - Streamlit chat UI that calls `rag_engine`.

Quickstart (local)
1. Copy `.env.example` to `.env` and fill `OPENAI_API_KEY` and `MODEL_NAME`.
2. Start rag_engine locally (recommended in a venv):

```powershell
cd "C:/Users/Bishoy Aboelsaad/Learning/Voice Agent/rag_engine"
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. Start the Twilio server (existing `twilio/` folder). If it has a local runner:

```powershell
cd "C:/Users/Bishoy Aboelsaad/Learning/Voice Agent/twilio"
# follow existing README.md inside twilio/
```

4. Start frontend (Streamlit):

```powershell
cd "C:/Users/Bishoy Aboelsaad/Learning/Voice Agent/frontend"
python -m pip install -r requirements.txt
streamlit run main.py
```

Docker Compose
- You can run everything with Docker Compose (requires a Dockerfile in `twilio/`):

```powershell
docker-compose up --build
```

Notes
- `rag_engine` builds a FAISS index under `rag_engine/index/` from docs saved via `/ingest`.
- The agent exposes `/chat` which uses LangChain tools: RAG Search, Trigger Call, Create Referral.
- A small Twilio HTTP client wrapper is in `rag_engine/clients/twilio_client.py` and used by agent tools.

Next steps / Improvements
- Add streaming (SSE) responses on the backend and streaming client in Streamlit.
- Harden intent/tool output parsing if structured responses are required.
- Add authentication and persistence for referrals (database).

