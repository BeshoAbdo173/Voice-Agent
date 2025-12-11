import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
import json
import asyncio
from pathlib import Path
import requests
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain_openai import ChatOpenAI as LCOpenAI
    from langchain.agents import Tool, initialize_agent, AgentType
except Exception:
    # If langchain is not installed the import will fail at runtime; requirements.txt includes it.
    LCOpenAI = None
    Tool = None
    initialize_agent = None
    AgentType = None

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.1")
TWILIO_SERVER_BASE = os.getenv("TWILIO_SERVER_BASE", "http://localhost:3000")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
RAG_ENGINE_SELF_BASE = os.getenv("RAG_ENGINE_SELF_BASE", "http://127.0.0.1:8000")

app = FastAPI(title="RAG Engine")


def check_model_available(model: str):
    """Check whether the OpenAI model name is available to the API key.

    Returns: (available: bool, detail: str|None)
    """
    if not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY not set"
    try:
        url = f"https://api.openai.com/v1/models/{model}"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return True, None
        # Some APIs return 404 or 403 if model is not available
        return False, f"Model check failed: {r.status_code} {r.text}"
    except Exception as e:
        return False, f"Model check error: {e}"


@app.get("/models")
async def list_models():
    """Optional helper: list available models for the API key (proxy to OpenAI)."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")
    try:
        r = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def send_openai_chat_request(payload: dict):
    """Send a chat/completions request to OpenAI, handling parameter differences across models.

    If the API returns the known error about 'max_tokens' being unsupported, retry using
    'max_completion_tokens'. Returns (status_code, response_json_or_text).
    """
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
    except requests.exceptions.RequestException as e:
        return None, f"request_error: {e}"

    # If the model complains about `max_tokens`, retry with `max_completion_tokens`
    if r.status_code == 400:
        try:
            data = r.json()
            err = data.get("error", {})
            msg = err.get("message", "")
        except Exception:
            msg = r.text
        if "max_tokens" in msg and "max_completion_tokens" in msg or "Unsupported parameter" in msg:
            alt = payload.copy()
            if "max_tokens" in alt:
                alt_val = alt.pop("max_tokens")
                alt["max_completion_tokens"] = alt_val
            try:
                r2 = requests.post("https://api.openai.com/v1/chat/completions", json=alt, headers=headers, timeout=30)
                return r2.status_code, r2
            except requests.exceptions.RequestException as e:
                return None, f"retry_request_error: {e}"

    return r.status_code, r


class QueryRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    message: str
    # optional originator phone number (E.164 or local formats)
    from_phone: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryRequest):
    if not OPENAI_API_KEY:
        # Return a friendly message instead of raising a 500 so the frontend
        # can show a clear explanation to the user when the OpenAI key
        # isn't configured in the environment.
        return {"answer": "OPENAI_API_KEY not set. Set the OPENAI_API_KEY environment variable in the backend to enable LLM responses."}

    # prompt = f"Answer the question using your knowledge and available context:\n\n{req.query}\n\nIf you don't know, say so."
     # intent detection and set intent to call. chat POST API function and
    prompt = (
    "You are a helpful, knowledgeable assistant. "
    "Whenever in the chat the user provides a phone number; Run the intent to 'call' with the provided phone number. "
    "Use your general knowledge and reasoning abilities to answer the user. "
    "If the answer depends on external documents and no documents were provided, "
    "Do NOT say phrases like 'as an AI model'. Just answer normally."
    "still give your best general answer.\n\n"
    f"User: {req.query}"
)

    # Validate the configured model before issuing the request so we return
    # a clear error when the model name is unknown or not permitted for this key.
    ok, detail = check_model_available(MODEL_NAME)
    if not ok:
        return {"answer": f"Model '{MODEL_NAME}' is not available: {detail}. Use a model your account can access (eg. gpt-4o, gpt-3.5-turbo)."}

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
    }
    # Send request using compatibility helper that retries with alternate params when needed
    status, resp = send_openai_chat_request(payload)
    if status is None:
        return {"answer": f"LLM request failed: {resp}. Check OPENAI_API_KEY, MODEL_NAME and network connectivity."}
    if status != 200:
        # Return the API body for diagnostics
        try:
            body = resp.json()
        except Exception:
            body = resp.text if hasattr(resp, 'text') else str(resp)
        return {"answer": f"OpenAI API error: {status} {body}. Check MODEL_NAME and API access."}
    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return {"answer": text}
    except Exception as e:
        return {"answer": f"Failed to parse LLM response: {e}"}


@app.post("/intent")
async def intent(req: ChatRequest):
    """LLM-based intent detection for `call` and `referral` intents.

    Returns JSON with keys depending on intent. Example responses:
    - call: {"intent":"call","phone":"+1555123"}
    - referral: {"intent":"referral","recipient_name":"X","recipient_phone":"+1555..","message":"..."}
    - none: {"intent":"none"}

    Falls back to a heuristic if `OPENAI_API_KEY` is not set or the LLM response cannot be parsed.
    """

    def heuristic():
        text = req.message.lower()
        if "call" in text:
            import re
            m = re.search(r"(\+?\d[\d\-\s]{6,}\d)", req.message)
            phone = m.group(1) if m else None
            return {"intent": "call", "phone": phone}
        # detect simple referral phrases heuristically
        if "share" in text or "refer" in text or "referral" in text:
            import re
            # try to find a phone in the text
            m = re.search(r"(\+?\d[\d\-\s]{6,}\d)", req.message)
            phone = m.group(1) if m else None
            # crude name extraction not reliable
            return {"intent": "referral", "recipient_name": None, "recipient_phone": phone, "message": req.message}
        return {"intent": "none"}

    if not OPENAI_API_KEY:
        return heuristic()

    system_prompt = (
        "You are an intent extractor. Given a user's message, output ONLY a JSON object describing the intent.\n"
        "Fields to include (use null if not present):\n"
        "- intent: one of [\"call\", \"referral\", \"none\"]\n"
        "- phone: phone number for call intent (string or null)\n"
        "- recipient_name: name for referral intent (string or null)\n"
        "- recipient_phone: phone number for referral intent (string or null)\n"
        "- message: the message to send for referral (string or null)\n\n"
        "Respond with JSON only, no markdown or additional text. Be conservative when assigning intents."
    )

    user_prompt = f"User message: {req.message}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
        "max_tokens": 300,
    }
    # Validate model availability first
    ok, detail = check_model_available(MODEL_NAME)
    if not ok:
        return heuristic()  # fallback to heuristic if the model isn't available

    # # Validate model availability first
    # ok, detail = check_model_available(MODEL_NAME)
    # if not ok:
    #     return heuristic()  # fallback to heuristic if the model isn't available

    # Use compatibility request sender
    status, resp = send_openai_chat_request(payload)
    if status is None or status != 200:
        return heuristic()
    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(text)
        # normalize keys to a canonical schema so downstream logic is consistent
        intent_value = parsed.get("intent", parsed.get("action", "none"))
        # helper to extract phone fields from various possible keys
        def extract_phone(d, *keys):
            for k in keys:
                v = d.get(k)
                if v:
                    return v
            return None

        if intent_value == "call":
            phone = extract_phone(parsed, "phone", "phone_number", "to", "number", "recipient_phone")
            return {"intent": "call", "phone": phone}
        if intent_value == "referral":
            recipient_phone = extract_phone(parsed, "recipient_phone", "phone", "phone_number", "to", "number")
            recipient_name = parsed.get("recipient_name") or parsed.get("name") or parsed.get("recipient")
            message = parsed.get("message") or parsed.get("body") or parsed.get("text")
            return {
                "intent": "referral",
                "recipient_name": recipient_name,
                "recipient_phone": recipient_phone,
                "message": message,
            }
        return {"intent": "none"}
    except Exception:
        return heuristic()


def build_langchain_agent():
    """Construct a simple LangChain agent with tools for placing calls, sending SMS, and running the local RAG query."""
    if LCOpenAI is None:
        raise RuntimeError("LangChain/OpenAI integrations not available. Install 'langchain' and 'openai'.")

    llm = LCOpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME)

    def place_call_tool(phone: str) -> str:
        phone = (phone or "").strip()
        if not phone:
            return "No phone number provided to place_call."
        try:
            resp = requests.post(f"{TWILIO_SERVER_BASE}/make-call", json={"to": phone}, timeout=15)
            resp.raise_for_status()
            try:
                body = resp.json()
            except Exception:
                body = resp.text if hasattr(resp, 'text') else str(resp)
            return f"Call placed to {phone}: {body}"
        except requests.exceptions.RequestException as e:
            return f"Failed to place call to {phone}: {e}"

    def send_sms_tool(payload: str) -> str:
        """Send SMS. Expect `to||body` format (double-pipe separator) or JSON string {to:'..',body:'..'}."""
        # try JSON first
        to = None
        body = None
        try:
            obj = json.loads(payload)
            to = obj.get("to")
            body = obj.get("body") or obj.get("message")
        except Exception:
            if "||" in payload:
                to, body = payload.split("||", 1)
            else:
                # last resort: treat whole payload as body with unknown recipient
                body = payload

        if not to:
            return "No recipient phone ('to') provided for send_sms. Use format 'to||body' or JSON {to,body}."
        try:
            resp = requests.post(f"{TWILIO_SERVER_BASE}/send-sms", json={"to": to.strip(), "body": body}, timeout=15)
            resp.raise_for_status()
            try:
                return f"SMS sent to {to}: {resp.json()}"
            except Exception:
                return f"SMS sent to {to}: {resp.text if hasattr(resp,'text') else str(resp)}"
        except requests.exceptions.RequestException as e:
            return f"Failed to send SMS to {to}: {e}"

    def query_knowledge_tool(q: str) -> str:
        # Use the local `query` function defined in this module. It's async, so run it in a fresh loop.
        try:
            qreq = QueryRequest(query=q)
            result = asyncio.run(query(qreq))
            # result may be dict like {"answer": ...}
            if isinstance(result, dict):
                return json.dumps(result)
            return str(result)
        except Exception as e:
            return f"Failed to run query tool: {e}"

    tools = [
        Tool(name="place_call", func=place_call_tool, description="Place an outbound phone call. Input: phone number string (E.164 preferred)."),
        Tool(name="send_sms", func=send_sms_tool, description="Send an SMS via Twilio. Input: 'to||body' or JSON {to,body}.'"),
        Tool(name="query_knowledge", func=query_knowledge_tool, description="Run the RAG query tool. Input: plain text question."),
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    return agent


# Simple file-backed referral store
REF_FILE = Path(__file__).parent / "referrals.json"
def load_referrals():
    if not REF_FILE.exists():
        return {}
    try:
        return json.loads(REF_FILE.read_text())
    except Exception:
        return {}

def save_referrals(d):
    REF_FILE.write_text(json.dumps(d))


@app.post("/ingest")
async def ingest(payload: dict):
    """Simple user-based ingestion: POST {"user_id":"u1","content":"..."}

    Stores user docs under `data/{user_id}/` for later indexing.
    """
    user_id = payload.get("user_id")
    content = payload.get("content")
    if not user_id or not content:
        raise HTTPException(status_code=400, detail="Missing user_id or content")
    data_dir = Path(__file__).parent / "data" / user_id
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"doc_{uuid4().hex}.txt"
    out.write_text(content)
    return {"status": "ok", "path": str(out)}


@app.post("/referral/get")
async def referral_get(payload: dict):
    """Return referral details by id: POST {"ref_id":"..."} -> mapping or 404"""
    ref_id = payload.get("ref_id")
    if not ref_id:
        raise HTTPException(status_code=400, detail="Missing ref_id")
    refs = load_referrals()
    if ref_id not in refs:
        raise HTTPException(status_code=404, detail="referral not found")
    return refs[ref_id]

@app.post("/chat")
async def chat(req: ChatRequest):
    # 1) Run intent detection first so we can enforce mandatory call placement
    intent_resp = await intent(req)

    # If a call intent is detected, enforce that we place the call before returning.
    if intent_resp.get("intent") == "call":
        phone = intent_resp.get("phone")
        if not phone:
            return {"result": "I detected a call intent but no phone number. Please provide a number."}
        # Place the call synchronously (mandatory)
        try:
            resp = requests.post(f"{TWILIO_SERVER_BASE}/make-call", json={"to": phone}, timeout=15)
            resp.raise_for_status()
            return {"result": f"Placing call to {phone}."}
        except requests.exceptions.ConnectionError as e:
            return {
                "result": (
                    f"Failed to trigger call: cannot connect to Twilio server at {TWILIO_SERVER_BASE}. "
                    "Ensure the Twilio server is running and TWILIO_SERVER_BASE is correct (or use ngrok if exposing locally)."
                ),
                "error": str(e),
            }
        except requests.exceptions.RequestException as e:
            return {"result": f"Failed to trigger call: {e}"}

    # If referral, keep existing referral flow (requires from_phone)
    if intent_resp.get("intent") == "referral":
        origin = req.from_phone
        recipient_phone = intent_resp.get("recipient_phone")
        recipient_name = intent_resp.get("recipient_name")
        message = intent_resp.get("message") or f"{req.message}"
        if not origin:
            return {"result": "Please provide your phone number in `from_phone` so the referral can connect you."}
        if not recipient_phone:
            return {"result": "Referral detected but no recipient phone found. Please include recipient number."}

        # create referral record
        ref_id = uuid4().hex
        refs = load_referrals()
        refs[ref_id] = {
            "origin": origin,
            "recipient": recipient_phone,
            "recipient_name": recipient_name,
            "message": message,
        }
        save_referrals(refs)

        # send SMS to recipient via Twilio server
        sms_body = f"{message}\nReply 'CALL {ref_id}' to connect a call with {origin}. RefID:{ref_id}"
        try:
            resp = requests.post(f"{TWILIO_SERVER_BASE}/send-sms", json={"to": recipient_phone, "body": sms_body}, timeout=15)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            return {
                "result": (
                    f"Referral created (RefID: {ref_id}) but failed to send SMS: cannot connect to Twilio server at {TWILIO_SERVER_BASE}. "
                    "Ensure the Twilio server is running and TWILIO_SERVER_BASE is correct. The referral is stored and can be retried later."
                ),
                "ref_id": ref_id,
                "error": str(e),
            }
        except requests.exceptions.RequestException as e:
            return {"result": f"Referral created (RefID: {ref_id}) but failed to send SMS: {e}", "ref_id": ref_id}

        return {"result": f"Referral sent to {recipient_phone} (RefID: {ref_id}).", "ref_id": ref_id}

    # Otherwise: use a LangChain agent to produce an agentic response with tools available.
    try:
        agent = build_langchain_agent()
    except Exception as e:
        # If LangChain is unavailable, fall back to simple query
        q = QueryRequest(query=req.message)
        return await query(q)

    # Run the blocking agent in a thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, lambda: agent.run(req.message))
        return {"result": result}
    except Exception as e:
        return {"result": f"Agent execution failed: {e}"}
