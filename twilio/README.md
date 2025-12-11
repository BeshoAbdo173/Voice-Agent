Twilio Python server (migrated)

Run (PowerShell):

```powershell
cd "c:\Users\Bishoy Aboelsaad\Learning\CNTXT\twilio"
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
# Create .env from provided .env and/or edit values
uvicorn main:app --host 0.0.0.0 --port 3000
```

Notes:
- Set your Twilio phone number's Messaging webhook to `https://<PUBLIC_URL>/sms-webhook`.
- For Voice Media Streams, set the Voice webhook to `https://<PUBLIC_URL>/twiml/media`.
- The server exposes a WebSocket `/media` for Twilio Media Streams.

Integration with Twilio Voice AI assistant
- See `twilio/voice_ai_integration.py` for a helper scaffold to integrate the Twilio blog realtime assistant (OpenAI Realtime API). Replace placeholders with your OpenAI Realtime and Twilio credentials.
