import os
import requests
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not GROQ_API_KEY:
        return jsonify(error="GROQ_API_KEY is not set on the server"), 500

    messages = (request.get_json(silent=True) or {}).get("messages", [])[-20:]
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": "You are a helpful assistant."}] + messages,
    }
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        data = r.json()
        if r.status_code != 200:
            return jsonify(error=data.get("error", {}).get("message", "Groq API error")), 502
        return jsonify(reply=data["choices"][0]["message"]["content"])
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
