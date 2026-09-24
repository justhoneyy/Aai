import os
import requests
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODELS = [m for m in [os.environ.get("GROQ_MODEL"), "openai/gpt-oss-120b", "openai/gpt-oss-20b"] if m]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not GROQ_API_KEY:
        return jsonify(error="GROQ_API_KEY is not set on the server"), 500

    messages = (request.get_json(silent=True) or {}).get("messages", [])

    # Keep only recent turns and cap total size so we stay under the free-tier TPM limit.
    CHAR_BUDGET = 12000  # rough ~4 chars/token budget for the prompt
    trimmed = []
    used = 0
    for m in reversed(messages[-20:]):
        content = (m.get("content") or "")[:4000]
        used += len(content)
        if used > CHAR_BUDGET and trimmed:
            break
        trimmed.append({"role": m.get("role", "user"), "content": content})
    trimmed.reverse()

    history = [{"role": "system", "content": "You are a helpful assistant. Keep answers concise."}] + trimmed
    last_error = "Groq API error"
    for model in MODELS:
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": history, "max_tokens": 1024},
                timeout=60,
            )
            data = r.json()
        except Exception as e:
            return jsonify(error=str(e)), 500
        if r.status_code == 200:
            return jsonify(reply=data["choices"][0]["message"]["content"], model=model)
        last_error = data.get("error", {}).get("message", "Groq API error")
        if "does not exist" not in last_error.lower():  # only retry other models for a missing-model error
            break
    return jsonify(error=last_error), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
