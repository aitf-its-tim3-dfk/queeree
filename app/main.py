import os
from sanic import Sanic
from sanic.response import json, file
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

app = Sanic("OpenRouterApp")

# Serve the static UI files (HTML, CSS, JS)
app.static("/static", "./static")

# Initialize OpenAI Async Client pointing to OpenRouter
# This relies on the openai python SDK which supports custom base URLs
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "dummy-key-if-not-set"),
)

@app.route("/")
async def index(request):
    """Serves the main frontend UI."""
    return await file("./static/index.html")

@app.post("/api/query")
async def handle_query(request):
    """Receives user query from UI, sends to OpenRouter, and returns response."""
    try:
        data = request.json
        user_query = data.get("query")
        
        if not user_query:
            return json({"error": "Query is required"}, status=400)
        
        # Call the OpenRouter API
        # By default we use 'openrouter/auto', but you can change this to any supported model,
        # e.g., 'anthropic/claude-3-haiku', 'google/gemini-2.5-pro', or 'meta-llama/llama-3-8b-instruct'
        response = await client.chat.completions.create(
            model="openrouter/auto",
            messages=[
                {"role": "user", "content": user_query}
            ],
            extra_headers={
                "HTTP-Referer": "http://localhost:8000", # Optional, for OpenRouter analytics
                "X-Title": "Sanic API App", # Optional, for OpenRouter analytics
            }
        )
        
        reply = response.choices[0].message.content
        return json({"response": reply, "success": True})
        
    except Exception as e:
        return json({"error": str(e), "success": False}, status=500)

if __name__ == "__main__":
    # Run the Sanic app locally
    app.run(host="127.0.0.1", port=8000, dev=True)
