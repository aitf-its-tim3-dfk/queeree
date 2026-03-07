import os
import json
import asyncio
from sanic import Sanic
from sanic.response import json as json_response, file
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pipeline import analyze_content, search_queue
from pipeline.law_retriever import local_law_retriever

# Load environment variables from .env file
load_dotenv()

app = Sanic("OpenRouterApp")
app.config.RESPONSE_TIMEOUT = 300

# Serve the static UI files (HTML, CSS, JS)
app.static("/static", "./static")

# Initialize OpenAI Async Client pointing to OpenRouter
# This relies on the openai python SDK which supports custom base URLs
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "dummy-key-if-not-set"),
)


@app.before_server_start
async def setup(app, loop):
    """Initialize background tasks and pre-load heavy models."""
    local_law_retriever.load()
    await search_queue.start()


@app.after_server_stop
async def teardown(app, loop):
    """Clean up background tasks on shutdown."""
    await search_queue.stop()


@app.route("/")
async def index(request):
    """Serves the main frontend UI."""
    return await file("./static/index.html")


@app.post("/api/analyze")
async def analyze_endpoint(request):
    """Receives content from UI, runs the moderation pipeline, and streams back progress."""
    data = request.json
    content = data.get("content")

    if not content:
        return json_response({"error": "Content is required"}, status=400)

    res = await request.respond(content_type="text/event-stream")

    async def progress_callback(progress_data):
        try:
            await res.send(
                f"data: {json.dumps({'type': 'progress', 'data': progress_data})}\n\n"
            )
        except BaseException:
            pass  # Connection likely dropped

    try:
        final_result = await analyze_content(
            client, content, emit_progress=progress_callback
        )
        await res.send(
            f"data: {json.dumps({'type': 'result', 'data': final_result})}\n\n"
        )
    except Exception as e:
        await res.send(f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n")

    await res.eof()


if __name__ == "__main__":
    # Run the Sanic app locally
    app.run(host="127.0.0.1", port=8000, dev=True)
