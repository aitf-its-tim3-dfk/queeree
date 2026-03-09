import os
import json
import asyncio
import mimetypes
from sanic import Sanic
from sanic.worker.manager import WorkerManager
from sanic.response import json as json_response, file
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pipeline import analyze_content, search_queue
from pipeline.law_retriever import local_law_retriever
from config import set_config, PipelineConfig

# Load environment variables from .env file
load_dotenv()

WorkerManager.THRESHOLD = 900

app = Sanic("OpenRouterApp")
app.config.RESPONSE_TIMEOUT = 900
app.config.KEEP_ALIVE_TIMEOUT = 900

# Serve the static UI files (HTML, CSS, JS)
app.static("/static", "./static")

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
    # Support both JSON and Multipart Form Data
    content_type = request.content_type or ""
    if content_type.startswith("multipart/form-data") or content_type.startswith(
        "application/x-www-form-urlencoded"
    ):
        content = request.form.get("content", "")
        image_file = request.files.get("image")
        image_data = None
        if image_file:
            mime_type = image_file.type
            if mime_type == "application/octet-stream" and image_file.name:
                guessed_type, _ = mimetypes.guess_type(image_file.name)
                if guessed_type:
                    mime_type = guessed_type
            
            # Fallback to a common image type if it's still generic or unknown
            if mime_type == "application/octet-stream" or not mime_type:
                mime_type = "image/jpeg"

            image_data = {
                "bytes": image_file.body,
                "mime_type": mime_type,
            }

        config_str = request.form.get("config", "{}")
        try:
            config_data = json.loads(config_str)
        except json.JSONDecodeError:
            config_data = {}

    else:
        try:
            data = request.json or {}
        except Exception:
            data = {}
        content = data.get("content", "")
        image_data = None
        config_data = data.get("config", {})

    if not content and not image_data:
        return json_response({"error": "Content or image is required"}, status=400)

    if config_data:
        try:
            cfg = PipelineConfig(**config_data)
            set_config(cfg)
        except Exception as e:
            print(f"Config parsing warning: {e}")

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
            client, content, image_data=image_data, emit_progress=progress_callback
        )
        await res.send(
            f"data: {json.dumps({'type': 'result', 'data': final_result})}\n\n"
        )
    except Exception as e:
        await res.send(f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n")

    await res.eof()


if __name__ == "__main__":
    # Run the Sanic app locally
    app.run(host="0.0.0.0", port=8000, dev=True)
