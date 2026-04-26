import os
import json
import asyncio
import mimetypes
import httpx
from textwrap import dedent
from sanic import Sanic
from sanic.worker.manager import WorkerManager
from sanic.response import json as json_response, file
from sanic_ext import openapi
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pipeline import analyze_content, search_queue
from pipeline.reranker import load as load_reranker
from config import set_config, PipelineConfig

# Load environment variables from .env file
load_dotenv()

WorkerManager.THRESHOLD = 900

app = Sanic("OpenRouterApp")
app.config.RESPONSE_TIMEOUT = 900
app.config.KEEP_ALIVE_TIMEOUT = 900

# OpenAPI metadata
app.ext.openapi.describe(
    "Queeree Content Moderation API",
    version="1.0.0",
    description=dedent("""
        AI-powered content moderation and fact-checking pipeline.

        ## Overview
        This API analyzes text and/or image content through a multi-stage pipeline:
        1. **Classification** — categorizes the content
        2. **Fact-checking** — verifies claims via web search
        3. **Final analysis** — produces a moderation verdict

        ## Response Format
        The `/api/analyze` endpoint uses **Server-Sent Events (SSE)** to stream
        progress updates in real-time, followed by a final result payload.

        Each SSE message is a JSON object with a `type` field:
        - `progress` — intermediate pipeline status
        - `result` — final analysis output
        - `error` — something went wrong
    """),
)

MEDIA_FETCH_TIMEOUT = 30  # seconds
MAX_MEDIA_SIZE = 20 * 1024 * 1024  # 20 MB


async def fetch_media_from_url(url: str) -> dict | None:
    """Download media from a URL and return an image_data dict compatible with the pipeline."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=MEDIA_FETCH_TIMEOUT) as http:
            resp = await http.get(url)
            resp.raise_for_status()

            content_length = resp.headers.get("content-length")
            if content_length and int(content_length) > MAX_MEDIA_SIZE:
                raise ValueError(f"Media too large ({int(content_length)} bytes, max {MAX_MEDIA_SIZE})")

            media_bytes = resp.content
            if len(media_bytes) > MAX_MEDIA_SIZE:
                raise ValueError(f"Media too large ({len(media_bytes)} bytes, max {MAX_MEDIA_SIZE})")

            # Determine MIME type: prefer Content-Type header, fall back to URL guessing
            mime_type = resp.headers.get("content-type", "").split(";")[0].strip()
            if not mime_type or mime_type == "application/octet-stream":
                guessed, _ = mimetypes.guess_type(url.split("?")[0])
                mime_type = guessed or "image/jpeg"

            return {"bytes": media_bytes, "mime_type": mime_type}
    except httpx.HTTPStatusError as e:
        raise ValueError(f"Failed to fetch media: HTTP {e.response.status_code}") from e
    except httpx.RequestError as e:
        raise ValueError(f"Failed to fetch media: {e}") from e

# Serve the static UI files (HTML, CSS, JS)
app.static("/static", "./static")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "dummy-key-if-not-set"),
)


@app.before_server_start
async def setup(app):
    """Initialize background tasks."""
    await search_queue.start()


@app.after_server_start
async def load_models(app):
    """Pre-load heavy models after server has started."""
    load_reranker()


@app.after_server_stop
async def teardown(app):
    """Clean up background tasks on shutdown."""
    await search_queue.stop()


@app.route("/")
@openapi.exclude()
async def index(request):
    """Serves the main frontend UI."""
    return await file("./static/index.html")


@app.post("/api/analyze")
@openapi.summary("Analyze content")
@openapi.description(dedent("""
    Run the full moderation pipeline on text and/or image content.

    **Accepts** either `multipart/form-data` or `application/json`.

    ### Multipart Form Fields
    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | `content` | string | conditional | Text content to analyze |
    | `image` | file | conditional | Image/media file to analyze |
    | `media_url` | string | no | URL to fetch media from (used when no file is uploaded) |
    | `config` | string (JSON) | no | Pipeline config overrides (JSON object) |

    At least one of `content` or `image`/`media_url` is required.

    ### JSON Body
    ```json
    {
      "content": "text to analyze",
      "media_url": "https://example.com/image.jpg",
      "config": {
        "classifier_model_name": "qwen/qwen3.5-27b",
        "reasoning_effort": "low"
      }
    }
    ```

    ### Config Overrides
    | Key | Type | Description |
    |-----|------|-------------|
    | `classifier_model_name` | string | Model for classification stage |
    | `fact_checker_model_name` | string | Model for fact-checking stage |
    | `classifier_n_samples` | int | Number of classification samples |
    | `fact_checker_n_samples` | int | Number of fact-checker samples |
    | `fact_checker_max_loops` | int | Max search loops for fact-checker |
    | `max_completion_tokens` | int | Max tokens per LLM call |
    | `reasoning_effort` | string | Global reasoning effort (`low`, `medium`, `high`) |
    | `verbose_logging` | bool | Enable verbose pipeline logs |

    ### Response (SSE stream)
    The response is a `text/event-stream`. Each event is a JSON line:
    - `{"type": "progress", "data": {...}}` — pipeline progress
    - `{"type": "result", "data": {...}}` — final analysis
    - `{"type": "error", "data": "..."}` — error message
"""))
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Text content to analyze",
                    },
                    "media_url": {
                        "type": "string",
                        "format": "uri",
                        "description": "URL to fetch media from (used when no file is uploaded)",
                    },
                    "config": {
                        "type": "object",
                        "description": "Pipeline config overrides",
                        "properties": {
                            "classifier_model_name": {
                                "type": "string",
                                "description": "Model for classification stage",
                            },
                            "fact_checker_model_name": {
                                "type": "string",
                                "description": "Model for fact-checking stage",
                            },
                            "classifier_n_samples": {
                                "type": "integer",
                                "description": "Number of classification samples",
                            },
                            "fact_checker_n_samples": {
                                "type": "integer",
                                "description": "Number of fact-checker samples",
                            },
                            "fact_checker_max_loops": {
                                "type": "integer",
                                "description": "Max search loops for fact-checker",
                            },
                            "max_completion_tokens": {
                                "type": "integer",
                                "description": "Max tokens per LLM call",
                            },
                            "reasoning_effort": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "description": "Global reasoning effort level",
                            },
                            "verbose_logging": {
                                "type": "boolean",
                                "description": "Enable verbose pipeline logs",
                            },
                        },
                    },
                },
            },
        },
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Text content to analyze",
                    },
                    "image": {
                        "type": "string",
                        "format": "binary",
                        "description": "Image/media file to analyze",
                    },
                    "media_url": {
                        "type": "string",
                        "format": "uri",
                        "description": "URL to fetch media from (used when no file is uploaded)",
                    },
                    "config": {
                        "type": "string",
                        "description": "Pipeline config overrides (JSON string)",
                    },
                },
            },
        },
    },
    description="Content to analyze — provide text, an image, a media URL, or a combination",
)
@openapi.tag("Analysis")
@openapi.response(200, description="SSE stream of progress updates and final result")
@openapi.response(400, description="Missing content/image or invalid media URL")
async def analyze_endpoint(request):
    """Receives content from UI, runs the moderation pipeline, and streams back progress.
    
    Accepts an optional `media_url` field — when provided (and no file is uploaded),
    the server will fetch the media from that URL automatically.
    """
    # Support both JSON and Multipart Form Data
    content_type = request.content_type or ""
    if content_type.startswith("multipart/form-data") or content_type.startswith(
        "application/x-www-form-urlencoded"
    ):
        content = request.form.get("content", "")
        media_url = request.form.get("media_url", "")
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
        elif media_url:
            try:
                image_data = await fetch_media_from_url(media_url)
            except ValueError as e:
                return json_response({"error": str(e)}, status=400)

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
        media_url = data.get("media_url", "")
        image_data = None
        config_data = data.get("config", {})

        if media_url and not image_data:
            try:
                image_data = await fetch_media_from_url(media_url)
            except ValueError as e:
                return json_response({"error": str(e)}, status=400)

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
