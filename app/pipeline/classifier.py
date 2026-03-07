import json
import asyncio
import base64
from typing import List, Dict, Optional, Callable
from openai import AsyncOpenAI
from .prompts import CLASSIFY_PROMPT

MODEL_NAME = "qwen/qwen3.5-35b-a3b"
N_SAMPLES = 3


async def _classify_single(
    client: AsyncOpenAI, content: str, image_data: Optional[Dict] = None
) -> List[str]:
    try:
        user_message_content = [{"type": "text", "text": content}]
        if image_data:
            # Construct base64 data URI
            b64_image = base64.b64encode(image_data["bytes"]).decode("utf-8")
            data_uri = f"data:{image_data['mime_type']};base64,{b64_image}"
            user_message_content.append(
                {"type": "image_url", "image_url": {"url": data_uri}}
            )

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": user_message_content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "classification_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "categories": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["categories"],
                        "additionalProperties": False,
                    },
                },
            },
            temperature=0.7,
        )

        reply = response.choices[0].message.content.strip()
        data = json.loads(reply)
        return data.get("categories", [])
    except Exception as e:
        print(f"Classification error: {e}")
        return []


async def classify_content(
    client: AsyncOpenAI,
    content: str,
    image_data: Optional[Dict] = None,
    emit_progress: Optional[Callable] = None,
) -> List[str]:
    """Runs classification N times and takes the majority vote for categories."""
    results = []

    # Run sequentially or in parallel? If we want verbose progress per vote, sequential might be better
    # but gather is faster. Let's do gather and update progress as they finish.
    tasks = [_classify_single(client, content, image_data) for _ in range(N_SAMPLES)]

    if emit_progress:
        # We can wrap tasks to report progress as they finish
        completed_count = 0

        async def wrap_task(t):
            res = await t
            nonlocal completed_count
            completed_count += 1
            await emit_progress(
                {
                    "stage": "classifying",
                    "message": f"Klasifikasi vote {completed_count}/{N_SAMPLES} selesai...",
                }
            )
            return res

        results = await asyncio.gather(*(wrap_task(t) for t in tasks))
    else:
        results = await asyncio.gather(*tasks)

    category_counts = {}
    for res in results:
        for cat in set(res):
            category_counts[cat] = category_counts.get(cat, 0) + 1

    threshold = N_SAMPLES // 2 + 1
    final_categories = [
        cat for cat, count in category_counts.items() if count >= threshold
    ]

    allowed = {
        "Provokasi",
        "SARA",
        "Separatisme",
        "Disinformasi",
        "Ujaran Kebencian",
        "Hoaks",
        "Penghinaan",
        "Makar",
        "Ancaman",
        "Pelanggaran Keamanan Informasi",
        "Kekerasan",
        "Penistaan Agama",
        "Misinformasi",
    }

    return [c for c in final_categories if c in allowed]

