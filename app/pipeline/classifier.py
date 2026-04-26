import json
import asyncio
import base64
from typing import List, Dict, Optional, Callable, Tuple
from openai import AsyncOpenAI
from .prompts import CLASSIFY_PROMPT, construct_grounded_prompt
import config


async def _classify_single(
    client: AsyncOpenAI, content: str, image_data: Optional[Dict] = None
) -> Tuple[List[str], bool]:
    user_message_content = [{"type": "text", "text": content}]
    if image_data:
        # Construct base64 data URI
        b64_image = base64.b64encode(image_data["bytes"]).decode("utf-8")
        mime_type = image_data.get("mime_type", "image/jpeg")
        if mime_type == "application/octet-stream":
            mime_type = "image/jpeg"
        data_uri = f"data:{mime_type};base64,{b64_image}"
        user_message_content.append(
            {"type": "image_url", "image_url": {"url": data_uri}}
        )

    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model=config.get_config_val("classifier_model_name"),
                messages=[
                    {"role": "system", "content": construct_grounded_prompt(CLASSIFY_PROMPT)},
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
                                "categories": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "needs_verification": {"type": "boolean"},
                            },
                            "required": ["categories", "needs_verification"],
                            "additionalProperties": False,
                        },
                    },
                },
                temperature=0.7,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
                **config.get_llm_kwargs("classifier"),
            )

            content_str = response.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - Classifier] LLM Response: {content_str}")

            if content_str is None:
                raise ValueError("Model returned None content")

            reply = content_str.strip()
            data = json.loads(reply)
            return data.get("categories", []), data.get("needs_verification", False)
        except Exception as e:
            print(f"Classification attempt {attempt+1} error: {e}")
            if attempt == 2:
                return [], False
            await asyncio.sleep(1)


async def classify_content(
    client: AsyncOpenAI,
    content: str,
    image_data: Optional[Dict] = None,
    emit_progress: Optional[Callable] = None,
) -> Tuple[List[str], bool, Dict[str, int]]:
    """Runs classification N times and takes the majority vote for categories."""
    results = []

    # Run sequentially or in parallel? If we want verbose progress per vote, sequential might be better
    # but gather is faster. Let's do gather and update progress as they finish.
    n_samples = config.get_config_val("classifier_n_samples")
    tasks = [_classify_single(client, content, image_data) for _ in range(n_samples)]

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
                    "message": f"Klasifikasi vote {completed_count}/{n_samples} selesai...",
                }
            )
            return res

        results = await asyncio.gather(*(wrap_task(t) for t in tasks))
    else:
        results = await asyncio.gather(*tasks)

    category_counts = {}
    verification_votes = 0

    for res in results:
        cats, needs_v = res
        for cat in set(cats):
            category_counts[cat] = category_counts.get(cat, 0) + 1
        if needs_v:
            verification_votes += 1

    threshold = n_samples // 2 + 1
    final_categories = [
        cat for cat, count in category_counts.items() if count >= threshold
    ]
    final_needs_verification = verification_votes >= threshold

    allowed = {
        "Disinformasi",
        "Fitnah",
        "Ujaran Kebencian",
    }

    return [c for c in final_categories if c in allowed], final_needs_verification, category_counts
