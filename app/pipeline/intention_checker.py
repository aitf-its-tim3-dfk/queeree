import json
import asyncio
from typing import Dict, Any, Optional, Callable
from openai import AsyncOpenAI
from .prompts import INTENTION_CHECK_PROMPT, construct_grounded_prompt
import config


async def check_intention(
    client: AsyncOpenAI, 
    content: str, 
    fact_check_reasoning: str,
    emit_progress: Optional[Callable] = None
) -> str:
    """
    Analyzes a false/debunked claim to determine if it's a Hostile Disinformation campaign
    or just a generic Hoax/Fabrication.
    Returns: "Disinformasi" or "Hoaks"
    """
    if emit_progress:
        await emit_progress("Menganalisis niat dan kemungkinan manipulasi di balik konten (Disinformasi/Hoaks)...")

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=config.get_config_val("classifier_model_name"), # We can reuse the classifier model here
                messages=[
                    {"role": "system", "content": construct_grounded_prompt(INTENTION_CHECK_PROMPT)},
                    {
                        "role": "user",
                        "content": f"Content:\n{content}\n\nFact-Checker Findings:\n{fact_check_reasoning}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "intention_check",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "has_malicious_intent": {"type": "boolean"},
                                "category": {"type": "string", "enum": ["Disinformasi", "Hoaks"]},
                                "reasoning": {"type": "string"}
                            },
                            "required": ["has_malicious_intent", "category", "reasoning"],
                            "additionalProperties": False,
                        },
                    },
                },
                temperature=0.3,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
                **config.get_llm_kwargs("classifier"),
            )
            
            content_str = response.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - IntentionChecker] LLM Response: {content_str}")
                
            if content_str is None:
                raise ValueError("Model returned None content")
                
            reply = content_str.strip()
            data = json.loads(reply)
            
            cat = data.get("category", "Hoaks")
            if cat not in ["Disinformasi", "Hoaks"]:
                cat = "Hoaks"
            
            if emit_progress:
                reason = data.get('reasoning', '')
                await emit_progress(f"Hasil analisis niat: {cat}. ({reason})")
                
            return cat
        except Exception as e:
            print(f"Intention check attempt {attempt+1} error: {e}")
            if attempt == 2:
                # Default to Hoaks on failure
                return "Hoaks"
            await asyncio.sleep(1)
