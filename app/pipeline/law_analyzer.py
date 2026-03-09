import json
import asyncio
import base64
from typing import Dict, Any, List, Optional, Callable
from difflib import SequenceMatcher
from openai import AsyncOpenAI
import config

from .prompts import (
    LAW_MULTIMODAL_ANALYZER_PROMPT,
    LAW_IMAGE_REASON_AGGREGATOR_PROMPT,
    LAW_IMAGE_FINAL_REASON_PROMPT,
    construct_grounded_prompt
)

def find_longest_common_substring(original: str, segment: str):
    matcher = SequenceMatcher(None, original, segment)
    match = matcher.find_longest_match(0, len(original), 0, len(segment))
    if match.size > 0:
        return match.a, match.a + match.size
    return -1, -1

async def _analyze_multimodal_single(
    client: AsyncOpenAI, 
    content: str, 
    law: Dict[str, str], 
    run_idx: int,
    image_data: Optional[Dict[str, Any]] = None,
    fact_check_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    
    # Build text context
    text_context = f"Law to evaluate:\nPasal: {law['pasal']}\nContent: {law['description']}\n\nUser Message/Text:\n{content}"
    if fact_check_result:
        text_context += f"\n\nFact-Checking Context:\nStatus: {fact_check_result.get('status', 'Unknown')}\nReasoning: {fact_check_result.get('reasoning', '')}"
        
    user_content = [{"type": "text", "text": text_context}]
    
    if image_data:
        b64_image = base64.b64encode(image_data["bytes"]).decode("utf-8")
        mime_type = image_data.get("mime_type", "image/jpeg")
        if mime_type == "application/octet-stream":
            mime_type = "image/jpeg"
        data_uri = f"data:{mime_type};base64,{b64_image}"
        user_content.append({"type": "image_url", "image_url": {"url": data_uri}})
    
    for attempt in range(3):
        try:
            res = await client.chat.completions.create(
                model=config.get_config_val("classifier_model_name"),
                messages=[
                    {"role": "system", "content": construct_grounded_prompt(LAW_MULTIMODAL_ANALYZER_PROMPT)},
                    {"role": "user", "content": user_content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "multimodal_law_analysis",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "is_violation": {"type": "boolean"},
                                "segments": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "reason": {"type": "string"}
                                        },
                                        "required": ["text", "reason"],
                                        "additionalProperties": False
                                    }
                                },
                                "reason": {"type": "string"}
                            },
                            "required": ["is_violation", "segments", "reason"],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.7,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
                **config.get_llm_kwargs("classifier"),
            )
            content_str = res.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - MultimodalLaw single] LLM Response: {content_str}")
                
            if content_str is None:
                raise ValueError("Model returned None content")
            
            data = json.loads(content_str.strip())
            return {"data": data, "run_idx": run_idx}
        except Exception as e:
            print(f"Multimodal law analysis attempt {attempt+1} error: {e}")
            if attempt == 2:
                return {"data": None, "run_idx": run_idx}
            await asyncio.sleep(1)

async def analyze_multimodal_laws(
    client: AsyncOpenAI, 
    content: str, 
    laws: List[Dict[str, str]], 
    emit_progress: Optional[Callable],
    image_data: Optional[Dict[str, Any]] = None,
    fact_check_result: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    
    analyzed_laws = []
    n_samples = config.get_config_val("classifier_n_samples") or 3
    
    async def process_law(law):
        if emit_progress:
            await emit_progress({"stage": "processing", "message": f"Menganalisis hukum {law['pasal']}..."})
            
        tasks = [_analyze_multimodal_single(client, content, law, i, image_data, fact_check_result) for i in range(n_samples)]
        results = await asyncio.gather(*tasks)
        
        # Voting on is_violation
        violations = sum(1 for r in results if r and r.get("data") and r["data"].get("is_violation", True))
        if violations < (n_samples / 2.0):
            return None
            
        all_segments = []
        overall_reasons = []
        valid_reasons = []
        
        for r in results:
            if not r or not r.get("data"): continue
            run_idx = r["run_idx"]
            data = r["data"]
            
            if not data.get("is_violation", True): continue
            
            overall_reasons.append(data.get("reason", ""))
            valid_reasons.append(data.get("reason", ""))
            
            for seg in data.get("segments", []):
                start, end = find_longest_common_substring(content, seg.get("text", ""))
                if start != -1: # valid match
                    all_segments.append({
                        "start": start,
                        "end": end,
                        "reason": seg.get("reason", ""),
                        "run_idx": run_idx
                    })
                    
        if not valid_reasons:
            return None
            
        # Group segments by overlap
        groups = []
        for seg in all_segments:
            matching_group = None
            for g in groups:
                if max(0, min(seg["end"], g["end"]) - max(seg["start"], g["start"])) > 0:
                    matching_group = g
                    break
            
            if matching_group:
                matching_group["start"] = min(matching_group["start"], seg["start"])
                matching_group["end"] = max(matching_group["end"], seg["end"])
                matching_group["runs"].add(seg["run_idx"])
                matching_group["reasons"].append(seg["reason"])
            else:
                groups.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "runs": {seg["run_idx"]},
                    "reasons": [seg["reason"]]
                })
                
        final_segments = []
        for g in groups:
            score = len(g["runs"])
            final_segments.append({
                "start": g["start"],
                "end": g["end"],
                "text": content[g["start"]:g["end"]],
                "score": score,
                "reason": g["reasons"][0] if g["reasons"] else ""
            })
            
        # Cluster reasons (especially useful if image is present)
        clustered = {}
        final_reason_text = overall_reasons[0] if overall_reasons else ""
        
        if len(valid_reasons) > 1:
            reason_list_text = "\n".join([f"- {r}" for r in valid_reasons[:10]]) # cap to 10 avoid huge context
            for attempt in range(3):
                try:
                    cluster_res = await client.chat.completions.create(
                        model=config.get_config_val("law_retriever_model_name"),
                        messages=[
                            {"role": "system", "content": construct_grounded_prompt(LAW_IMAGE_REASON_AGGREGATOR_PROMPT)},
                            {"role": "user", "content": reason_list_text},
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "reason_cluster",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "clustered_reasons": {
                                            "type": "object",
                                            "additionalProperties": {"type": "integer"}
                                        }
                                    },
                                    "required": ["clustered_reasons"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        temperature=0.1,
                        max_completion_tokens=config.get_config_val("max_completion_tokens"),
                        **config.get_llm_kwargs("law_retriever"),
                    )
                    content_str = cluster_res.choices[0].message.content
                    cluster_data = json.loads(content_str.strip())
                    if "clustered_reasons" in cluster_data:
                        clustered = cluster_data["clustered_reasons"]
                        
                        # Generate final cohesive reason
                        top_reasons = "\n".join([f"- {r} (count: {c})" for r, c in clustered.items()])
                        final_res = await client.chat.completions.create(
                            model=config.get_config_val("law_retriever_model_name"),
                            messages=[
                                {"role": "system", "content": construct_grounded_prompt(LAW_IMAGE_FINAL_REASON_PROMPT)},
                                {"role": "user", "content": f"Law: {law['pasal']}\nCollected reasons:\n{top_reasons}"},
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "final_reason",
                                    "strict": True,
                                    "schema": {
                                        "type": "object",
                                        "properties": {"reason": {"type": "string"}},
                                        "required": ["reason"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            temperature=0.3,
                            max_completion_tokens=config.get_config_val("max_completion_tokens"),
                            **config.get_llm_kwargs("law_retriever"),
                        )
                        final_data = json.loads(final_res.choices[0].message.content.strip())
                        final_reason_text = final_data.get("reason", final_reason_text)
                        
                        break
                except Exception as e:
                    print(f"Cluster/Final reason attempt {attempt+1} error: {e}")
                    await asyncio.sleep(1)

        return {
            "pasal": law["pasal"],
            "description": law["description"],
            "segments": final_segments,
            "overall_reason": final_reason_text,
            "clustered_reason_counts": clustered
        }

    tasks = [process_law(law) for law in laws]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]
