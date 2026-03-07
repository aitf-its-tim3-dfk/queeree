import json
import asyncio
import base64
from typing import Dict, Any, List, Optional, Callable
from difflib import SequenceMatcher
from openai import AsyncOpenAI
import config

from .prompts import (
    LAW_TEXT_ANALYZER_PROMPT,
    LAW_IMAGE_ANALYZER_PROMPT,
    LAW_IMAGE_REASON_AGGREGATOR_PROMPT,
    LAW_IMAGE_FINAL_REASON_PROMPT
)

def find_longest_common_substring(original: str, segment: str):
    matcher = SequenceMatcher(None, original, segment)
    match = matcher.find_longest_match(0, len(original), 0, len(segment))
    if match.size > 0:
        return match.a, match.a + match.size
    return -1, -1

async def _analyze_text_single(client: AsyncOpenAI, content: str, law: Dict[str, str], run_idx: int) -> Dict[str, Any]:
    user_content = f"Law:\nPasal: {law['pasal']}\nContent: {law['description']}\n\nUser Text:\n{content}"
    
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=config.get_config_val("law_retriever_model_name"),
                messages=[
                    {"role": "system", "content": LAW_TEXT_ANALYZER_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "text_law_analysis",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
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
                                "overall_reason": {"type": "string"}
                            },
                            "required": ["segments", "overall_reason"],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.7,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
            )
            content_str = response.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - TextLaw] LLM Response: {content_str}")
                
            if content_str is None:
                raise ValueError("Model returned None content")
            
            data = json.loads(content_str.strip())
            return {"data": data, "run_idx": run_idx}
        except Exception as e:
            print(f"Text law analysis attempt {attempt+1} error: {e}")
            if attempt == 2:
                return {"data": None, "run_idx": run_idx}
            await asyncio.sleep(1)

async def analyze_text_laws(client: AsyncOpenAI, content: str, laws: List[Dict[str, str]], emit_progress: Optional[Callable]) -> List[Dict[str, Any]]:
    analyzed_laws = []
    
    # default N=3 for consistency
    n_samples = config.get_config_val("classifier_n_samples") or 3
    
    async def process_law(law):
        if emit_progress:
            await emit_progress({"stage": "processing", "message": f"Menganalisis hukum {law['pasal']} pada teks..."})
            
        tasks = [_analyze_text_single(client, content, law, i) for i in range(n_samples)]
        results = await asyncio.gather(*tasks)
        
        all_segments = []
        overall_reasons = []
        for r in results:
            if not r["data"]: continue
            run_idx = r["run_idx"]
            data = r["data"]
            overall_reasons.append(data.get("overall_reason", ""))
            
            for seg in data.get("segments", []):
                start, end = find_longest_common_substring(content, seg.get("text", ""))
                if start != -1: # valid match
                    all_segments.append({
                        "start": start,
                        "end": end,
                        "reason": seg.get("reason", ""),
                        "run_idx": run_idx
                    })
        
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
        
        # Format groups for output
        final_segments = []
        for g in groups:
            # only keep if it appeared at least once, or maybe apply threshold?
            # User suggested returning all with their score
            score = len(g["runs"])
            final_segments.append({
                "start": g["start"],
                "end": g["end"],
                "text": content[g["start"]:g["end"]],
                "score": score,
                "reason": g["reasons"][0] if g["reasons"] else ""
            })
            
        return {
            "pasal": law["pasal"],
            "description": law["description"],
            "segments": final_segments,
            "overall_reason": overall_reasons[0] if overall_reasons else ""
        }

    tasks = [process_law(law) for law in laws]
    return await asyncio.gather(*tasks)

async def _analyze_image_single(client: AsyncOpenAI, content: str, image_data: Dict[str, Any], law: Dict[str, str], run_idx: int) -> Dict[str, Any]:
    b64_image = base64.b64encode(image_data["bytes"]).decode("utf-8")
    data_uri = f"data:{image_data['mime_type']};base64,{b64_image}"
    
    user_content = [
        {"type": "text", "text": f"Law:\nPasal: {law['pasal']}\nContent: {law['description']}\n\nUser Context:\n{content}"},
        {"type": "image_url", "image_url": {"url": data_uri}}
    ]
    
    for attempt in range(3):
        try:
            res = await client.chat.completions.create(
                model=config.get_config_val("classifier_model_name"),
                messages=[
                    {"role": "system", "content": LAW_IMAGE_ANALYZER_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "image_law_analysis",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string"}
                            },
                            "required": ["reason"],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.7,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
            )
            content_str = res.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - ImageLaw single] LLM Response: {content_str}")
                
            if content_str is None:
                raise ValueError("Model returned None content")
            
            data = json.loads(content_str.strip())
            return {"reason": data.get("reason", "")}
        except Exception as e:
            print(f"Image law analysis attempt {attempt+1} error: {e}")
            if attempt == 2:
                return {"reason": None}
            await asyncio.sleep(1)

async def analyze_image_laws(client: AsyncOpenAI, content: str, image_data: Dict[str, Any], laws: List[Dict[str, str]], emit_progress: Optional[Callable]) -> List[Dict[str, Any]]:
    analyzed_laws = []
    
    n_samples = config.get_config_val("classifier_n_samples") or 3
    
    async def process_law(law):
        if emit_progress:
            await emit_progress({"stage": "processing", "message": f"Menganalisis gambar untuk hukum {law['pasal']}..."})
            
        tasks = [_analyze_image_single(client, content, image_data, law, i) for i in range(n_samples)]
        results = await asyncio.gather(*tasks)
        
        valid_reasons = [r["reason"] for r in results if r["reason"]]
        if not valid_reasons:
            return None
            
        # cluster reasons
        reason_list_text = "\n".join([f"- {r}" for r in valid_reasons])
        clustered = {valid_reasons[0]: len(valid_reasons)}
        for attempt in range(3):
            try:
                cluster_res = await client.chat.completions.create(
                    model=config.get_config_val("law_retriever_model_name"),
                    messages=[
                        {"role": "system", "content": LAW_IMAGE_REASON_AGGREGATOR_PROMPT},
                        {"role": "user", "content": reason_list_text},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "image_reason_cluster",
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
                )
                content_str = cluster_res.choices[0].message.content
                if config.get_config_val("verbose_logging"):
                    print(f"[VERBOSE - ImageLaw Cluster] LLM Response: {content_str}")
                    
                if content_str is None:
                    raise ValueError("Model returned None content")
                cluster_data = json.loads(content_str.strip())
                if "clustered_reasons" in cluster_data:
                    clustered = cluster_data["clustered_reasons"]
                    break
            except Exception as e:
                print(f"Cluster extraction attempt {attempt+1} error: {e}")
                await asyncio.sleep(1)
            
        # Generate final reason based on top clusters
        # Filter reasons by count (e.g., above threshold or just take all generated)
        top_reasons = "\n".join([f"- {r} (count: {c})" for r, c in clustered.items()])
        final_reason = "Tidak ada alasan final."
        for attempt in range(3):
            try:
                final_res = await client.chat.completions.create(
                    model=config.get_config_val("law_retriever_model_name"),
                    messages=[
                        {"role": "system", "content": LAW_IMAGE_FINAL_REASON_PROMPT},
                        {"role": "user", "content": f"Law: {law['pasal']}\nCollected reasons:\n{top_reasons}"},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "image_final_reason",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "reason": {"type": "string"}
                                },
                                "required": ["reason"],
                                "additionalProperties": False
                            }
                        }
                    },
                    temperature=0.3,
                    max_completion_tokens=config.get_config_val("max_completion_tokens"),
                )
                content_str = final_res.choices[0].message.content
                if config.get_config_val("verbose_logging"):
                    print(f"[VERBOSE - ImageLaw Final] LLM Response: {content_str}")
                    
                if content_str is None:
                    raise ValueError("Model returned None content")
                final_data = json.loads(content_str.strip())
                final_reason = final_data.get("reason", "Tidak ada alasan final.")
                break
            except Exception as e:
                print(f"Final reason generation attempt {attempt+1} error: {e}")
                if attempt == 2:
                    final_reason = "Error generating final reason."
                await asyncio.sleep(1)
            
        return {
            "pasal": law["pasal"],
            "description": law["description"],
            "segments": [],
            "overall_reason": final_reason,
            "clustered_reason_counts": clustered
        }

    tasks = [process_law(law) for law in laws]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]
