from openai import AsyncOpenAI
import json
import asyncio
from typing import Dict, Any, Callable

# Import pipeline components
from .classifier import classify_content
from .law_retriever import retrieve_laws
from .fact_checker import fact_check
from .retrieval import RetrievalQueue, search_queue

async def generate_final_summary(client: AsyncOpenAI, user_content: str, pipeline_result: Dict[str, Any]) -> str:
    from .prompts import PIPELINE_SUMMARY_PROMPT, construct_grounded_prompt
    import config
    
    # We want to exclude base64 or heavy items if any, but our result dict is fairly clean.
    # We can just serialize the result dict.
    clean_result = {k: v for k, v in pipeline_result.items() if k != "image_data"}
    
    context = f"User Content:\n{user_content}\n\nPipeline Results:\n{json.dumps(clean_result, indent=2, ensure_ascii=False)}"
    
    for attempt in range(3):
        try:
            res = await client.chat.completions.create(
                model=config.get_config_val("law_retriever_model_name"),
                messages=[
                    {"role": "system", "content": construct_grounded_prompt(PIPELINE_SUMMARY_PROMPT)},
                    {"role": "user", "content": context},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pipeline_final_summary",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "final_summary": {"type": "string"}
                            },
                            "required": ["final_summary"],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.3,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
                **config.get_llm_kwargs("law_retriever"),
            )
            content_str = res.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - Final Summary] LLM Response: {content_str}")
            
            if content_str is None:
                raise ValueError("Model returned None content")
                
            data = json.loads(content_str.strip())
            return data.get("final_summary", "Tidak ada ringkasan yang dihasilkan.")
        except Exception as e:
            print(f"Final summary attempt {attempt+1} error: {e}")
            await asyncio.sleep(1)
            
    return "Gagal menyusun ringkasan akhir."

async def extract_image_context(client: AsyncOpenAI, image_data: Dict[str, Any]) -> str:
    from .prompts import IMAGE_CONTEXT_EXTRACTION_PROMPT, construct_grounded_prompt
    import config
    import base64
    
    b64_image = base64.b64encode(image_data["bytes"]).decode("utf-8")
    mime_type = image_data.get("mime_type", "image/jpeg")
    if mime_type == "application/octet-stream":
        mime_type = "image/jpeg"
    data_uri = f"data:{mime_type};base64,{b64_image}"
    
    for attempt in range(3):
        try:
            res = await client.chat.completions.create(
                model=config.get_config_val("classifier_model_name"),
                messages=[
                    {"role": "system", "content": construct_grounded_prompt(IMAGE_CONTEXT_EXTRACTION_PROMPT)},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "Ekstrak informasi dari gambar ini."},
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        ]
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "image_context",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "extracted_text": {"type": "string"},
                                "visual_context": {"type": "string"}
                            },
                            "required": ["extracted_text", "visual_context"],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.3,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
                **config.get_llm_kwargs("classifier"),
            )
            content_str = res.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - Image Context Extraction] LLM Response: {content_str}")
            
            if content_str is None:
                raise ValueError("Model returned None content")
                
            data = json.loads(content_str.strip())
            
            extracted_text = data.get("extracted_text", "").strip()
            visual_context = data.get("visual_context", "").strip()
            
            context_parts = []
            if extracted_text:
                context_parts.append(f"Teks dalam gambar: {extracted_text}")
            if visual_context:
                context_parts.append(f"Konteks visual: {visual_context}")
                
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"Image context extraction attempt {attempt+1} error: {e}")
            await asyncio.sleep(1)
            
    return ""

async def analyze_content(
    client: AsyncOpenAI,
    content: str,
    image_data: Dict[str, Any] = None,
    emit_progress: Callable = None,
) -> Dict[str, Any]:
    """
    Main orchestration logic for the content moderation pipeline.
    """
    result = {
        "categories": [],
        "laws_summary": "",
        "fact_check": None,
        "is_flagged": False,
    }

    try:
        # Step 0: Image Context Extraction
        if image_data:
            if emit_progress:
                await emit_progress(
                    {"stage": "processing", "message": "Mengekstrak konteks dari gambar..."}
                )
            
            image_context = await extract_image_context(client, image_data)
            if image_context:
                content = content.strip() + f"\n\n[Konteks Gambar yang Diekstrak]:\n{image_context}"
                content = content.strip()

        # Step 1: Classification
        if emit_progress:
            await emit_progress(
                {"stage": "classifying", "message": "Menganalisis konten..."}
            )

        categories, needs_verification, category_counts = await classify_content(
            client, content, image_data=image_data, emit_progress=emit_progress
        )
        
        result["category_votes"] = category_counts

        if not categories and not needs_verification:
            if emit_progress:
                await emit_progress({"stage": "done", "message": "Selesai: Aman."})
            return result

        result["is_flagged"] = True

        # Step 2: Fact-Checking & Intention (if needed)
        if needs_verification:
            if emit_progress:
                await emit_progress(
                    {"stage": "fact_checking", "message": "Klaim faktual terdeteksi. Memulai verifikasi fakta..."}
                )

            async def fc_progress(msg: str):
                if emit_progress:
                    await emit_progress({"stage": "fact_checking", "message": msg})

            fc_result = await fact_check(client, content, search_queue, fc_progress)
            result["fact_check"] = fc_result
            
            status = fc_result.get("status")
            if status == "FALSE":
                # Check Intention
                from .intention_checker import check_intention
                intent_cat = await check_intention(client, content, fc_result.get("reasoning", ""), fc_progress)
                if intent_cat not in categories:
                    categories.append(intent_cat)
            elif status == "UNVERIFIED":
                if "Misinformasi" not in categories:
                    categories.append("Misinformasi")
            # If TRUE, we don't add any misinformation categories.
            
        result["categories"] = categories

        # Double check if categories is actually empty after fact check
        if not categories:
            if emit_progress:
                await emit_progress({"stage": "done", "message": "Selesai: Fakta terverifikasi, konten aman."})
            result["is_flagged"] = False
            return result

        # Step 3: Law Retrieval
        if emit_progress:
            await emit_progress(
                {
                    "stage": "processing",
                    "message": f"Kategori final: {', '.join(categories)}. Mengumpulkan dasar hukum...",
                }
            )

        laws_data = await retrieve_laws(
            client, content, categories, emit_progress=emit_progress
        )
        summary = laws_data.get("summary", "")
        articles = laws_data.get("articles", [])
        
        analyzed_laws = []
        if articles:
            from .law_analyzer import analyze_multimodal_laws
            analyzed_laws = await analyze_multimodal_laws(
                client, 
                content, 
                articles, 
                emit_progress, 
                image_data=image_data, 
                fact_check_result=result.get("fact_check")
            )
                
        result["laws_summary"] = summary
        result["law_analysis"] = analyzed_laws

        if emit_progress:
            await emit_progress({"stage": "processing", "message": "Menyusun ringkasan akhir..."})

        final_summary = await generate_final_summary(client, content, result)
        result["final_summary"] = final_summary

        if emit_progress:
            await emit_progress({"stage": "done", "message": "Analisis selesai."})

        return result

    except Exception as e:
        print(f"Pipeline orchestration error: {e}")
        if emit_progress:
            await emit_progress(
                {
                    "stage": "error",
                    "message": f"Terjadi kesalahan di pipeline: {str(e)}",
                }
            )
        result["error"] = str(e)
        return result

