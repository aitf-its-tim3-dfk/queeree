from openai import AsyncOpenAI
import json
from typing import Dict, Any, Callable

# Import pipeline components
from .classifier import classify_content
from .law_retriever import retrieve_laws
from .fact_checker import fact_check
from .retrieval import RetrievalQueue, search_queue


async def analyze_content(
    client: AsyncOpenAI, content: str, emit_progress: Callable = None
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
        # Step 1: Classification
        if emit_progress:
            await emit_progress(
                {"stage": "classifying", "message": "Klasifikasi konten..."}
            )

        categories = await classify_content(client, content)
        result["categories"] = categories

        if not categories:
            if emit_progress:
                await emit_progress({"stage": "done", "message": "Selesai: Aman."})
            return result

        result["is_flagged"] = True

        # Step 2: Parallelized Law Retrieval and Fact-Checking
        if emit_progress:
            await emit_progress(
                {
                    "stage": "processing",
                    "message": f"Ditemukan pelanggaran: {', '.join(categories)}. Mengumpulkan bukti...",
                }
            )

        task_laws = retrieve_laws(client, content, categories)

        # Only fact-check if Disinformasi or Hoaks
        needs_fact_check = "Disinformasi" in categories or "Hoaks" in categories

        if needs_fact_check:
            # We wrap emit_progress to only pass string messages to the fact checker
            async def fc_progress(msg: str):
                if emit_progress:
                    await emit_progress({"stage": "fact_checking", "message": msg})

            task_fact = fact_check(client, content, search_queue, fc_progress)

            fc_result = await task_fact
            laws_summary = await task_laws
            
            result["laws_summary"] = laws_summary
            result["fact_check"] = fc_result

            # Fallback to Misinformasi if we can't verify (counterfactuals not found)
            if fc_result and fc_result.get("verified") is None:
                categories = [
                    c for c in categories if c not in ("Hoaks", "Disinformasi")
                ]
                if "Misinformasi" not in categories:
                    categories.append("Misinformasi")
                result["categories"] = categories
        else:
            # Only await laws
            laws_summary = await task_laws
            result["laws_summary"] = laws_summary

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
