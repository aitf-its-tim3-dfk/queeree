import json
import asyncio
from typing import Dict, Any, List
from openai import AsyncOpenAI
from .prompts import FACT_CHECK_QUERY_PROMPT, SUFFICIENCY_PROMPT, REFINED_QUERY_PROMPT
from .retrieval import RetrievalQueue

MODEL_NAME = "qwen/qwen3.5-27b"
N_SAMPLES = 3
MAX_LOOPS = 3


async def _check_sufficiency_single(
    client: AsyncOpenAI, content: str, results_context: str
) -> Dict[str, Any]:
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SUFFICIENCY_PROMPT},
                {
                    "role": "user",
                    "content": f"Claim:\n{content}\n\nSearch Results:\n{results_context}",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sufficiency_check",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sufficient": {"type": "boolean"},
                            "verified": {"type": ["boolean", "null"]},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["sufficient", "verified", "reasoning"],
                        "additionalProperties": False,
                    },
                },
            },
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        if reply.startswith("```json"):
            reply = reply[7:-3].strip()
        elif reply.startswith("```"):
            reply = reply[3:-3].strip()

        return json.loads(reply)
    except Exception as e:
        print(f"Sufficiency check error: {e}")
        return {
            "sufficient": False,
            "verified": None,
            "reasoning": "Error in LLM evaluation",
        }


async def check_sufficiency(
    client: AsyncOpenAI, content: str, results_context: str
) -> Dict[str, Any]:
    """Run N=3 self-consistency check on whether the retrieved evidence is sufficient to debunk/verify."""
    tasks = [
        _check_sufficiency_single(client, content, results_context)
        for _ in range(N_SAMPLES)
    ]
    evals = await asyncio.gather(*tasks)

    # Majority vote
    sufficient_votes = sum(1 for e in evals if e.get("sufficient") is True)
    if sufficient_votes >= (N_SAMPLES // 2 + 1):
        # We have sufficiency. Take the majority verifiable state
        verified_votes = sum(1 for e in evals if e.get("verified") is True)
        unverified_votes = sum(1 for e in evals if e.get("verified") is False)

        is_verified = True if verified_votes >= unverified_votes else False

        # Take the reasoning from the first valid positive vote
        reasoning = next(
            (e.get("reasoning") for e in evals if e.get("sufficient") is True),
            "Sufficient evidence found.",
        )
        return {"sufficient": True, "verified": is_verified, "reasoning": reasoning}

    # Not sufficient
    reasoning = next(
        (e.get("reasoning") for e in evals if e.get("sufficient") is False),
        "Evidence insufficient.",
    )
    return {"sufficient": False, "verified": None, "reasoning": reasoning}


async def generate_query(client: AsyncOpenAI, prompt: str, content: str) -> str:
    try:
        res = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "search_query",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            temperature=0.3,
        )
        data = json.loads(res.choices[0].message.content)
        return data.get("query", "hoaks indonesia")
    except Exception as e:
        print(f"Query generation error: {e}")
        return "hoaks indonesia"


async def fact_check(
    client: AsyncOpenAI, content: str, search_queue: RetrievalQueue, emit_progress=None
) -> Dict[str, Any]:
    """Iterative fact-checking retrieval loop."""
    scratchpad = []
    accumulated_sources = []

    # Initial query
    if emit_progress:
        await emit_progress("Membuat kueri pencarian awal...")

    query = await generate_query(client, FACT_CHECK_QUERY_PROMPT, content)

    for loop in range(MAX_LOOPS):
        if emit_progress:
            await emit_progress(
                f"Iterasi {loop+1}/{MAX_LOOPS}: Mencari '{query}'..."
            )

        try:
            results = await search_queue.retrieve(query)
        except Exception:
            results = []

        if not results:
            scratchpad.append(f"Q: {query} -> A: No results found.")
            results_context = "No search results returned."
        else:
            top_results = results[:4]
            # Accumulate and deduplicate
            current_urls = {s["url"] for s in accumulated_sources}
            for r in top_results:
                if r["url"] not in current_urls:
                    accumulated_sources.append(r)
                    current_urls.add(r["url"])

            results_context = "\n\n".join(
                [
                    f"[{i+1}] {r['title']}\n{r['description']}\n(Source: {r['url']})"
                    for i, r in enumerate(top_results)
                ]
            )
            scratchpad.append(f"Q: {query} -> Found {len(results)} results.")

        if emit_progress:
            await emit_progress(
                f"Evaluasi bukti: {len(accumulated_sources)} sumber unik terkumpul..."
            )

        eval_result = await check_sufficiency(client, content, results_context)

        if eval_result.get("sufficient") is True:
            # Done!
            if emit_progress:
                await emit_progress("Verifikasi selesai: Bukti cukup ditemukan.")

            return {
                "verified": eval_result.get("verified"),
                "reasoning": eval_result.get("reasoning"),
                "sources": accumulated_sources[:6],  # Top 6 total unique sources
            }

        # Not sufficient, refine the query
        if loop < MAX_LOOPS - 1:
            if emit_progress:
                await emit_progress(
                    f"Bukti belum cukup (Iterasi {loop+1}). Menyaring kueri..."
                )
            refine_context = (
                f"Original Claim:\n{content}\n\nPast queries and results:\n"
                + "\n".join(scratchpad)
            )
            query = await generate_query(client, REFINED_QUERY_PROMPT, refine_context)

    # Failed to find sufficient evidence after MAX_LOOPS
    if emit_progress:
        await emit_progress("Batas pencarian tercapai. Hasil tidak konklusif.")

    return {
        "verified": None,
        "reasoning": "Tidak dapat menemukan bukti yang meyakinkan setelah beberapa pencarian.",
        "sources": accumulated_sources[:6],
    }

