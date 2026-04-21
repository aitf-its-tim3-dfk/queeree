import json
import asyncio
import statistics
from typing import Dict, Any, List, Callable
from openai import AsyncOpenAI
from .prompts import (
    FACT_CHECK_STANDARD_QUERY_PROMPT,
    FACT_CHECK_CONTRARY_QUERY_PROMPT,
    PURE_REASONING_PROMPT,
    SUFFICIENCY_PROMPT,
    REFINED_QUERY_PROMPT,
    construct_grounded_prompt,
)
from .retrieval import RetrievalQueue
from .reranker import rerank
import config


async def _check_sufficiency_single(
    client: AsyncOpenAI, content: str, results_context: str
) -> Dict[str, Any]:
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=config.get_config_val("fact_checker_model_name"),
                messages=[
                    {
                        "role": "system",
                        "content": construct_grounded_prompt(SUFFICIENCY_PROMPT),
                    },
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
                                "sufficient_evidence": {"type": "boolean"},
                                "factuality_score": {"type": "integer"},
                                "reasoning": {"type": "string"},
                            },
                            "required": [
                                "sufficient_evidence",
                                "factuality_score",
                                "reasoning",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                temperature=0.7,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
                **config.get_llm_kwargs("fact_checker"),
            )
            content_str = response.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(
                    f"[VERBOSE - FactChecker Sufficiency] LLM Response: {content_str}"
                )

            if content_str is None:
                raise ValueError("Model returned None content")

            reply = content_str.strip()
            return json.loads(reply)
        except Exception as e:
            print(f"Sufficiency check attempt {attempt+1} error: {e}")
            if attempt == 2:
                return {
                    "sufficient_evidence": False,
                    "factuality_score": 50,
                    "reasoning": "Error in LLM evaluation",
                }
            await asyncio.sleep(1)


async def check_sufficiency(
    client: AsyncOpenAI, content: str, results_context: str
) -> Dict[str, Any]:
    """Run N self-consistency check on whether the retrieved evidence is sufficient to debunk/verify."""
    n_samples = config.get_config_val("fact_checker_n_samples") or 3
    tasks = [
        _check_sufficiency_single(client, content, results_context)
        for _ in range(n_samples)
    ]
    evals = await asyncio.gather(*tasks)

    # Extract scores
    scores = [e.get("factuality_score", 50) for e in evals]
    mean_score = statistics.mean(scores)

    sufficient_votes = sum(1 for e in evals if e.get("sufficient_evidence") is True)
    is_sufficient = sufficient_votes >= (n_samples // 2 + 1)

    status = "UNVERIFIED"

    if is_sufficient:
        if mean_score >= 70:
            status = "TRUE"
        elif mean_score <= 30:
            status = "FALSE"

    # Take reasoning from a representative vote
    if status == "TRUE":
        reason_vote = next(
            (e for e in evals if e.get("factuality_score", 0) >= 70), evals[0]
        )
    elif status == "FALSE":
        reason_vote = next(
            (e for e in evals if e.get("factuality_score", 100) <= 30), evals[0]
        )
    else:
        reason_vote = next(
            (e for e in evals if not e.get("sufficient_evidence", True)), evals[0]
        )

    reasoning = reason_vote.get("reasoning", "Analysis ambiguous.")

    return {
        "status": status,
        "mean": mean_score,
        "sufficient": is_sufficient,
        "reasoning": reasoning,
    }


async def _check_reasoning_single(client: AsyncOpenAI, content: str) -> Dict[str, Any]:
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=config.get_config_val("fact_checker_model_name"),
                messages=[
                    {
                        "role": "system",
                        "content": construct_grounded_prompt(PURE_REASONING_PROMPT),
                    },
                    {"role": "user", "content": f"Claim:\n{content}"},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "reasoning_check",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "factuality_score": {"type": "integer"},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["factuality_score", "reasoning"],
                            "additionalProperties": False,
                        },
                    },
                },
                temperature=0.7,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
                **config.get_llm_kwargs("fact_checker"),
            )
            content_str = response.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - FactChecker Reasoning] LLM Response: {content_str}")

            if content_str is None:
                raise ValueError("Model returned None content")

            reply = content_str.strip()
            return json.loads(reply)
        except Exception as e:
            if attempt == 2:
                return {
                    "factuality_score": 50,
                    "reasoning": "Error in LLM evaluation",
                }
            await asyncio.sleep(1)


async def check_reasoning_likelihood(
    client: AsyncOpenAI, content: str
) -> Dict[str, Any]:
    """Path 3: Run self-consistency check on factuality purely based on reasoning (no external context)."""
    n_samples = config.get_config_val("fact_checker_n_samples") or 3
    tasks = [_check_reasoning_single(client, content) for _ in range(n_samples)]
    evals = await asyncio.gather(*tasks)

    # Extract scores
    scores = [e.get("factuality_score", 50) for e in evals]
    mean_score = statistics.mean(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

    status = "UNVERIFIED"

    # Strict logic for self-consistency path
    if mean_score >= 70 and std_score < 20:
        status = "TRUE"
    elif mean_score <= 30 and std_score < 20:
        status = "FALSE"

    if status == "TRUE":
        reason_vote = next(
            (e for e in evals if e.get("factuality_score", 0) >= 70), evals[0]
        )
    elif status == "FALSE":
        reason_vote = next(
            (e for e in evals if e.get("factuality_score", 100) <= 30), evals[0]
        )
    else:
        reason_vote = evals[0]

    return {
        "status": status,
        "mean": mean_score,
        "std": std_score,
        "reasoning": reason_vote.get("reasoning", "Analisis logika ambigu."),
        "sufficient": False,
    }


async def generate_query(client: AsyncOpenAI, prompt: str, content: str) -> str:
    for attempt in range(3):
        try:
            res = await client.chat.completions.create(
                model=config.get_config_val("fact_checker_model_name"),
                messages=[
                    {"role": "system", "content": construct_grounded_prompt(prompt)},
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
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
                **config.get_llm_kwargs("fact_checker"),
            )
            content_str = res.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - FactChecker Query] LLM Response: {content_str}")

            if content_str is None:
                raise ValueError("Model returned None content")

            data = json.loads(content_str.strip())
            return data.get("query", "hoaks indonesia")
        except Exception as e:
            print(f"Query generation attempt {attempt+1} error: {e}")
            if attempt == 2:
                return "hoaks indonesia"
            await asyncio.sleep(1)


async def run_search_path_iterative(
    client: AsyncOpenAI,
    content: str,
    search_queue: RetrievalQueue,
    initial_prompt: str,
    emit_progress: Callable,
    path_name: str,
) -> Dict[str, Any]:
    scratchpad = []
    accumulated_sources = []

    if emit_progress:
        await emit_progress(f"[{path_name}] Membuat kueri pencarian awal...")

    query = await generate_query(client, initial_prompt, content)

    max_loops = config.get_config_val("fact_checker_max_loops") or 1
    # Fallback
    eval_result = {
        "status": "UNVERIFIED",
        "mean": 50,
        "sufficient": False,
        "reasoning": "No search iterations performed.",
    }
    for loop in range(max_loops):
        if emit_progress:
            await emit_progress(
                f"[{path_name}] Iterasi {loop+1}/{max_loops}: Mencari '{query}'..."
            )

        try:
            results = await search_queue.retrieve(query)
        except Exception:
            results = []

        if not results:
            scratchpad.append(
                f'Iteration {loop+1} — Query: "{query}"\n'
                f"  Result: No results found."
            )
        else:
            # Rerank results by relevance to the claim
            top_results = await rerank(query, results, top_k=4)
            current_urls = {s["url"] for s in accumulated_sources}
            for r in top_results:
                if r["url"] not in current_urls:
                    accumulated_sources.append(r)
                    current_urls.add(r["url"])

            result_summaries = "; ".join([f"\"{r['title']}\"" for r in top_results[:3]])
            scratchpad.append(
                f'Iteration {loop+1} — Query: "{query}"\n'
                f"  Result: Found {len(results)} results. Top hits: {result_summaries}"
            )

        # Build results_context from ALL accumulated sources, not just current batch
        results_context = (
            "\n\n".join(
                [
                    f"[{i+1}] {r['title']}\n{r['description']}\n(Source: {r['url']})"
                    for i, r in enumerate(accumulated_sources)
                ]
            )
            if accumulated_sources
            else "No search results returned."
        )

        if emit_progress:
            await emit_progress(
                f"[{path_name}] Evaluasi bukti: {len(accumulated_sources)} sumber terkumpul..."
            )

        eval_result = await check_sufficiency(client, content, results_context)

        status = eval_result.get("status")

        if status in ["TRUE", "FALSE"] or (eval_result.get("sufficient") is True):
            eval_result["sources"] = accumulated_sources
            return eval_result

        # Not sufficient, refine the query
        if loop < max_loops - 1:
            if emit_progress:
                await emit_progress(
                    f"[{path_name}] Bukti belum cukup (Iterasi {loop+1}). Menyaring kueri..."
                )
            # Feed sufficiency reasoning back so the LLM knows why evidence was lacking
            insufficiency_reason = eval_result.get("reasoning", "Unknown")
            scratchpad.append(f"  → Insufficient because: {insufficiency_reason}")

            refine_context = (
                f"Original Claim:\n{content}\n\nPast queries and results:\n"
                + "\n".join(scratchpad)
            )
            query = await generate_query(client, REFINED_QUERY_PROMPT, refine_context)

    # Failed to find sufficient evidence after MAX_LOOPS
    eval_result["sources"] = accumulated_sources
    return eval_result


async def fact_check(
    client: AsyncOpenAI, content: str, search_queue: RetrievalQueue, emit_progress=None
) -> Dict[str, Any]:
    """Execute 3 parallel fact-checking paths: standard, contrary, and pure reasoning."""
    if emit_progress:
        await emit_progress(
            "Memulai verifikasi fakta (Standar, Kontradiksi, Logika Dasar)..."
        )

    path1_task = run_search_path_iterative(
        client,
        content,
        search_queue,
        FACT_CHECK_STANDARD_QUERY_PROMPT,
        emit_progress,
        "Standar",
    )
    path2_task = run_search_path_iterative(
        client,
        content,
        search_queue,
        FACT_CHECK_CONTRARY_QUERY_PROMPT,
        emit_progress,
        "Kontradiksi",
    )
    path3_task = check_reasoning_likelihood(client, content)

    p1, p2, p3 = await asyncio.gather(path1_task, path2_task, path3_task)

    # Accumulate & Deduplicate sources
    sources = p1.get("sources", []) + p2.get("sources", [])
    unique_sources = []
    seen_urls = set()
    for s in sources:
        if s["url"] not in seen_urls:
            unique_sources.append(s)
            seen_urls.add(s["url"])

    final_status = "UNVERIFIED"
    final_reasoning = ""
    mean_score = 50.0

    # Decision logic with conflict reconciliation
    p1_decided = p1["sufficient"] and p1["status"] in ["TRUE", "FALSE"]
    p2_decided = p2["sufficient"] and p2["status"] in ["TRUE", "FALSE"]

    # Happy path: both agree or only one decided
    if p1_decided and p2_decided and p1["status"] == p2["status"]:
        final_status = p1["status"]
        final_reasoning = f"(Kedua jalur pencarian sepakat)\n{p1['reasoning']}"
        mean_score = (p1["mean"] + p2["mean"]) / 2.0
    elif p1_decided and not p2_decided:
        final_status = p1["status"]
        final_reasoning = f"(Berdasarkan Pencarian Bukti Standar)\n{p1['reasoning']}"
        mean_score = p1["mean"]
    elif p2_decided and not p1_decided:
        final_status = p2["status"]
        final_reasoning = (
            f"(Berdasarkan Pencarian Bukti Kontradiktif)\n{p2['reasoning']}"
        )
        mean_score = p2["mean"]
    else:
        # Conflict or both inconclusive, run combined sufficiency on pooled evidence
        if unique_sources:
            if emit_progress:
                await emit_progress(
                    "Jalur pencarian tidak konklusif. Mengevaluasi seluruh bukti gabungan..."
                )

            combined_context = "\n\n".join(
                [
                    f"[{i+1}] {r['title']}\n{r['description']}\n(Source: {r['url']})"
                    for i, r in enumerate(unique_sources)
                ]
            )
            combined_eval = await check_sufficiency(client, content, combined_context)

            combined_status = combined_eval.get("status")
            if combined_status in ["TRUE", "FALSE"]:
                final_status = combined_status
                final_reasoning = (
                    f"(Berdasarkan Evaluasi Bukti Gabungan — {len(unique_sources)} sumber)\n"
                    f"{combined_eval['reasoning']}"
                )
                mean_score = combined_eval["mean"]
            elif p3["status"] in ["TRUE", "FALSE"]:
                # Combined still inconclusive, fall back to reasoning path
                final_status = p3["status"]
                final_reasoning = (
                    f"(Berdasarkan Konsistensi Logis Internal Model)\n{p3['reasoning']}"
                )
                mean_score = p3["mean"]
            else:
                final_status = "UNVERIFIED"
                final_reasoning = (
                    f"Bukti gabungan dan logika dasar tidak cukup untuk memutuskan.\n\n"
                    f"Evaluasi Gabungan: {combined_eval.get('reasoning', '')}\n\n"
                    f"Logika Internal: {p3['reasoning']}"
                )
                mean_score = (combined_eval["mean"] + p3["mean"]) / 2.0
        elif p3["status"] in ["TRUE", "FALSE"]:
            final_status = p3["status"]
            final_reasoning = (
                f"(Berdasarkan Konsistensi Logis Internal Model)\n{p3['reasoning']}"
            )
            mean_score = p3["mean"]
        else:
            final_status = "UNVERIFIED"
            final_reasoning = (
                f"Tidak ada bukti eksternal dan logika dasar menghasilkan probabilitas ragu-ragu.\n\n"
                f"Logika Internal: {p3['reasoning']}"
            )
            mean_score = p3["mean"]

    if emit_progress:
        await emit_progress(
            f"Verifikasi selesai: {final_status} (Skor Fakta: {mean_score:.1f}/100)."
        )

    return {
        "status": final_status,
        "mean": mean_score,
        "std": p3.get("std", 0.0),  # std from reasoning path
        "sufficient": p1["sufficient"] or p2["sufficient"],
        "reasoning": final_reasoning,
        "sources": unique_sources[:6],
    }
