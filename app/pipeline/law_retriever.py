import os
import json
import numpy as np
import hnswlib
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
import asyncio
from .prompts import LAW_QUERY_PROMPT, LAW_SUMMARY_PROMPT
import config


class LocalLawRetriever:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.embed_model = None
        self.index = None
        self.titles = []
        self.passages = []
        self.title_embs = None
        self.laws = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        print("Loading local law index...")
        self.embed_model = SentenceTransformer(
            config.get_config_val("embedding_model_name"), trust_remote_code=True
        )

        with open(
            os.path.join(self.data_dir, "law_metadata.json"), "r", encoding="utf-8"
        ) as f:
            self.laws = json.load(f)

        self.title_embs = np.load(os.path.join(self.data_dir, "law_title_embs.npy"))

        dim = self.title_embs.shape[1]
        self.index = hnswlib.Index(space="ip", dim=dim)
        index_path = os.path.join(self.data_dir, "law_passages.bin")
        self.index.load_index(index_path, max_elements=len(self.laws))
        self._loaded = True

    def unpack_binary(self, emb):
        if emb.dtype == np.uint8 or emb.dtype == np.int8:
            if emb.shape[1] < 1000:
                unpacked = np.unpackbits(emb.view(np.uint8), axis=1)
                return unpacked.astype(np.float32) * 2 - 1
            else:
                float_emb = emb.astype(np.float32)
                if np.all(float_emb >= 0):
                    return float_emb * 2 - 1
                return float_emb
        return emb.astype(np.float32)

    def retrieve_top_k(self, query: str, k=5, prefilter_k=50):
        if not self._loaded:
            self.load()

        # 1. Embed query (binary quantized)
        q_emb = self.embed_model.encode([query], quantization="binary")
        q_emb = self.unpack_binary(q_emb)  # shape (1, D)

        # 2. Retrieve top passages
        actual_prefilter = min(prefilter_k, len(self.laws))
        labels, distances = self.index.knn_query(q_emb, k=actual_prefilter)

        labels = labels[0]  # top K labels

        # 3. Hierarchical Re-ranking: passage_sim + title_sim
        results = []
        for idx in labels:
            label_idx = np.where(labels == idx)[0][0]
            passage_sim_dist = distances[0][label_idx]
            passage_sim = 1.0 - passage_sim_dist

            t_emb = self.title_embs[idx]
            title_sim = np.dot(q_emb[0], t_emb)

            score = passage_sim + title_sim
            results.append((score, idx))

        results.sort(key=lambda x: x[0], reverse=True)
        top_results = results[:k]

        retrieved_laws = [self.laws[idx] for score, idx in top_results]
        return retrieved_laws


# Global singleton
local_law_retriever = LocalLawRetriever()


async def retrieve_laws(
    client: AsyncOpenAI,
    content: str,
    categories: list[str],
    emit_progress=None,
) -> Dict[str, Any]:
    """Finds and formats relevant Indonesian laws using local hnswlib index, returning a dict with summary and articles."""
    if not categories:
        return {"summary": "", "articles": []}

    cats_str = ", ".join(categories)

    # 1. Ask LLM to generate search query
    if emit_progress:
        await emit_progress(
            {"stage": "law_retrieval", "message": "Menyiapkan kueri pencarian pasal..."}
        )

    for attempt in range(3):
        try:
            res = await client.chat.completions.create(
                model=config.get_config_val("law_retriever_model_name"),
                messages=[
                    {"role": "system", "content": LAW_QUERY_PROMPT},
                    {
                        "role": "user",
                        "content": f"Categories: {cats_str}\n\nContent snippet: {content[:500]}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "law_query",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                },
                temperature=0.1,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
            )
            content_str = res.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - LawRetriever Target Query] LLM Response: {content_str}")
                
            if content_str is None:
                raise ValueError("Model returned None content")
            data = json.loads(content_str.strip())
            query = data.get("query", cats_str)
            break
        except Exception as e:
            print(f"Law query generation attempt {attempt+1} error: {e}")
            if attempt == 2:
                query = cats_str  # Fallback to categories
            await asyncio.sleep(1)

    # 2. Run retrieval
    if emit_progress:
        await emit_progress(
            {"stage": "law_retrieval", "message": f"Mencari pasal untuk: {query}..."}
        )

    try:
        results = local_law_retriever.retrieve_top_k(query, k=5)
    except Exception as e:
        print(f"Retrieval error: {e}")
        return "Gagal mencari pasal (Local index error)."

    if not results:
        return {
            "summary": f"*(Dicari: {query})*\nTidak ditemukan pasal hukum secara spesifik.",
            "articles": [],
        }

    if emit_progress:
        await emit_progress(
            {
                "stage": "law_retrieval",
                "message": f"Ditemukan {len(results)} pasal hukum. Merangkum...",
            }
        )

    # 3. Format raw results into readable text
    search_context = "\n".join([f"- {r['pasal']}: {r['description']}" for r in results])

    for attempt in range(3):
        try:
            final_res = await client.chat.completions.create(
                model=config.get_config_val("law_retriever_model_name"),
                messages=[
                    {"role": "system", "content": LAW_SUMMARY_PROMPT},
                    {
                        "role": "user",
                        "content": f"Categories: {cats_str}\n\nSearch Results:\n{search_context}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "law_summary",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "summary": {"type": "string"},
                                "articles": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "pasal": {"type": "string"},
                                            "description": {"type": "string"},
                                        },
                                        "required": ["pasal", "description"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["summary", "articles"],
                            "additionalProperties": False,
                        },
                    },
                },
                temperature=0.2,
                max_completion_tokens=config.get_config_val("max_completion_tokens"),
            )
            content_str = final_res.choices[0].message.content
            if config.get_config_val("verbose_logging"):
                print(f"[VERBOSE - LawRetriever Summary] LLM Response: {content_str}")
                
            if content_str is None:
                raise ValueError("Model returned None content")
            summary_data = json.loads(content_str.strip())
            return summary_data
        except Exception as e:
            print(f"Law summarization attempt {attempt+1} error: {e}")
            if attempt == 2:
                return {
                    "summary": "Gagal merangkum hukum (Error in LLM).",
                    "articles": [],
                }
            await asyncio.sleep(1)
