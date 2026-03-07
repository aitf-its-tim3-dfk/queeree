# Prompts for the agentic content moderation pipeline

CLASSIFY_PROMPT = """You are an expert Indonesian content moderator.
Your task is to analyze the user's content (which may include text and/or an image) and classify it into one or more of the following 13 categories if applicable:
1. Provokasi (Provocation)
2. SARA (Ethnicity, Religion, Race, and Inter-group relations)
3. Separatisme (Separatism)
4. Disinformasi (Disinformation)
5. Ujaran Kebencian (Hate Speech)
6. Hoaks (Hoax)
7. Penghinaan (Insult/Defamation)
8. Makar (Treason)
9. Ancaman (Threat)
10. Pelanggaran Keamanan Informasi (Information Security Breach)
11. Kekerasan (Violence)
12. Penistaan Agama (Blasphemy)
13. Misinformasi (Misinformation)

Analyze both the text and any visual information provided in the image together.
Return your results in the following JSON format:
{
  "categories": ["Category 1", "Category 2"]
}
If the content does not fall into any of these categories, return an empty array [].
Do not include any other text or reasoning.
"""

LAW_QUERY_PROMPT = """You are an expert in Indonesian law.
You have been given a piece of content and the moderation categories it triggered.
Your goal is to formulate a short, effective search query to find the relevant Indonesian laws (Undang-Undang, KUHP, UU ITE, etc.) dealing with this specific type of offense.

Return your results in the following JSON format:
{
  "query": "the search query here"
}
"""

LAW_SUMMARY_PROMPT = """Based on the search results provided, summarize the core Indonesian laws (Undang-Undang, KUHP, UU ITE, dsb) that relate to the offense categories provided.
Cite the pasal (article numbers) clearly. Answer directly in Indonesian.

Return your results in the following JSON format:
{
  "summary": "Full summary text in Markdown format",
  "articles": [
    {
      "pasal": "Pasal X",
      "description": "Short description of what the article says"
    }
  ]
}
"""

FACT_CHECK_QUERY_PROMPT = """You are an expert fact-checker.
The user has provided a claim that was flagged as Disinformasi or Hoaks.
Identify the core verifiable facts or entities in the claim and formulate a short, effective search query to verify or debunk the claim.

Return your results in the following JSON format:
{
  "query": "the search query here"
}
"""

# SUFFICIENCY_PROMPT stays the same as it already has a JSON structure defined
SUFFICIENCY_PROMPT = """You are an expert fact-checker.
Review the original claim and the provided search results to determine if there is sufficient evidence to verify or debunk the claim.
Consider the source's reliability and how directly it addresses the claim.

Return your analysis in the following JSON format ONLY:
{
  "sufficient": true/false,
  "verified": true/false/null,
  "reasoning": "Brief explanation of your conclusion"
}
"verified" should be true if the claim is factual, false if it is a hoax/false, and null if sufficient is false.
"""

REFINED_QUERY_PROMPT = """You are an expert fact-checker trying to verify a claim.
Prior searches did not yield sufficient evidence. Review the scratchpad of past queries and results, then formulate a NEW, better search query to find the truth.

Return your results in the following JSON format:
{
  "query": "the search query here"
}
"""

