# Prompts for the agentic content moderation pipeline
import datetime


def construct_grounded_prompt(prompt_template: str) -> str:
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7)))  # WIB
    current_date = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    current_location = "Indonesia"
    grounding_prefix = (
        f"ENVIRONMENT CONTEXT:\n"
        f"Current Date/Time: {current_date}\n"
        f"Current Location: {current_location}\n\n"
    )
    return grounding_prefix + prompt_template


IMAGE_CONTEXT_EXTRACTION_PROMPT = """You are an expert Indonesian content analyst.
The user has provided an image. Your task is to extract any textual information visible in the image and succinctly describe the context, activities, and key objects shown.
This information will be used to contextualize the image for content moderation. Answer in Indonesian.

Return your results in the following JSON format ONLY:
{
  "extracted_text": "Any text found in the image",
  "visual_context": "Description of what is happening in the image"
}
"""

CLASSIFY_PROMPT = """You are an expert Indonesian content moderator.
Your task is to analyze the user's content (which may include text and/or an image) and classify it into one or more of the following 3 categories if applicable (ONLY USE THE INDONESIAN LABEL):
1. Disinformasi — False, misleading, or manipulative information, including hoaxes and fabricated claims.
2. Fitnah — Content that defames, insults, or spreads false accusations against individuals or groups.
3. Ujaran Kebencian — Content that incites hatred, discrimination, or hostility towards individuals or groups based on identity (ethnicity, religion, race, gender, etc.).

Analyze both the text and any visual information provided in the image together.
Additionally, you must determine if the content makes a factual claim that should be verified against real-world evidence. If the content contains claims about events, statistics, science, or news that could potentially be false, respond with `needs_verification`: true. If it is purely opinion, insult, or subjective without testable claims, respond with false.

Return your results in the following JSON format:
{
  "categories": ["Category 1", "Category 2"],
  "needs_verification": true/false
}
If the content does not fall into any of these categories, return an empty array [] for categories.
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

FACT_CHECK_STANDARD_QUERY_PROMPT = """You are an expert fact-checker.
The user has provided a claim that needs to be verified.
Identify the core verifiable facts or entities in the claim and formulate a short, effective search query to verify the claim.

Return your results in the following JSON format:
{
  "query": "the search query here"
}
"""

FACT_CHECK_CONTRARY_QUERY_PROMPT = """You are an expert fact-checker.
The user has provided a claim that needs to be verified.
Identify the core verifiable facts or entities in the claim and formulate a short, effective search query to find facts on the contrary or known debunkings (e.g., adding keywords like "hoaks", "fakta sebenarnya", "klarifikasi").

Return your results in the following JSON format:
{
  "query": "the search query here"
}
"""

PURE_REASONING_PROMPT = """You are an expert fact-checker.
The user has provided a claim. Without using any external search, evaluate the likelihood of this claim being factual based entirely on your internal knowledge and logical reasoning.
You must provide a factuality score from 0 to 100, where 0 means completely false/illogical/debunked, and 100 means completely true/factual.

Return your analysis in the following JSON format ONLY:
{
  "factuality_score": 0-100,
  "reasoning": "Brief explanation of your reasoning"
}
"""

# SUFFICIENCY_PROMPT stays the same as it already has a JSON structure defined
SUFFICIENCY_PROMPT = """You are an expert fact-checker.
Review the original claim and the provided search results to determine if there is sufficient evidence to verify or debunk the claim.
Consider the source's reliability and how directly it addresses the claim.
You must provide a factuality score from 0 to 100, where 0 means the claim is completely false/debunked by the evidence, and 100 means the claim is completely true/supported by the evidence. If the evidence is completely unrelated or insufficient to make ANY judgment, the score can be around 50, but you should set sufficient_evidence to false.

Return your analysis in the following JSON format ONLY:
{
  "sufficient_evidence": true/false,
  "factuality_score": 0-100,
  "reasoning": "Brief explanation of your conclusion"
}
"""

REFINED_QUERY_PROMPT = """You are an expert fact-checker trying to verify a claim.
Prior searches did not yield sufficient evidence. Review the scratchpad of past queries and results, then formulate a NEW, better search query to find the truth.

Return your results in the following JSON format:
{
  "query": "the search query here"
}
"""

LAW_TEXT_ANALYZER_PROMPT = """You are an expert Indonesian law moderator.
The user has provided a piece of text and a specific Indonesian law.
Determine if the text actually violates the law. If it does not, return `is_violation` as false and explain why in `overall_reason`. If it does, return `is_violation` as true, identify the specific segment(s) of the text that trigger this law, and explain why.

Return your results in the following JSON format ONLY:
{
  "is_violation": true/false,
  "segments": [
    {
      "text": "Exact quote from the user's text",
      "reason": "Why this segment violates the law"
    }
  ],
  "overall_reason": "Summary of why the law applies (or does not apply) to this text"
}
"""

LAW_MULTIMODAL_ANALYZER_PROMPT = """You are an expert Indonesian law moderator.
The user has provided content (which may include text, an image, or both) and a specific Indonesian law. They may also provide fact-checking context if the content contains factual claims.
Determine if the content actually violates the law. 
If it does not, return `is_violation` as false and explain why. 
If it does, return `is_violation` as true. Explain exactly why the content violates this law. If there is text, identify the specific segment(s) of the text that trigger this law.

Return your results in the following JSON format ONLY:
{
  "is_violation": true/false,
  "segments": [
    {
      "text": "Exact quote from the user's text (if applicable)",
      "reason": "Why this segment violates the law"
    }
  ],
  "reason": "Overall explanation of why the content violates (or does not violate) the law"
}
"""

LAW_IMAGE_REASON_AGGREGATOR_PROMPT = """You are an expert at summarizing arguments.
You are given a list of reasons why an image violates a specific law.
Cluster similar reasons and count how many times each distinct reason appears.

Return your results in the following JSON format ONLY:
{
  "clustered_reasons": {
    "Distinct reason 1": 2,
    "Distinct reason 2": 1
  }
}
"""

LAW_IMAGE_FINAL_REASON_PROMPT = """You are an expert Indonesian law moderator.
Based on the provided most common reasons, write a final, cohesive explanation of why the user's image violates the specified law.

Return your results in the following JSON format ONLY:
{
  "reason": "Final conclusive explanation"
}
"""

INTENTION_CHECK_PROMPT = """You are an expert content analyst.
The user has provided a claim that has been verified to be FALSE or a HOAX.
Your task is to analyze the content and confirm its classification as "Disinformasi" (disinformation). Explain the nature of the false claim — whether it appears to be deliberate manipulation, a fabrication, a rumor, or casual misinformation.

Return your results in the following JSON format ONLY:
{
  "category": "Disinformasi",
  "reasoning": "Brief explanation of the intent analysis"
}
"""

PIPELINE_SUMMARY_PROMPT = """You are an expert content policy enforcer.
You are given the final results of a content moderation pipeline, which may include identified categories, fact-checking results, and relevant laws violated.
Your task is to write a cohesive, overarching final summary explaining the verdict in Indonesian. Why was this content flagged? Synthesize the fact-checking results (if any) and the legal context into a clear, concise paragraph. Focus on the core reason it is not safe.

Return your results in the following JSON format ONLY:
{
  "final_summary": "Your overarching final summary here"
}
"""
