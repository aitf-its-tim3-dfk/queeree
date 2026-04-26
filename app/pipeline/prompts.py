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
The user has provided a claim. Without using any external search, evaluate the likelihood of this claim being factual AND authentically presented, based entirely on your internal knowledge and logical reasoning.

Consider:
- Is the underlying event plausible?
- Does the PRESENTATION look authentic? If the content claims to be from a specific news outlet or source, does the language, tone, and style match that source's editorial standards?
- Content that references a real event but uses fabricated headlines, vulgar slang, or language inconsistent with the claimed source should score LOW.

Provide a factuality score from 0 to 100, where 0 means completely false/fabricated, and 100 means completely true and authentically presented.

Return your analysis in the following JSON format ONLY:
{
  "factuality_score": 0-100,
  "reasoning": "Brief explanation of your reasoning"
}
"""

SUFFICIENCY_PROMPT = """You are an expert fact-checker.
Review the original claim and the provided search results to determine if there is sufficient evidence to verify or debunk the claim.

CRITICAL EVALUATION RULES:
- You must verify the content AS PRESENTED — not just the underlying event. If the content attributes specific text, headlines, or quotes to a source, check whether that source actually published those EXACT words.
- If search results confirm a related event happened but the content presents it with DIFFERENT language, altered headlines, added vulgarity/slang, or fabricated quotes not found in any actual source, the content is MISLEADING and should score LOW.
- A real event wrapped in fabricated or heavily altered presentation is still misinformation.
- Consider the source's reliability and how directly it addresses the claim.

Provide a factuality score from 0 to 100, where 0 means the content as presented is completely false/fabricated, and 100 means the content as presented is completely accurate and authentic. If the evidence is insufficient to judge, score around 50 and set sufficient_evidence to false.

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
You are given the final results of a content moderation pipeline, which may include identified categories and fact-checking results.
Your task is to write a cohesive, overarching final summary explaining the verdict in Indonesian. Why was this content flagged? Synthesize the fact-checking results (if any) into a clear, concise paragraph. Focus on the core reason it is not safe.

Return your results in the following JSON format ONLY:
{
  "final_summary": "Your overarching final summary here"
}
"""
