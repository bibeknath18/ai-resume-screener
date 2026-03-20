"""
Step 6 — LLM Insights Pipeline
Extracts: strengths, gaps, reasoning, recommendation
Includes: strict JSON output (Fix 3) + pydantic validation
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
import anthropic

load_dotenv()


# ── Pydantic model for validated output ─────────────────────
class InsightsOutput(BaseModel):
    strengths:      list[str]
    gaps:           list[str]
    reasoning:      str
    recommendation: str


# ── Prompt template ─────────────────────────────────────────
INSIGHTS_PROMPT = """You are a senior technical recruiter evaluating a candidate.

Job Role: {role}
Required Skills: {required_skills}
Optional Skills: {optional_skills}
Required Experience: {required_yoe}+ years
Required Education: {required_education}

Job Description:
{jd_text}

Candidate Profile:
- File: {filename}
- Skills found: {candidate_skills}
- Years of experience: {candidate_yoe}
- Education: {candidate_edu}
- Resume text (excerpt): {resume_excerpt}

Evaluate this candidate against the job description.
Respond ONLY with valid JSON. No preamble, no markdown, no text outside the JSON.

Use exactly this schema:
{{
  "strengths": ["specific strength 1 tied to JD", "specific strength 2"],
  "gaps": ["specific gap 1 from JD requirements", "specific gap 2"],
  "reasoning": "one concise sentence explaining overall fit",
  "recommendation": "Strong Fit"
}}

Rules:
- strengths: exactly 2-3 bullet points, specific to this candidate
- gaps: exactly 2-3 bullet points, only real gaps from JD requirements
- reasoning: one sentence, factual and concise
- recommendation: must be exactly one of: "Strong Fit", "Moderate Fit", "Not Fit"
"""


def build_prompt(jd_profile: dict, candidate_profile: dict) -> str:
    """Build the prompt string for a candidate."""
    resume_excerpt = candidate_profile.get("raw", "")[:600]

    return INSIGHTS_PROMPT.format(
        role=jd_profile.get("role", ""),
        required_skills=", ".join(jd_profile.get("required_skills", [])),
        optional_skills=", ".join(jd_profile.get("optional_skills", [])),
        required_yoe=jd_profile.get("required_yoe", 3),
        required_education=jd_profile.get("required_education", "bachelors"),
        jd_text=jd_profile.get("raw", "")[:800],
        filename=candidate_profile.get("filename", ""),
        candidate_skills=", ".join(candidate_profile.get("skills", [])),
        candidate_yoe=candidate_profile.get("years_of_experience", 0),
        candidate_edu=candidate_profile.get("education_level", "unknown"),
        resume_excerpt=resume_excerpt,
    )


def parse_llm_response(raw: str) -> InsightsOutput:
    """Parse and validate LLM JSON response."""
    # Strip markdown code fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    data = json.loads(clean)

    # Validate recommendation value
    valid_recs = {"Strong Fit", "Moderate Fit", "Not Fit"}
    if data.get("recommendation") not in valid_recs:
        data["recommendation"] = "Moderate Fit"

    return InsightsOutput(**data)


def get_insights_sync(
    jd_profile: dict,
    candidate_profile: dict,
    retries: int = 2
) -> InsightsOutput:
    """Get LLM insights for one candidate (synchronous)."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = build_prompt(jd_profile, candidate_profile)

    for attempt in range(retries):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = message.content[0].text
            return parse_llm_response(raw)

        except json.JSONDecodeError as e:
            if attempt == retries - 1:
                # Final fallback
                return InsightsOutput(
                    strengths=["Could not parse LLM response"],
                    gaps=["Could not parse LLM response"],
                    reasoning="LLM response parsing failed.",
                    recommendation="Moderate Fit"
                )
        except Exception as e:
            if attempt == retries - 1:
                return InsightsOutput(
                    strengths=["LLM call failed"],
                    gaps=["LLM call failed"],
                    reasoning=f"Error: {str(e)}",
                    recommendation="Moderate Fit"
                )

    return InsightsOutput(
        strengths=["Unknown"],
        gaps=["Unknown"],
        reasoning="Unknown error.",
        recommendation="Moderate Fit"
    )


def get_all_insights(
    jd_profile: dict,
    candidate_profiles: list
) -> list:
    """Get insights for all candidates."""
    results = []
    total = len(candidate_profiles)

    for i, profile in enumerate(candidate_profiles, 1):
        filename = profile.get("filename", "unknown")
        print(f"  [{i}/{total}] Analyzing {filename}...")
        insights = get_insights_sync(jd_profile, profile)
        results.append({
            "filename":       filename,
            "strengths":      insights.strengths,
            "gaps":           insights.gaps,
            "reasoning":      insights.reasoning,
            "llm_recommendation": insights.recommendation,
        })

    return results


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    from parser import parse_all_resumes, ParseStatus
    from extractor import extract_candidate_profile, extract_jd_profile

    with open("data/jd.txt") as f:
        jd_text = f.read()
    jd_profile = extract_jd_profile(jd_text)

    parse_results = parse_all_resumes("data/resumes")
    profiles = [
        extract_candidate_profile(pr)
        for pr in parse_results
        if pr["status"] == ParseStatus.OK
    ]

    print("\n" + "=" * 55)
    print("LLM Insights")
    print("=" * 55 + "\n")

    # Test on first 2 candidates only to save API cost
    test_profiles = profiles[:2]
    insights_list = get_all_insights(jd_profile, test_profiles)

    for ins in insights_list:
        print(f"\nFile: {ins['filename']}")
        print(f"Recommendation: {ins['llm_recommendation']}")
        print(f"Reasoning: {ins['reasoning']}")
        print("Strengths:")
        for s in ins["strengths"]:
            print(f"  + {s}")
        print("Gaps:")
        for g in ins["gaps"]:
            print(f"  - {g}")