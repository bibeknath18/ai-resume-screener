"""
Step 7 — Recommendation + Merge Logic
Combines score + LLM insights into final candidate result.
Includes: ParseStatus edge case handling (Fix 5)
"""

from parser import ParseStatus


def get_recommendation(score: float) -> str:
    """Apply score thresholds to get recommendation label."""
    if score >= 75:
        return "Strong Fit"
    elif score >= 50:
        return "Moderate Fit"
    else:
        return "Not Fit"


def get_recommendation_color(recommendation: str) -> str:
    """Return color code for recommendation label."""
    colors = {
        "Strong Fit":   "#3B6D11",
        "Moderate Fit": "#854F0B",
        "Not Fit":      "#A32D2D",
        "Parsing Failed": "#5F5E5A",
    }
    return colors.get(recommendation, "#5F5E5A")


def merge_result(
    parse_result:      dict,
    score_dict:        dict,
    candidate_profile: dict,
    insights_dict:     dict
) -> dict:
    """
    Merge all components into one final candidate result.
    Handles failed parses gracefully (Fix 5).
    """
    filename = parse_result.get("filename", "unknown")

    # Fix 5 — handle failed parses
    if parse_result.get("status") != ParseStatus.OK:
        return {
            "filename":           filename,
            "name":               filename.replace(".pdf", ""),
            "status":             parse_result["status"].value,
            "total":              None,
            "skill_score":        None,
            "req_match_pct":      None,
            "opt_match_pct":      None,
            "exp_score":          None,
            "edu_score":          None,
            "kw_score":           None,
            "candidate_yoe":      None,
            "candidate_edu":      None,
            "req_skills_matched": [],
            "opt_skills_matched": [],
            "req_skills_missing": [],
            "strengths":          [],
            "gaps":               [],
            "reasoning":          "Resume could not be parsed.",
            "recommendation":     "Parsing Failed",
            "llm_recommendation": "Parsing Failed",
            "error":              parse_result.get("error", ""),
        }

    # Score-based recommendation
    score_recommendation = get_recommendation(score_dict.get("total", 0))

    # Check if LLM and score recommendations agree
    llm_rec   = insights_dict.get("llm_recommendation", "Moderate Fit")
    score_rec = score_recommendation
    flag_disagreement = llm_rec != score_rec

    return {
        "filename":              filename,
        "name":                  candidate_profile.get("name", filename.replace(".pdf","")),
        "status":                "ok",
        "total":                 score_dict.get("total"),
        "skill_score":           score_dict.get("skill_score"),
        "req_match_pct":         score_dict.get("req_match_pct"),
        "opt_match_pct":         score_dict.get("opt_match_pct"),
        "exp_score":             score_dict.get("exp_score"),
        "edu_score":             score_dict.get("edu_score"),
        "kw_score":              score_dict.get("kw_score"),
        "candidate_yoe":         score_dict.get("candidate_yoe"),
        "candidate_edu":         score_dict.get("candidate_edu"),
        "req_skills_matched":    score_dict.get("req_skills_matched", []),
        "opt_skills_matched":    score_dict.get("opt_skills_matched", []),
        "req_skills_missing":    score_dict.get("req_skills_missing", []),
        "strengths":             insights_dict.get("strengths", []),
        "gaps":                  insights_dict.get("gaps", []),
        "reasoning":             insights_dict.get("reasoning", ""),
        "recommendation":        score_recommendation,
        "llm_recommendation":    llm_rec,
        "flag_disagreement":     flag_disagreement,
    }


def rank_candidates(results: list) -> list:
    """
    Sort candidates by score descending.
    Parsing Failed candidates go to the bottom.
    """
    scored   = [r for r in results if r.get("total") is not None]
    failed   = [r for r in results if r.get("total") is None]
    scored.sort(key=lambda x: x["total"], reverse=True)

    # Add rank numbers
    for i, r in enumerate(scored, 1):
        r["rank"] = i
    for r in failed:
        r["rank"] = None

    return scored + failed


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Recommender module loaded successfully.")
    print(f"Score 80 → {get_recommendation(80)}")
    print(f"Score 60 → {get_recommendation(60)}")
    print(f"Score 40 → {get_recommendation(40)}")