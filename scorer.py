"""
Step 5 — Weighted Scoring System
Formula: Skills(40%) + Experience(30%) + Education(15%) + Keywords(15%)
Includes: required vs optional skill weighting (Fix 1)
"""

from extractor import EDU_TABLE


def compute_skill_score(jd_profile: dict, candidate_profile: dict) -> dict:
    """
    Fix 1 — Weighted skill matching.
    Required skills worth 70%, optional skills worth 30%.
    """
    req_skills  = set(jd_profile.get("required_skills", []))
    opt_skills  = set(jd_profile.get("optional_skills", []))
    cand_skills = set(candidate_profile.get("skills", []))

    # Required match
    req_matched = req_skills & cand_skills
    req_match   = len(req_matched) / len(req_skills) if req_skills else 0.0

    # Optional match
    opt_matched = opt_skills & cand_skills
    opt_match   = len(opt_matched) / len(opt_skills) if opt_skills else 0.0

    # Weighted skill score
    skill_score = 0.70 * req_match + 0.30 * opt_match

    return {
        "skill_score":       round(skill_score, 4),
        "req_match_pct":     round(req_match * 100, 1),
        "opt_match_pct":     round(opt_match * 100, 1),
        "req_skills_matched": list(req_matched),
        "opt_skills_matched": list(opt_matched),
        "req_skills_missing": list(req_skills - cand_skills),
    }


def compute_experience_score(jd_profile: dict, candidate_profile: dict) -> dict:
    """Score experience based on years."""
    required_yoe   = jd_profile.get("required_yoe", 3)
    candidate_yoe  = candidate_profile.get("years_of_experience", 0)

    # Cap at 1.0 — extra experience doesn't hurt
    exp_score = min(candidate_yoe / required_yoe, 1.0) if required_yoe > 0 else 1.0

    return {
        "exp_score":     round(exp_score, 4),
        "candidate_yoe": candidate_yoe,
        "required_yoe":  required_yoe,
    }


def compute_education_score(jd_profile: dict, candidate_profile: dict) -> dict:
    """Score education level against JD requirement."""
    required_edu      = jd_profile.get("required_education", "bachelors")
    required_edu_score = EDU_TABLE.get(required_edu, 0.70)
    candidate_edu     = candidate_profile.get("education_level", "none")
    candidate_score   = candidate_profile.get("education_score", 0.20)

    # Ratio of candidate edu vs required edu
    edu_score = min(candidate_score / required_edu_score, 1.0)

    return {
        "edu_score":       round(edu_score, 4),
        "candidate_edu":   candidate_edu,
        "required_edu":    required_edu,
    }


def compute_final_score(
    jd_profile: dict,
    candidate_profile: dict,
    section_sims: dict,
    w_skills: float = 0.40,
    w_exp:    float = 0.30,
    w_edu:    float = 0.15,
    w_kw:     float = 0.15,
) -> dict:
    """
    Master scoring function.
    Combines all components into final 0-100 score.
    Weights are configurable for different role types.
    """
    # Component scores
    skill_result = compute_skill_score(jd_profile, candidate_profile)
    exp_result   = compute_experience_score(jd_profile, candidate_profile)
    edu_result   = compute_education_score(jd_profile, candidate_profile)

    skill_score = skill_result["skill_score"]
    exp_score   = exp_result["exp_score"]
    edu_score   = edu_result["edu_score"]
    kw_score    = section_sims.get("keyword_sim", 0.0)

    # Weighted formula with configurable weights
    final = (
        skill_score * w_skills +
        exp_score   * w_exp +
        edu_score   * w_edu +
        kw_score    * w_kw
    ) * 100

    return {
        "total":              round(final, 1),
        "skill_score":        round(skill_score * 100, 1),
        "req_match_pct":      skill_result["req_match_pct"],
        "opt_match_pct":      skill_result["opt_match_pct"],
        "req_skills_matched": skill_result["req_skills_matched"],
        "opt_skills_matched": skill_result["opt_skills_matched"],
        "req_skills_missing": skill_result["req_skills_missing"],
        "exp_score":          round(exp_score * 100, 1),
        "candidate_yoe":      exp_result["candidate_yoe"],
        "edu_score":          round(edu_score * 100, 1),
        "candidate_edu":      edu_result["candidate_edu"],
        "kw_score":           round(kw_score * 100, 1),
    }


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    from parser import parse_all_resumes, ParseStatus
    from extractor import extract_candidate_profile, extract_jd_profile
    from embedder import compute_section_similarities

    with open("data/jd.txt") as f:
        jd_text = f.read()
    jd_profile = extract_jd_profile(jd_text)

    parse_results = parse_all_resumes("data/resumes")

    print("\n" + "=" * 55)
    print("Candidate Scores")
    print("=" * 55)

    scores = []
    for pr in parse_results:
        if pr["status"] == ParseStatus.OK:
            profile  = extract_candidate_profile(pr)
            sims     = compute_section_similarities(jd_profile, profile)
            result   = compute_final_score(jd_profile, profile, sims)
            result["filename"] = profile["filename"]
            scores.append(result)

    # Sort by total score
    scores.sort(key=lambda x: x["total"], reverse=True)

    for i, s in enumerate(scores, 1):
        print(f"\n#{i} {s['filename']}")
        print(f"  Total Score     : {s['total']}/100")
        print(f"  Skill Score     : {s['skill_score']}%  "
              f"(Required: {s['req_match_pct']}%, Optional: {s['opt_match_pct']}%)")
        print(f"  Experience      : {s['exp_score']}%  ({s['candidate_yoe']} yrs)")
        print(f"  Education       : {s['edu_score']}%  ({s['candidate_edu']})")
        print(f"  Keyword Score   : {s['kw_score']}%")
        print(f"  Matched skills  : {s['req_skills_matched']}")
        print(f"  Missing skills  : {s['req_skills_missing']}")