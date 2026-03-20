"""
Step 8 — Pipeline Orchestrator
Runs all steps end-to-end for a batch of resumes.
"""

import os
import json
import pandas as pd
from parser import parse_pdf, parse_all_resumes, ParseStatus
from extractor import extract_candidate_profile, extract_jd_profile
from embedder import compute_section_similarities
from scorer import compute_final_score
from llm_insights import get_all_insights
from recommender import merge_result, rank_candidates


def run_pipeline(
    jd_path:      str,
    resumes_dir:  str,
    jd_meta_path: str = "jd_meta.json",
    use_llm:      bool = True
) -> list:
    """
    Full end-to-end pipeline.
    Returns list of ranked candidate result dicts.
    """

    print("\n" + "=" * 55)
    print("  AI Resume Screener — Running Pipeline")
    print("=" * 55)

    # ── Step 1: Load JD ─────────────────────────────────────
    print("\n[1/5] Loading Job Description...")
    with open(jd_path) as f:
        jd_text = f.read()
    jd_profile = extract_jd_profile(jd_text, jd_meta_path)
    print(f"      Role: {jd_profile['role']}")
    print(f"      Required skills: {len(jd_profile['required_skills'])}")
    print(f"      Optional skills: {len(jd_profile['optional_skills'])}")

    # ── Step 2: Parse resumes ────────────────────────────────
    print("\n[2/5] Parsing resumes...")
    parse_results = parse_all_resumes(resumes_dir)
    ok_results     = [r for r in parse_results if r["status"] == ParseStatus.OK]
    failed_results = [r for r in parse_results if r["status"] != ParseStatus.OK]
    print(f"      Parsed: {len(ok_results)} OK, {len(failed_results)} failed")

    # ── Step 3: Extract features ─────────────────────────────
    print("\n[3/5] Extracting features...")
    profiles = [extract_candidate_profile(r) for r in ok_results]
    print(f"      Extracted profiles for {len(profiles)} candidates")

    # ── Step 4: Score candidates ─────────────────────────────
    print("\n[4/5] Scoring candidates...")
    score_dicts = []
    for profile in profiles:
        sims  = compute_section_similarities(jd_profile, profile)
        score = compute_final_score(jd_profile, profile, sims)
        score_dicts.append(score)
        print(f"      {profile['filename']}: {score['total']}/100")

    # ── Step 5: LLM insights ─────────────────────────────────
    insights_list = []
    if use_llm:
        print("\n[5/5] Generating LLM insights...")
        insights_list = get_all_insights(jd_profile, profiles)
    else:
        print("\n[5/5] Skipping LLM insights (use_llm=False)")
        for profile in profiles:
            insights_list.append({
                "filename":           profile["filename"],
                "strengths":          ["LLM disabled"],
                "gaps":               ["LLM disabled"],
                "reasoning":          "LLM insights not generated.",
                "llm_recommendation": "Moderate Fit",
            })

    # ── Merge + rank ─────────────────────────────────────────
    results = []

    # Merge OK candidates
    for parse_result, profile, score_dict, insights in zip(
        ok_results, profiles, score_dicts, insights_list
    ):
        result = merge_result(parse_result, score_dict, profile, insights)
        results.append(result)

    # Add failed candidates
    for parse_result in failed_results:
        result = merge_result(parse_result, {}, {}, {})
        results.append(result)

    # Rank all
    ranked = rank_candidates(results)

    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    for r in ranked:
        if r["total"] is not None:
            print(f"  #{r['rank']} {r['filename']}")
            print(f"     Score: {r['total']}/100 — {r['recommendation']}")
        else:
            print(f"  FAILED {r['filename']} — {r['recommendation']}")

    return ranked


def save_results(ranked: list, output_path: str = "outputs/results.csv"):
    """Save ranked results to CSV."""
    os.makedirs("outputs", exist_ok=True)

    rows = []
    for r in ranked:
        rows.append({
            "rank":               r.get("rank"),
            "filename":           r.get("filename"),
            "total_score":        r.get("total"),
            "recommendation":     r.get("recommendation"),
            "skill_score":        r.get("skill_score"),
            "req_match_pct":      r.get("req_match_pct"),
            "opt_match_pct":      r.get("opt_match_pct"),
            "exp_score":          r.get("exp_score"),
            "edu_score":          r.get("edu_score"),
            "kw_score":           r.get("kw_score"),
            "candidate_yoe":      r.get("candidate_yoe"),
            "candidate_edu":      r.get("candidate_edu"),
            "strengths":          " | ".join(r.get("strengths", [])),
            "gaps":               " | ".join(r.get("gaps", [])),
            "reasoning":          r.get("reasoning"),
            "llm_recommendation": r.get("llm_recommendation"),
            "status":             r.get("status"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    return df


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    ranked = run_pipeline(
        jd_path     = "data/jd.txt",
        resumes_dir = "data/resumes",
        jd_meta_path= "jd_meta.json",
        use_llm     = False      # Set True once you have API credits
    )

    df = save_results(ranked)
    print("\n--- Final DataFrame ---")
    print(df[["filename", "total_score", "recommendation"]].to_string(index=False))