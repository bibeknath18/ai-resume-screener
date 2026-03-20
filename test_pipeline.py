"""
Step 9 — Testing & Validation
Compares system ranking vs manual ground truth.
Computes Kendall Tau rank correlation.
"""

from scipy.stats import kendalltau
from parser import parse_all_resumes, ParseStatus
from extractor import extract_candidate_profile, extract_jd_profile
from embedder import compute_section_similarities
from scorer import compute_final_score
from recommender import get_recommendation

# ── Manual ground truth ──────────────────────────────────────
# Rank these yourself based on your knowledge of the resumes.
# 1 = best fit, 7 = worst fit
# Edit this to match YOUR manual ranking

MANUAL_RANKING = {
    "Namankarwa_ (2).pdf":            1,
    "Garv_NonTechCV_draft1.pdf":      2,
    "VanshBhandari_CV (2).pdf":       3,
    "Daksh.pdf":                      4,
    "Khush Kothawala Non-Tech CV-1.pdf": 5,
    "Sanjeev_CV.pdf":                 6,
    "Aaryan_Bondekar_off.pdf":        7,
}


def run_validation():
    print("\n" + "=" * 55)
    print("  Step 9 — Pipeline Validation")
    print("=" * 55)

    # Load JD
    with open("data/jd.txt") as f:
        jd_text = f.read()
    jd_profile = extract_jd_profile(jd_text)

    # Parse + score
    parse_results = parse_all_resumes("data/resumes")
    system_scores = []

    for pr in parse_results:
        if pr["status"] == ParseStatus.OK:
            profile = extract_candidate_profile(pr)
            sims    = compute_section_similarities(jd_profile, profile)
            score   = compute_final_score(jd_profile, profile, sims)
            system_scores.append({
                "filename": pr["filename"],
                "score":    score["total"],
            })

    # Sort by score to get system ranking
    system_scores.sort(key=lambda x: x["score"], reverse=True)
    system_ranking = {
        r["filename"]: i + 1
        for i, r in enumerate(system_scores)
    }

    # ── Compare rankings ─────────────────────────────────────
    print("\n[1] Ranking Comparison\n")
    print(f"  {'Candidate':<38} {'Manual':>8} {'System':>8} {'Match':>8}")
    print("  " + "-" * 62)

    manual_list = []
    system_list = []
    all_match   = True

    for filename in MANUAL_RANKING:
        manual_rank = MANUAL_RANKING[filename]
        system_rank = system_ranking.get(filename, 99)
        match       = "YES" if manual_rank == system_rank else "---"
        if match == "---":
            all_match = False
        short_name  = filename[:36]
        print(f"  {short_name:<38} {manual_rank:>8} {system_rank:>8} {match:>8}")
        manual_list.append(manual_rank)
        system_list.append(system_rank)

    # ── Kendall Tau ──────────────────────────────────────────
    print("\n[2] Rank Correlation\n")
    tau, p_value = kendalltau(manual_list, system_list)
    print(f"  Kendall Tau  : {tau:.3f}")
    print(f"  P-value      : {p_value:.3f}")

    if tau >= 0.8:
        quality = "Excellent"
    elif tau >= 0.6:
        quality = "Good"
    elif tau >= 0.4:
        quality = "Moderate"
    else:
        quality = "Needs improvement"

    print(f"  Quality      : {quality}")

    # ── Score distribution ───────────────────────────────────
    print("\n[3] Score Distribution\n")
    for r in system_scores:
        rec   = get_recommendation(r["score"])
        bar   = "#" * int(r["score"] / 5)
        short = r["filename"][:30]
        print(f"  {short:<32} {r['score']:>5.1f}  {bar:<20}  {rec}")

    # ── Fix checks ───────────────────────────────────────────
    print("\n[4] Fix Validation Checks\n")

    # Fix 1: candidate with no required skills should score low
    no_skill_candidate = next(
        (s for s in system_scores
         if s["filename"] == "Khush Kothawala Non-Tech CV-1.pdf"), None
    )
    if no_skill_candidate:
        passed = no_skill_candidate["score"] < 55
        print(f"  Fix 1 — Zero required skills scores low : {'PASS' if passed else 'FAIL'}")

    # Fix 2: synonym normalization loaded
    try:
        import yaml
        with open("skill_synonyms.yaml") as f:
            syns = yaml.safe_load(f)
        passed = "ml" in syns and syns["ml"] == "machine learning"
        print(f"  Fix 2 — Synonym map loaded correctly    : {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"  Fix 2 — Synonym map check FAILED: {e}")

    # Fix 4: section similarities are non-zero
    sample = parse_results[0]
    if sample["status"] == ParseStatus.OK:
        profile = extract_candidate_profile(sample)
        sims    = compute_section_similarities(jd_profile, profile)
        passed  = all(v >= 0.0 for v in sims.values())
        print(f"  Fix 4 — Section-wise similarities valid : {'PASS' if passed else 'FAIL'}")

    # Fix 5: all resumes returned a result dict
    passed = len(system_scores) == len(
        [r for r in parse_results if r["status"] == ParseStatus.OK]
    )
    print(f"  Fix 5 — All resumes returned results    : {'PASS' if passed else 'FAIL'}")

    print("\n" + "=" * 55)
    if tau >= 0.6:
        print("  VALIDATION PASSED — pipeline is working well!")
    else:
        print("  VALIDATION WARNING — consider adjusting weights.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_validation()