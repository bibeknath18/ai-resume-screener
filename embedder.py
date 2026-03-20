"""
Step 4 — Embedding Engine
Computes semantic similarity between JD and resume sections.
Includes: section-wise matching (Fix 4) + embedding cache for speed.
"""

import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once at startup
print("Loading embedding model...")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

CACHE_PATH = "outputs/embedding_cache.pkl"


def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for a single text."""
    if not text or not text.strip():
        return np.zeros((1, 384))
    return MODEL.encode([text[:512]])  # cap at 512 chars for speed


def get_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts."""
    if not text_a or not text_b:
        return 0.0
    emb_a = get_embedding(text_a)
    emb_b = get_embedding(text_b)
    score = cosine_similarity(emb_a, emb_b)[0][0]
    return round(float(score), 4)


def compute_section_similarities(jd_profile: dict, candidate_profile: dict) -> dict:
    """
    Fix 4 — Section-wise matching.
    Match each resume section against the corresponding JD section.
    Returns similarity scores per section.
    """
    jd_sections = jd_profile.get("raw", "")
    resume_sections = candidate_profile.get("sections", {})

    # Skills section similarity
    jd_skills_text = " ".join(
        jd_profile.get("required_skills", []) +
        jd_profile.get("optional_skills", [])
    )
    resume_skills_text = resume_sections.get("skills", "")
    if not resume_skills_text:
        resume_skills_text = candidate_profile.get("raw", "")[:300]

    skills_sim = get_similarity(jd_skills_text, resume_skills_text)

    # Experience section similarity
    jd_exp_text = jd_sections
    resume_exp_text = resume_sections.get("experience", "")
    if not resume_exp_text:
        resume_exp_text = candidate_profile.get("raw", "")
    exp_sim = get_similarity(jd_exp_text, resume_exp_text)

    # Education section similarity
    resume_edu_text = resume_sections.get("education", "")
    jd_edu_text = f"{jd_profile.get('required_education', '')} degree computer science engineering"
    edu_sim = get_similarity(jd_edu_text, resume_edu_text) if resume_edu_text else 0.5

    # Summary / keyword similarity (full JD vs full resume)
    resume_summary = resume_sections.get("summary", "")
    if not resume_summary:
        resume_summary = candidate_profile.get("raw", "")[:500]
    kw_sim = get_similarity(jd_sections[:500], resume_summary)

    return {
        "skills_sim":     skills_sim,
        "experience_sim": exp_sim,
        "education_sim":  edu_sim,
        "keyword_sim":    kw_sim,
    }


def load_cache() -> dict:
    """Load embedding cache from disk."""
    if os.path.exists(CACHE_PATH):
        return joblib.load(CACHE_PATH)
    return {}


def save_cache(cache: dict):
    """Save embedding cache to disk."""
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(cache, CACHE_PATH)


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    from parser import parse_all_resumes, ParseStatus
    from extractor import extract_candidate_profile, extract_jd_profile

    # Load JD
    with open("data/jd.txt") as f:
        jd_text = f.read()
    jd_profile = extract_jd_profile(jd_text)

    # Parse resumes
    parse_results = parse_all_resumes("data/resumes")

    print("\n" + "=" * 55)
    print("Section-wise Similarity Scores")
    print("=" * 55)

    for pr in parse_results:
        if pr["status"] == ParseStatus.OK:
            profile = extract_candidate_profile(pr)
            sims = compute_section_similarities(jd_profile, profile)
            print(f"\n{profile['filename']}")
            print(f"  Skills sim     : {sims['skills_sim']}")
            print(f"  Experience sim : {sims['experience_sim']}")
            print(f"  Education sim  : {sims['education_sim']}")
            print(f"  Keyword sim    : {sims['keyword_sim']}")