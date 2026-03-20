"""
Bias Audit Module
Detects potential bias indicators in resume screening results.
Flags demographic keywords that may unfairly influence scores.
"""

import re


# Potentially biasing terms to watch for
GENDER_CODED_POSITIVE = [
    "aggressive", "ambitious", "analytical", "assertive", "autonomous",
    "confident", "decisive", "determined", "dominant", "driven",
    "independent", "leader", "outspoken", "strong"
]

GENDER_CODED_NEGATIVE = [
    "collaborative", "committed", "compassionate", "connected",
    "cooperative", "dependable", "empathetic", "enthusiastic",
    "honest", "kind", "loyal", "nurturing", "patient",
    "responsible", "supportive", "trustworthy", "warm"
]

ELITE_INSTITUTION_KEYWORDS = [
    "iit", "iim", "bits", "nit", "iisc", "aiims",
    "mit", "stanford", "harvard", "oxford", "cambridge",
    "ivy league", "premier institute"
]

DEMOGRAPHIC_KEYWORDS = [
    "he ", "she ", "his ", "her ", "him ",
    "mr.", "mrs.", "ms.", "miss",
    "male", "female", "gender",
    "age", "years old", "born in",
    "nationality", "religion", "caste",
    "married", "single", "divorced"
]


def audit_resume(resume_text: str, filename: str) -> dict:
    """
    Audit a single resume for potential bias indicators.
    Returns a dict with flags and explanations.
    """
    text_lower = resume_text.lower()
    flags      = []
    warnings   = []

    # Check for demographic keywords
    found_demographic = []
    for kw in DEMOGRAPHIC_KEYWORDS:
        if kw in text_lower:
            found_demographic.append(kw.strip())
    if found_demographic:
        flags.append("demographic_info")
        warnings.append(
            f"Resume contains demographic keywords: {', '.join(found_demographic[:3])}. "
            f"Consider anonymizing before screening."
        )

    # Check for elite institution mentions
    found_elite = []
    for kw in ELITE_INSTITUTION_KEYWORDS:
        if kw in text_lower:
            found_elite.append(kw.upper())
    if found_elite:
        flags.append("elite_institution")
        warnings.append(
            f"Resume mentions elite institutions: {', '.join(found_elite)}. "
            f"Ensure scoring is skill-based, not prestige-based."
        )

    # Check JD for gender-coded language
    masculine_count = sum(
        1 for w in GENDER_CODED_POSITIVE if w in text_lower
    )
    feminine_count = sum(
        1 for w in GENDER_CODED_NEGATIVE if w in text_lower
    )
    if masculine_count > 3:
        flags.append("masculine_coded_language")
        warnings.append(
            f"Resume uses {masculine_count} masculine-coded words. "
            f"May disadvantage certain candidates."
        )

    return {
        "filename":          filename,
        "flags":             flags,
        "warnings":          warnings,
        "has_bias_risk":     len(flags) > 0,
        "demographic_found": found_demographic,
        "elite_found":       found_elite,
        "risk_level":        "High" if len(flags) >= 2
                             else "Medium" if len(flags) == 1
                             else "Low",
    }


def audit_jd(jd_text: str) -> dict:
    """
    Audit the Job Description for biased language.
    """
    text_lower = jd_text.lower()
    flags      = []
    warnings   = []

    # Gender-coded language in JD
    masculine_words = [w for w in GENDER_CODED_POSITIVE if w in text_lower]
    feminine_words  = [w for w in GENDER_CODED_NEGATIVE if w in text_lower]

    if len(masculine_words) > 3:
        flags.append("masculine_coded_jd")
        warnings.append(
            f"JD contains masculine-coded words: {', '.join(masculine_words[:5])}. "
            f"This may discourage diverse applicants."
        )

    if len(feminine_words) > 3:
        flags.append("feminine_coded_jd")
        warnings.append(
            f"JD contains feminine-coded words: {', '.join(feminine_words[:5])}."
        )

    # Check for unnecessary requirements
    unnecessary = []
    if re.search(r'\b(degree|phd|masters)\s+required\b', text_lower):
        if not re.search(r'(research|academia|scientist)', text_lower):
            unnecessary.append("degree requirement may be unnecessary")

    if unnecessary:
        flags.append("unnecessary_requirements")
        warnings.append(
            f"Potential unnecessary requirements: {', '.join(unnecessary)}"
        )

    return {
        "flags":        flags,
        "warnings":     warnings,
        "has_bias_risk": len(flags) > 0,
        "masculine_coded_words": masculine_words,
        "feminine_coded_words":  feminine_words,
        "risk_level":   "High" if len(flags) >= 2
                        else "Medium" if len(flags) == 1
                        else "Low",
    }


def run_full_audit(ranked: list, jd_text: str) -> dict:
    """Run bias audit on all candidates and JD."""
    jd_audit       = audit_jd(jd_text)
    candidate_audits = []

    for r in ranked:
        if r.get("status") == "ok":
            audit = audit_resume(
                r.get("raw", ""),
                r.get("filename", "")
            )
            candidate_audits.append(audit)

    high_risk   = sum(1 for a in candidate_audits if a["risk_level"] == "High")
    medium_risk = sum(1 for a in candidate_audits if a["risk_level"] == "Medium")

    return {
        "jd_audit":          jd_audit,
        "candidate_audits":  candidate_audits,
        "high_risk_count":   high_risk,
        "medium_risk_count": medium_risk,
        "overall_risk":      "High" if high_risk > 0
                             else "Medium" if medium_risk > 0
                             else "Low",
    }


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    with open("data/jd.txt") as f:
        jd_text = f.read()

    jd_result = audit_jd(jd_text)
    print("JD Bias Audit:")
    print(f"  Risk level : {jd_result['risk_level']}")
    print(f"  Flags      : {jd_result['flags']}")
    for w in jd_result["warnings"]:
        print(f"  Warning    : {w}")