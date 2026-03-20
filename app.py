"""
Step 10 — Streamlit Demo UI
Full interactive web app for AI Resume Screener.
"""

# Auto-download spaCy model if not present
import subprocess
import sys
try:
    import spacy
    spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
from bias_audit import run_full_audit
from pdf_report import generate_pdf_report
import os
import tempfile
import pandas as pd
import streamlit as st

from parser import parse_pdf, ParseStatus
from extractor import extract_candidate_profile, extract_jd_profile
from embedder import compute_section_similarities
from scorer import compute_final_score
from llm_insights import get_insights_sync
from recommender import merge_result, rank_candidates, get_recommendation_color, get_recommendation

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.big-score {
    font-size: 48px;
    font-weight: 700;
    line-height: 1;
}
.rec-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
}
.strong  { background: #EAF3DE; color: #3B6D11; }
.moderate{ background: #FAEEDA; color: #854F0B; }
.notfit  { background: #FCEBEB; color: #A32D2D; }
.failed  { background: #F1EFE8; color: #5F5E5A; }
.section-header {
    font-size: 13px;
    font-weight: 600;
    color: #888780;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


def get_badge_html(recommendation: str) -> str:
    cls_map = {
        "Strong Fit":   "strong",
        "Moderate Fit": "moderate",
        "Not Fit":      "notfit",
        "Parsing Failed": "failed",
    }
    cls = cls_map.get(recommendation, "failed")
    return f'<span class="rec-badge {cls}">{recommendation}</span>'


def score_bar(label: str, value: float, max_val: float = 100):
    """Render a labeled progress bar."""
    pct = min(value / max_val, 1.0) if max_val > 0 else 0
    st.markdown(f"<div class='section-header'>{label}</div>",
                unsafe_allow_html=True)
    st.progress(pct)
    st.caption(f"{value:.1f} / {max_val:.0f}")


@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("AI Resume Screener")
    st.markdown("---")

    st.subheader("Job Description")
    jd_source = st.radio(
        "JD Source",
        ["Use sample JD", "Upload JD file", "Paste JD text"]
    )

    jd_meta_path = "jd_meta.json"

    # Multiple JD support
    st.markdown("**How many JDs to compare?**")
    num_jds = st.radio("", [1, 2, 3], horizontal=True)

    jd_texts  = []
    jd_labels = []

    for i in range(num_jds):
        st.markdown(f"**JD {i+1}**" if num_jds > 1 else "")
        jd_source_i = st.selectbox(
            f"Source for JD {i+1}" if num_jds > 1 else "JD Source",
            ["Use sample JD", "Upload JD file", "Paste JD text"],
            key=f"jd_source_{i}"
        )
        label_i = st.text_input(
            "JD Label",
            value=f"Role {i+1}" if num_jds > 1 else "ML Engineer",
            key=f"jd_label_{i}"
        )
        jd_labels.append(label_i)

        if jd_source_i == "Use sample JD":
            if os.path.exists("data/jd.txt"):
                with open("data/jd.txt") as f:
                    jd_texts.append(f.read())
                st.success("Sample JD loaded")
            else:
                st.error("data/jd.txt not found")
                jd_texts.append("")

        elif jd_source_i == "Upload JD file":
            jd_file = st.file_uploader(
                "Upload JD (.txt)", type=["txt"], key=f"jd_file_{i}"
            )
            if jd_file:
                jd_texts.append(jd_file.read().decode("utf-8"))
                st.success("JD uploaded")
            else:
                jd_texts.append("")

        elif jd_source_i == "Paste JD text":
            txt = st.text_area("Paste JD here", height=150, key=f"jd_text_{i}")
            jd_texts.append(txt)

    # Use first JD as primary for backward compatibility
    jd_text = jd_texts[0] if jd_texts else ""

    st.markdown("---")
    st.subheader("Resumes")
    resume_source = st.radio(
        "Resume Source",
        ["Use sample resumes", "Upload resumes"]
    )

    uploaded_files = []
    if resume_source == "Upload resumes":
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF or DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )

    st.subheader("Settings")
    use_llm = st.toggle("Enable LLM insights", value=False)
    if use_llm:
        st.info("LLM insights require Anthropic API credits.")

    st.markdown("---")
    st.subheader("Scoring Weights")
    st.caption("Must add up to 100%")
    w_skills = st.slider("Skills weight %",     10, 60, 40, 5)
    w_exp    = st.slider("Experience weight %", 10, 50, 30, 5)
    w_edu    = st.slider("Education weight %",   5, 30, 15, 5)
    w_kw     = st.slider("Keywords weight %",    5, 30, 15, 5)
    total_w  = w_skills + w_exp + w_edu + w_kw
    if total_w != 100:
        st.warning(f"Weights sum to {total_w}% — adjust to reach 100%")
    else:
        st.success("Weights sum to 100%")

    run_btn = st.button("Screen Candidates", type="primary", use_container_width=True)


# ── Main area ────────────────────────────────────────────────
st.title("AI Resume Screener")
st.markdown("Semantic matching + LLM insights for smarter hiring.")

if not run_btn:
    st.info("Configure your JD and resumes in the sidebar, then click **Screen Candidates**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Embedding Model", "all-MiniLM-L6-v2")
    with col2:
        st.metric("LLM Model", "Claude Haiku")
    with col3:
        st.metric("Scoring Formula", "Skills 40% + Exp 30% + Edu 15% + KW 15%")
    st.stop()


# ── Run pipeline ─────────────────────────────────────────────
if not jd_text:
    st.error("Please provide a Job Description first.")
    st.stop()

with st.spinner("Loading embedding model..."):
    load_embedding_model()

# Parse resumes
parse_results = []

if resume_source == "Use sample resumes":
    resumes_dir = "data/resumes"
    pdf_files   = [f for f in os.listdir(resumes_dir) if f.endswith(".pdf")]
    for fname in pdf_files:
        path   = os.path.join(resumes_dir, fname)
        result = parse_pdf(path)
        parse_results.append(result)

elif resume_source == "Upload resumes" and uploaded_files:
    for uf in uploaded_files:
        suffix = ".docx" if uf.name.endswith(".docx") else ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        result = parse_pdf(tmp_path)
        result["filename"] = uf.name
        parse_results.append(result)
        os.unlink(tmp_path)

if not parse_results:
    st.error("No resumes found. Please upload resumes or use sample resumes.")
    st.stop()

# Extract JD profile
jd_profile = extract_jd_profile(jd_text, jd_meta_path)

# Process candidates
ok_results     = [r for r in parse_results if r["status"] == ParseStatus.OK]
failed_results = [r for r in parse_results if r["status"] != ParseStatus.OK]

profiles    = [extract_candidate_profile(r) for r in ok_results]
score_dicts = []

with st.spinner(f"Scoring {len(profiles)} candidates..."):
    for profile in profiles:
        sims  = compute_section_similarities(jd_profile, profile)
        score = compute_final_score(
            jd_profile, profile, sims,
            w_skills=w_skills/100,
            w_exp=w_exp/100,
            w_edu=w_edu/100,
            w_kw=w_kw/100
        )
        score_dicts.append(score)

# LLM insights
insights_list = []
if use_llm:
    with st.spinner("Generating LLM insights..."):
        from llm_insights import get_all_insights
        insights_list = get_all_insights(jd_profile, profiles)
else:
    for profile in profiles:
        insights_list.append({
            "filename":           profile["filename"],
            "strengths":          ["Enable LLM insights for detailed analysis"],
            "gaps":               ["Enable LLM insights for detailed analysis"],
            "reasoning":          "LLM insights disabled.",
            "llm_recommendation": get_recommendation(score_dicts[
                profiles.index(profile)]["total"]),
        })

# Merge + rank
results = []
for parse_result, profile, score_dict, insights in zip(
    ok_results, profiles, score_dicts, insights_list
):
    result = merge_result(parse_result, score_dict, profile, insights)
    results.append(result)

for parse_result in failed_results:
    result = merge_result(parse_result, {}, {}, {})
    results.append(result)

ranked = rank_candidates(results)

# ── Multi JD comparison ──────────────────────────────────────
if num_jds > 1:
    st.markdown("---")
    st.subheader("Multi-JD Comparison")
    st.info("Showing how candidates rank across all job descriptions.")

    all_jd_scores = {}
    for ji, jd_txt in enumerate(jd_texts):
        if not jd_txt:
            continue
        jd_prof_i = extract_jd_profile(jd_txt, jd_meta_path)
        scores_i  = []
        for profile in profiles:
            sims_i  = compute_section_similarities(jd_prof_i, profile)
            score_i = compute_final_score(
                jd_prof_i, profile, sims_i,
                w_skills=w_skills/100,
                w_exp=w_exp/100,
                w_edu=w_edu/100,
                w_kw=w_kw/100
            )
            scores_i.append(score_i["total"])
        all_jd_scores[jd_labels[ji]] = scores_i

    candidate_names = [
        p.get("name", p.get("filename","").replace(".pdf",""))
        for p in profiles
    ]
    compare_df = pd.DataFrame(all_jd_scores, index=candidate_names)
    st.dataframe(compare_df, use_container_width=True)
    st.bar_chart(compare_df)


# ── Summary metrics ──────────────────────────────────────────
st.markdown("---")
st.subheader("Results Summary")

total     = len(ranked)
strong    = sum(1 for r in ranked if r.get("recommendation") == "Strong Fit")
moderate  = sum(1 for r in ranked if r.get("recommendation") == "Moderate Fit")
not_fit   = sum(1 for r in ranked if r.get("recommendation") == "Not Fit")
failed    = sum(1 for r in ranked if r.get("recommendation") == "Parsing Failed")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Screened", total)
col2.metric("Strong Fit",     strong)
col3.metric("Moderate Fit",   moderate)
col4.metric("Not Fit",        not_fit)
col5.metric("Parse Failed",   failed)

# ── Results table ────────────────────────────────────────────
st.markdown("---")
st.subheader("Ranked Candidates")

table_rows = []
for r in ranked:
    if r.get("total") is not None:
        table_rows.append({
            "Rank":           r.get("rank"),
            "Candidate": r.get("name", r.get("filename", "").replace(".pdf", "")),
            "Score":          r.get("total"),
            "Skill Match":    f"{r.get('req_match_pct', 0):.0f}%",
            "Experience":     f"{r.get('candidate_yoe', 0):.0f} yrs",
            "Education":      r.get("candidate_edu", ""),
            "Recommendation": r.get("recommendation"),
        })

if table_rows:
    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart comparing all candidates
    st.markdown("---")
    st.subheader("Score Comparison")
    chart_data = pd.DataFrame({
        "Candidate": [r.get("name", r.get("filename","").replace(".pdf",""))
                      for r in ranked if r.get("total") is not None],
        "Score":     [r.get("total") for r in ranked if r.get("total") is not None],
    })
    st.bar_chart(chart_data.set_index("Candidate"))

# ── Detailed candidate cards ─────────────────────────────────
st.markdown("---")
st.subheader("Detailed Candidate Analysis")

for r in ranked:
    if r.get("total") is None:
        continue

    with st.expander(
        f"#{r['rank']} {r.get('name', r['filename'].replace('.pdf',''))} — {r['total']}/100"
    ):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(
                f"<div class='big-score'>{r['total']}</div><div style='color:#888'>/ 100</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                get_badge_html(r["recommendation"]),
                unsafe_allow_html=True
            )
            st.markdown("")
            st.caption(f"Experience: {r.get('candidate_yoe', 0):.0f} years")
            st.caption(f"Education: {r.get('candidate_edu', 'unknown')}")

        with col2:
            score_bar("Skill match (required)",
                      r.get("req_match_pct", 0), 100)
            score_bar("Skill match (optional)",
                      r.get("opt_match_pct", 0), 100)
            score_bar("Experience score",
                      r.get("exp_score", 0), 100)
            score_bar("Education score",
                      r.get("edu_score", 0), 100)
            score_bar("Keyword relevance",
                      r.get("kw_score", 0), 100)

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Strengths**")
            for s in r.get("strengths", []):
                st.markdown(f"+ {s}")

            st.markdown("**Matched required skills**")
            matched = r.get("req_skills_matched", [])
            if matched:
                st.markdown(", ".join(f"`{s}`" for s in matched))
            else:
                st.markdown("_None matched_")

        with col4:
            st.markdown("**Gaps**")
            for g in r.get("gaps", []):
                st.markdown(f"- {g}")

            st.markdown("**Missing required skills**")
            missing = r.get("req_skills_missing", [])
            if missing:
                st.markdown(", ".join(f"`{s}`" for s in missing))
            else:
                st.markdown("_No gaps_")

        if r.get("reasoning"):
            st.info(f"Reasoning: {r['reasoning']}")

# ── Bias audit ───────────────────────────────────────────────
st.markdown("---")
with st.expander("Bias Audit Report", expanded=False):
    st.markdown("Checks for potential bias indicators in resumes and JD.")
    audit_results = run_full_audit(ranked, jd_text)

    jd_audit = audit_results["jd_audit"]
    risk_color = {
        "Low":    "green",
        "Medium": "orange",
        "High":   "red"
    }

    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Overall Risk",    audit_results["overall_risk"])
    col_b2.metric("High Risk Resumes",   audit_results["high_risk_count"])
    col_b3.metric("Medium Risk Resumes", audit_results["medium_risk_count"])

    st.markdown("**JD Bias Check**")
    if jd_audit["has_bias_risk"]:
        for w in jd_audit["warnings"]:
            st.warning(w)
    else:
        st.success("No bias indicators found in Job Description.")

    st.markdown("**Resume Bias Flags**")
    for audit in audit_results["candidate_audits"]:
        if audit["has_bias_risk"]:
            with st.expander(f"{audit['filename']} — Risk: {audit['risk_level']}"):
                for w in audit["warnings"]:
                    st.warning(w)
        else:
            st.success(f"{audit['filename']} — Low risk")


# ── Failed resumes ───────────────────────────────────────────
failed_list = [r for r in ranked if r.get("recommendation") == "Parsing Failed"]
if failed_list:
    st.markdown("---")
    st.subheader("Failed to Parse")
    for r in failed_list:
        st.warning(f"{r['filename']} — {r.get('error', 'Unknown error')}")

# ── Export ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Export Results")
col_exp1, col_exp2 = st.columns(2)

# ── Export ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Export Results")
col_exp1, col_exp2 = st.columns(2)

if table_rows:
    with col_exp1:
        df_export = pd.DataFrame([{
            "rank":           r.get("rank"),
            "filename":       r.get("filename"),
            "total_score":    r.get("total"),
            "recommendation": r.get("recommendation"),
            "req_match_pct":  r.get("req_match_pct"),
            "opt_match_pct":  r.get("opt_match_pct"),
            "exp_score":      r.get("exp_score"),
            "edu_score":      r.get("edu_score"),
            "candidate_yoe":  r.get("candidate_yoe"),
            "candidate_edu":  r.get("candidate_edu"),
            "strengths":      " | ".join(r.get("strengths", [])),
            "gaps":           " | ".join(r.get("gaps", [])),
            "reasoning":      r.get("reasoning"),
        } for r in ranked if r.get("total") is not None])

        csv = df_export.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="screening_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_exp2:
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_path = generate_pdf_report(ranked, jd_profile)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f.read(),
                        file_name="screening_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )