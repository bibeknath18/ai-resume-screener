"""
PDF Report Generator
Generates a formatted PDF report with candidate cards.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# Color scheme
COLOR_STRONG   = colors.HexColor("#3B6D11")
COLOR_MODERATE = colors.HexColor("#854F0B")
COLOR_NOT_FIT  = colors.HexColor("#A32D2D")
COLOR_HEADER   = colors.HexColor("#1a1a2e")
COLOR_LIGHT_BG = colors.HexColor("#F8F9FA")
COLOR_BORDER   = colors.HexColor("#DEE2E6")


def get_rec_color(recommendation: str):
    if recommendation == "Strong Fit":
        return COLOR_STRONG
    elif recommendation == "Moderate Fit":
        return COLOR_MODERATE
    else:
        return COLOR_NOT_FIT


def generate_pdf_report(
    ranked: list,
    jd_profile: dict,
    output_path: str = "outputs/screening_report.pdf"
) -> str:
    """Generate a formatted PDF report from ranked candidates."""

    os.makedirs("outputs", exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Title page header ────────────────────────────────────
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=24,
        textColor=COLOR_HEADER,
        spaceAfter=6,
        alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.gray,
        spaceAfter=20,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=COLOR_HEADER,
        spaceBefore=16,
        spaceAfter=8
    )
    normal_style = ParagraphStyle(
        "Normal2",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=4,
        leading=14
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontSize=10,
        leftIndent=16,
        spaceAfter=3,
        leading=14
    )

    story.append(Paragraph("AI Resume Screening Report", title_style))
    story.append(Paragraph(f"Role: {jd_profile.get('role', 'N/A')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_BORDER))
    story.append(Spacer(1, 0.5*cm))

    # ── Summary table ────────────────────────────────────────
    scored    = [r for r in ranked if r.get("total") is not None]
    strong    = sum(1 for r in scored if r.get("recommendation") == "Strong Fit")
    moderate  = sum(1 for r in scored if r.get("recommendation") == "Moderate Fit")
    not_fit   = sum(1 for r in scored if r.get("recommendation") == "Not Fit")

    story.append(Paragraph("Executive Summary", heading_style))

    summary_data = [
        ["Total Screened", "Strong Fit", "Moderate Fit", "Not Fit"],
        [str(len(scored)), str(strong), str(moderate), str(not_fit)],
    ]
    summary_table = Table(summary_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), COLOR_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 11),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWHEIGHT",   (0,0), (-1,-1), 28),
        ("GRID",        (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ("BACKGROUND",  (0,1), (-1,1), COLOR_LIGHT_BG),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Rankings overview table ───────────────────────────────
    story.append(Paragraph("Candidate Rankings", heading_style))

    rank_data = [["Rank", "Candidate", "Score", "Skills", "Exp", "Recommendation"]]
    for r in scored:
        rank_data.append([
            str(r.get("rank", "")),
            r.get("name", r.get("filename",""))[:30],
            f"{r.get('total', 0):.1f}",
            f"{r.get('req_match_pct', 0):.0f}%",
            f"{r.get('candidate_yoe', 0):.1f} yrs",
            r.get("recommendation", ""),
        ])

    rank_table = Table(
        rank_data,
        colWidths=[1.5*cm, 5.5*cm, 2*cm, 2*cm, 2.5*cm, 3.5*cm]
    )
    rank_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), COLOR_HEADER),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWHEIGHT",   (0,0), (-1,-1), 22),
        ("GRID",        (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [COLOR_LIGHT_BG, colors.white]),
    ]))
    story.append(rank_table)
    story.append(Spacer(1, 0.8*cm))

    # ── Individual candidate cards ────────────────────────────
    story.append(Paragraph("Detailed Candidate Analysis", heading_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER))

    for r in scored:
        story.append(Spacer(1, 0.4*cm))

        # Candidate header
        rec_color = get_rec_color(r.get("recommendation", "Not Fit"))
        name      = r.get("name", r.get("filename","").replace(".pdf",""))

        header_data = [[
            Paragraph(f"<b>#{r.get('rank')} {name}</b>", ParagraphStyle(
                "CH", parent=styles["Normal"],
                fontSize=12, textColor=colors.white
            )),
            Paragraph(
                f"<b>{r.get('total', 0):.1f}/100</b>",
                ParagraphStyle("CS", parent=styles["Normal"],
                               fontSize=14, textColor=colors.white,
                               alignment=TA_CENTER)
            ),
            Paragraph(
                f"<b>{r.get('recommendation','')}</b>",
                ParagraphStyle("CR", parent=styles["Normal"],
                               fontSize=11, textColor=colors.white,
                               alignment=TA_CENTER)
            ),
        ]]
        header_table = Table(header_data, colWidths=[9*cm, 3*cm, 5*cm])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), rec_color),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("ROWHEIGHT",  (0,0), (-1,-1), 32),
            ("LEFTPADDING",(0,0), (0,-1), 12),
        ]))
        story.append(header_table)

        # Score breakdown
        score_data = [
            ["Component", "Score", "Details"],
            ["Skills Match",
             f"{r.get('skill_score',0):.1f}%",
             f"Required: {r.get('req_match_pct',0):.0f}% | Optional: {r.get('opt_match_pct',0):.0f}%"],
            ["Experience",
             f"{r.get('exp_score',0):.1f}%",
             f"{r.get('candidate_yoe',0):.1f} years"],
            ["Education",
             f"{r.get('edu_score',0):.1f}%",
             r.get("candidate_edu", "unknown")],
            ["Keywords",
             f"{r.get('kw_score',0):.1f}%",
             "Semantic relevance to JD"],
        ]
        score_table = Table(score_data, colWidths=[5*cm, 3*cm, 9*cm])
        score_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#E9ECEF")),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("ALIGN",         (1,0), (1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ROWHEIGHT",     (0,0), (-1,-1), 20),
            ("GRID",          (0,0), (-1,-1), 0.5, COLOR_BORDER),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, COLOR_LIGHT_BG]),
        ]))
        story.append(score_table)

        # Strengths and Gaps
        col_data = [[
            Paragraph("<b>Strengths</b>", ParagraphStyle(
                "SH", parent=styles["Normal"],
                fontSize=10, textColor=COLOR_STRONG
            )),
            Paragraph("<b>Gaps</b>", ParagraphStyle(
                "GH", parent=styles["Normal"],
                fontSize=10, textColor=COLOR_NOT_FIT
            )),
        ]]
        for i in range(max(
            len(r.get("strengths",[])),
            len(r.get("gaps",[]))
        )):
            s = r.get("strengths",[])[i] if i < len(r.get("strengths",[])) else ""
            g = r.get("gaps",[])[i]      if i < len(r.get("gaps",[]))      else ""
            col_data.append([
                Paragraph(f"+ {s}" if s else "", bullet_style),
                Paragraph(f"- {g}" if g else "", bullet_style),
            ])

        col_table = Table(col_data, colWidths=[8.5*cm, 8.5*cm])
        col_table.setStyle(TableStyle([
            ("VALIGN",     (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("GRID",       (0,0), (-1,-1), 0.5, COLOR_BORDER),
            ("BACKGROUND", (0,0), (0,0), colors.HexColor("#F0F7E8")),
            ("BACKGROUND", (1,0), (1,0), colors.HexColor("#FDF0F0")),
        ]))
        story.append(col_table)

        # Reasoning
        if r.get("reasoning"):
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(
                f"<i>Reasoning: {r.get('reasoning')}</i>",
                ParagraphStyle("R", parent=styles["Normal"],
                               fontSize=9, textColor=colors.gray,
                               leftIndent=8)
            ))

        story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER))

    # Build PDF
    doc.build(story)
    return output_path


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    from pipeline import run_pipeline
    from extractor import extract_jd_profile

    with open("data/jd.txt") as f:
        jd_text = f.read()
    jd_profile = extract_jd_profile(jd_text)

    ranked = run_pipeline(
        jd_path="data/jd.txt",
        resumes_dir="data/resumes",
        use_llm=False
    )

    path = generate_pdf_report(ranked, jd_profile)
    print(f"Report generated: {path}")