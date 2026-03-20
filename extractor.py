"""
Step 3 — Feature Extractor
Extracts: skills (required/optional), years of experience, education level
Includes: skill synonym normalization (Fix 2) + weighted skill tagging (Fix 1)
"""

import re
import json
import yaml
import spacy
from rapidfuzz import process, fuzz

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load skill synonyms
with open("skill_synonyms.yaml") as f:
    SKILL_MAP = yaml.safe_load(f)

# Education level lookup table
EDU_TABLE = {
    "phd":        1.00,
    "doctorate":  1.00,
    "masters":    0.85,
    "msc":        0.85,
    "mtech":      0.85,
    "mba":        0.80,
    "bachelors":  0.70,
    "btech":      0.70,
    "bsc":        0.70,
    "be":         0.70,
    "bca":        0.65,
    "diploma":    0.50,
    "associate":  0.50,
    "12th":       0.30,
    "high school":0.30,
    "none":       0.20,
}

# Master skill list for matching
KNOWN_SKILLS = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c", "go", "ruby",
    "scala", "kotlin", "swift", "r", "matlab", "bash", "sql", "html", "css",
    "php", "rust", "dart", "julia", "haskell", "perl", "shell",

    # ML & AI Core
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "large language model",
    "generative ai", "machine learning operations", "feature engineering",
    "model deployment", "model optimization", "transfer learning",
    "fine tuning", "prompt engineering", "retrieval augmented generation",
    "vector database", "embeddings", "neural networks", "transformers",

    # ML Frameworks
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm",
    "catboost", "hugging face", "transformers", "langchain", "spacy", "nltk",
    "gensim", "opencv", "pillow", "fastai", "jax", "flax", "paddlepaddle",

    # MLOps & Serving
    "mlflow", "kubeflow", "wandb", "dvc", "torchserve", "triton", "bentoml",
    "ray", "seldon", "bento", "model serving", "mlops",

    # Data Science
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy",
    "statsmodels", "bokeh", "altair", "tableau", "powerbi", "looker",

    # Cloud & DevOps
    "amazon web services", "google cloud platform", "microsoft azure",
    "docker", "kubernetes", "git", "github", "gitlab", "terraform",
    "ansible", "jenkins", "ci cd", "linux", "bash",

    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "dynamodb", "sqlite", "snowflake", "bigquery", "redshift",
    "cassandra", "neo4j", "pinecone", "chromadb", "faiss", "weaviate",

    # Data Engineering
    "apache spark", "hadoop", "apache kafka", "apache airflow", "dbt",
    "fivetran", "dask", "ray", "celery", "rabbitmq",

    # Web & APIs
    "fastapi", "flask", "django", "rest api", "graphql", "grpc",
    "react", "node js", "vue js", "angular", "streamlit", "gradio",

    # NLP Specific
    "bert", "gpt", "llama", "mistral", "gemini", "stable diffusion",
    "whisper", "text classification", "named entity recognition",
    "sentiment analysis", "summarization", "question answering",
    "text generation", "tokenization", "word embeddings", "word2vec",

    # Computer Vision Specific
    "image classification", "object detection", "image segmentation",
    "yolo", "resnet", "vgg", "efficientnet", "detectron",

    # Testing & Quality
    "pytest", "unittest", "selenium", "test driven development",

    # Practices
    "agile", "scrum", "object oriented programming", "functional programming",
    "microservices", "system design", "data structures", "algorithms",
]

def normalize_skill(skill: str) -> str:
    """Normalize skill using synonym map + fuzzy matching."""
    s = skill.lower().strip()
    # Direct match in synonym map
    if s in SKILL_MAP:
        return SKILL_MAP[s]
    # Fuzzy match against synonym map keys
    if len(s) > 2:
        match, score, _ = process.extractOne(
            s, SKILL_MAP.keys(), scorer=fuzz.ratio
        )
        if score > 88:
            return SKILL_MAP[match]
    return s

def extract_candidate_name(text: str, filename: str) -> str:
    """
    Extract candidate name from resume text using spaCy NER.
    Falls back to filename if name not found.
    """
    # Try spaCy NER on first 300 chars (name is usually at top)
    doc = nlp(text[:300])
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
            return ent.text.strip()

    # Fallback: first line of resume often has the name
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        first_line = lines[0]
        # If first line looks like a name (2-4 words, no special chars)
        words = first_line.split()
        if 2 <= len(words) <= 4 and all(w.replace('.','').isalpha() for w in words):
            return first_line

    # Final fallback: clean up filename
    name = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    # Remove trailing numbers and extra words
    import re
    name = re.sub(r'\s*\(?\d+\)?\s*$', '', name)
    name = re.sub(r'\s*(cv|resume|draft|final|off|new)\s*$', '', name, flags=re.IGNORECASE)
    return name.strip().title()

def extract_skills(text: str) -> list:
    """Extract and normalize skills from text."""
    text_lower = text.lower()
    found_skills = []

    for skill in KNOWN_SKILLS:
        # Check if skill appears in text
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    # Also check for abbreviations via synonym map
    words = re.findall(r'\b\w+\b', text_lower)
    for word in words:
        normalized = normalize_skill(word)
        if normalized in KNOWN_SKILLS and normalized not in found_skills:
            found_skills.append(normalized)

    return list(set(found_skills))


def extract_years_of_experience(text: str, sections: dict = None) -> float:
    """
    Extract total years of experience from resume text.
    Improved: only looks at experience section, not education years.
    """
    # Use only experience section if available — avoids counting college years
    exp_text = ""
    if sections:
        exp_text = sections.get("experience", "")
    if not exp_text:
        exp_text = text

    text_lower = exp_text.lower()

    # Pattern 1: explicit "X years of experience" statement
    explicit_patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'experience\s+of\s+(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s+of\s+experience',
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return float(match.group(1))

    # Pattern 2: month ranges like "Jan 2023 - Mar 2024" (most accurate for students)
    month_map = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
        'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
    }
    month_pattern = (
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
        r'[a-z]*\.?\s*(20\d{2})\s*[-–—]\s*'
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|current)'
        r'[a-z]*\.?\s*(20\d{2})?'
    )
    month_ranges = re.findall(month_pattern, text_lower)
    if month_ranges:
        total_months = 0
        for m1, y1, m2, y2 in month_ranges:
            start = int(y1) * 12 + month_map.get(m1[:3], 1)
            if m2 in ['present', 'current']:
                end = 2026 * 12 + 3
            else:
                end_yr = int(y2) if y2 else 2026
                end = end_yr * 12 + month_map.get(m2[:3], 1)
            total_months += max(0, end - start)
        return round(min(total_months / 12, 20), 1)

    # Pattern 3: year ranges ONLY in experience section
    year_ranges = re.findall(
        r'(20\d{2})\s*[-–—]\s*(20\d{2}|present|current)',
        text_lower
    )
    if year_ranges:
        total_months = 0
        for start, end in year_ranges:
            start_yr = int(start)
            end_yr = 2026 if end in ['present', 'current'] else int(end)
            # Cap each single job at 5 years to avoid college year inflation
            years = min(end_yr - start_yr, 5)
            total_months += max(0, years * 12)
        # Cap total at 10 years for students
        total_years = min(total_months / 12, 10)
        return round(total_years, 1)

    # Pattern 4: count internship mentions as 0.5 years each
    internship_count = len(re.findall(r'\bintern(ship)?\b', text_lower))
    if internship_count > 0:
        return round(min(internship_count * 0.5, 2.0), 1)

    return 0.0


def extract_education(text: str) -> tuple:
    """
    Extract education level. Returns (level_string, score).
    Improved: looks specifically in education section,
    uses context-aware matching to avoid false positives.
    """
    text_lower = text.lower()

    # Specific degree patterns with context
    degree_patterns = [
        # PhD patterns
        (r'\bph\.?d\b', "phd", 1.00),
        (r'\bdoctor(ate)?\b', "phd", 1.00),
        # Masters patterns
        (r'\bm\.?tech\b', "mtech", 0.85),
        (r'\bm\.?e\.?\b', "mtech", 0.85),
        (r'\bm\.?sc\b', "msc", 0.85),
        (r'\bmaster\'?s?\b', "masters", 0.85),
        (r'\bm\.?b\.?a\b(?!\s*college)(?!\s*institute)(?!\s*school)(?!\s*university)', "mba", 0.80),
        # Bachelors patterns
        (r'\bb\.?tech\b', "btech", 0.70),
        (r'\bb\.?e\.?\b', "btech", 0.70),
        (r'\bb\.?sc\b', "bsc", 0.70),
        (r'\bbachelor\'?s?\b', "bachelors", 0.70),
        (r'\bb\.?c\.?a\b', "bca", 0.65),
        # Diploma
        (r'\bdiploma\b', "diploma", 0.50),
    ]

    for pattern, edu_key, score in degree_patterns:
        if re.search(pattern, text_lower):
            return edu_key, score

    return "none", EDU_TABLE["none"]

def extract_candidate_profile(parse_result: dict) -> dict:
    """
    Build full candidate profile from parse result.
    Input: output from parser.parse_pdf()
    Output: structured profile dict
    """
    raw = parse_result.get("raw", "")
    sections = parse_result.get("sections", {})

    # Use skills section primarily, fall back to full text
    skills_text = sections.get("skills", "") + " " + raw
    exp_text = sections.get("experience", "") + " " + raw
    edu_text = sections.get("education", "") + " " + raw

    skills = extract_skills(skills_text)
    yoe = extract_years_of_experience(exp_text, sections)
    edu_level, edu_score = extract_education(edu_text)

    name = extract_candidate_name(raw, parse_result.get("filename", ""))

    return {
        "filename": parse_result.get("filename", ""),
        "name": name,
        "skills": skills,
        "years_of_experience": yoe,
        "education_level": edu_level,
        "education_score": edu_score,
        "raw": raw,
        "sections": sections,
    }


def extract_jd_profile(jd_text: str, jd_meta_path: str = "jd_meta.json") -> dict:
    """
    Build JD profile with required/optional skill split.
    Input: raw JD text + path to jd_meta.json
    Output: structured JD profile dict
    """
    with open(jd_meta_path) as f:
        meta = json.load(f)

    required = [normalize_skill(s) for s in meta["required_skills"]]
    optional = [normalize_skill(s) for s in meta["optional_skills"]]

    return {
        "role": meta.get("role", ""),
        "required_skills": required,
        "optional_skills": optional,
        "all_skills": required + optional,
        "required_yoe": meta.get("required_yoe", 3),
        "required_education": meta.get("required_education", "bachelors"),
        "required_edu_score": EDU_TABLE.get(
            meta.get("required_education", "bachelors"), 0.70
        ),
        "raw": jd_text,
    }


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    from parser import parse_all_resumes, ParseStatus

    # Load JD
    with open("data/jd.txt") as f:
        jd_text = f.read()

    jd_profile = extract_jd_profile(jd_text)

    print("=" * 55)
    print("JD Profile")
    print("=" * 55)
    print(f"Role          : {jd_profile['role']}")
    print(f"Required YoE  : {jd_profile['required_yoe']} years")
    print(f"Required edu  : {jd_profile['required_education']}")
    print(f"Required skills ({len(jd_profile['required_skills'])}):")
    for s in jd_profile["required_skills"]:
        print(f"  - {s}")
    print(f"Optional skills ({len(jd_profile['optional_skills'])}):")
    for s in jd_profile["optional_skills"]:
        print(f"  - {s}")

    # Parse resumes
    print("\n" + "=" * 55)
    print("Candidate Profiles")
    print("=" * 55)

    parse_results = parse_all_resumes("data/resumes")
    for pr in parse_results:
        if pr["status"] == ParseStatus.OK:
            profile = extract_candidate_profile(pr)
            print(f"\nFile    : {profile['filename']}")
            print(f"Skills  : {profile['skills']}")
            print(f"YoE     : {profile['years_of_experience']} years")
            print(f"Edu     : {profile['education_level']} (score: {profile['education_score']})")