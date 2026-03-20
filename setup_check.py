import sys
import os

REQUIRED = [
    ("pdfplumber",            "pdfplumber"),
    ("fitz",                  "pymupdf"),
    ("spacy",                 "spacy"),
    ("sentence_transformers", "sentence-transformers"),
    ("sklearn",               "scikit-learn"),
    ("numpy",                 "numpy"),
    ("pandas",                "pandas"),
    ("anthropic",             "anthropic"),
    ("dotenv",                "python-dotenv"),
    ("streamlit",             "streamlit"),
    ("pydantic",              "pydantic"),
    ("rapidfuzz",             "rapidfuzz"),
    ("yaml",                  "PyYAML"),
    ("scipy",                 "scipy"),
    ("joblib",                "joblib"),
]

print("\n" + "="*55)
print("  AI Resume Screener — Step 0 Setup Check")
print("="*55)

all_ok = True

print("\n[1] Checking Python packages...\n")
for import_name, pip_name in REQUIRED:
    try:
        __import__(import_name)
        print(f"  OK  {pip_name}")
    except ImportError:
        print(f"  MISSING  {pip_name}  →  pip install {pip_name}")
        all_ok = False

print("\n[2] Checking spaCy model...\n")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("  OK  en_core_web_sm")
except OSError:
    print("  MISSING  en_core_web_sm  →  python -m spacy download en_core_web_sm")
    all_ok = False

print("\n[3] Checking project files...\n")
required_files = [
    "data/jd.txt",
    "jd_meta.json",
    "skill_synonyms.yaml",
    ".env",
    "requirements.txt",
]
for f in required_files:
    exists = os.path.exists(f)
    status = "OK" if exists else "MISSING"
    print(f"  {status}  {f}")
    if not exists:
        all_ok = False

print("\n[4] Checking .env API key...\n")
try:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key and key != "your_api_key_here":
        print("  OK  ANTHROPIC_API_KEY is set")
    else:
        print("  WARNING  ANTHROPIC_API_KEY not set in .env")
except Exception as e:
    print(f"  ERROR  {e}")

print("\n[5] Quick embedding test...\n")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode(["hello world"])
    assert vec.shape[1] == 384
    print("  OK  SentenceTransformer loaded, embedding dim=384")
except Exception as e:
    print(f"  ERROR  {e}")
    all_ok = False

print("\n" + "="*55)
if all_ok:
    print("  ALL CHECKS PASSED — ready to move to Step 1!")
else:
    print("  SOME CHECKS FAILED — fix the above before proceeding.")
print("="*55 + "\n")