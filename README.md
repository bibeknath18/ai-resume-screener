**# AI Resume Screener**



**An intelligent resume screening system that uses semantic embeddings and LLM insights to rank candidates against a job description.**



**## Demo**



**Upload a Job Description + resumes and get:**

**- Match Score (0-100)**

**- Candidate Ranking**

**- Key Strengths \& Gaps**

**- Final Recommendation (Strong Fit / Moderate Fit / Not Fit)**



**## Architecture**

**```**

**PDF Resumes + Job Description**

&#x20;       **↓**

&#x20;  **PDF Parser (pdfplumber)**

&#x20;       **↓**

&#x20; **Feature Extractor (spaCy)**

&#x20;       **↓**

&#x20; **Embedding Engine (all-MiniLM-L6-v2)**

&#x20;       **↓**

&#x20; **Weighted Scorer (Skills 40% + Exp 30% + Edu 15% + KW 15%)**

&#x20;       **↓**

&#x20; **LLM Insights (Claude Haiku)**

&#x20;       **↓**

&#x20; **Ranked Output + Streamlit UI**

**```**



**## Tech Stack**



**| Component | Technology |**

**|---|---|**

**| PDF Parsing | pdfplumber, PyMuPDF |**

**| NLP / NER | spaCy (en\_core\_web\_sm) |**

**| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |**

**| LLM Insights | Anthropic Claude Haiku |**

**| Scoring | scikit-learn cosine similarity |**

**| UI | Streamlit |**

**| Validation | Pydantic, scipy (Kendall Tau) |**



**## Scoring Formula**

**```**

**Score = (Skills × 0.40) + (Experience × 0.30) + (Education × 0.15) + (Keywords × 0.15)**



**Skills Score = 0.70 × required\_match + 0.30 × optional\_match**

**```**



**## Setup \& Run**

**```bash**

**# 1. Clone the repo**

**git clone https://github.com/yourusername/ai-resume-screener**

**cd ai-resume-screener**



**# 2. Create virtual environment**

**python -m venv venv**

**venv\\Scripts\\activate  # Windows**

**source venv/bin/activate  # Mac/Linux**



**# 3. Install dependencies**

**pip install -r requirements.txt**

**python -m spacy download en\_core\_web\_sm**



**# 4. Add API key**

**echo ANTHROPIC\_API\_KEY=your\_key\_here > .env**



**# 5. Run**

**streamlit run app.py**

**```**



**## Project Structure**

**```**

**ai-resume-screener/**

**├── app.py              # Streamlit UI**

**├── pipeline.py         # Orchestrator**

**├── parser.py           # PDF parsing**

**├── extractor.py        # Feature extraction**

**├── embedder.py         # Semantic embeddings**

**├── scorer.py           # Scoring formula**

**├── llm\_insights.py     # LLM insights**

**├── recommender.py      # Recommendation logic**

**├── skill\_synonyms.yaml # Skill normalization map**

**├── jd\_meta.json        # JD required/optional skills**

**└── data/**

&#x20;   **├── jd.txt**

&#x20;   **└── resumes/**

**```**



**## Key Design Decisions**



**- \*\*Embeddings over TF-IDF\*\*: Captures semantic meaning — "ML Engineer" matches "Machine Learning" correctly**

**- \*\*Hybrid approach\*\*: Fast embeddings for scoring + LLM for qualitative insights**

**- \*\*Section-wise matching\*\*: Skills vs Skills, Experience vs Experience for higher precision**

**- \*\*Required vs Optional skills\*\*: Required skills weighted 70%, optional 30%**

**- \*\*Synonym normalization\*\*: "ML" → "machine learning", "k8s" → "kubernetes"**



**## Validation**



**Kendall Tau rank correlation: \*\*0.714\*\* (Good) against manual ground truth ranking.**

