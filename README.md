"""AI Resume ATS Score Checker
# AI Resume ATS Score Checker

A Streamlit app that scores a resume for Applicant Tracking System (ATS) compatibility and suggests concrete improvements, using Google's Gemini Flash model.

## Features

- Upload a resume as **PDF** or **DOCX**
- Overall **ATS score (0-100)** with a breakdown: keywords, formatting, experience, education, readability
- Strengths, prioritized improvements, and before/after rewrite examples
- Optional **job description** input for keyword-gap analysis
- Download the report as JSON

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally

1. Get a free Gemini API key at https://aistudio.google.com/apikey
2. Install and run:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"        # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

You can also skip the environment variable and paste the key into the app's sidebar.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit your API key).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:

```toml
GEMINI_API_KEY = "your-key"
```

5. Click **Deploy**.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets or environment variable | none (or enter in sidebar) |
| Model name | Sidebar field | `gemini-flash-latest` |

## Notes and limitations

- Scanned (image-only) PDFs have no extractable text; use a text-based PDF or DOCX.
- The score is an AI estimate, not the output of any real ATS.
- Resume text is sent to the Gemini API; avoid uploading data you are not comfortable sharing.
