# DocMind — AI Document & Image Q&A

Upload a PDF or image. Ask anything. Get instant answers powered by open-source AI.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)

**Live Demo:** [your-app.streamlit.app](https://docmindswaroop.streamlit.app)

---

## Models

| Task | Model |
|------|-------|
| Text generation | `meta-llama/Llama-3.1-8B-Instruct` via Cerebras → Groq → Together → Fireworks |
| PDF Q&A | `deepset/roberta-base-squad2` |
| Image captioning | `Salesforce/blip-image-captioning-large` |

---

## Setup

**1. Clone & install**
```bash
git clone https://github.com/swaroop-bug/DocMind.git
cd DocMind
pip install -r requirements.txt
```

**2. Add your Hugging Face token**

Create `.streamlit/secrets.toml` (never commit this):
```toml
HF_TOKEN = "hf_your_token_here"
```

Update `app.py` line 17:
```python
HF_TOKEN = st.secrets["HF_TOKEN"]
```

**3. Run**
```bash
streamlit run app.py
```

---

## Deploy on Streamlit Cloud

1. Push code to GitHub — ensure `.streamlit/secrets.toml` is in `.gitignore`
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo → set `app.py`
3. **App Settings → Secrets** → add `HF_TOKEN = "hf_your_token_here"`
4. Deploy

> The token lives in Streamlit's encrypted vault — never in your repo.

---

## Stack

- **Frontend:** Streamlit
- **PDF extraction:** PyMuPDF
- **HTTP client:** httpx
- **Language:** Python 3.10+

---

## Author

Swaroop · [github.com/swaroop-bug](https://github.com/swaroop-bug)
