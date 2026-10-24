# DocMind — AI Document, Spreadsheet & Image Q&A

Upload PDFs, Excel spreadsheets, or images. Ask questions in natural language and receive instant, structured answers powered by Groq and open-source AI models.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Fast%20Inference-f55036?style=flat)](https://groq.com)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)

**Live Demo:** [docmindswaroop.streamlit.app](https://docmindswaroop.streamlit.app)

---

## ✨ Features

- 📄 **PDF Q&A & Document Extraction**: Multi-page document text extraction using PyMuPDF, intelligent text chunking with overlap, semantic candidate extraction with RoBERTa, and contextual synthesis using high-speed LLMs.
- 📊 **Excel & Spreadsheet Intelligence**: Support for multi-sheet `.xlsx`, `.xls`, `.xlsb`, and `.xlsm` workbooks with automatic engine routing (`openpyxl`, `pyxlsb`, `xlrd`). Answers questions about row counts, column schemas, data trends, extremes, and statistical distributions.
- 🖼️ **Image & Visual Understanding**: Captions and visual descriptions generated using Salesforce BLIP, combined with reasoning models to analyze objects, mood, settings, and visible text.
- ⚡ **High-Speed Free LLM Inference via Groq**: Powered by Groq's high-throughput LPU inference engine with zero cost using free developer API keys.
- 🔄 **Resilient Multi-Tier Fallback Architecture**:
  - **Primary (Groq)**: `qwen/qwen3.8-27b` → `openai/gpt-oss-120b` → `openai/gpt-oss-20b`
  - **Secondary (Hugging Face Router)**: `meta-llama/Llama-3.1-8B-Instruct` (Novita → DeepInfra → Nscale → base router)
  - Automatic provider detection, rate-limit retries with backoff, and graceful error handling.
- 💡 **Dynamic Smart Suggestions**: Instant pre-populated prompts tailored to the uploaded file type (PDF, Spreadsheet, or Image).
- 🎨 **Modern Minimalist UI**: Dark-mode glassmorphic interface built with custom CSS, Outfit typography, responsive file preview, metadata stats, and clean conversation threads.

---

## 📁 Supported File Formats

| Format | Extensions | Engine / Processing |
|--------|------------|---------------------|
| **PDF** | `.pdf` | PyMuPDF (`fitz`) |
| **Spreadsheets** | `.xlsx`, `.xls`, `.xlsb`, `.xlsm` | pandas (`openpyxl`, `pyxlsb`, `xlrd`) |
| **Images** | `.png`, `.jpg`, `.jpeg`, `.webp` | Pillow (`PIL`) & Salesforce BLIP |

---

## 🤖 AI Models & Pipeline

| Task | Primary Provider & Model | Fallback Providers & Models |
|------|--------------------------|-----------------------------|
| **Text Generation & Reasoning** | **Groq**: `qwen/qwen3.8-27b` | **Groq**: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`<br>**Hugging Face**: `meta-llama/Llama-3.1-8B-Instruct` (Novita, DeepInfra, Nscale) |
| **Extractive Document Q&A** | **Hugging Face**: `deepset/roberta-base-squad2` | Extractive chunk scoring before LLM synthesis |
| **Image Captioning & Vision** | **Hugging Face**: `Salesforce/blip-image-captioning-large` | Vision caption piped into LLM visual reasoning prompt |

---

## 🚀 Getting Started

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/swaroop-bug/DocMind-Swaroop.git
cd DocMind-Swaroop
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.streamlit/secrets.toml` file in the project root:

```toml
# Recommended: Free & ultra-fast LLM inference from https://console.groq.com/keys
GROQ_API_KEY = "gsk_your_groq_api_key_here"

# Optional: For Hugging Face extractive QA & image captioning from https://huggingface.co/settings/tokens
HF_TOKEN = "hf_your_huggingface_token_here"
```

> **Note:** The application automatically detects configured credentials. If `GROQ_API_KEY` is present, it uses Groq's high-speed inference. If only `HF_TOKEN` is provided, it falls back to Hugging Face router endpoints.

### 3. Run Locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy on Streamlit Cloud

1. Push your repository to GitHub (ensure `.streamlit/secrets.toml` is ignored by `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository (`DocMind-Swaroop`), branch (`main`), and set the main file path to `app.py`.
4. Open **Advanced settings → Secrets** and add your credentials:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   HF_TOKEN = "hf_your_huggingface_token_here"
   ```
5. Click **Deploy**.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM Inference:** [Groq Cloud API](https://groq.com/) & [Hugging Face Inference API](https://huggingface.co/docs/api-inference/)
- **PDF Extraction:** [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Data & Spreadsheet Engine:** [pandas](https://pandas.pydata.org/), [openpyxl](https://openpyxl.readthedocs.io/), [pyxlsb](https://github.com/dimitrje/pyxlsb), [xlrd](https://xlrd.readthedocs.io/)
- **Image Processing:** [Pillow (PIL)](https://pillow.readthedocs.io/)
- **HTTP Client:** [httpx](https://www.python-httpx.org/)
- **Language:** Python 3.10+

---

## 👤 Author

**Swaroop**
- GitHub: [@swaroop-bug](https://github.com/swaroop-bug)
