import streamlit as st
import httpx
import fitz  # PyMuPDF
import tempfile
import os
import asyncio
from PIL import Image
import io
import base64

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — AI Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Dark background */
.stApp {
    background-color: #0a0c0a;
    color: #e2e8e2;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #101410 !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Header */
.doc-header {
    background: linear-gradient(135deg, #101410, #141914);
    border: 1px solid rgba(74,222,128,0.15);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    text-align: center;
}
.doc-header h1 {
    color: #4ade80;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}
.doc-header p {
    color: rgba(226,232,226,0.55);
    font-size: 1rem;
    margin-top: 6px;
}

/* Cards */
.info-card {
    background: #101410;
    border: 1px solid rgba(74,222,128,0.12);
    border-left: 4px solid #4ade80;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* Chat messages */
.user-msg {
    background: linear-gradient(135deg, #4ade80, #22c55e);
    color: #0a0c0a;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
    margin: 8px 0;
    font-weight: 500;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
}
.ai-msg {
    background: #141914;
    border: 1px solid rgba(255,255,255,0.07);
    color: #e2e8e2;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 85%;
    font-size: 0.95rem;
    line-height: 1.7;
    white-space: pre-wrap;
}
.ai-label {
    color: #4ade80;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.msg-time {
    color: rgba(255,255,255,0.2);
    font-size: 0.7rem;
    margin-bottom: 2px;
    text-align: right;
}

/* Suggestion chips */
.sugg-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0;
}

/* File info */
.file-info {
    background: rgba(74,222,128,0.05);
    border: 1px solid rgba(74,222,128,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-size: 0.85rem;
}

/* Input styling */
.stTextInput > div > div > input {
    background: #141914 !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8e2 !important;
    font-family: 'Outfit', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(74,222,128,0.5) !important;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4ade80, #22c55e) !important;
    color: #0a0c0a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(74,222,128,0.3) !important;
}

/* Secondary button */
.secondary-btn > button {
    background: rgba(255,255,255,0.05) !important;
    color: rgba(226,232,226,0.6) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #101410 !important;
    border: 2px dashed rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(74,222,128,0.4) !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.07) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #4ade80 !important;
}

/* Success/error */
.stSuccess {
    background: rgba(74,222,128,0.1) !important;
    border: 1px solid rgba(74,222,128,0.2) !important;
    color: #4ade80 !important;
}
.stError {
    background: rgba(248,113,113,0.1) !important;
    border: 1px solid rgba(248,113,113,0.2) !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else "")

CHAT_URL   = "https://router.huggingface.co/v1/chat/completions"
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
CHAT_FALLBACKS = [
    "meta-llama/Llama-3.1-8B-Instruct:groq",
    "meta-llama/Llama-3.1-8B-Instruct:together",
    "meta-llama/Llama-3.1-8B-Instruct:fireworks-ai",
]
QA_URL     = "https://router.huggingface.co/hf-inference/models/deepset/roberta-base-squad2"
VISION_URL = "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-large"

JSON_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

SUGGESTIONS = {
    "pdf": [
        "📋 What is the main topic?",
        "🔍 What are the key findings?",
        "📝 Summarize briefly",
        "❓ What problem does this address?",
        "📌 List the most important points",
    ],
    "image": [
        "🖼️ What is shown in this image?",
        "🔎 Describe the main subject in detail",
        "📝 What text appears in this image?",
        "🎨 What colors and objects are visible?",
        "💡 What is the overall theme or mood?",
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_pdf_text(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    finally:
        os.unlink(path)

def chunk_text(text: str, size=3500, overlap=200):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i: i + size])
        i += size - overlap
    return chunks

def call_chat(system: str, user: str) -> str:
    models = [CHAT_MODEL] + CHAT_FALLBACKS
    last_error = ""
    with httpx.Client(timeout=60) as client:
        for model in models:
            try:
                r = client.post(CHAT_URL, headers=JSON_HEADERS, json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.4,
                    "stream": False,
                })
                if len(r.content) == 0 or r.status_code in (503, 404):
                    last_error = f"{model} unavailable"
                    continue
                if not r.is_success:
                    last_error = f"{model} error {r.status_code}"
                    continue
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_error = str(e)
                continue
    raise ValueError(f"All models failed. Last: {last_error}")

def call_qa(question: str, context: str) -> dict:
    with httpx.Client(timeout=45) as client:
        for _ in range(3):
            r = client.post(QA_URL, headers=JSON_HEADERS, json={
                "inputs": {"question": question, "context": context}
            })
            if r.status_code == 503:
                import time; time.sleep(10)
                continue
            if r.is_success:
                return r.json()
    return {"answer": "", "score": 0}

def call_vision(image_bytes: bytes, mime: str) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": mime}
    with httpx.Client(timeout=45) as client:
        for _ in range(3):
            r = client.post(VISION_URL, headers=headers, content=image_bytes)
            if r.status_code == 503:
                import time; time.sleep(10)
                continue
            if r.is_success:
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "an image")
                return data.get("generated_text", "an image")
    return "an uploaded image"

def answer_question(file_bytes: bytes, file_type: str, mime: str, question: str) -> str:
    if not HF_TOKEN:
        return "⚠️ HF_TOKEN is not set. Please add it in your environment or Streamlit secrets."

    if file_type == "image":
        caption = call_vision(file_bytes, mime)
        return call_chat(
            system="You are a helpful image analyst. Answer questions about images clearly and in detail.",
            user=f'The image shows: "{caption}"\n\nQuestion: {question}\n\nAnswer helpfully and in detail.',
        )
    else:  # PDF
        doc_text = extract_pdf_text(file_bytes)
        if not doc_text or len(doc_text) < 50:
            return "❌ No text found in PDF. It may be a scanned image-only file."

        chunks = chunk_text(doc_text)
        best_ans, best_score = "", -1.0
        for chunk in chunks[:6]:
            try:
                res = call_qa(question, chunk)
                if res.get("score", 0) > best_score:
                    best_score = res.get("score", 0)
                    best_ans = res.get("answer", "")
            except Exception:
                continue

        context_snippet = doc_text[:4000]
        if best_ans and best_score > 0.05:
            user_msg = (
                f"A document Q&A system found this snippet: '{best_ans}' "
                f"for the question: '{question}'.\n\n"
                f"Using the document below, give a complete and clear answer:\n\n{context_snippet}"
            )
        else:
            user_msg = (
                f"Read this document and answer the question.\n\n"
                f"Document:\n{context_snippet}\n\n"
                f"Question: {question}\n\n"
                f"Answer clearly and accurately. If the answer is not in the document, say so."
            )
        return call_chat(
            system="You are an expert document analyst. Answer questions about documents precisely. Use bullet points where helpful.",
            user=user_msg,
        )

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes = None
if "file_mime" not in st.session_state:
    st.session_state.file_mime = None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px 0;'>
        <div style='background: linear-gradient(135deg, #4ade80, #16a34a);
                    border-radius: 12px; width: 48px; height: 48px;
                    display: flex; align-items: center; justify-content: center;
                    margin: 0 auto 10px auto; font-size: 24px;'>📄</div>
        <div style='font-size: 1.4rem; font-weight: 800; color: #4ade80; letter-spacing: -0.5px;'>DocMind</div>
        <div style='color: rgba(226,232,226,0.4); font-size: 0.75rem; margin-top: 2px;'>AI Document Q&A</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # File uploader
    st.markdown("<div style='color: rgba(226,232,226,0.5); font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Upload File</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF or Image",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        mime = uploaded.type
        if mime == "application/pdf":
            ftype = "pdf"
        elif mime.startswith("image/"):
            ftype = "image"
        else:
            ftype = None

        if ftype and (st.session_state.uploaded_file != uploaded.name):
            st.session_state.uploaded_file = uploaded.name
            st.session_state.file_type = ftype
            st.session_state.file_bytes = uploaded.read()
            st.session_state.file_mime = mime
            st.session_state.messages = []

    # File info
    if st.session_state.uploaded_file:
        size_kb = round(len(st.session_state.file_bytes) / 1024, 1)
        st.markdown(f"""
        <div class='file-info'>
            <div style='color: #4ade80; font-weight: 700; margin-bottom: 4px;'>
                {'📄' if st.session_state.file_type == 'pdf' else '🖼️'} {st.session_state.uploaded_file}
            </div>
            <div style='color: rgba(226,232,226,0.4);'>
                {st.session_state.file_type.upper()} &nbsp;·&nbsp; {size_kb} KB &nbsp;·&nbsp; {len(st.session_state.messages)//2} Q&A
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Image preview
        if st.session_state.file_type == "image":
            img = Image.open(io.BytesIO(st.session_state.file_bytes))
            st.image(img, use_container_width=True)

        st.markdown("---")

        # Suggestions
        st.markdown("<div style='color: rgba(226,232,226,0.3); font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Suggestions</div>", unsafe_allow_html=True)

        for sugg in SUGGESTIONS.get(st.session_state.file_type, []):
            if st.button(sugg, key=f"sugg_{sugg}", use_container_width=True):
                st.session_state["pending_question"] = sugg
                st.rerun()

        st.markdown("---")

        # Clear button
        if st.button("🗑️ New File", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_file = None
            st.session_state.file_type = None
            st.session_state.file_bytes = None
            st.session_state.file_mime = None
            st.rerun()

    else:
        st.markdown("""
        <div style='color: rgba(226,232,226,0.25); font-size: 0.85rem; text-align: center; padding: 20px 0; line-height: 1.7;'>
            Upload a PDF or image<br>to get started
        </div>
        """, unsafe_allow_html=True)

    # Token status
    st.markdown("---")
    if HF_TOKEN:
        st.markdown("<div style='text-align:center; color: #4ade80; font-size: 0.75rem;'>🟢 HF Token Connected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center; color: #f87171; font-size: 0.75rem;'>🔴 HF Token Missing</div>", unsafe_allow_html=True)

# ── MAIN AREA ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class='doc-header'>
    <h1>📄 DocMind</h1>
    <p>Upload a PDF or image — ask anything about it. Powered by open-source AI.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.uploaded_file:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='info-card'>
            <div style='font-size: 1.8rem; margin-bottom: 8px;'>📄</div>
            <div style='font-weight: 700; color: #4ade80; margin-bottom: 4px;'>PDF Support</div>
            <div style='color: rgba(226,232,226,0.5); font-size: 0.85rem;'>Upload any PDF and ask questions about its content using AI</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-card'>
            <div style='font-size: 1.8rem; margin-bottom: 8px;'>🖼️</div>
            <div style='font-weight: 700; color: #4ade80; margin-bottom: 4px;'>Image Analysis</div>
            <div style='color: rgba(226,232,226,0.5); font-size: 0.85rem;'>Upload photos or screenshots and get AI-powered visual analysis</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='info-card'>
            <div style='font-size: 1.8rem; margin-bottom: 8px;'>⚡</div>
            <div style='font-weight: 700; color: #4ade80; margin-bottom: 4px;'>Instant Answers</div>
            <div style='color: rgba(226,232,226,0.5); font-size: 0.85rem;'>Powered by Llama 3.1, RoBERTa & BLIP via Hugging Face</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; padding: 40px 0; color: rgba(226,232,226,0.2); font-size: 0.9rem;'>
        ← Upload a file from the sidebar to begin
    </div>
    """, unsafe_allow_html=True)

else:
    # Chat area
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            icon = "🖼️" if st.session_state.file_type == "image" else "📄"
            st.markdown(f"""
            <div style='text-align: center; padding: 40px 0; color: rgba(226,232,226,0.2);'>
                <div style='font-size: 3rem; margin-bottom: 12px;'>{icon}</div>
                <div style='font-size: 1.1rem; font-weight: 600;'>Ready to answer</div>
                <div style='font-size: 0.85rem; margin-top: 4px;'>Ask anything about <span style='color: rgba(226,232,226,0.4);'>{st.session_state.uploaded_file}</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style='display: flex; justify-content: flex-end; margin: 8px 0;'>
                        <div class='user-msg'>{msg['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='margin: 8px 0;'>
                        <div class='ai-label'>✦ DocMind</div>
                        <div class='ai-msg'>{msg['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")

    # Input area
    col_input, col_btn = st.columns([6, 1])

    with col_input:
        question = st.text_input(
            "Ask a question",
            placeholder=f"Ask anything about your {st.session_state.file_type}…",
            label_visibility="collapsed",
            key="question_input",
        )

    with col_btn:
        send = st.button("Send ➤", use_container_width=True)

    # Handle pending question from suggestion buttons
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        send = True

    # Process question
    if send and question and question.strip():
        st.session_state.messages.append({"role": "user", "content": question.strip()})

        with st.spinner("🤔 Thinking…"):
            try:
                answer = answer_question(
                    st.session_state.file_bytes,
                    st.session_state.file_type,
                    st.session_state.file_mime,
                    question.strip(),
                )
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    st.markdown("<div style='text-align:center; color: rgba(255,255,255,0.1); font-size: 0.75rem; margin-top: 6px;'>Press Enter or click Send</div>", unsafe_allow_html=True)