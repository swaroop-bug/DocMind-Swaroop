import streamlit as st
import httpx
import fitz  # PyMuPDF
import tempfile
import os
import time
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — AI Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── HF Token (hardcoded) ──────────────────────────────────────────────────────
HF_TOKEN = "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"   # ← Replace with your actual token

# ── API config ────────────────────────────────────────────────────────────────
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
        "📝 Summarize this document briefly",
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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

.stApp {
    background-color: #0a0c0a;
    color: #e2e8e2;
}

/* Hide sidebar */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Header */
.doc-header {
    background: linear-gradient(135deg, #101410, #141914);
    border: 1px solid rgba(74,222,128,0.15);
    border-radius: 20px;
    padding: 36px 40px 28px;
    margin-bottom: 32px;
    text-align: center;
}
.doc-header h1 {
    color: #4ade80;
    font-size: 3rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    letter-spacing: -1.5px;
}
.doc-header p {
    color: rgba(226,232,226,0.45);
    font-size: 1rem;
    margin: 0;
}

/* Feature cards */
.feature-row {
    display: flex;
    gap: 16px;
    max-width: 800px;
    margin: 0 auto 28px auto;
}
.feat-card {
    flex: 1;
    background: #101410;
    border: 1px solid rgba(74,222,128,0.1);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.feat-card .icon  { font-size: 1.8rem; margin-bottom: 8px; }
.feat-card .title { color: #4ade80; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px; }
.feat-card .desc  { color: rgba(226,232,226,0.4); font-size: 0.8rem; line-height: 1.5; }

/* Upload zone */
.upload-zone {
    background: #101410;
    border: 2px dashed rgba(74,222,128,0.25);
    border-radius: 24px;
    padding: 40px 40px 20px;
    text-align: center;
    margin: 0 auto 32px auto;
}
.upload-zone:hover { border-color: rgba(74,222,128,0.45); }
.upload-icon  { font-size: 3.5rem; display: block; margin-bottom: 12px; }
.upload-title { color: #e2e8e2; font-size: 1.3rem; font-weight: 700; margin-bottom: 6px; }
.upload-sub   { color: rgba(226,232,226,0.35); font-size: 0.88rem; margin-bottom: 20px; }

/* File uploader widget - blend into upload zone */
[data-testid="stFileUploader"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploader"] section { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stFileUploader"] label { display: none !important; }

/* File info bar */
.file-bar {
    background: rgba(74,222,128,0.07);
    border: 1px solid rgba(74,222,128,0.18);
    border-radius: 12px;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}
.file-bar-name { color: #4ade80; font-weight: 700; }
.file-bar-meta { color: rgba(226,232,226,0.4); font-size: 0.82rem; }

/* Chat messages */
.user-msg {
    background: linear-gradient(135deg, #4ade80, #22c55e);
    color: #0a0c0a;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    font-weight: 600;
    max-width: 72%;
    margin-left: auto;
    font-size: 0.95rem;
}
.ai-label {
    color: #4ade80;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 4px;
}
.ai-msg {
    background: #141914;
    border: 1px solid rgba(255,255,255,0.07);
    color: #e2e8e2;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.75;
    white-space: pre-wrap;
}

/* All buttons default: chip style */
.stButton > button {
    background: rgba(74,222,128,0.08) !important;
    color: rgba(226,232,226,0.7) !important;
    border: 1px solid rgba(74,222,128,0.2) !important;
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(74,222,128,0.15) !important;
    color: #e2e8e2 !important;
}

/* Send button */
.send-btn > button {
    background: linear-gradient(135deg, #4ade80, #22c55e) !important;
    color: #0a0c0a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}
.send-btn > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(74,222,128,0.35) !important;
}

/* New file button */
.newfile-btn > button {
    background: rgba(248,113,113,0.09) !important;
    color: #f87171 !important;
    border: 1px solid rgba(248,113,113,0.25) !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
}

/* Text input */
.stTextInput > div > div > input {
    background: #141914 !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e2e8e2 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(74,222,128,0.45) !important;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.08) !important;
}

hr { border-color: rgba(255,255,255,0.06) !important; }
.stSpinner > div { border-top-color: #4ade80 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 960px !important; }
</style>
""", unsafe_allow_html=True)

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
                if r.status_code in (503, 404) or len(r.content) == 0:
                    last_error = f"{model} unavailable"
                    continue
                if not r.is_success:
                    last_error = f"{model} error {r.status_code}"
                    continue
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_error = str(e)
    raise ValueError(f"All models failed. Last: {last_error}")

def call_qa(question: str, context: str) -> dict:
    with httpx.Client(timeout=45) as client:
        for _ in range(3):
            r = client.post(QA_URL, headers=JSON_HEADERS, json={
                "inputs": {"question": question, "context": context}
            })
            if r.status_code == 503:
                time.sleep(10)
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
                time.sleep(10)
                continue
            if r.is_success:
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "an image")
                return data.get("generated_text", "an image")
    return "an uploaded image"

def answer_question(file_bytes: bytes, file_type: str, mime: str, question: str) -> str:
    if not HF_TOKEN or HF_TOKEN == "your_huggingface_token_here":
        return "⚠️ Please replace `your_huggingface_token_here` in app.py with your actual Hugging Face token."

    if file_type == "image":
        caption = call_vision(file_bytes, mime)
        return call_chat(
            system="You are a helpful image analyst. Answer questions about images clearly and in detail.",
            user=f'The image shows: "{caption}"\n\nQuestion: {question}\n\nAnswer helpfully and in detail.',
        )
    else:
        doc_text = extract_pdf_text(file_bytes)
        if not doc_text or len(doc_text) < 50:
            return "❌ No readable text found. This PDF may be a scanned image-only file."

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
            system="You are an expert document analyst. Answer questions precisely. Use bullet points where helpful.",
            user=user_msg,
        )

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("uploaded_file", None),
    ("file_type", None),
    ("file_bytes", None),
    ("file_mime", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='doc-header'>
    <h1>📄 DocMind</h1>
    <p>Upload a PDF or image — ask anything about it. Powered by open-source AI.</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STATE A: No file → centered upload zone
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.uploaded_file:

    # Feature cards
    st.markdown("""
    <div class='feature-row'>
        <div class='feat-card'>
            <div class='icon'>📄</div>
            <div class='title'>PDF Q&A</div>
            <div class='desc'>Upload any PDF and ask questions about its content using AI</div>
        </div>
        <div class='feat-card'>
            <div class='icon'>🖼️</div>
            <div class='title'>Image Analysis</div>
            <div class='desc'>Upload photos or screenshots for AI-powered visual analysis</div>
        </div>
        <div class='feat-card'>
            <div class='icon'>⚡</div>
            <div class='title'>Instant Answers</div>
            <div class='desc'>Llama 3.1 · RoBERTa · BLIP via Hugging Face free inference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Centered upload — 3-column layout to constrain width
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        st.markdown("""
        <div class='upload-zone'>
            <span class='upload-icon'>⬆️</span>
            <div class='upload-title'>Drop your file here</div>
            <div class='upload-sub'>Supports PDF · PNG · JPG · WEBP</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload file",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            mime = uploaded.type
            ftype = "pdf" if mime == "application/pdf" else ("image" if mime.startswith("image/") else None)
            if ftype:
                st.session_state.uploaded_file = uploaded.name
                st.session_state.file_type = ftype
                st.session_state.file_bytes = uploaded.read()
                st.session_state.file_mime = mime
                st.session_state.messages = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STATE B: File uploaded → chat UI
# ═══════════════════════════════════════════════════════════════════════════════
else:
    file_icon = "📄" if st.session_state.file_type == "pdf" else "🖼️"
    size_kb   = round(len(st.session_state.file_bytes) / 1024, 1)
    qa_count  = len(st.session_state.messages) // 2

    # File info bar + New File button
    col_info, col_clear = st.columns([5, 1])
    with col_info:
        st.markdown(f"""
        <div class='file-bar'>
            <span style='font-size:1.5rem'>{file_icon}</span>
            <div>
                <div class='file-bar-name'>{st.session_state.uploaded_file}</div>
                <div class='file-bar-meta'>
                    {st.session_state.file_type.upper()} &nbsp;·&nbsp;
                    {size_kb} KB &nbsp;·&nbsp;
                    {qa_count} Q&amp;A pair{'s' if qa_count != 1 else ''}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_clear:
        st.markdown("<div class='newfile-btn'>", unsafe_allow_html=True)
        if st.button("🗑️ New File", use_container_width=True):
            for k in ["messages", "uploaded_file", "file_type", "file_bytes", "file_mime"]:
                st.session_state[k] = [] if k == "messages" else None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Image preview
    if st.session_state.file_type == "image":
        _, img_col, _ = st.columns([2, 1, 2])
        with img_col:
            img = Image.open(io.BytesIO(st.session_state.file_bytes))
            st.image(img, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Suggestion chips
    sugg_list = SUGGESTIONS.get(st.session_state.file_type, [])
    if sugg_list:
        cols = st.columns(len(sugg_list))
        for i, sugg in enumerate(sugg_list):
            with cols[i]:
                if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                    st.session_state["pending_question"] = sugg
                    st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Chat history
    if not st.session_state.messages:
        st.markdown(f"""
        <div style='text-align:center; padding: 48px 0; color: rgba(226,232,226,0.18);'>
            <div style='font-size: 3rem; margin-bottom: 12px;'>{file_icon}</div>
            <div style='font-size: 1.05rem; font-weight: 600;'>Ready to answer</div>
            <div style='font-size: 0.82rem; margin-top: 4px;'>
                Ask anything about
                <span style='color: rgba(226,232,226,0.35);'>{st.session_state.uploaded_file}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='display:flex; justify-content:flex-end; margin: 8px 0;'>
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

    st.markdown("<hr>", unsafe_allow_html=True)

    # Input row
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        question = st.text_input(
            "question",
            placeholder=f"Ask anything about your {st.session_state.file_type}…",
            label_visibility="collapsed",
            key="question_input",
        )
    with col_btn:
        st.markdown("<div class='send-btn'>", unsafe_allow_html=True)
        send = st.button("Send ➤", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Handle suggestion click
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

    st.markdown("""
    <div style='text-align:center; color:rgba(255,255,255,0.08); font-size:0.72rem; margin-top:6px;'>
        Press Enter or click Send
    </div>
    """, unsafe_allow_html=True)

