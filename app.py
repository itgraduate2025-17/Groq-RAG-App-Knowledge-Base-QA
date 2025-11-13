import os
import re
import requests
import tempfile
import faiss
import numpy as np
import gradio as gr
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ====== CONFIG: Hugging Face secret ======
# Make sure you added your GROQ_API_KEY in HF Space secrets
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("Please add GROQ_API_KEY to your Hugging Face Space secrets.")

client = Groq(api_key=api_key)

# ====== LOAD EMBEDDING MODEL ======
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ====== GOOGLE DRIVE PDF LINKS ======
drive_links = [
    "https://drive.google.com/file/d/1sFbj-m4gHPU1vaY49nTtcGSsOkQ87lJM/view?usp=sharing",
    "https://drive.google.com/file/d/1javaraiucbTQfzKc465YKJtz30SN3OIA/view?usp=sharing"
]

# ====== DOWNLOAD PDF FROM DRIVE ======
def download_from_drive(link):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", link)
    if not match:
        raise ValueError(f"Invalid Google Drive link: {link}")
    file_id = match.group(1)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download PDF: {link}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

# ====== EXTRACT TEXT FROM PDF ======
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Failed to read PDF {pdf_path}: {e}")
        return ""

# ====== CHUNK TEXT ======
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ====== BUILD KNOWLEDGE BASE ======
def build_knowledge_base():
    print("📚 Building knowledge base from Google Drive PDFs...")
    all_chunks = []
    for link in drive_links:
        pdf_path = download_from_drive(link)
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    embeddings = embedder.encode(all_chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    print(f"✅ Loaded {len(all_chunks)} chunks into FAISS index.")
    return index, all_chunks, embeddings

index, all_chunks, embeddings = build_knowledge_base()

# ====== QUESTION ANSWERING ======
def answer_question(question):
    if not question.strip():
        return "⚠️ Please enter a question."

    q_emb = embedder.encode([question])
    D, I = index.search(np.array(q_emb).astype("float32"), k=3)
    retrieved_texts = [all_chunks[i] for i in I[0] if i < len(all_chunks)]
    context = "\n\n".join(retrieved_texts)

    prompt = f"""
You are a helpful document QA assistant. Use ONLY the following context to answer.
If the answer is not in the context, respond exactly:
"Sorry, I don't know — this is not in my knowledge base."
CONTEXT:
{context}
QUESTION:
{question}
"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    answer = chat_completion.choices[0].message.content

    formatted = f"### 🧠 Answer\n\n{answer}"
    return formatted

# ====== GRADIO UI ======
title = "🌈 Groq RAG App — Knowledge Base QA"
description = """
Ask any question based on the documents in my Google Drive knowledge base.  
If your question is outside the documents, the app will say it does not know 💬
"""

custom_css = """
body {
  background: linear-gradient(135deg, #ff6ec4, #7873f5);
}
.gradio-container {
  color: white;
  font-family: 'Poppins', sans-serif;
}
h1, h2, h3 { color: #fff; }
textarea, input { border-radius: 12px !important; border: 2px solid #fff !important; }
button { background: #ff6ec4 !important; color: white !important; border: none !important; border-radius: 12px !important; font-weight: bold !important; }
button:hover { background: #7873f5 !important; }
.output-markdown {
  background: rgba(255,255,255,0.15);
  padding: 15px;
  border-radius: 12px;
  color: #fff;
  white-space: pre-wrap;
}
footer { display: none !important; }
"""

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="💬 Ask a Question", placeholder="e.g., What is the main idea in the documents?"),
    outputs=gr.Markdown(label="🧠 Answer Box"),
    title=title,
    description=description,
    css=custom_css,
    theme="gradio/soft",
    allow_flagging="never"  # prevents creating local flagged dataset
)

iface.launch()
