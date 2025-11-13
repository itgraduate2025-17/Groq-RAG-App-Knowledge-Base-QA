# Groq RAG App — Knowledge Base QA

## Overview

This project is a **Retrieval-Augmented Generation (RAG) application** built with **Gradio, Groq SDK, and FAISS**. It automatically downloads PDFs from Google Drive, extracts text, creates embeddings using **SentenceTransformers**, and stores them in a **FAISS index**. Users can ask questions based on the documents, and the app retrieves relevant text chunks to generate accurate answers using Groq’s **Llama 3.3-70B model**.

This project is designed for deployment on **Hugging Face Spaces** with secure secret management for the API key.

---

## Features

* Downloads and processes PDFs directly from Google Drive
* Extracts and chunks text automatically
* Generates embeddings with **all-MiniLM-L6-v2**
* Builds a **FAISS index** for similarity-based retrieval
* Uses Groq’s Llama model for question answering
* Clean, minimal Gradio interface with custom styling

---

## Setup on Hugging Face

### Add Your API Key

To run this project on Hugging Face Spaces, you need to add your Groq API key as a repository secret.

**Steps:**

1. Open your Hugging Face Space.
2. Go to **Settings → Repository Secrets**.
3. Add a new secret with:

   * **Name:** GROQ_API_KEY
   * **Value:** your Groq API key (for example: gsk_abc123xyz456...)
4. Save the secret.

In the code, the key is automatically loaded from the environment:

```python
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("Please add GROQ_API_KEY to your Hugging Face Space secrets.")
```

> Important: Never write your API key directly in the code or upload it publicly.

---

## How It Works

1. **PDF Download:** The app fetches PDF files from Google Drive links.
2. **Text Extraction:** Extracts text from each page using **PyPDF2**.
3. **Chunking:** Splits text into smaller parts (default 500 words).
4. **Embeddings:** Converts chunks into embeddings using **SentenceTransformer**.
5. **Vector Index:** Stores embeddings in a **FAISS index** for fast search.
6. **Question Answering:** Retrieves the top similar chunks and sends them with the question to Groq’s Llama model to generate an answer.
7. **Interface:** Displays the result in a responsive **Gradio web UI**.

---

## Deployment

Once your Space is created and the API key is added:

1. Upload your project files (**app.py, requirements.txt, and this README.md**).
2. Commit and push to your Hugging Face Space repository.
3. The build will run automatically and the app will launch.
4. Share the Space link to allow others to use it.

---

## Example Interaction

**Question:** What is covered in the second document?
**Answer:** The document discusses methods for improving neural network training and preprocessing steps for large datasets.

---

## Common Issues

| Problem                 | Cause                      | Solution                                       |
| ----------------------- | -------------------------- | ---------------------------------------------- |
| Please add GROQ_API_KEY | Missing API key            | Add it under Hugging Face → Repository Secrets |
| Failed to download PDF  | Private or invalid link    | Make sure the Google Drive link is public      |
| Empty response          | Non-selectable text in PDF | Use PDFs with actual text (not scanned images) |

Notes:

* You can update the Google Drive links inside the **drive_links** list in the code.
* Supports multiple PDFs to build one combined knowledge base.
* Works best with text-based PDFs.

---

## Developer

**Amjad Ali**

**Purpose:** Research and educational use
**License:** Free for learning and non-commercial use
