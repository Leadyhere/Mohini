import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document


env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

_GENAI_CONFIGURED = False


def _reset_file_pointer(uploaded_file):
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)


def _configure_genai():
    global _GENAI_CONFIGURED

    if _GENAI_CONFIGURED:
        return

    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

    genai.configure(api_key=google_api_key)
    _GENAI_CONFIGURED = True


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    return " ".join(text.split()).strip()


def chunk_text(text, chunk_size=120, overlap=30):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def get_embedding(text, task_type="RETRIEVAL_DOCUMENT"):
    _configure_genai()
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type=task_type
    )
    return result["embedding"]


def read_txt(uploaded_file):
    _reset_file_pointer(uploaded_file)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def read_csv(uploaded_file):
    _reset_file_pointer(uploaded_file)
    df = pd.read_csv(uploaded_file)
    text = ""

    for _, row in df.iterrows():
        row_text = " ".join([str(value) for value in row.values])
        text += row_text + "\n"

    return text


def read_pdf(uploaded_file):
    _reset_file_pointer(uploaded_file)
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def read_docx(uploaded_file):
    _reset_file_pointer(uploaded_file)
    document = Document(uploaded_file)
    text = ""

    for para in document.paragraphs:
        text += para.text + "\n"

    return text


def extract_text(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return read_txt(uploaded_file)

    elif file_name.endswith(".csv"):
        return read_csv(uploaded_file)

    elif file_name.endswith(".pdf"):
        return read_pdf(uploaded_file)

    elif file_name.endswith(".docx"):
        return read_docx(uploaded_file)

    return ""
