import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from .ingestion_utils import get_embedding
except ImportError:
    from ingestion_utils import get_embedding

BASE_DIR = os.path.dirname(__file__)
ENV_PATH = os.path.join(BASE_DIR, ".env")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

load_dotenv(dotenv_path=ENV_PATH)

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")

_GENAI_CONFIGURED = False


def configure_genai():
    global _GENAI_CONFIGURED

    if _GENAI_CONFIGURED:
        return

    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise RuntimeError(f"GOOGLE_API_KEY not found in {ENV_PATH}")

    genai.configure(api_key=google_api_key)
    _GENAI_CONFIGURED = True

app = FastAPI(title="Digital Clone Brain API")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="person_memory")


class CloneRequest(BaseModel):
    person_name: str
    sender_name: str
    message_text: str


class CloneResponse(BaseModel):
    original_sender: str
    original_message: str
    clone_draft: str
    confidence_score: int
    reasoning: str
    status: str


def retrieve_memory(person_name, query, n_results=5):
    query_embedding = get_embedding(query, task_type="RETRIEVAL_QUERY")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"person_name": person_name}
    )

    memories = []

    if results["documents"] and results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            memories.append({
                "text": doc,
                "type": meta.get("type"),
                "source": meta.get("source")
            })

    return memories


def build_prompt(person_name, sender_name, message_text, memories):
    memory_text = ""

    for memory in memories:
        memory_text += f"""
Source Type: {memory['type']}
Source File: {memory['source']}
Memory: {memory['text']}
"""

    prompt = f"""
You are acting as the digital clone of {person_name}.

Your job:
Generate an email/message reply that sounds like {person_name}, based on their past communication style and persona memory.

Rules:
1. Reply in the same tone as the person.
2. Keep the reply realistic and professional.
3. Do not invent personal facts.
4. If the message involves money, legal, confidential, or sensitive matters, do not auto-commit.
5. Keep the reply concise unless the context requires detail.

Relevant persona memories:
{memory_text}

Incoming message from {sender_name}:
{message_text}

Return ONLY valid JSON in this exact format:

{{
  "draft": "reply text here",
  "confidence": 0,
  "reasoning": "short explanation"
}}

Confidence should be between 0 and 100.
"""
    return prompt


def generate_clone_reply(person_name, sender_name, message_text):
    configure_genai()
    memories = retrieve_memory(person_name, message_text)

    prompt = build_prompt(
        person_name=person_name,
        sender_name=sender_name,
        message_text=message_text,
        memories=memories
    )

    model = genai.GenerativeModel(GENERATION_MODEL)

    response = model.generate_content(prompt)

    if not getattr(response, "text", "").strip():
        raise RuntimeError("Gemini returned an empty response.")

    return response.text


@app.post("/clone/process", response_model=CloneResponse)
def process_clone(request: CloneRequest):
    try:
        raw_response = generate_clone_reply(
            person_name=request.person_name,
            sender_name=request.sender_name,
            message_text=request.message_text
        )
    except Exception as exc:
        return CloneResponse(
            original_sender=request.sender_name,
            original_message=request.message_text,
            clone_draft="",
            confidence_score=0,
            reasoning=f"Clone generation failed: {exc}",
            status="error"
        )

    # Simple safe fallback parsing
    import json

    try:
        cleaned = raw_response.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        data = json.loads(cleaned)

        confidence = int(data.get("confidence", 50))
        status = "auto_sent" if confidence >= 90 else "pending"

        return CloneResponse(
            original_sender=request.sender_name,
            original_message=request.message_text,
            clone_draft=data.get("draft", ""),
            confidence_score=confidence,
            reasoning=data.get("reasoning", "Generated using retrieved persona memory."),
            status=status
        )

    except Exception:
        return CloneResponse(
            original_sender=request.sender_name,
            original_message=request.message_text,
            clone_draft=raw_response,
            confidence_score=50,
            reasoning="Model response could not be parsed as JSON, returned raw draft.",
            status="pending"
        )