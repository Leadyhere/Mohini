import os
import uuid
import streamlit as st
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

from ingestion_utils import (
    clean_text,
    chunk_text,
    get_embedding,
    extract_text
)
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Check your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="person_memory"
)


def store_in_chroma(text, data_type, source_name, person_name):
    text = clean_text(text)
    chunks = chunk_text(text)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:
        ids.append(str(uuid.uuid4()))
        documents.append(chunk)
        embeddings.append(get_embedding(chunk))
        metadatas.append({
            "person_name": person_name,
            "type": data_type,
            "source": source_name
        })

    if documents:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    return len(documents)


def search_memory(query, person_name, n_results=5):
    query_embedding = get_embedding(query, task_type="RETRIEVAL_QUERY")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"person_name": person_name}
    )

    return results


st.title("Digital Clone - Persona Data Upload")

person_name = st.text_input("Enter person's name")

st.subheader("Upload Persona Files")

data_type = st.selectbox(
    "What type of data are you uploading?",
    ["email", "whatsapp", "blog"]
)

uploaded_files = st.file_uploader(
    "Upload Gmail, WhatsApp, or Blog files",
    type=["txt", "csv", "pdf", "docx"],
    accept_multiple_files=True
)

if st.button("Store Uploaded Files"):
    if not person_name:
        st.error("Please enter the person's name.")
    elif not uploaded_files:
        st.error("Please upload at least one file.")
    else:
        total_chunks = 0

        for uploaded_file in uploaded_files:
            text = extract_text(uploaded_file)

            if not text.strip():
                st.warning(f"No readable text found in {uploaded_file.name}")
                continue

            chunks_stored = store_in_chroma(
                text=text,
                data_type=data_type,
                source_name=uploaded_file.name,
                person_name=person_name
            )

            total_chunks += chunks_stored
            st.success(f"{uploaded_file.name}: stored {chunks_stored} chunks")

        st.success(f"Done. Total chunks stored: {total_chunks}")


st.divider()

st.subheader("Persona Questions")

q1 = st.text_area("1. How do you usually reply to emails? Formal, casual, or direct?")
q2 = st.text_area("2. Do you prefer short replies or detailed replies?")
q3 = st.text_area("3. How do you reject a request politely?")
q4 = st.text_area("4. How do you accept a meeting invitation?")
q5 = st.text_area("5. How do you ask someone to complete work faster?")
q6 = st.text_area("6. How do you respond when someone delays work?")
q7 = st.text_area("7. What words or phrases do you often use?")
q8 = st.text_area("8. What words should your clone avoid using?")
q9 = st.text_area("9. Are you strict with deadlines or flexible?")
q10 = st.text_area("10. What topics should your clone never answer automatically?")

if st.button("Save Persona Answers"):
    if not person_name:
        st.error("Please enter the person's name first.")
    else:
        persona_text = f"""
        Q: How do you usually reply to emails?
        A: {q1}

        Q: Do you prefer short replies or detailed replies?
        A: {q2}

        Q: How do you reject a request politely?
        A: {q3}

        Q: How do you accept a meeting invitation?
        A: {q4}

        Q: How do you ask someone to complete work faster?
        A: {q5}

        Q: How do you respond when someone delays work?
        A: {q6}

        Q: What words or phrases do you often use?
        A: {q7}

        Q: What words should your clone avoid using?
        A: {q8}

        Q: Are you strict with deadlines or flexible?
        A: {q9}

        Q: What topics should your clone never answer automatically?
        A: {q10}
        """

        chunks_stored = store_in_chroma(
            text=persona_text,
            data_type="persona_questions",
            source_name="app_persona_questions",
            person_name=person_name
        )

        st.success(f"Persona answers saved. Stored {chunks_stored} chunks in Chroma.")


st.divider()

st.subheader("Test Persona Memory Search")

search_person = st.text_input("Enter person name to search")
query = st.text_input("Enter search query")

if st.button("Search Memory"):
    if not search_person:
        st.error("Enter the person's name.")
    elif not query:
        st.error("Enter a search query.")
    else:
        results = search_memory(query, search_person)

        if not results["documents"][0]:
            st.warning("No matching memory found.")
        else:
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0]
            ):
                st.write("**Type:**", meta.get("type"))
                st.write("**Source:**", meta.get("source"))
                st.write("**Memory:**", doc)
                st.divider()
