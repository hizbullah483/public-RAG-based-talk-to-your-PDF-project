import streamlit as st
import fitz  
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq


st.set_page_config(page_title="PDF RAG (Groq)", layout="wide")
st.title("PDF Q&A")

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "gsk_nicetry"
client = Groq(api_key=GROQ_API_KEY)

if "text_chunks" not in st.session_state:
    st.session_state.update({
        "text_chunks": [],
        "tfidf_matrix": None,
        "vectorizer": None,
        "chat_history": []
    })

@st.cache_data(show_spinner=False)
def process_pdf(file_bytes: bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    chunk_size = 600
    overlap = 100
    chunks = []

    for page in doc:
        text = page.get_text("text")
        if not text.strip():
            continue

        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])

        if len(chunks) > 1500:
            break

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000
    )

    tfidf_matrix = vectorizer.fit_transform(chunks)

    return chunks, vectorizer, tfidf_matrix

def get_context(query, chunks, vectorizer, tfidf_matrix, top_k=2):
    query_vec = vectorizer.transform([query])
    sims = (tfidf_matrix @ query_vec.T).toarray().ravel()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return "\n".join(chunks[i] for i in top_idx)

with st.sidebar:
    st.header("Settings")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file and st.button("Index Document"):
        with st.spinner("Processing..."):
            c, v, m = process_pdf(uploaded_file.getvalue())
            st.session_state.text_chunks = c
            st.session_state.vectorizer = v
            st.session_state.tfidf_matrix = m
            st.success(f"Indexed {len(c)} chunks")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

for q, a in st.session_state.chat_history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

if question := st.chat_input("Ask about the document..."):

    if not st.session_state.text_chunks:
        st.error("Upload and index a PDF first!")
    else:
        st.chat_message("user").write(question)

        context = get_context(
            question,
            st.session_state.text_chunks,
            st.session_state.vectorizer,
            st.session_state.tfidf_matrix
        )


        prompt = f"""
Context:
{context}

Question:
{question}

Answer briefly and only from context:
"""

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            start_time = time.time()

            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",  
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150,
                    stream=True
                )

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"Error: {str(e)}"
                placeholder.markdown(full_response)

            st.session_state.chat_history.append((question, full_response))
            st.caption(f"⏱ {time.time() - start_time:.2f}s")
