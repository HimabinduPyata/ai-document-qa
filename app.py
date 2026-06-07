import os
import streamlit as st
from openai import OpenAI
from document_reader import extract_text, chunk_text, get_top_chunks
from vector_store import VectorStore

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("📚 AI Document Q&A (RAG Assistant)")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

question = st.text_input("Ask a question about your document")

if st.button("Ask"):

    if uploaded_file is None:
        st.warning("Please upload a PDF.")

    elif question.strip() == "":
        st.warning("Please enter a question.")

    else:
        with st.spinner("Processing document..."):

            document_text = extract_text(uploaded_file)
            chunks = chunk_text(document_text)

            store = VectorStore()
            store.build_index(chunks)

            top_chunks = store.search(question)
            context = "\n\n".join(top_chunks)

        with st.spinner("Thinking..."):

            prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers strictly from documents."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

        # Save chat history
        st.session_state.messages.append(("user", question))
        st.session_state.messages.append(("assistant", answer))

# Display chat history
st.markdown("---")
st.subheader("💬 Chat History")

for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")