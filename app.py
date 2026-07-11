import streamlit as st
import faiss
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# -----------------------------------------------------
# Load API Key (Works both Locally and on Streamlit Cloud)
# -----------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    api_key = st.secrets["OPENAI_API_KEY"]

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="🎬 Subtitle Search Chatbot",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------------------------------
# Load All Resources (Runs only once)
# -----------------------------------------------------
@st.cache_resource
def load_all_resources():
    print("⏳ Loading all resources...")

    # Load Embedding Model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load FAISS Index
    index = faiss.read_index("faiss_index.bin")

    # Load Subtitle Chunks
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load Metadata
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Initialize LLM
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=300
    )

    print("✅ All resources loaded!")

    return model, index, chunks, metadata, llm


# Load resources
model, index, chunks, metadata, llm = load_all_resources()

# -----------------------------------------------------
# RAG Function
# -----------------------------------------------------
def retrieve_and_answer(query, top_k=5):

    # Convert query to embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    context_parts = []
    sources = []

    for dist, idx in zip(distances[0], indices[0]):

        chunk_text = chunks[idx]
        movie_name = metadata[idx]["name"]
        similarity = round(float(dist), 4)

        context_parts.append(
            f"Source: {movie_name}\n{chunk_text}"
        )

        sources.append({
            "name": movie_name,
            "score": similarity,
            "text": chunk_text[:200]
        })

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a helpful movie and TV show assistant.

You have access to subtitle data from various movies and TV shows.

Use the following subtitle excerpts as context to answer the user's question.

Be conversational and specific.

If the context is not sufficient, honestly say that you don't know.

CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "answer": response.content,
        "sources": sources
    }

# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
st.title("🎬 Subtitle Search Chatbot - Made by Roshan 😊")
st.subheader("Ask anything about Movies and TV Shows!")
st.divider()

# Chat History
if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! 👋 I'm your Movie & TV Show assistant. Ask me anything about movies, TV shows, scenes, dialogues or characters!"
        }
    ]

# Display Previous Messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
if user_input := st.chat_input("Ask me about any movie or TV show..."):

    # Show User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Generate Response
    with st.chat_message("assistant"):

        with st.spinner("🔍 Searching subtitles and generating answer..."):

            result = retrieve_and_answer(user_input)

            answer = result["answer"]
            sources = result["sources"]

            st.write(answer)

            with st.expander("📚 Sources Used"):

                for i, source in enumerate(sources):

                    st.markdown(f"**{i+1}. 🎬 {source['name']}**")
                    st.markdown(f"📊 Similarity Score: `{source['score']}`")
                    st.markdown(f"📝 {source['text']}...")
                    st.divider()

    # Save Assistant Message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )