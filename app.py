import streamlit as st
import faiss
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load API key from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🎬 Subtitle Search Chatbot",
    page_icon="🎬",
    layout="wide"
)

# This runs only once when app starts
# @st.cache_resource means dont reload every time user types
@st.cache_resource
def load_all_resources():
    print("⏳ Loading all resources...")
    
    # Load BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load FAISS index
    index = faiss.read_index('faiss_index.bin')
    
    # Load chunks
    with open('chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    
    # Load metadata
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # Load GPT model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=300
    )
    
    print("✅ All resources loaded!")
    return model, index, chunks, metadata, llm

# Load everything
model, index, chunks, metadata, llm = load_all_resources()

def retrieve_and_answer(query, top_k=5):
    """Full RAG pipeline"""
    
    # Step 1: Convert query to embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)
    
    # Step 2: Search FAISS
    distances, indices = index.search(query_embedding, top_k)
    
    # Step 3: Build context
    context_parts = []
    sources = []
    
    for dist, idx in zip(distances[0], indices[0]):
        chunk_text = chunks[idx]
        movie_name = metadata[idx]['name']
        similarity = round(float(dist), 4)
        
        context_parts.append(
            f"Source: {movie_name}\n{chunk_text}"
        )
        sources.append({
            'name': movie_name,
            'score': similarity,
            'text': chunk_text[:200]
        })
    
    # Step 4: Build prompt
    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are a helpful movie and TV show assistant.
You have access to subtitle data from various movies and TV shows.
Use the following subtitle excerpts as context to answer the question.
Be conversational and specific.
If context is not enough say so honestly.

CONTEXT FROM SUBTITLES:
{context}

USER QUESTION:
{query}

YOUR ANSWER:"""
    
    # Step 5: Get GPT answer
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        'answer': response.content,
        'sources': sources
    }

# App title
st.title("🎬 Subtitle Search Chatbot - Made by Roshan😊")
st.subheader("Ask anything about movies and TV shows!")
st.divider()

# Initialize chat history
# This keeps messages visible during the session
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message from bot
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! 👋 I'm your Movie & TV Show assistant! Ask me anything about movies, scenes, dialogues or characters!"
    })

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input box at bottom
if user_input := st.chat_input("Ask me about any movie or TV show..."):
    
    # Show user message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get answer from RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching subtitles and generating answer..."):
            result = retrieve_and_answer(user_input)
            answer = result['answer']
            sources = result['sources']
            
            # Show answer
            st.write(answer)
            
            # Show sources in expandable section
            with st.expander("📚 Sources Used"):
                for i, source in enumerate(sources):
                    st.markdown(f"**{i+1}. 🎬 {source['name']}**")
                    st.markdown(f"📊 Similarity Score: `{source['score']}`")
                    st.markdown(f"📝 {source['text']}...")
                    st.divider()
    
    # Save assistant answer to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

