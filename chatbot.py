import streamlit as st
import os
import json
from dotenv import load_dotenv
from groq import Groq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Config
VECTOR_PATH = "data/chat_memory"
os.makedirs("data", exist_ok=True)

# Cache the embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

EMBED_MODEL = load_embeddings()

# Load or initialize FAISS
if os.path.exists(VECTOR_PATH + ".faiss") and os.path.exists(VECTOR_PATH + ".pkl"):
    db = FAISS.load_local(VECTOR_PATH, EMBED_MODEL, allow_dangerous_deserialization=True)
else:
    db = None

# Groq completion with full history context
def ask_groq(history, new_input):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for u, b in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": new_input})

    response = groq_client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )
    return response.choices[0].message.content

# Store messages to FAISS
def store_in_vectorstore(text):
    global db
    doc = Document(page_content=text)
    if db is None:
        db = FAISS.from_documents([doc], EMBED_MODEL)
    else:
        db.add_documents([doc])
    db.save_local(VECTOR_PATH)

# Streamlit app
st.set_page_config(page_title="🧠 Smart Chatbot", page_icon="💬")
st.title("💬 Memory Chatbot (Groq + FAISS)")

# Init session state
if "history" not in st.session_state:
    st.session_state.history = []

# Show conversation
for user_msg, bot_reply in st.session_state.history:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_reply)

# User input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show user input
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Get response
    bot_reply = ask_groq(st.session_state.history, user_input)

    # Show bot reply
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_reply)

    # Save to memory + FAISS
    st.session_state.history.append((user_input, bot_reply))
    store_in_vectorstore(f"User: {user_input}\nAssistant: {bot_reply}")
