import streamlit as st
import time
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="LawGPT – Indian Penal Code", layout="wide")

st.title("⚖️ LawGPT")
st.caption("AI Legal Assistant for Indian Penal Code (IPC)")

# ---------------- SESSION STATE ----------------
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        return_messages=True
    )

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- VECTOR DB ----------------
db = FAISS.load_local(
    "ipc_vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 4})

# ---------------- PROMPT ----------------
prompt_template = """
You are a legal assistant specializing in the Indian Penal Code (IPC).

Answer strictly from the provided context.
Mention IPC sections wherever applicable.
If the answer is not present, say:
"I do not have enough information from the IPC to answer this."

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"]
)

# ---------------- LLM (GROQ – FREE) ----------------
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.3
)

# ---------------- RAG CHAIN ----------------
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# ---------------- CHAT UI ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask an IPC-related question...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result = qa.invoke({"question": user_input})
            answer = result["answer"]

            output = ""
            placeholder = st.empty()
            for char in answer:
                output += char
                placeholder.markdown(output + "▌")
                time.sleep(0.01)

            placeholder.markdown(output)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

st.button("🗑️ Reset Chat", on_click=reset_conversation)
