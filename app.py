import os
from dotenv import load_dotenv
import streamlit as st

# LangChain/LLM imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory

# --- ENVIRONMENT SETUP ---
load_dotenv()

# --- INITIALIZE VECTOR STORE (with caching) ---
@st.cache_resource(show_spinner="Indexing medical PDF. Please wait...")
def setup_vectordb(pdf_path):
    if not os.path.exists(pdf_path):
        st.error(f"File `{pdf_path}` not found.")
        st.stop()
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db

DB = setup_vectordb("Medical_book.pdf")

# --- LLM & RETRIEVER ---
llm = ChatOpenAI(temperature=0)
retriever = DB.as_retriever()
base_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False  # Toggle True to display sources if you prefer
)

# --- SESSION-BASED CHAT MEMORY ---
chat_histories = {}
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: chat_histories.setdefault(session_id, ChatMessageHistory()),
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# --- STREAMLIT UI ---
st.title("🩺 Medical Chatbot")

st.markdown("Ask your medical questions.")

if "session_id" not in st.session_state:
    st.session_state.session_id = "user-session"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your medical question:")

if query:
    with st.spinner("Thinking..."):
        try:
            result = chain_with_history.invoke(
                {"question": query},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            answer = result.get("answer") or result.get("result") or result.get("output_text", "No response.")
        except Exception as e:
            answer = f"❌ Error: {str(e)}"
    st.session_state.chat_history.append({"question": query, "answer": answer})

# --- DISPLAY CHAT HISTORY ---
if st.session_state.chat_history:
    st.markdown("---")
    for item in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {item['question']}")
        st.markdown(f"**Bot:** {item['answer']}")

if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    chat_histories[st.session_state.session_id] = ChatMessageHistory()
    st.rerun()
