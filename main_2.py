import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import create_retriever_tool

# Initialize UI FIRST to prevent blank screen while loading models
st.set_page_config(page_title="FastAPI AI Expert", layout="wide")
st.title("🐍 FastAPI Mastery Assistant")

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY in .env file")
    st.stop()

# Use a function to load heavy models only when needed
@st.cache_resource
def load_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Rest of your logic...



# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 2. Sidebar: File Processing
with st.sidebar:
    st.header("1. Knowledge Base")
    uploaded_files = st.file_uploader("Upload FastAPI PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Analyzing PDFs..."):
            all_docs = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    path = os.path.join(temp_dir, uploaded_file.name)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    loader = PyMuPDFLoader(path)
                    all_docs.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(all_docs)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.success(f"Indexed {len(splits)} chunks!")

# 3. Agent Setup
if st.session_state.vectorstore:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    retriever_tool = create_retriever_tool(
        st.session_state.vectorstore.as_retriever(),
        name="fastapi_docs",
        description="Searches the uploaded FastAPI documentation for specific technical details."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a FastAPI expert. Use the 'fastapi_docs' tool to answer questions. Provide code snippets in every response."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

    # 4. Chat Interface
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if user_query := st.chat_input("Ask a FastAPI question..."):
        st.chat_message("user").markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent_executor.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                full_response = response["output"]
                st.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload and process PDF documents in the sidebar to start.")
