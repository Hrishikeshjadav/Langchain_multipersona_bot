import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Langchain core imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Agent imports
from langgraph.prebuilt import create_react_agent

load_dotenv()

st.set_page_config(page_title="Chat Bot", layout="centered", page_icon="💬")

st.title("✨Amusing Bot✨")
st.markdown("Super bot for your daily purposes powered with groq_api key")

# Initialize session state for vectorstore and messages
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# sidebar
with st.sidebar:
    st.header("Settings ⚙️")
    # 1. Theme Selection Menu
    theme_choice = st.selectbox(
        "Choose Theme", 
        ["Light Mode ☀️", "Dark Mode 🌙"],
        index=0  # Default to Dark Mode
    )
    
    # 2. Define Theme CSS
    if theme_choice == "Dark Mode 🌙":
        st.markdown("""
            <style>
            .stApp { background-color: #0E1117; color: #FFFFFF; }
            [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
            .stChatMessage { background-color: #1e1e1e !important; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp { background-color: #FFFFFF; color: #000000; }
            [data-testid="stSidebar"] { background-color: #F0F2F6; border-right: 1px solid #e6e9ef; }
            .stChatMessage { background-color: #f0f2f6 !important; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
            </style>
            """, unsafe_allow_html=True)

    # api key if not set
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("API Key loaded from .env")
    else:
        api_key = st.text_input(
            "GROQ_API_KEY", 
            type="password", 
            help="You can get your API key from [console.groq.com](https://console.groq.com)"
        )
        
    # Model selection
    TOOL_CAPABLE_MODELS = ["llama-3.3-70b-versatile", "llama3-groq-70b-8192-tool-use-preview", "llama3-groq-8b-8192-tool-use-preview"]
    ALL_MODELS = TOOL_CAPABLE_MODELS + ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
    model_name = st.selectbox(
        "Select the model you want:",
        ALL_MODELS,
        index=0
    )
    
    # System prompt selection
    _tool_suffix = " When you have access to search tools or documents, always use them to look up current information before answering questions about recent events or news."
    system_prompts = {
        "Girlfriend 💖": "You are exceptionaly special girlfriend. Answer in a way that is easy to understand and fun." + _tool_suffix,
        "Helpful Assistant 🤖": "You are a helpful conversation assistant." + _tool_suffix,
        "Friendly Assistant 😊": "You are a friendly conversation assistant." + _tool_suffix,
        "Skillful Assistant 💻": "You are a skillful conversation assistant." + _tool_suffix
    }
    prompt_type = st.selectbox(
        "Select Assistant Persona:",
        list(system_prompts.keys()),
        index=0
    )
    selected_system_prompt = system_prompts[prompt_type]

    st.markdown("---")
    st.header("Tools & Capabilities 🧰")
    
    # Tool toggles
    enable_wikipedia = st.checkbox("Enable Wikipedia Search", value=False)
    enable_tavily = st.checkbox("Enable Web Search (Tavily)", value=False)
    
    if (enable_wikipedia or enable_tavily) and model_name not in TOOL_CAPABLE_MODELS:
        st.warning(f"⚠️ **{model_name}** doesn't support tool calling. Please select `llama-3.3-70b-versatile` or one of the `tool-use` models above.")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if enable_tavily:
        if tavily_api_key:
            st.success("Tavily API Key loaded from .env")
        else:
            tavily_api_key = st.text_input(
                "TAVILY_API_KEY", 
                type="password", 
                help="Get it from [app.tavily.com](https://app.tavily.com)"
            )
            if tavily_api_key:
                os.environ["TAVILY_API_KEY"] = tavily_api_key
    
    st.markdown("---")
    st.header("Knowledge Base 📚")
    uploaded_files = st.file_uploader("Upload any PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Analyzing PDFs..."):
            # Lazy load heavy dependencies
            from langchain_community.document_loaders import PyMuPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma
            
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

    if st.button("clear chat📜"):
        st.session_state.messages= []
        st.session_state.vectorstore = None
        st.rerun()

# ---------------------------------------------
# Bot / Agent setup
# ---------------------------------------------

def get_tools():
    active_tools = []
    
    if enable_wikipedia:
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
        active_tools.append(WikipediaQueryRun(api_wrapper=wiki_wrapper))
        
    if enable_tavily and os.getenv("TAVILY_API_KEY"):
        active_tools.append(TavilySearch(max_results=3))
        
    if st.session_state.vectorstore:
        from langchain_core.tools import create_retriever_tool
        retriever_tool = create_retriever_tool(
            st.session_state.vectorstore.as_retriever(),
            name="fastapi_docs",
            description="Searches the uploaded FastAPI documentation for specific technical details. Use this tool when asked about the uploaded PDFs."
        )
        active_tools.append(retriever_tool)
        
    return active_tools

if not api_key:
    st.warning("Please provide your GROQ_API_KEY")
    st.stop()

llm_model = ChatGroq(
    api_key=api_key, 
    model=model_name,
    temperature=0.7,
    streaming=True
)

active_tools = get_tools()

# ---------------------------------------------
# Chat Execution
# ---------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# chat input
if input_text := st.chat_input("Ask anything"):
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.write(input_text)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            if active_tools:
                # We have tools, we must use an Agent
                with st.spinner("Thinking and looking up information..."):
                    # Construct chat history for the agent
                    chat_history = []
                    for msg in st.session_state.messages[:-1]: # exclude the current user input
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))
                            
                    chat_history.append(HumanMessage(content=input_text))
                    
                    # Prepend system message to chat history for ReAct agent
                    messages_with_system = [SystemMessage(content=selected_system_prompt)] + chat_history
                    
                    agent = create_react_agent(llm_model, tools=active_tools)
                    
                    response = agent.invoke({"messages": messages_with_system})
                    # Find the last AIMessage that has real text content (not a tool call)
                    final_text = ""
                    for msg in reversed(response["messages"]):
                        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                            final_text = msg.content
                            break
                    if not final_text:
                        final_text = "I searched but couldn't find a clear answer. Please try rephrasing."
                    message_placeholder.markdown(final_text)
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
            else:
                # No tools active, standard chain streaming
                prompt = ChatPromptTemplate.from_messages([
                    ("system", selected_system_prompt),
                    ("user", "{input}")
                ])
                chain = prompt | llm_model | StrOutputParser()
                
                full_response = ""
                for chunk in chain.stream({"input": input_text}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
        except Exception as e:
            message_placeholder.markdown(f"**Error:** {str(e)}")

st.markdown("---")
st.markdown("### Try these examples🛠️ ")
col1,col2 = st.columns(2)
with col1:
    st.markdown("- What is langchain?")
    st.markdown("- What is groq?")
    st.markdown("- What is openai?")
with col2:
    st.markdown("- What is streamlit?")
    st.markdown("- What is python?")
    st.markdown("- What is machine learning?")
## footer
st.markdown("---")
st.markdown("@2026 by Jadav")
