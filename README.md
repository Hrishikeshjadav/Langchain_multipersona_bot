✨ Amusing Bot ✨
An interactive, multi-persona Streamlit chatbot powered by Groq's ultra-fast inference. This bot features RAG (Retrieval-Augmented Generation) capabilities, web search integration, and dynamic persona switching.
🚀 Features
⚡ Blazing Fast AI: Powered by Groq (Llama 3.1/3.3 models).
🧠 Knowledge Base (RAG): Upload PDFs to create a local vector store (using ChromaDB and HuggingFace embeddings) for the bot to reference.
🛠️ Tool Integration:
Tavily Search: Real-time web browsing for current events.
Wikipedia: Quick factual lookups.
PDF Retriever: Query your own uploaded documents.
🎭 AI Personas: Toggle between "Girlfriend", "Helpful Assistant", "Friendly", or "Skillful" personalities.
🌓 UI/UX: Built-in Light and Dark mode support with custom CSS.
🛠️ Tech Stack
Frontend: Streamlit
Orchestration: LangChain & LangGraph
LLM API: Groq
Vector Store: ChromaDB
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
Search: Tavily API & Wikipedia API
📋 Prerequisites
Python 3.9+
A Groq API Key (Get it at console.groq.com)
(Optional) A Tavily API Key for web search.
⚙️ Installation & Setup
Clone the repository:
bash
git clone https://github.com
cd amusing-bot
Use code with caution.

Install dependencies:
bash
pip install -r requirements.txt
Use code with caution.

Environment Variables:
Create a .env file in the root directory and add your keys:
env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
Use code with caution.

Run the App:
bash
streamlit run app.py
Use code with caution.

💡 How to Use
Select a Model: Choose a tool-capable model (like llama-3.3-70b-versatile) if you plan to use Web Search or PDF RAG.
Choose Persona: Pick a vibe from the sidebar.
Upload Knowledge: Drop a PDF into the "Knowledge Base" section and click Process Documents.
Chat: Ask questions! The agent will automatically decide whether to use its internal knowledge, search the web, or check your PDFs.
Note: Ensure you use llama-3.3-70b-versatile or tool-use-preview models when enabling Wikipedia or Tavily to ensure the Agent functions correctly.
