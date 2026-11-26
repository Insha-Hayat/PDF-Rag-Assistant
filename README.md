# 📄 PDF RAG Assistant

An AI-powered document question-answering application built using Retrieval-Augmented Generation (RAG). It enables users to upload PDF documents and ask natural language questions to receive context-aware responses based on document content.

## 🚀 Features

- PDF document upload and text extraction
- Intelligent text chunking with overlap
- Vector embeddings for semantic search
- FAISS-based vector storage and similarity search
- Context-aware answer generation using LLMs
- Reduced hallucinations through document grounding
- Interactive and user-friendly Streamlit interface

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Streamlit | Interactive user interface |
| LangChain | RAG pipeline orchestration |
| OpenAI API | Embeddings and answer generation |
| FAISS | Vector storage and similarity search |
| PyPDF | PDF text extraction |

## 🧠 System Architecture

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Store
    ↓
User Query
    ↓
Semantic Similarity Search
    ↓
Relevant Context Retrieval
    ↓
LLM Answer Generation
    ↓
Context-Aware Response
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-assistant.git
cd pdf-rag-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_api_key
```

### 6. Run the application

```bash
streamlit run app.py
```

## 📖 How It Works

1. The user uploads a PDF document.
2. The application extracts and divides the text into smaller chunks.
3. Text chunks are converted into vector embeddings.
4. Embeddings are indexed using FAISS.
5. The user's question is converted into an embedding.
6. Relevant document chunks are retrieved using semantic similarity.
7. The LLM generates an answer using the retrieved context.

## 🎯 Use Cases

- Research paper analysis
- Educational document assistance
- Enterprise knowledge retrieval
- Technical documentation search
- Intelligent document question answering

## 📊 Future Enhancements

- RAGAS evaluation framework
- Hybrid search combining semantic and keyword retrieval
- Cross-encoder reranking
- Source citations and page references
- Multi-document retrieval
- Conversational memory
- Docker containerization
- FastAPI backend integration
- LLM observability and performance monitoring

## 🔐 Security

API keys are managed through environment variables and excluded from version control using `.gitignore`.

## 👩‍💻 Author

**Insha Hayat**

AI Engineer | Generative AI | Python | LLM Applications | RAG

---

Built to demonstrate practical AI engineering, semantic retrieval, vector databases and LLM application development.
