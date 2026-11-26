import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄")
st.title("📄 PDF RAG Assistant")
st.write("Upload a PDF and ask a question about its content.")

if not os.getenv("OPENAI_API_KEY"):
     st.error("OpenAI API key is missing.")
     st.stop()

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file is not None:

    # Process PDF only when a new file is uploaded
    file_id = uploaded_file.file_id

    if st.session_state.get("file_id") != file_id:

        with st.spinner("Processing PDF..."):
            reader = PdfReader(uploaded_file)

            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"

            if not text.strip():
                st.error("No readable text found in this PDF.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = splitter.split_text(text)

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )

            st.session_state.vectorstore = FAISS.from_texts(
                texts=chunks,
                embedding=embeddings
            )

            st.session_state.file_id = file_id

        st.success("PDF processed successfully!")

    question = st.text_input("Ask a question about your PDF")

    if question:
        with st.spinner("Generating answer..."):

            vectorstore = st.session_state.vectorstore

            docs = vectorstore.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0
            )

            prompt = f"""
            Answer the question using only the context below.
            If the answer is unavailable, say:
            "I couldn't find that information in the document."

            Context:
            {context}

            Question:
            {question}
            """

            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)
