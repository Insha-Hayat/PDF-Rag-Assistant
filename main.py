import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key not found!")

embeddings= OpenAIEmbeddings(
    model="text-embedding-3-small"
)



# Load PDF
reader = PdfReader("data/Sample_Final.pdf")
text=""
for page in reader.pages:
    text+=(page.extract_text() or "") + "\n"
print("Extracted text: ")
print(text[:1000])


# Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(text)
print("Total chunks:", len(chunks))
print("First chunk:", chunks[0] if chunks else "No text found")

if not chunks:
    raise ValueError("No text chunks found in PDF")

vectorstore= FAISS.from_texts(
    texts=chunks,
    embedding=embeddings
)

vectorstore.save_local("faiss_index")
print("Embeddings are generated and stored successfully!")

# Ask a question about the PDF
query ="What is the salary of manager?"

# Retrieve the 3 most relevant chunks
retrieved_docs =vectorstore.similarity_search(query,k=3)

print("\nRetrieved Documents:")

for i, doc in enumerate(retrieved_docs):
    print(f"\nChunk{i+1}:")
    print(doc.page_content)


# Initialize the language model

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature = 0
)

# Combine retrieved chunks
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

# Created a grounded prompt 
prompt=f"""
You are a helpful PDF assistant

Answer the question using only the provided context.
If the answer is not present, say:
"I couldn't find that information in the document."

Context:
{context}

Question:
{query}
"""


# Generate the answer

response = llm.invoke(prompt)
print("\n AI Answer:")
print(response.content)