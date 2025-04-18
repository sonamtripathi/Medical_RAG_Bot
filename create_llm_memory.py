import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore_db_faiss/"

# Step 1: Load raw PDF(s)
def load_pdf_files(data):
    print("Files in data folder:", os.listdir(data))
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Total documents (pages) loaded: {len(documents)}")
    return documents

documents = load_pdf_files(DATA_PATH)
print("length of the documents:", len(documents))

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# Step 4: Store in FAISS
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print(f"FAISS index saved to: {DB_FAISS_PATH}")
