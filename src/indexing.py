from langchain_community.vectorstores import FAISS
from config import embeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
from functools import partial

INDEX_DIR = "faiss_index"
def build_vector_store():
    # Charger documents
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FOLDER = os.path.join(BASE_DIR, "data", "raw")
    try:
        txt_loader = DirectoryLoader(DATA_FOLDER, glob="**/*.txt", loader_cls=partial(TextLoader, encoding="utf-8"))
        pdf_loader = DirectoryLoader(DATA_FOLDER, glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs = txt_loader.load() + pdf_loader.load()
    except Exception as e:
        print(f"Erreur lors du chargement des documents : {e}")
        raise e
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

        # Étape 3 : FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    
    print("Indexation terminée.")