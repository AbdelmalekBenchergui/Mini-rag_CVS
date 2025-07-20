
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings

load_dotenv()


# Récupérer une variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Embeddings OpenAI
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key = openai_api_key
)
