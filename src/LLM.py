import os
import re
import shutil
from config import embeddings, openai_api_key
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from typing import List, TypedDict, Tuple
from collections import defaultdict

llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

class State(TypedDict):
    question: str
    context: List[Tuple[Document, float]]
    results: List[dict]

INDEX_DIR = "faiss_index"
vector_store = None
graph = None

def merge_context_by_file(context: List[Tuple[Document, float]]) -> List[Tuple[str, str, str, float]]:
    grouped = defaultdict(list)
    for doc, score in context:
        source = doc.metadata.get("source", "inconnu")
        grouped[source].append((doc.page_content, score))

    merged = []
    for filepath, entries in grouped.items():
        contents = [e[0] for e in entries]
        scores = [e[1] for e in entries]
        full_content = "\n".join(contents)
        avg_score = sum(scores) / len(scores)
        filename = os.path.basename(filepath)
        merged.append((filepath, filename, full_content, avg_score))
    return merged

def build_graph():
    global vector_store, graph

    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

        def retrieve(state: State):
            docs_with_scores = vector_store.similarity_search_with_score(state["question"], k=10)

            return {"context": docs_with_scores}

        def generate(state: State):
            filtered = []
            merged_context = merge_context_by_file(state["context"])
            
            DOSSIER_SELECTION = "selected_cvs"
            os.makedirs(DOSSIER_SELECTION, exist_ok=True)  

            for filepath, filename, content, score_faiss in merged_context:
                prompt = f"""
                Tu es un recruteur en ressources humaines. Ta tâche est d'évaluer si ce candidat correspond à l'offre suivante :

                 Besoin de l'entreprise :
                "{state['question']}"

                Tu dois :
                1. Lire le contenu du CV.
                2. Attribuer une note sur 10 selon la pertinence du profil.
                3. Prendre une décision : **À conserver** ou **À écarter**.
                4. Donner une justification claire et concise, **en un seul paragraphe**, en expliquant les points forts et les faiblesses par rapport au besoin.

                 Format attendu **obligatoire** :
                NOTE: X/10 — Décision : À conserver / À écarter  
                Justification : [un seul paragraphe sans saut de ligne, 3-4 phrases max]


                Texte du CV :
                {content}
                """

                response = llm.invoke([
                    {"role": "system", "content": "Tu es un assistant RH."},
                    {"role": "user", "content": prompt}
                ])

                match = re.search(r"NOTE\s*:\s*(\d+)", response.content)
                score_llm = int(match.group(1)) if match else 0
                 


                filtered.append({
                    "score_faiss": round(float(score_faiss), 3),
                    "score_llm": score_llm,
                    "justification": response.content.strip(),
                    "filename": filename,
                    "filepath": filepath,
                })

            filtered = sorted(filtered, key=lambda x: x["score_llm"], reverse=True)
            return {"results": filtered}

        builder = StateGraph(State)
        builder.add_node("retrieve", retrieve)
        builder.add_node("generate", generate)
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        graph = builder.compile()

    else:
        vector_store = None
        graph = None

build_graph()
