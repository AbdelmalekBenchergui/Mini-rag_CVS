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

#  Fonction d'extraction regex
def extract_infos_cv(text: str) -> dict:
    infos = {}
    infos["emails"] = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    infos["linkedin"] = re.findall(r"(https?://)?(www\.)?linkedin\.com/in/[a-zA-Z0-9\-_/]+", text)
    infos["github"] = re.findall(r"(https?://)?(www\.)?github\.com/[a-zA-Z0-9\-_/]+", text)
    infos["formations"] = re.findall(r"\b(?:Licence|Master|Ing[ée]nieur|Doctorat|Ph\.?D|Bac\+3|Bac\+5)[^,\n]*", text, re.IGNORECASE)

    # 🔍 Extraction projets : chaque ligne qui commence par "projet" ou "-"/"•" et contient assez de texte
    projet_pattern = re.compile(
        r"(?:^|\n)[\s\-•]*projet[s]?\s*[:\-–]\s*(.+?)(?=\n[\s\-•]*projet[s]?\s*[:\-–]|$)", 
        re.IGNORECASE | re.DOTALL
    )
    infos["projets"] = [p.strip().replace('\n', ' ') for p in projet_pattern.findall(text)]
    for key in infos:
    infos[key] = [
        " ".join(item) if isinstance(item, tuple) else str(item)
        for item in infos[key]
    ]

    return infos


def build_graph():
    global vector_store, graph

    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

        def retrieve(state: State):
            docs_with_scores = vector_store.similarity_search_with_score(state["question"])
            return {"context": docs_with_scores}

        def generate(state: State):
            filtered = []
            merged_context = merge_context_by_file(state["context"])
            SEUIL_SIMILARITE_FAISS = 0.75
            SEUIL_SCORE = 6
            DOSSIER_SELECTION = "selected_cvs"
            os.makedirs(DOSSIER_SELECTION, exist_ok=True)  

            for filepath, filename, content, score_faiss in merged_context:
                if score_faiss < SEUIL_SIMILARITE_FAISS:
                    continue

                # 🟩 EXTRACTION INFOS CV
                info_extracted = extract_infos_cv(content)
                extracted_summary = f"Emails: {', '.join(info_extracted['emails'])}\n"
                extracted_summary += f"LinkedIn: {', '.join(info_extracted['linkedin'])}\n"
                extracted_summary += f"GitHub: {', '.join(info_extracted['github'])}\n"
                extracted_summary += f"Formations: {', '.join(info_extracted['formations'])}\n"
                extracted_summary += f"Projets: {', '.join(info_extracted['projets'])}\n"

                # Prompt enrichi
                prompt = f"""Tu es un recruteur. Évalue si ce candidat correspond à :"{state['question']}"
Voici les infos extraites automatiquement du CV :
{extracted_summary}
Attribue une note de 0 à 10, avec une justification.
Commence ta réponse par : NOTE: X/10
Texte :
{content}

"""

                response = llm.invoke([
                    {"role": "system", "content": "Tu es un assistant RH."},
                    {"role": "user", "content": prompt}
                ])

                match = re.search(r"NOTE\s*:\s*(\d+)", response.content)
                score_llm = int(match.group(1)) if match else 0
                score_total = 4 * score_faiss + 6 * (score_llm / 10)

                if score_total >= SEUIL_SCORE:
                    try:
                        shutil.move(filepath, os.path.join(DOSSIER_SELECTION, filename))
                    except Exception as e:
                        print(f"Erreur copie fichier {filepath} : {e}")

                filtered.append({
                    "score_total": round(float(score_total), 3),
                    "score_faiss": round(float(score_faiss), 3),
                    "score_llm": score_llm,
                    "justification": response.content.strip(),
                    "filename": filename,
                    "filepath": filepath,
                    # 🟩 Infos extraites ajoutées au résultat
                    "emails": info_extracted['emails'],
                    "linkedin": info_extracted['linkedin'],
                    "github": info_extracted['github'],
                    "formations": info_extracted['formations'],
                    "projets": info_extracted['projets'],
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
