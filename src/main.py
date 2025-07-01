from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import shutil
import os
from indexing import build_vector_store
from typing import List
import LLM

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(DATA_FOLDER, exist_ok=True)

@app.get("/welcome")
def welcome():
    return {"message": "Hello World!"}

@app.post("/upload-cvs/")
async def upload_cvs(files: List[UploadFile] = File(...)):
    try:
        # Supprimer les anciens fichiers du dossier
        for f in os.listdir(DATA_FOLDER):
            file_path = os.path.join(DATA_FOLDER, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Sauvegarder les nouveaux fichiers
        for file in files:
            print("FICHIER RECU :", file.filename)
            file_path = os.path.join(DATA_FOLDER, file.filename)
            print("Sauvegarde dans :", file_path)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        print("Contenu du dossier :", os.listdir(DATA_FOLDER))
        return {"message": f"{len(files)} fichiers CV sauvegardés avec succès."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur upload : {str(e)}")



@app.post("/index-cvs/")
def index_cvs():
    try:
        build_vector_store()
        LLM.build_graph() 
        return {"message": "✅ Indexation terminée avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur indexation : {str(e)}")

@app.get("/ask-cv/")
def ask_cv(question: str = Query(..., min_length=5)):
    if LLM.graph is None:
        raise HTTPException(
            status_code=400,
            detail="Index FAISS non disponible. Exécutez /index-cvs/ d’abord."
        )
    try:
        result = LLM.graph.invoke({"question": question})
        return {

            "question": question,
            "results": [
                { 
                    "score_llm": cv.get("score_llm", None),
                    "score_faiss": cv.get("score_faiss", None),
                    "justification": cv.get("justification", "Pas de justification."),
                    "filename": cv["filename"],
                }
                for cv in result.get("results", [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur traitement : {str(e)}")
