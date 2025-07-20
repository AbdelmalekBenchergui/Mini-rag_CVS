import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Chatbot RH - Analyse de CV", layout="centered")

st.title("🤖 Chatbot RH - Matching CVs")

# Identifiant de session utilisateur (facultatif pour future personnalisation)
user_id = st.text_input("🧑 Identifiant utilisateur", value="demo")

# Upload des fichiers
st.subheader("1️⃣ Upload de fichiers CV")
uploaded_files = st.file_uploader("Sélectionne les fichiers PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("📤 Envoyer les CVs"):
    if uploaded_files:
        files = [("files", (file.name, file.getvalue())) for file in uploaded_files]
        params = {"user_id": user_id}
        response = requests.post(f"{API_URL}/upload-cvs/", files=files, params=params)
        st.success(response.json()["message"])
    else:
        st.warning("Ajoutez au moins un fichier.")

# Indexation
st.subheader("2️⃣ Indexation des CVs")
if st.button("⚙️ Lancer l’indexation"):
    params = {"user_id": user_id}
    response = requests.post(f"{API_URL}/index-cvs/", params=params)
    if response.status_code == 200:
        st.success(response.json()["message"])
    else:
        st.error(response.json()["detail"])

# Poser une question (offre d’emploi)
st.subheader("3️⃣ Poser une question (offre d’emploi)")
question = st.text_area("🔍 Exemple : Développeur Python avec 2 ans d’expérience en IA", height=100)

if st.button("🤖 Lancer l’analyse"):
    if len(question) < 5:
        st.warning("Pose une question un peu plus longue.")
    else:
        params = {"question": question, "user_id": user_id}
        response = requests.get(f"{API_URL}/ask-cv/", params=params)
        if response.status_code == 200:
            results = response.json()["results"]
            if not results:
                st.info("Aucun résultat pertinent trouvé.")
            for res in results:
                st.markdown(f"""
                ---  
                📄 **{res['filename']}**  
                ✅ **Note LLM :** {res['score_llm']} /10  
                📊 **Score FAISS :** {res['score_faiss']}  
                📝 **Justification :**  
                {res['justification']}
                """)
        else:
            st.error(response.json()["detail"])


