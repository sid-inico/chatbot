import json
import ollama
import chromadb as db
from sentence_transformers import SentenceTransformer
import logging

# Silencia los logs de ChromaDB (solo mostrará ERROR o superior)
logging.getLogger('chromadb').setLevel(logging.ERROR)

# Cargar el JSON
with open("json/train.json", "r") as f:
    data = json.load(f)

# Extraer pares
pairs = [(item["query"], item["response"]) for item in data]

# Inicializar ChromaDB
client = db.PersistentClient("./prueba", )
collection = client.get_or_create_collection(name="prueba")

# Cargar un modelo de embeddings
embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Generar embeddings y almacenarlos
for idx, (query, response) in enumerate(pairs):
    embedding = embedder.encode(query)
    collection.add(
        ids=[str(idx)],
        embeddings=[embedding.tolist()],
        metadatas=[{"response": response}]
    )

def generate_response(prompt):
    response = ollama.generate(
        model="llama3.1",
        prompt=prompt
    )
    return response["response"]

def chatbot(query):
    input_embedding = embedder.encode(query)
    
    results = collection.query(
        query_embeddings=[input_embedding.tolist()],
        n_results=1
    )
    
    # Usar el umbral de similitud para decidir
    if results["distances"] and results["distances"][0][0] < 0.3:  # Ajusta este valor
        closest_output = results["metadatas"][0][0].get("response")
        return closest_output
    else:
        return generate_response(query)

# Probar el chatbot
print("Nico: ¡Hola! Soy tu asistente. Escribe 'salir' para terminar.")

user_input = "" 

while (user_input.lower() != "salir") and (user_input.lower() != "adios"):
    user_input = input("Tú: ")
    if user_input.lower() in ["salir", "adios"]:
        break
    response = chatbot(user_input)
    print(f"Nico: {response}")

print("Nico: ¡Hasta luego!")