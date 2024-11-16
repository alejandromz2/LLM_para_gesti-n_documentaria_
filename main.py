import pandas as pd
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import numpy as np
import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from procesar_data import limpieza_datos
from generar_embeddings import buscar_embeddings, procesar_documentos_y_crear_chunks
from agente_llm import ejecutar_modelo
from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

model = None 
def initialize_data():
    global df_pdfs
    data_path = '/Users/alejandromunoz/Desktop/Proyectos/LLM para gestión documentaria/data/df_pdfs.pkl'
    
    try:
        # Asegurarse de que la carpeta existe
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        if os.path.exists(data_path):
            df_pdfs = load_data(data_path)
            print("Datos cargados desde cache.")
        else:
            df_pdfs = limpieza_datos()
            save_data(df_pdfs, data_path)
            print("Datos procesados y guardados en cache.")
        
        print(df_pdfs.head())
    except Exception as e:
        print(f"Error al inicializar los datos: {e}")

def load_data(file_path):
    with open(file_path, "rb") as archivo:
        return pickle.load(archivo)

def save_data(data, file_path):
    with open(file_path, "wb") as archivo:
        pickle.dump(data, archivo)


@app.route("/chat", methods=['POST'])
def chat():
    """
    Maneja tanto la obtención de la pregunta como el envío de la respuesta.
    Espera un JSON con la clave 'question'.
    """
    data = request.json
    if not data or "question" not in data:
        return jsonify({"error": "Falta el parámetro 'question'"}), 400

    question = data["question"]
    
    # Aquí puedes integrar toda la lógica de procesamiento de tu pipeline:
    collection, embedding_pregunta = procesar_documentos_y_crear_chunks(df_pdfs, question, model)
    top_3emb = buscar_embeddings(collection, embedding_pregunta, top_k=3)
    respuesta = ejecutar_modelo(top_3emb, question, df_pdfs)
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    initialize_data()
    app.run(port=3001)



