from flask import Flask, request, jsonify
from langchain_community.llms import Ollama


app = Flask(__name__)
@app.route("/chat",methods=['GET'])
def obtener_pregunta():
    data = request.json
    question=data["question"]
    return question
    
@app.route("/chat",methods=['POST'])
def enviar_respuesta(resumen):
    app.run(port=3000) 
    
    return resumen

