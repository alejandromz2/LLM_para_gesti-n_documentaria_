# MODELO DE LLM
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Instanciar el modelo Ollama
llm = Ollama(model="llama3.1:8b")



prompt_template = """
Quiero que respondas a la pregunta que te hago a partir de la información que te estoy dando como contexto

Contexto: {context}

Además, te estoy entregando el nombre del archivo del cual he extraido ese contexto, para que puedas hacer referencia a este al momento de dar tu respuesta. No tienes que acceder al archivo ya que todo lo que necesitas te lo estoy dando en el contexto.

Nombre del Documento: {titulo}

Pregunta: {pregunta}

Quiero que respondas a mi pregunta de manera clara. Primero me das la respuesta y luego me dices de que documento has sacado la información
 
"""

# Función para obtener el resumen utilizando Ollama
def get_summary_ollama(context_text,titulo, pregunta):
    # Crear el prompt final utilizando el contexto
    final_prompt = prompt_template.format(context=context_text, titulo=titulo, pregunta=pregunta)
    
    # Generar la respuesta utilizando el modelo Ollama
    response = llm.invoke(final_prompt)
    
    return response

# Función para procesar la respuesta de Ollama
def get_answer_ollama(context_text, titulo, pregunta):
    # Obtener la respuesta de Ollama
    summary = get_summary_ollama(context_text,titulo, pregunta)
    return summary


def ejecutar_modelo(top_3emb, pregunta,df_pdfs):
    context_text = top_3emb[0]['text']
    id_documento = top_3emb[0]['document_id']
    titulo_documento = df_pdfs['titulo'].iloc[int(id_documento)]

    # Pregunta de ejemplo
    pregunta = pregunta

    # Obtener y mostrar la respuesta de Ollama
    print(pregunta)
    print()
    respuesta = get_answer_ollama(context_text,titulo_documento, pregunta)
    print(respuesta)

    return respuesta