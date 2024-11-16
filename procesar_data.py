import pandas as pd
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import numpy as np
import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



def extraer_texto_de_pdf_con_ocr(ruta_pdf):
    """
    Extrae el texto de un documento PDF utilizando OCR (Tesseract) y pdf2image.
    :param ruta_pdf: Ruta del archivo PDF.
    :return: Texto extraído del PDF.
    """
    texto_completo = ""

    # Convierte el PDF a una lista de imágenes (una imagen por página)
    paginas = convert_from_path(ruta_pdf)

    # Realiza OCR en cada imagen de página
    for numero_pagina, pagina in enumerate(paginas, start=1):
        print(f"Procesando página {numero_pagina}...")  # Informar sobre el progreso
        # Convierte la imagen a un formato soportado por Tesseract (RGB)
        pagina_rgb = pagina.convert('RGB')
        
        # Realiza OCR en la imagen de la página
        texto = pytesseract.image_to_string(pagina_rgb, lang='spa')  # 'lang' establece el idioma (ej. español)
        texto_completo += f"\n\n--- Página {numero_pagina} ---\n" + texto  # Añade el número de página para referencia

    return texto_completo


def crear_df_de_directorio(directorio):
    """
    Lee todos los archivos PDF en un directorio y crea un DataFrame con tres columnas:
    id, titulo (nombre del archivo), texto (contenido extraído).
    :param directorio: Ruta del directorio que contiene los archivos PDF.
    :return: DataFrame con columnas id, titulo, texto.
    """
    archivos = [f for f in os.listdir(directorio) if f.endswith('.pdf')]
    
    datos = []  # Lista para almacenar los datos de los PDFs
    
    for idx, archivo in enumerate(archivos, 1):
        ruta_pdf = os.path.join(directorio, archivo)
        texto = extraer_texto_de_pdf_con_ocr(ruta_pdf)  # Extraemos el texto del PDF
        
        datos.append({
            'id': idx,
            'titulo': archivo,
            'texto': texto
        })
    
    # Crear un DataFrame con los datos recopilados
    df = pd.DataFrame(datos, columns=['id', 'titulo', 'texto'])
    
    return df


nltk.download('stopwords')
nltk.download('wordnet')

# Función para limpiar el texto
def limpiar_texto(texto):
    texto = re.sub(r'•|–|-|–|—', '', texto)  # Eliminar viñetas comunes y guiones
    texto = re.sub(r'\n\s*\n', '\n', texto)  # Eliminar saltos de línea excesivos
    texto = re.sub(r'\s+', ' ', texto)       # Reducir múltiples espacios a uno solo
    texto = texto.strip()                    # Eliminar espacios en blanco al inicio y al final
    return texto

# Función para preprocesar el texto (en español)
def preprocesar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñü ]', '', texto)  # Filtrar caracteres no alfabéticos y tildes
    
    stop_words = set(stopwords.words('spanish'))
    palabras = texto.split()
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
    
    lemmatizer = WordNetLemmatizer()  # Nota: Este lematizador es para inglés. Considera spaCy para español.
    palabras_lemmatizadas = [lemmatizer.lemmatize(palabra) for palabra in palabras_filtradas]
    
    texto_procesado = ' '.join(palabras_lemmatizadas)
    
    return texto_procesado


def limpieza_datos():
    directorio = "/Users/alejandromunoz/Desktop/Proyectos/LLM para gestión documentaria/libros"
    df_pdfs = crear_df_de_directorio(directorio)

    # Aplicar las funciones al DataFrame
    df_pdfs['texto'] = df_pdfs['texto'].apply(limpiar_texto)
    df_pdfs['texto'] = df_pdfs['texto'].apply(preprocesar_texto)
    return df_pdfs

