from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus.exceptions import MilvusException
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from multiprocessing import Pool, cpu_count

# Conexión a Milvus
def conectar_milvus(host="localhost", port="19530"):
    connections.connect("default", host=host, port=port)
    print("Conexión exitosa a Base de datos Vectorial")

# Creación de colección
def crear_coleccion(nombre_coleccion, embedding_size):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_size),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="document_id", dtype=DataType.INT64)
    ]
    schema = CollectionSchema(fields, "Colección para almacenar embeddings, textos y document_id")
    
    try:
        collection = Collection(name=nombre_coleccion, schema=schema)
        print(f"Colección '{nombre_coleccion}' creada con éxito")
    except MilvusException as e:
        print(f"Error al crear la colección: {e}")
        collection = Collection(name=nombre_coleccion)

    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Índice creado con éxito")

    return collection

# Inserción de embeddings, textos y document_id
def insertar_embeddings_y_textos(collection, chunked_documents, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = []
    texts = []
    document_ids = []

    for doc in chunked_documents:
        text = doc['text']
        document_id = doc['document_id']
        embedding = model.encode(text)
        embeddings.append(embedding)
        texts.append(text)
        document_ids.append(document_id)

    embeddings_flat = [embedding.tolist() for embedding in embeddings]
    data = [embeddings_flat, texts, document_ids]
    
    collection.insert(data)
    print(f"Insertados {len(embeddings)} embeddings en la colección '{collection.name}'")

# Búsqueda de embeddings en Milvus
def buscar_embeddings(collection, embedding_query, top_k=5):
    embedding_query_flat = embedding_query.tolist()
    collection.load()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    try:
        resultados = collection.search([embedding_query_flat], "embedding", search_params, limit=top_k, output_fields=["id", "text", "document_id"])

        chunks_encontrados = []
        for resultado in resultados:
            for hit in resultado:
                chunk_info = {
                    "id": hit.id,
                    "text": hit.entity.get('text'),
                    "document_id": hit.entity.get('document_id'),
                    "distance": hit.distance
                }
                chunks_encontrados.append(chunk_info)
                print(f"ID: {chunk_info['id']}, Distancia: {chunk_info['distance']}, Texto: {chunk_info['text']}, Document ID: {chunk_info['document_id']}")

        return chunks_encontrados
    except MilvusException as e:
        print(f"Error durante la búsqueda: {e}")
        raise

# Obtener embedding
def get_embedding(text, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Buscar otro chunk en la lista `chunked_documents` con el mismo document_id y la palabra "PROBLEM"
def buscar_chunk_con_problema(chunked_documents, document_id):
    for chunk in chunked_documents:
        if chunk['document_id'] == document_id and 'PROBLEM' in chunk['text']:
            return chunk
    return None



def procesar_documentos_y_crear_chunks(df_pdfs, pregunta):
    """
    Procesa documentos en un DataFrame y los divide en chunks para su almacenamiento en Milvus.
    
    Args:
        df_pdfs (DataFrame): DataFrame con columnas 'id', 'titulo' y 'texto'.
        nombre_coleccion (str): Nombre de la colección en Milvus.
        embedding_size (int): Tamaño del embedding.

    Returns:
        list: Lista de diccionarios con chunks divididos y sus metadatos.
    """
    # Conectar a Milvus
    conectar_milvus()
    nombre_coleccion = "embedding_collection"
    embedding_size = 384
    # Crear la colección en Milvus
    collection = crear_coleccion(nombre_coleccion, embedding_size)
    
    # Inicializar el divisor de texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    
    # Lista para almacenar chunks
    chunked_documents = []
    chunk_unique_id = 0
    
    # Procesar cada documento en el DataFrame
    for doc_id, row in df_pdfs.iterrows():
        document = row['texto']  # Extraer texto de la columna 'texto'
        chunks = text_splitter.split_text(document)
        
        # Crear chunks con metadatos
        for chunk_id, chunk in enumerate(chunks):
            chunked_documents.append({
                "id": chunk_unique_id,
                "document_id": doc_id,  # Usamos el id de df_pdfs como document_id
                "text": chunk
            })
            chunk_unique_id += 1
            
    insertar_embeddings_y_textos(collection, chunked_documents)

    pregunta = pregunta
    embedding_pregunta = get_embedding(pregunta)
    
    return collection, embedding_pregunta


