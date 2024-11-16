from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus.exceptions import MilvusException
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from multiprocessing import Pool, cpu_count
import pandas as pd


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

# Paralelismo para procesar documentos
def procesar_documento(row):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    doc_id = row[0]
    document = row[1]['texto']
    chunks = text_splitter.split_text(document)

    chunked_documents = []
    for chunk in chunks:
        chunked_documents.append({
            "document_id": doc_id,
            "text": chunk
        })
    return chunked_documents

def insertar_embeddings_y_textos(collection, chunked_documents, model, batch_size=1000):
    for i in range(0, len(chunked_documents), batch_size):
        batch = chunked_documents[i:i + batch_size]
        embeddings = [model.encode(doc['text']).tolist() for doc in batch]
        texts = [doc['text'] for doc in batch]
        document_ids = [doc['document_id'] for doc in batch]

        data = [embeddings, texts, document_ids]
        collection.insert(data)
    print(f"Insertados {len(chunked_documents)} embeddings en la colección '{collection.name}'")

# Obtener embedding
def get_embedding(text, model):
    return model.encode(text)

def procesar_documentos_y_crear_chunks(df_pdfs, pregunta, model):
    conectar_milvus()
    nombre_coleccion = "embedding_collection"
    embedding_size = 384
    collection = crear_coleccion(nombre_coleccion, embedding_size)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    with Pool(cpu_count()) as pool:
        chunked_documents = pool.map(procesar_documento, df_pdfs.iterrows())
    
    chunked_documents = [item for sublist in chunked_documents for item in sublist]
    insertar_embeddings_y_textos(collection, chunked_documents, model)

    embedding_pregunta = get_embedding(pregunta, model)
    return collection, embedding_pregunta

def buscar_embeddings(collection, embedding_query, top_k=5):
    embedding_query_flat = embedding_query.tolist()
    collection.load(timeout=60)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    try:
        resultados = collection.search(
            [embedding_query_flat], "embedding", search_params,
            limit=top_k, output_fields=["id", "text", "document_id"]
        )

        chunks_encontrados = []
        for resultado in resultados:
            for hit in resultado:
                chunks_encontrados.append({
                    "id": hit.id,
                    "text": hit.entity.get('text'),
                    "document_id": hit.entity.get('document_id'),
                    "distance": hit.distance
                })
        return chunks_encontrados
    except MilvusException as e:
        print(f"Error durante la búsqueda: {e}")
        raise
