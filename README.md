## Implementación Full Stack de chatbot para estudio de abogados a partir de una arquitectura RAG

### 1. Introducción
Hoy en día, los bufetes de abogados se enfrentan a un problema creciente en cuanto a la revisión y análisis a gran escala de documentos judiciales. Dado el gran tamaño de información legal existente y nuevas leyes y reformas que son creadas todos los días, realizar búsquedas con alta precisión sobre un determinado caso judicial representa un desafío. No solo se requiere una gran cantidad de horas hombre para analizar esta documentación, sino que también existe un alto riesgo de perder información importante, lo cual afectaría la calidad del análisis que se realiza posteriormente con esta información.
Ante este escenario, la implementación de chatbots y herramientas de inteligencia artificial representa una solución potencial e innovadora para consultar información de manera eficiente. Los chatbots pueden ser utilizados en diversas áreas dentro de un estudio de abogados, desde la automatización de tareas repetitivas hasta el soporte en la clasificación y segmentación de documentos jurídicos. Gracias a su capacidad para procesar grandes volúmenes de datos y responder de manera rápida y estructurada, estas herramientas no solo optimizan el tiempo de los abogados, sino que también garantizan una cobertura más exhaustiva de la información disponible.

Este informe explora las posibles aplicaciones de arquitecturas RAG en los estudios de abogados, enfocándose en cómo pueden ser utilizados para mejorar la eficiencia en la búsqueda y análisis de documentos judiciales. 

### 2. Problemática

La revisión de documentos judiciales provenientes de diversas fuentes a gran escala representa un desafío crítico para los estudios de abogados, dado el tiempo y esfuerzo que requiere analizar manualmente la información contenida en cada documento para determinar su relevancia en un caso específico. La creciente cantidad de información legal disponible, así como la complejidad inherente a su procesamiento, ralentiza considerablemente el desarrollo de procesos legales, incrementando no solo los tiempos de respuesta, sino también el riesgo de omitir información crucial para la estrategia jurídica. Estas dificultades impactan directamente en la calidad y exhaustividad del análisis, lo que puede comprometer la toma de decisiones fundamentadas y afectar los resultados esperados en cada caso.

Adicionalmente, esta problemática genera ineficiencias operativas significativas, al requerir un mayor número de horas hombre dedicadas a tareas repetitivas de clasificación y análisis de documentos, desviando recursos valiosos que podrían ser enfocados en actividades de mayor valor agregado. Frente a esta situación, es indispensable contar con una solución escalable que analizar los documentos que el estudio de abogados tiene actualmente, como incorporar progresivamente nuevos documentos y fuentes de información, asegurando la adaptabilidad a las necesidades cambiantes del mercado. Asimismo, es fundamental que esta solución sea sostenible, de manera que pueda ser actualizada por otros desarrolladores, fomentando un ciclo continuo de innovación que garantice su evolución y efectividad a largo plazo.


### 3. Solución

Se planteó una arquitectura de proyecto fullstack, en la cual se cuenta tanto con interfaz gráfica frontend para interactuar con el usuario desarrollada en React como con un backend en Python, el cual implementa una arquitectura Retrieval-Augmented Generation (RAG), utilizando Llama3 como modelo de IA generativa.

![Sin título (2)](https://github.com/user-attachments/assets/4932d0ac-6d9b-482a-8737-6f6a8c1c45db)

En la Figura 1 se observa la arquitectura de la aplicación realizada, en la cual el usuario realiza una consulta mediante la interfaz frontend, la cual fue realizada con React. Esta interfaz cuenta con un mensaje predeterminado de bienvenida como se muestra en la Figura 2.

![image](https://github.com/user-attachments/assets/68925d3a-8088-40f7-b5e3-627f3d4a3731)

La aplicación envía la pregunta mediante una solicitud POST a una API desarrollada en Node.js, que actúa como intermediaria entre el frontend y el backend. Esta API recibe las preguntas del usuario y las reenvía al backend implementado en Flask, encargado de acceder a la arquitectura RAG desarrollada. Para la arquitectura RAG, inicialmente se realizó un proceso de limpieza y preparación de los datos, ya que estos se encontraban en formato PDF. Fue necesario utilizar una librería OCR para extraer la información textual de los documentos. Una vez obtenidos los datos, estos se dividieron en chunks o fragmentos más pequeños. Posteriormente, se utilizó el tokenizer “all-MiniLM-L6-v2” de Hugging Face para convertir dichos chunks en embeddings (vectores numéricos). Estos embeddings se almacenaron en una base de datos vectorial Milvus, la cual se encuentra desplegada en un contenedor Docker.

El backend en Flask queda a la espera de recibir una solicitud POST con una pregunta del usuario. Cuando la petición es recibida, la pregunta es transformada en un vector de consulta y enviada a la base de datos vectorial Milvus. Milvus devuelve los 3 embeddings más cercanos a la pregunta, los cuales se utilizan como contexto para enriquecer la interacción con el modelo de IA generativa. En este caso, el modelo utilizado es Llama3, encargado de generar la respuesta final. Una vez que la respuesta es generada por Llama3, esta es retornada al backend Flask, reenviada a la API en Node.js y, finalmente, mostrada al usuario en la interfaz Frontend.

A continuación se muestran capturas de la aplicación en funcionamiento. Primero, en la Figura 3 se observa cómo interactúa el API de Node.js con la API de Flask para responder una pregunta.

![image](https://github.com/user-attachments/assets/605ca9ee-cf36-44c4-829d-177a8a2b0e0b)

Posteriormente, se muestra en la Figura 4 una prueba realizando una consulta general sobre un documento en específico. Podemos observar que se responde correctamente la pregunta en base a la información recuperada del documento “Vigencia Ing. Sandro Che - Chimu - Jefe de Sistemas - 29.11.23.pdf”. En el archivo main.py se puede observar como esta funcionando el modelo RAG, además se pueden observar los embeddings devueltos que han sido utilizados como contexto para responder la pregunta.

![image](https://github.com/user-attachments/assets/9665a09d-7660-455f-93ea-56bb634200ae)

En la Figura 5, realizamos una prueba con uno de los documentos que se encuentran almacenados en la carpeta el cual habla sobre normas legales. Para ello, realizamos la pregunta sobre la ley de servicio civil y podemos observar cómo nos devuelve la respuesta correcta corroborando con el documento del costado. Además, al preguntarle sobre su función obtenemos una respuesta sobre el prompt que le hemos dado.

![image](https://github.com/user-attachments/assets/45ad7cf3-964b-40ee-988b-6b2864808761)

### 4. Conclusiones

La implementación de esta arquitectura permite transformar documentos judiciales en información accesible y procesable mediante un flujo eficiente. La integración con el modelo de IA generativa Llama3 en una arquitectura RAG permitió generar respuestas contextualizadas y de alta calidad. La combinación de un backend desarrollado en Flask, una API intermediaria en Node.js y un Frontend interactivo en React ofrece una solución escalable y eficiente que optimiza la gestión y búsqueda de información compleja.

Para un estudio de abogados, contar con esta aplicación facilita el proceso de análisis de grandes volúmenes de información judicial. La posibilidad de realizar consultas específicas y obtener respuestas en base al conocimiento almacenado permite la toma de decisiones estratégicas y un mejor manejo de la información. Esta herramienta no solo agiliza los procesos repetitivos de revisión de información de manera manual, sino que también permite a los especialistas poder tomar conclusiones más acertadas al contar con toda la información requerida. 

