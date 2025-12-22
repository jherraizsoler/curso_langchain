import os
# Usamos la nueva ruta de importación recomendada
from langchain_google_community import GoogleDriveLoader

# Es buena práctica asegurar que las rutas sean tratadas correctamente por el sistema operativo
credentials_path = os.path.join("Nombre carpeta padre", "Archivo .json ejemplo credentials_google.json")
token_path = os.path.join("Nombre carpeta padre", "Archivo .json ejemplo token_google.json")

# Configuración del loader
loader = GoogleDriveLoader(
    # Entras en la carpeta en drive y arriba en la url 
    # Ejemplo de url, no valida:   https://drive.google.com/drive/folders/1A2b3C4d5E6f7G8h9I0jKLmNoPqRsTuVw
    # el id de la carpeta es lo que va después de folders/ y termina en ? si hay algo más
    folder_id="tu_id_de_carpeta_aquí",
    credentials_path=credentials_path,
    token_path=token_path,
    recursive=True
)

documents = loader.load()
print(f"Metadatos: {documents[0].metadata}")
print(f"Contenido: {documents[0].page_content}")

# Carga de documentos
#try:
#    documents = loader.load()
#    print(f"Se han cargado {len(documents)} documentos.")
#    for doc in documents:
#        print("-" * 30)
#        print(f"Contenido: {doc.page_content[:100]}...") # Imprime los primeros 100 caracteres
#except Exception as e:
#    print(f"Ocurrió un error al cargar los documentos: {e}")
    