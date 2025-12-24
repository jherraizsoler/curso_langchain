import hashlib
from typing import List
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os 

from config import * 

class DocumentProcessor:
    """Procesador de documentos para el sistema RAG."""
    
    def __init__(self, docs_path: str = "docs", chroma_path: str = "./chroma_db"):
        # 1. Detectar de forma dinámica la carpeta donde está este archivo script
        # Esto asegura que funcione aunque lances Streamlit desde la raíz del curso
        base_dir = Path(__file__).resolve().parent
    
        # 2. Construir rutas absolutas uniendo la base con el nombre de la carpeta
        self.docs_path = base_dir / "docs"
        self.chroma_path = base_dir / "chroma_db"
        
        # 3. Log para verificar en la terminal de Streamlit qué ruta está usando
        print(f"📍 Buscando documentos en: {self.docs_path}")
        
        # 4. Configuración normal de LangChain
        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
    def load_documents(self) -> List[Document]:
        """Carga documentos markdown del directorio docs."""
        print(f"📚 Cargando documentos desde {self.docs_path}")
        
        # Cargar archivos markdown
        loader = DirectoryLoader(
            str(self.docs_path),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        documents = loader.load()
        
        # Enriquecer metadatos
        for doc in documents:
            filename = Path(doc.metadata["source"]).stem
            doc.metadata.update({
                "filename": filename,
                "doc_type": self._get_doc_type(filename),
                "doc_id": self._generate_doc_id(doc.page_content)
            })
        
        print(f"✅ Cargados {len(documents)} documentos")
        return documents
    
    def _get_doc_type(self, filename: str) -> str:
        """Determina el tipo de documento basado en el nombre."""
        if "faq" in filename.lower():
            return "faq"
        elif "manual" in filename.lower():
            return "manual"
        elif "troubleshooting" in filename.lower():
            return "troubleshooting"
        else:
            return "general"
    
    def _generate_doc_id(self, content: str) -> str:
        """Genera un ID único para el documento."""
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Divide documentos en chunks más pequeños."""
        print("✂️  Dividiendo documentos en chunks...")
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Agregar metadatos de chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content)
            })
        
        print(f"✅ Creados {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Crea el vectorstore con ChromaDB."""
        print("🔄 Creando vectorstore con ChromaDB...")
        
        # 1. Intentar limpiar el directorio anterior
        if self.chroma_path.exists():
            import shutil
            try:
                shutil.rmtree(self.chroma_path)
                print("🧹 Directorio previo eliminado.")
            except PermissionError:
                # Si Windows bloquea la carpeta, no detenemos el programa
                print("⚠️ No se pudo borrar la carpeta (archivo en uso). Actualizando colección existente...")

        # 2. Crear o actualizar el vectorstore
        # Al no borrar la carpeta (si falló el rmtree), Chroma intentará añadir los docs
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.chroma_path),
            collection_name="helpdesk_knowledge"
        )
        
        # 3. Forzar persistencia (importante en versiones antiguas de Chroma)
        # En versiones muy nuevas esto se hace solo, pero no estorba
        if hasattr(vectorstore, 'persist'):
            vectorstore.persist()
        
        print(f"✅ Vectorstore listo en {self.chroma_path}")
        print(f"📊 Total de vectores procesados: {len(documents)}")
        
        return vectorstore
    
    def load_existing_vectorstore(self) -> Chroma:
        """Carga vectorstore existente."""
        if not self.chroma_path.exists():
            raise FileNotFoundError(f"Vectorstore no encontrado en {self.chroma_path}")
        
        vectorstore = Chroma(
            persist_directory=str(self.chroma_path),
            embedding_function=self.embeddings,
            collection_name="helpdesk_knowledge"
        )
        
        return vectorstore
    
    def setup_rag_system(self, force_rebuild: bool = False):
        """Configura el sistema RAG completo."""
        print("🚀 Configurando sistema RAG...")
        
        # Verificar si ya existe y no forzar rebuild
        if self.chroma_path.exists() and not force_rebuild:
            print("📦 Vectorstore existente encontrado")
            return self.load_existing_vectorstore()
        
        # Cargar y procesar documentos
        documents = self.load_documents()
        if not documents:
            print("⚠️  No se encontraron documentos para procesar")
            return None
        
        # Dividir documentos
        chunks = self.split_documents(documents)
        
        # Crear vectorstore
        vectorstore = self.create_vectorstore(chunks)
        
        print("✅ Sistema RAG configurado exitosamente")
        return vectorstore
    
    def test_search(self, vectorstore: Chroma, query: str = "resetear contraseña"):
        """Prueba la funcionalidad de búsqueda."""
        print(f"\n🔍 Probando búsqueda: '{query}'")
        
        results = vectorstore.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"\n📄 Resultado {i}:")
            print(f"Tipo: {doc.metadata.get('doc_type', 'unknown')}")
            print(f"Archivo: {doc.metadata.get('filename', 'unknown')}")
            print(f"Contenido: {doc.page_content[:200]}...")
        
        return results


def main():
    """Función principal para configurar RAG."""
    print("🎧 Configuración RAG - Helpdesk 2.0")
    print("=" * 40)
    
    # Configurar procesador
    processor = DocumentProcessor(docs_path=DOCS_PATH, chroma_path=CHROMADB_PATH)
    
    # Configurar sistema RAG
    vectorstore = processor.setup_rag_system(force_rebuild=True)
    
    if vectorstore:
        # Probar búsquedas
        test_queries = [
            "resetear contraseña",
            "error 500",
            "cancelar suscripción",
            "aplicación lenta"
        ]
        
        for query in test_queries:
            processor.test_search(vectorstore, query)
    
    print("\n✅ Configuración completada")


if __name__ == "__main__":
    main()