from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from pathlib import Path
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any

from config import *


class VectorRAGSystem:
    """Sistema RAG avanzado con ChromaDB y MultiQueryRetriever."""
    
    def __init__(self, chroma_path: str = "chroma_db"):
        self.chroma_path = Path(chroma_path)
        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = None
        self.retriever = None
        
        # Configurar logging para MultiQueryRetriever
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Carga el vectorstore de ChromaDB."""
        try:
            if not self.chroma_path.exists():
                print(f"⚠️ Vectorstore no encontrado en {self.chroma_path}")
                return
                
            self.vectorstore = Chroma(
                persist_directory=str(self.chroma_path),
                embedding_function=self.embeddings,
                collection_name="helpdesk_knowledge"
            )
            
            # Crear MultiQueryRetriever
            self.retriever = MultiQueryRetriever.from_llm(
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}  # Más documentos para mejor contexto
                ),
                llm=self.llm,
                prompt=self._get_multi_query_prompt()
            )
            
            print("✅ VectorRAGSystem inicializado correctamente")
            
        except Exception as e:
            print(f"❌ Error cargando vectorstore: {str(e)}")
            self.vectorstore = None
            self.retriever = None
    
    def _get_multi_query_prompt(self):
        """Prompt personalizado para MultiQueryRetriever."""
        return ChatPromptTemplate.from_template(
            """Eres un asistente de helpdesk experto. Tu tarea es generar múltiples 
versiones de la consulta del usuario para recuperar documentos relevantes de una 
base de conocimiento de soporte técnico.

Genera 3 versiones diferentes de la consulta original, considerando:
- Sinónimos técnicos
- Diferentes formas de expresar el mismo problema
- Variaciones en terminología de helpdesk

Consulta original: {question}

Versiones alternativas:"""
        )
    
    def buscar(self, consulta: str) -> Dict[str, Any]:
        """Busca respuestas usando MultiQueryRetriever."""
        if not self.retriever:
            return {
                "respuesta": "Sistema RAG no disponible. Verifique la configuración.",
                "confianza": 0.0,
                "fuentes": []
            }
        
        try:
            # Buscar documentos relevantes con MultiQueryRetriever
            documentos = self.retriever.invoke(consulta)
            
            if not documentos:
                return {
                    "respuesta": "No encontré información relevante en la base de conocimiento.",
                    "confianza": 0.1,
                    "fuentes": []
                }
            
            # Extraer información de los documentos
            contexto_partes = []
            fuentes = []
            
            for i, doc in enumerate(documentos[:3]):  # Usar top 3 documentos
                contenido = doc.page_content.strip()
                if contenido:
                    contexto_partes.append(f"Documento {i+1}: {contenido}")
                    
                    # Extraer fuentes
                    filename = doc.metadata.get('filename', f'doc_{i+1}')
                    if filename not in fuentes:
                        fuentes.append(filename)
            
            if not contexto_partes:
                return {
                    "respuesta": "Documentos encontrados pero sin contenido útil.",
                    "confianza": 0.2,
                    "fuentes": fuentes
                }
            
            # Generar respuesta usando el contexto encontrado
            contexto = "\n\n".join(contexto_partes)
            respuesta = self._generar_respuesta(consulta, contexto)
            
            # Calcular confianza basada en la relevancia
            confianza = self._calcular_confianza(consulta, documentos)
            
            return {
                "respuesta": respuesta,
                "confianza": confianza,
                "fuentes": fuentes
            }
            
        except Exception as e:
            print(f"❌ Error en búsqueda RAG: {str(e)}")
            return {
                "respuesta": f"Error interno en la búsqueda: {str(e)}",
                "confianza": 0.0,
                "fuentes": []
            }
    
    def _generar_respuesta(self, consulta: str, contexto: str) -> str:
        """Genera una respuesta basada en el contexto encontrado."""
        prompt = ChatPromptTemplate.from_template(
            """Eres un asistente de helpdesk experto. Responde a la consulta del usuario 
basándote únicamente en el contexto proporcionado de la base de conocimiento.

Instrucciones:
- Proporciona una respuesta clara, directa y útil
- Si el contexto no contiene información suficiente, dilo claramente
- Mantén un tono profesional pero amigable
- No inventes información que no esté en el contexto

Contexto de la base de conocimiento:
{contexto}

Consulta del usuario: {consulta}

Respuesta:"""
        )
        
        try:
            response = self.llm.invoke(prompt.format(consulta=consulta, contexto=contexto))
            return response.content.strip()
        except Exception as e:
            return f"Error generando respuesta: {str(e)}"
    
    def _calcular_confianza(self, consulta: str, documentos: List) -> float:
        """Calcula la confianza basada en la relevancia de los documentos."""
        if not documentos:
            return 0.0
        
        # Factores para calcular confianza
        num_docs = len(documentos)
        palabras_consulta = set(consulta.lower().split())
        
        puntuacion_relevancia = 0
        total_contenido = 0
        
        for doc in documentos[:3]:  # Evaluar top 3
            contenido = doc.page_content.lower()
            total_contenido += len(contenido.split())
            
            # Contar coincidencias de palabras clave
            coincidencias = sum(1 for palabra in palabras_consulta 
                              if palabra in contenido and len(palabra) > 2)
            
            puntuacion_relevancia += coincidencias
        
        # Normalizar puntuación
        if palabras_consulta and total_contenido > 0:
            confianza_base = min(puntuacion_relevancia / len(palabras_consulta), 1.0)
            
            # Bonus por tener múltiples documentos relevantes
            bonus_documentos = min(num_docs / 4.0, 0.2)
            
            # Bonus por longitud adecuada del contenido
            bonus_contenido = min(total_contenido / 1000.0, 0.1)
            
            confianza_final = min(confianza_base + bonus_documentos + bonus_contenido, 1.0)
            
            return round(confianza_final, 2)
        
        return 0.3  # Confianza mínima si se encontraron documentos