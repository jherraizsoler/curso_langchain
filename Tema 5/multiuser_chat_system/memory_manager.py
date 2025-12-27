import os
import uuid
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from typing_extensions import TypedDict, Annotated
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import shutil
import gc
import time

from config import USERS_DIR, MAX_VECTOR_RESULTS, DEFAULT_MODEL

# Estado extendido que combina mensajes con memoria vectorial
class MemoryState(TypedDict):
    """Estado que combina mensajes de LangGraph con memoria vectorial."""
    messages: Annotated[List[BaseMessage], add_messages]
    vector_memories: List[str] # IDs de memorias vectoriales activas
    user_profile: Dict[str, Any] # Perfil del usuario
    last_memory_extraction: Optional[str] # Ultimo mensaje procesado para memorias

class ExtractedMemory(BaseModel):
    """Modelo para memoria extraída estructurada."""
    category: str = Field(description="Categoria: personal, profesional, preferencias, hecho_importantes")
    content: str = Field(description="Contenido de la memoria")
    importance: int = Field(description="Importancia del 1 al 5", ge=1, le=5)

class ModernMemoryManager:

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.user_dir = os.path.join(USERS_DIR, user_id)
        os.makedirs(self.user_dir, exist_ok=True)

        # Base de datos vectorial chromadb para memoria transversal
        self.chromadb_path = os.path.join(self.user_dir, "chromadb")
        self._init_vector_db()

        # Sistema de extracción inteligente de memoria transversal
        self._init_extraction_system()

        # Ruta de la base de datos LangGraph
        self.langgraph_db_path = os.path.join(self.user_dir, "langgraph_memory.db")

    def _init_vector_db(self):
        """Inicializa la base de datos vectorial chromadb"""
        try:
            self.vectorstore = Chroma(
                collection_name=f"memoria_{self.user_id}",
                embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
                persist_directory=self.chromadb_path
            )

            self.client = chromadb.PersistentClient(path=self.chromadb_path)
            try:
                self.collection = self.client.get_collection(f"memoria_{self.user_id}")
            except:
                self.collection = self.client.create_collection(f"memoria_{self.user_id}")

        except Exception as e:
            print(f"Error inicializando Chromadb: {e}")
            self.vectorstore = None
            self.collection = None
    
    def _init_extraction_system(self):
        """Inicializa el sistema de extracción inteligente de memoria transversal."""
        try:
            self.extraction_llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
            self.memory_parser = PydanticOutputParser(pydantic_object=ExtractedMemory)

            self.extraction_template = PromptTemplate(
                template="""Analiza el siguiente mensaje del usuario y determina si contiene información importante que deba recordarse.

Categorías disponibles:
- personal: Nombre, edad, ubicación, familia, etc.
- profesional: Trabajo, empresa, proyectos, habilidades
- preferencias: Gustos, disgustos, preferencias personales
- hechos_importantes: Información relevante que debe recordarse

Mensaje del usuario: "{user_message}"

Si el mensaje contiene información importante, extrae UNA memoria (la más importante).
Si no contiene información relevante para recordar, responde con categoría "none".

{format_instructions}""",
                input_variables=["user_message"],
                partial_variables={"format_instructions": self.memory_parser.get_format_instructions()}
            )

            self.extraction_chain = self.extraction_template | self.extraction_llm | self.memory_parser
        
        except Exception as e:
            print(f"Error inicializando el sistema de extracción: {e}")
            self.extraction_chain = None

    # === GESTIÓN DE CHATS (híbridos: JSON ligero + LangGraph para persistencia) ===

    def get_user_chats(self):
        """Obtiene todos los chats del usuario."""
        try:
            # Si no existe archivo de metadatos, retornar vació
            chats_meta_file = os.path.join(self.user_dir, "chats_meta.json")
            if not os.path.exists(chats_meta_file):
                return []
            
            # Cargar metadatos
            with open(chats_meta_file, 'r', encoding='utf-8') as f:
                chats_data = json.load(f)

            # Ordenar por ultima actualización
            chats_data.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
            return chats_data
        
        except Exception as e:
            print(f"Error obteniendo chats: {e}")
            return []
        
    def _save_chats_metadata(self, chats_data):
        """Guarda metadatos ligeros del chat."""
        try:
            chats_meta_file = os.path.join(self.user_dir, "chats_meta.json")
            with open(chats_meta_file, 'w', encoding='utf-8') as f:
                json.dump(chats_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error guardando metadatos de chats {e}")
        
    def create_new_chat(self, first_message: str = ""):
        """Crea un nuevo chat y actualiza metadatos."""
        chat_id = str(uuid.uuid4())

        # Generar un titulo basado en el primer mensaje
        title = self._generate_chat_title(first_message) if first_message else "Nuevo chat"

        # Crear metadatos del chat
        new_chat = {
            'chat_id': chat_id,
            'title': title,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'message_count': 0
        }

        # Cargar chats existentes y agregar el nuevo
        chats_data = self.get_user_chats()
        chats_data.append(new_chat)
        self._save_chats_metadata(chats_data)

        return chat_id

    def update_chat_metadata(self, chat_id, title: str = None, increment_messages: bool = False):
        """Actualiza metadatos de un chat."""
        chats_data = self.get_user_chats()

        for chat in chats_data:
            if chat['chat_id'] == chat_id:
                if title:
                    chat['title'] = title
                if increment_messages:
                    chat['message_count'] = chat.get('message_count', 0) + 1
                chat['update_at'] = datetime.now().isoformat()
                break
        else:
            # Si no existe chat, crear entrada
            if chat_id:
                new_chat = {
                    'chat_id': chat_id,
                    'title': title or "Chat sin titulo",
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'message_count': 1 if increment_messages else 0
                }
                chats_data.append(new_chat)

        self._save_chats_metadata(chats_data)

    def delete_chat(self, chat_id):
        """Elimina un chat de los metadatos."""
        try:
            # Eliminar metadatos del chat
            chats_data = self.get_user_chats()
            chats_data = [chat for chat in chats_data if chat["chat_id"] != chat_id]
            self._save_chats_metadata(chats_data)
            return True
        except Exception as e:
            print(f"Error eliminando chat: {e}")
            return False
        
    def get_chat_info(self, chat_id):
        """Obtiene los metadatos de un chat especifico."""
        chats = self.get_user_chats()
        for chat in chats:
            if chat['chat_id'] == chat_id:
                return chat
        return None
    
    def _generate_chat_title(self, first_message):
        """Genera un titulo para el chat basado en el primer mensaje."""
        try:
            if not self.extraction_llm:
                return first_message[:30] + "..." if len(first_message) > 30 else first_message
            
            title_prompt = PromptTemplate(
                template="""Genera un título corto (máximo 4-5 palabras) para una conversación que comienza con este mensaje:

"{message}"

El título debe:
- Ser conciso y descriptivo
- Capturar el tema principal
- Ser apropiado para un historial de chat
- No incluir comillas

Título:""",
                input_variables=["message"]
            )

            title_chain = title_prompt | self.extraction_llm

            response = title_chain.invoke({"message": first_message[:200]})

            title = response.content.strip().strip('"').strip("'")
            return title if len(title) <= 50 else title[:47] + "..."
        
        except Exception as e:
            print(f"Error generando titulo: {e}")
            return first_message[:30] + "..." if len(first_message) > 30 else first_message


    # === MEMORIA VECTORIAL ===

    def save_vector_memory(self, text: str, metadata: Optional[Dict] = None):
        """Guarda información en la memoria vectorial."""
        if not self.collection:
            return ""
        
        try:
            memory_id = str(uuid.uuid4())
            doc_metadata = metadata or {}
            doc_metadata.update({
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "memory_id": memory_id
            })

            self.collection.add(
                documents=[text],
                ids=[memory_id],
                metadatas=[doc_metadata]
            )

            return memory_id
        
        except Exception as e:
            print(f"Error guardando memoria vectorial {e}")
            return ''
        
    
    def search_vector_memory(self, query: str, k: int = MAX_VECTOR_RESULTS):
        """Busca información relevante en la memoria vectorial."""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            return results['documents'][0] if results['documents'] else []
        
        except Exception as e:
            print(f"Error buscando en la memoria vectorial {e}")
            return []
        
    
    def get_all_vector_memories(self):
        """Obtiene todas las memorias vectoriales del usuario."""
        if not self.collection:
            return []
        
        try:
            results = self.collection.get()
            memories = []

            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    memory = {
                        'id': results['ids'][i],
                        'content': doc,
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    }
                    memories.append(memory)
            
            return memories
        
        except Exception as e:
            print(f"Error obteniendo memorias vectoriales: {e}")
            return []
        
    # === EXTRACCIÓN INTELIGENTE ===

    def extract_and_store_memories(self, user_message: str):
        """Extrae y almacena memorias usando LLM"""
        if not self.extraction_chain:
            return self._extract_memories_manual(user_message)
        
        try:
            extracted_memory = self.extraction_chain.invoke({
                "user_message": user_message
            })

            if extracted_memory.category != "none" and extracted_memory.importance >= 2:
                memory_id = self.save_vector_memory(
                    extracted_memory.content,
                    {
                        'category': extracted_memory.category,
                        'importance': extracted_memory.importance,
                        'original_message': user_message[:200]
                    }
                )
                return bool(memory_id)
            return False
        
        except Exception as e:
            print(f"Error en extraccion automatica: {e}")
            return self._extract_memories_manual(user_message)

    def _extract_memories_manual(self, user_message: str) -> bool:
        """Método manual de extracción (fallback)"""
        message_lower = user_message.lower()
        
        memory_rules = [
            (["me llamo", "mi nombre es", "soy"], "personal", f"Info personal: {user_message}"),
            (["trabajo en", "trabajo como", "mi profesión"], "profesional", f"Info profesional: {user_message}"),
            (["me gusta", "me encanta", "prefiero", "odio"], "preferencias", f"Preferencia: {user_message}"),
            (["importante", "recuerda que", "no olvides"], "hechos_importantes", f"Hecho importante: {user_message}")
        ]
        
        for phrases, category, memory_text in memory_rules:
            if any(phrase in message_lower for phrase in phrases):
                memory_id = self.save_vector_memory(memory_text, {'category': category})
                return bool(memory_id)
        
        return False
    
    # === CERRAR CONEXIONES BORRAR USUARIO ===
    def close_connections(self):
        """Cierre ultra-agresivo para liberar archivos bloqueados por Chroma/SQLite."""
        try:
            # 1. Liberar el vectorstore de LangChain
            if hasattr(self, 'vectorstore') and self.vectorstore:
                # Accedemos al cliente interno si existe para forzar cierre
                if hasattr(self.vectorstore, "_client"):
                    self.vectorstore._client.clear_system_cache()
                self.vectorstore = None
            
            # 2. Liberar el cliente de Chroma
            if hasattr(self, 'client') and self.client:
                try:
                    self.client.clear_system_cache()
                except:
                    pass
                self.client = None
            
            # 3. Forzar a Python a soltar los 'File Handles'
            import gc
            import time
            gc.collect()
            time.sleep(0.5) # Pausa para que el SO registre el cierre
            
            print("[+] Conexiones cerradas y memoria liberada.")
        except Exception as e:
            print(f"[-] Error en el cierre nuclear: {e}")

class UserManager:
    """Gestor simplificado de usuarios"""

    @staticmethod
    def get_users():
        """Obtiene un listado de usuarios existentes."""
        if not os.path.exists(USERS_DIR):
            return []
        
        users = []
        for item in os.listdir(USERS_DIR):
            user_path = os.path.join(USERS_DIR, item)
            if os.path.isdir(user_path):
                users.append(item)

        return sorted(users)
    
    @staticmethod
    def user_exists(user_id):
        """Verifica si un usuario existe."""
        user_path = os.path.join(USERS_DIR, user_id)
        return os.path.exists(user_path)
    
    @staticmethod
    def create_user(user_id):
        """Crea un nuevo usuario"""
        try:
            user_path = os.path.join(USERS_DIR, user_id)
            os.makedirs(user_path, exist_ok=True)
            return True
        
        except Exception as e:
            print(f"Error creando usuario: {e}")
            return False

    @staticmethod
    def delete_user_completely(user_id):
        import shutil, time, gc, os
        from config import USERS_DIR
        user_path = os.path.join(USERS_DIR, user_id)
        
        if not os.path.exists(user_path):
            return True

        for i in range(5):
            try:
                gc.collect()
                # Pausa incremental más larga: 1s, 2s, 3s...
                time.sleep(i + 1) 
                
                # Intentamos borrar
                shutil.rmtree(user_path)
                return True
            except PermissionError:
                print(f"[!] Archivo bloqueado por Windows (Intento {i+1}). Reintentando...")
                # OPCIONAL: Si falla mucho, podrías intentar borrar archivo por archivo
                # pero normalmente rmtree es lo mejor.
                continue
        return False