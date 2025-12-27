from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from memory_manager import ModernMemoryManager, MemoryState
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
import os

class ModernChatbot:

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_manager = ModernMemoryManager(user_id)

        # Configuracion del modelo LLM
        self.llm = ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE
        )

        # Template del sistema con contexto dinámico
        self.system_template = """Eres un asistente personal inteligente y amigable.

Características de tu personalidad:
- Eres útil, empático y conversacional
- Recuerdas información importante de conversaciones anteriores
- Adaptas tu estilo a las preferencias del usuario
- Eres proactivo ofreciendo sugerencias relevantes
- Mantienes un tono profesional pero cercano

{context}

Usa esta información para personalizar tus respuestas, pero no menciones explícitamente que tienes memoria a menos que sea relevante para la conversación."""

        # Configurar el trimming de mensajes para gestión del contexto
        self.message_trimmer = trim_messages(
            strategy="last",
            max_tokens=4000,
            token_counter=self.llm,
            start_on="human",
            include_system=True
        )

        # Crear aplicacion de LangGraph
        self.app = self._create_app()

    def _create_app(self):
        """Crea la aplicación de LangGraph con estado extendido."""
        workflow = StateGraph(state_schema=MemoryState)

        def memory_retrieval_node(state):
            """Nodo que recupera memorias relevantes."""
            messages = state['messages']

            if not messages:
                return {"vector_memories": []}
            
            # Obtener el ultimo mensaje del usuario
            last_user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg
                    break

            if not last_user_message:
                return {"vector_memories": []}
            
            # Buscar memorias vectoriales relevantes
            relevant_memories = self.memory_manager.search_vector_memory(
                last_user_message.content
            )

            return {"vector_memories": relevant_memories}
        
        def context_optimization_node(state):
            """Nodo que optimiza el contexto usando trim_messages."""
            messages = state['messages']

            # Aplicar trimming inteligente
            trimmed_messages = self.message_trimmer.invoke(messages)

            return {"messages": trimmed_messages}
        
        def response_generation_node(state):
            """Nodo que genera la respuesta usando el contexto optimizado."""
            messages = state['messages']
            vector_memories = state.get('vector_memories', [])

            if not messages:
                return {"messages": []}
            
            # Construir contexto con memorias vectoriales
            if vector_memories:
                context_parts = ["Información relevante que recuerdas del usuario:"]
                for memory in vector_memories:
                    context_parts.append(f"- {memory}")
                context = "\n".join(context_parts)
            else:
                context = "No hay information previa relevante disponible."

            # Crear el prompt con el contexto dinamico
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_template.format(context=context)),
                MessagesPlaceholder(variable_name="messages")
            ])

            # Generar la respuesta
            chain = prompt | self.llm
            response = chain.invoke({"messages": messages})

            return {"messages": response}

        def memory_extraction_node(state):
            """Nodo que extrae y guarda nuevas memorias vectoriales."""
            messages = state['messages']
            last_extraction = state.get('last_memory_extraction')

            # Obtener ultimo mensaje del usuario
            last_user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg
                    break

            if not last_user_message:
                return {}
            
            # Solo procesar si no hemos extraído memorias de este mensaje
            if last_extraction != last_user_message.content:
                self.memory_manager.extract_and_store_memories(last_user_message.content)
                return {"last_memory_extraction": last_user_message.content}
            
            return {}
        
        # Configurar el grafo con flujo secuencial
        workflow.add_node("memory_retrieval", memory_retrieval_node)
        workflow.add_node("context_optimization", context_optimization_node)
        workflow.add_node("response_generation", response_generation_node)
        workflow.add_node("memory_extraction", memory_extraction_node)

        # Definir el flujo del grafo
        workflow.add_edge(START, "memory_retrieval")
        workflow.add_edge("memory_retrieval", "context_optimization")
        workflow.add_edge("context_optimization", "response_generation")
        workflow.add_edge("response_generation", "memory_extraction")
        workflow.add_edge("memory_extraction", END)

        # Configurar persistente con SqliteSaver
        db_path = os.path.join(
            self.memory_manager.user_dir,
            "langgraph_memory.db"
        )

        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        return workflow.compile(checkpointer=checkpointer)
    
    def chat(self, message: str, chat_id: str = "default"):
        """Envía un mensaje y obtiene respuesta del chatbot."""
        try:
            # Configuración para el thread especifico del chat
            config = {"configurable": {"thread_id": f"user_{self.user_id}_chat_{chat_id}"}}

            # Actualizamos el titulo del chat si es necesario
            chat_info = self.memory_manager.get_chat_info(chat_id)
            if chat_info["title"] == "Nuevo chat":
                chat_title = self.memory_manager._generate_chat_title(message)
                self.memory_manager.update_chat_metadata(chat_id, chat_title)

            # Invocar el chatbot con el nuevo mensaje
            result = self.app.invoke(
                {"messages": [HumanMessage(content=message)]},
                config
            )

            # Extraer respuesta
            assistant_response = result["messages"][-1].content

            return {
                "success": True,
                "response": assistant_response,
                "error": None,
                "memories_used": len(result.get("vector_memories", [])),
                "context_optimized": True
            }
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e),
                "memories_used": 0,
                "context_optimized": False
            }
        
    def get_conversation_history(self, chat_id: str = "default", limit: int = 50):
        """Obtiene el historial de conversación usando el estado de LangGraph."""
        try:
            config = {"configurable": {"thread_id": f"user_{self.user_id}_chat_{chat_id}"}}

            # Obtener el estado actual
            state = self.app.get_state(config)

            if not state.values or "messages" not in state.values:
                return []
            
            messages = state.values["messages"]

            # Convertir a formato para la UI
            history = []
            for msg in messages[-limit:]:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    history.append({
                        'role': 'user' if isinstance(msg, HumanMessage) else 'assistant',
                        'content': msg.content,
                        'timestamp': getattr(msg, 'timestamp', None) or "2026-01-01T00:00:00"
                    })
            return history
        
        except Exception as e:
            print(f"Error obteniendo el historial: {e}")
            return []
        
    def clear_conversation(self, chat_id: str = "default") -> bool:
        """Limpia el historial de conversación"""
        try:
            config = {"configurable": {"thread_id": f"user_{self.user_id}_chat_{chat_id}"}}
            
            # Crear un estado vacío para "resetear" la conversación
            self.app.invoke({"messages": []}, config)
            return True
            
        except Exception as e:
            print(f"Error limpiando conversación: {e}")
            return False
    
    def delete_chat_from_langgraph(self, chat_id: str) -> bool:
        """Elimina un chat de la base de datos SQLite de LangGraph del usuario."""
        conn = None
        try:
            thread_id = f"user_{self.user_id}_chat_{chat_id}"
            # Usamos la ruta que ya definiste en el __init__
            db_path = self.langgraph_db_path 
            
            # Conectar específicamente a la DB del usuario
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Eliminar registros de las tablas de LangGraph
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error eliminando chat de LangGraph: {e}")
            return False
        finally:
            if conn:
                conn.close() # ESTO LIBERA EL ARCHIVO PARA QUE WINDOWS PERMITA BORRARLO
class ChatbotManager:

    _instances = {}

    @classmethod
    def get_chatbot(cls, user_id):
        """Obtiene o crea una instancia de chatbot para un usuario"""
        if user_id not in cls._instances:
            cls._instances[user_id] = ModernChatbot(user_id)

        return cls._instances[user_id]
    
    @classmethod
    def remove_chatbot(cls, user_id):
        """Elimina una instancia de chatbot"""
        if user_id in cls._instances:
            del cls._instances[user_id]

    @classmethod
    def clear_all(cls):
        """Limpia toas las instancias de chatbot"""
        cls._instances.clear()