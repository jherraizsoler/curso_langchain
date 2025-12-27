import streamlit as st
from datetime import datetime
from memory_manager import ModernMemoryManager, UserManager
from chatbot import ChatbotManager
from utils import (
    format_timestamp,
    truncate_text,
    validate_user_id,
    get_memory_category_icon
)
from config import PAGE_TITLE, PAGE_ICON

# Configuración de la página
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Inicializa el estado de la sesión"""
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = None
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'memory_manager' not in st.session_state:
        st.session_state.memory_manager = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'show_memories' not in st.session_state:
        st.session_state.show_memories = False

def user_selection_sidebar():
    """Sidebar para selección/creación de usuarios"""
    st.sidebar.header("👤 Gestión de Usuarios")
    
    existing_users = UserManager.get_users()
    
    if existing_users:
        st.sidebar.subheader("Seleccionar o Eliminar")
        for u_id in existing_users:
            col_user, col_del = st.sidebar.columns([4, 1])
            
            # Botón para seleccionar usuario
            with col_user:
                is_current = u_id == st.session_state.current_user
                if st.button(f"{'➡️ ' if is_current else ''}{u_id}", key=f"sel_{u_id}", use_container_width=True):
                    st.session_state.current_user = u_id
                    st.session_state.chatbot = ChatbotManager.get_chatbot(u_id)
                    st.session_state.memory_manager = ModernMemoryManager(u_id)
                    st.session_state.current_chat = None
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Botón de basura para eliminar
            with col_del:
                if st.button("🗑️", key=f"del_user_{u_id}", help=f"Eliminar TODO de {u_id}"):
                    # Guardamos en sesión que queremos borrar este usuario para mostrar el diálogo
                    st.session_state.user_to_delete = u_id
        
        # Ventana de confirmación (Modal)
        if 'user_to_delete' in st.session_state and st.session_state.user_to_delete:
            u_to_del = st.session_state.user_to_delete
            
            @st.dialog("⚠️ ¿Eliminar usuario?") # Requiere Streamlit 1.34.0+
            def confirm_delete_dialog():
                st.warning(f"Estás a punto de borrar a **{u_to_del}**. Esta acción eliminará permanentemente todos los chats, la base de datos y la memoria vectorial.")
                st.write("¿Estás seguro?")
                
                c1, c2 = st.columns(2)
                if c1.button("SÍ, BORRAR TODO", type="primary", use_container_width=True):
                    # 1. Liberar al usuario actual si es el que se va a borrar
                    if st.session_state.current_user == u_to_del:
                        if st.session_state.memory_manager:
                            st.session_state.memory_manager.close_connections()
                        
                        # Seteamos a None para que Streamlit no intente acceder más
                        st.session_state.current_user = None
                        st.session_state.chatbot = None
                        st.session_state.memory_manager = None

                    # 2. MUY IMPORTANTE: Quitar la instancia del ChatbotManager
                    # El chatbot tiene una conexión abierta a la DB de LangGraph
                    ChatbotManager.remove_chatbot(u_to_del)

                    # 3. Darle tiempo al Sistema Operativo para desbloquear los archivos .bin y .sqlite3
                    import time
                    import gc
                    gc.collect() # Una última limpieza global
                    time.sleep(1.0) # Subimos a 1 segundo por seguridad en Windows

                    # 4. Intentar el borrado
                    if UserManager.delete_user_completely(u_to_del):
                        st.success(f"Usuario {u_to_del} eliminado.")
                        st.session_state.user_to_delete = None
                        st.rerun()
                    else:
                        st.error(f"Error: El archivo .bin sigue bloqueado por Windows.")
                
                if c2.button("Cancelar", use_container_width=True):
                    st.session_state.user_to_delete = None
                    st.rerun()

            confirm_delete_dialog()

    else:
        st.sidebar.info("No hay usuarios creados")
    
    # Crear nuevo usuario
    with st.sidebar.expander("Crear nuevo usuario", expanded=not existing_users):
        new_user_id = st.text_input(
            "ID de usuario:",
            placeholder="usuario123",
            help="Solo letras, números, - y _",
            key="new_user_input"
        )
        if st.button("Crear Usuario", type="primary", key="create_user_btn"):
            if not new_user_id:
                st.error("Ingresa un ID de usuario")
            elif not validate_user_id(new_user_id):
                st.error("ID inválido. Solo letras, números, - y _")
            elif UserManager.user_exists(new_user_id):
                st.error("El usuario ya existe")
            else:
                if UserManager.create_user(new_user_id):
                    st.session_state.current_user = new_user_id
                    st.session_state.chatbot = ChatbotManager.get_chatbot(new_user_id)
                    st.session_state.memory_manager = ModernMemoryManager(new_user_id)
                    st.session_state.current_chat = None
                    st.session_state.chat_history = []
                    st.success(f"Usuario '{new_user_id}' creado")
                    st.rerun()
                else:
                    st.error("Error creando usuario")

def chat_history_sidebar():
    """Sidebar estilo ChatGPT con historial de chats"""
    if not st.session_state.current_user:
        return
    
    st.sidebar.header("💬 Chats")
    memory_manager = st.session_state.memory_manager
    
    # Botón para nuevo chat
    if st.sidebar.button("➕ Nuevo Chat", type="primary", use_container_width=True):
        # Crear nuevo chat vacío
        new_chat_id = memory_manager.create_new_chat()
        st.session_state.current_chat = new_chat_id
        st.session_state.chat_history = []
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Obtener historial de chats
    chats = memory_manager.get_user_chats()
    if chats:
        st.sidebar.subheader("Historial")
        for chat in chats:
            chat_id = chat['chat_id']
            title = chat['title']
            message_count = chat.get('message_count', 0)
            updated_at = format_timestamp(chat['updated_at'])
            
            # Contenedor para cada chat
            chat_container = st.sidebar.container()
            with chat_container:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Botón principal del chat
                    is_active = st.session_state.current_chat == chat_id
                    # Crear argumentos del botón dinámicamente
                    button_args = {
                        "label": f"💬 {truncate_text(title, 25)}",
                        "key": f"chat_{chat_id}",
                        "help": f"Mensajes: {message_count} | Actualizado: {updated_at}",
                        "use_container_width": True
                    }
                    # Solo agregar type si es activo
                    if is_active:
                        button_args["type"] = "secondary"
                    if st.button(**button_args):
                        if st.session_state.current_chat != chat_id:
                            st.session_state.current_chat = chat_id
                            # Cargar historial del chat
                            st.session_state.chat_history = st.session_state.chatbot.get_conversation_history(chat_id)
                            st.rerun()
                with col2:
                    # Botón de eliminar
                    if st.button(
                        "🗑️",
                        key=f"delete_{chat_id}",
                        help="Eliminar chat"
                    ):
                        if memory_manager.delete_chat(chat_id):
                            # También eliminar de LangGraph si existe chatbot
                            if st.session_state.chatbot:
                                st.session_state.chatbot.delete_chat_from_langgraph(chat_id)
                            # Si eliminamos el chat activo, deseleccionar
                            if st.session_state.current_chat == chat_id:
                                st.session_state.current_chat = None
                                st.session_state.chat_history = []
                            st.rerun()
        
        # Información adicional
        st.sidebar.markdown(f"**Total de chats:** {len(chats)}")
    else:
        st.sidebar.info("No hay chats todavía.\nHaz clic en 'Nuevo Chat' para comenzar.")

def main_chat_interface():
    """Interfaz principal de chat estilo ChatGPT"""
    if not st.session_state.current_user:
        st.title(PAGE_TITLE)
        st.info("👈 Selecciona o crea un usuario en la barra lateral para comenzar")
        return
    
    chatbot = st.session_state.chatbot
    if not chatbot:
        st.error("Error inicializando chatbot")
        return
    
    # Si no hay chat seleccionado, mostrar pantalla de bienvenida
    if not st.session_state.current_chat:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.title("🤖 Asistente IA")
            st.markdown(f"**Hola, {st.session_state.current_user}!**")
            st.markdown("¿En qué puedo ayudarte hoy?")
            # Sugerencias de inicio
            st.markdown("### Puedes preguntarme sobre:")
            st.markdown("""
            - 💼 **Trabajo y proyectos**
            - 📚 **Aprender algo nuevo**
            - 🤔 **Resolver problemas**
            - 💡 **Ideas creativas**
            - 📋 **Planificación y tareas**
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Input para comenzar nueva conversación
        user_input = st.chat_input("Comienza una nueva conversación...")
        if user_input:
            # Crear nuevo chat con el primer mensaje
            memory_manager = st.session_state.memory_manager
            new_chat_id = memory_manager.create_new_chat(user_input)
            st.session_state.current_chat = new_chat_id
            # Procesar el primer mensaje
            process_user_message(user_input)
        return
    
    # Mostrar chat activo
    current_chat_info = st.session_state.memory_manager.get_chat_info(st.session_state.current_chat)
    if not current_chat_info:
        st.error("Chat no encontrado")
        return
    
    # Header del chat
    st.title(f"💬 {current_chat_info['title']}")
    st.caption(f"Usuario: {st.session_state.current_user}")
    
    # Cargar historial si no está cargado
    if not st.session_state.chat_history:
        st.session_state.chat_history = chatbot.get_conversation_history(st.session_state.current_chat)
    
    # Mostrar historial de conversación
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                timestamp = format_timestamp(message.get('timestamp', ''))
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                        if timestamp:
                            st.caption(f"📅 {timestamp}")
                else:
                    with st.chat_message("assistant"):
                        st.write(message['content'])
                        if timestamp:
                            st.caption(f"📅 {timestamp}")
        else:
            st.info("Comienza la conversación escribiendo un mensaje.")
    
    # Input del usuario
    user_input = st.chat_input("Escribe tu mensaje aquí...")
    if user_input:
        process_user_message(user_input)

def process_user_message(user_input: str):
    """Procesa un mensaje del usuario"""
    # Mostrar mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.write(user_input)
        st.caption(f"📅 {format_timestamp(datetime.now().isoformat())}")
    
    # Obtener respuesta del chatbot
    with st.spinner("Pensando..."):
        response = st.session_state.chatbot.chat(user_input, st.session_state.current_chat)
    
    if response['success']:
        # Mostrar respuesta del asistente
        with st.chat_message("assistant"):
            st.write(response['response'])
            caption_parts = [f"📅 {format_timestamp(datetime.now().isoformat())}"]
            if response.get('memories_used', 0) > 0:
                caption_parts.append(f"🧠 {response['memories_used']} memorias")
            if response.get('context_optimized'):
                caption_parts.append("⚡ Optimizado")
            st.caption(" | ".join(caption_parts))
        
        # Actualizar metadatos del chat
        st.session_state.memory_manager.update_chat_metadata(
            st.session_state.current_chat,
            increment_messages=True
        )
        
        # Recargar historial
        st.session_state.chat_history = st.session_state.chatbot.get_conversation_history(
            st.session_state.current_chat
        )
        st.rerun()
    else:
        st.error(f"Error: {response['error']}")

def show_memory_interface(container=st):
    """Interfaz moderna para mostrar memorias vectoriales"""
    container.subheader("🧠 Memoria Vectorial")
    if container.button("Cerrar", key="close_memories"):
        st.session_state.show_memories = False
        st.rerun()
    if not st.session_state.memory_manager:
        container.error("No hay gestor de memoria disponible")
        return
    
    memories = st.session_state.memory_manager.get_all_vector_memories()
    if not memories:
        container.info("No hay memorias guardadas todavía. El sistema guardará automáticamente información importante de tus conversaciones.")
        return
    
    # Estadísticas de memorias
    col1, col2, col3 = container.columns(3)
    with col1:
        st.metric("Total Memorias", len(memories))
    with col2:
        categories = [mem['metadata'].get('category', 'sin_categoria') for mem in memories]
        unique_categories = len(set(categories))
        st.metric("Categorías", unique_categories)
    with col3:
        high_importance = sum(1 for mem in memories if mem['metadata'].get('importance', 0) >= 4)
        st.metric("Alta Importancia", high_importance)
    
    # Filtros
    categories = list(set(mem['metadata'].get('category', 'sin_categoria') for mem in memories))
    selected_category = container.selectbox(
        "Filtrar por categoría:",
        ["Todas"] + sorted(categories)
    )
    
    # Filtrar memorias
    filtered_memories = memories
    if selected_category != "Todas":
        filtered_memories = [
            mem for mem in memories
            if mem['metadata'].get('category') == selected_category
        ]
    
    # Ordenar por importancia (si existe) y fecha
    filtered_memories.sort(
        key=lambda x: (
            x['metadata'].get('importance', 0),
            x['metadata'].get('timestamp', '')
        ),
        reverse=True
    )
    
    # Mostrar memorias
    container.write(f"Mostrando {len(filtered_memories)} de {len(memories)} memorias")
    for memory in filtered_memories:
        category = memory['metadata'].get('category', 'sin_categoria')
        timestamp = memory['metadata'].get('timestamp', '')
        importance = memory['metadata'].get('importance', 0)
        
        # Crear título con iconos y metadatos
        title_parts = [get_memory_category_icon(category)]
        title_parts.append(truncate_text(memory['content'], 60))
        if importance > 0:
            title_parts.append(f"({'⭐' * importance})")
        title = " ".join(title_parts)
        
        with container.expander(title, expanded=False):
            st.write(memory['content'])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"**Categoría:** {category}")
            with col2:
                if importance > 0:
                    st.caption(f"**Importancia:** {'⭐' * importance}")
            with col3:
                st.caption(f"**Fecha:** {format_timestamp(timestamp)}")

def main():
    """Función principal de la aplicación"""
    init_session_state()
    
    # Sidebar
    user_selection_sidebar()
    
    if st.session_state.current_user:
        # Historial de chats estilo ChatGPT
        chat_history_sidebar()
        
        # Información del usuario actual
        st.sidebar.markdown("---")
        st.sidebar.info(f"**Usuario:** {st.session_state.current_user}")
        
        # Botón de memorias globales
        if st.sidebar.button("🧠 Ver Todas las Memorias", use_container_width=True):
            st.session_state.show_memories = True
    
    # Interfaz principal con layout dinámico
    if st.session_state.show_memories:
        chat_col, mem_col = st.columns([3, 2])
        with chat_col:
            main_chat_interface()
        with mem_col:
            show_memory_interface(container=mem_col)
    else:
        main_chat_interface()

if __name__ == "__main__":
    main()