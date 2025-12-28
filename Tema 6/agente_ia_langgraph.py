from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import GmailToolkit
import os
from langchain_core.tools import tool
import base64
from email.mime.text import MIMEText



# Configurar el directorio de trabajo
original_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

@tool
def create_gmail_reply_draft(message: str, to: str, subject: str, thread_id: str, in_reply_to: str = None) -> str:
    """
    Crea un borrador de respuesta en Gmail que SÍ funciona como respuesta en el hilo.
    Usa esta herramienta cuando quieras crear una RESPUESTA a un email existente.

    Args:
        message: Contenido del mensaje de respuesta
        to: Email del destinatario  
        subject: Asunto (debería empezar con Re:)
        thread_id: ID del hilo para que sea una respuesta (OBLIGATORIO)
        in_reply_to: Message-ID del email original (opcional)
    """
    try:
        # Obtener service de Gmail del toolkit
        gmail_toolkit = GmailToolkit()
        service = gmail_toolkit.api_resource

        # Crear mensaje MIME con headers de respuesta
        mime_message = MIMEText(message, 'plain', 'utf-8')
        mime_message['To'] = to
        mime_message['Subject'] = subject

        # Headers para que sea una respuesta
        if in_reply_to:
            mime_message['In-Reply-To'] = in_reply_to
            mime_message['References'] = in_reply_to

        # Codificar mensaje
        encoded_message = base64.urlsafe_b64encode(
            mime_message.as_bytes()
        ).decode('utf-8')

        # Crear borrador CON thread_id
        draft_body = {
            'message': {
                'raw': encoded_message,
                'threadId': thread_id  # ¡Esto es lo crítico!
            }
        }

        draft = service.users().drafts().create(
            userId='me',
            body=draft_body
        ).execute()

        return f"Borrador de RESPUESTA creado exitosamente. Draft ID: {draft['id']}, Thread ID: {thread_id}"

    except Exception as e:
        return f"Error creando borrador de respuesta: {str(e)}"

# Configurar el toolkit de Gmail
gmail_toolkit = GmailToolkit()
tools = gmail_toolkit.get_tools()
tools.append(create_gmail_reply_draft)

# Configurar modelo del agente que soporte tool calling
model = init_chat_model("openai:gpt-4o", temperature=0)

# Prompt de agente que define su comportamiento
system_prompt = """Eres un asistente de email profesional. Para procesar emails sigue EXACTAMENTE estos pasos:

    1. PRIMERO: Usa 'search_gmail' con query 'in:inbox' para obtener la lista de mensajes en la bandeja de entrada.
    
    2. SEGUNDO: De la lista obtenida, identifica el message_id del email más reciente (el primer resultado).
    
    3. TERCERO: Usa 'get_gmail_message' con el message_id real obtenido en el paso anterior para obtener el contenido completo.
    
    4. CUARTO: Analiza el email y EXTRAE esta información crítica:
       - Thread ID (busca "Thread ID:" en el contenido)
       - Remitente original (busca "From:" y extrae el email)
       - Asunto original (busca "Subject:")
       - Contenido principal del mensaje
    
    5. QUINTO: Genera una respuesta profesional y apropiada en español.
    
    6. SEXTO: Usa 'create_gmail_draft' para crear un borrador de RESPUESTA (no email nuevo) con:
       - "message": tu respuesta generada
       - "subject": "Re: [asunto original]" (si no empieza ya con "Re:")
       - "to": email del remitente original
       - "thread_id": el Thread ID extraído del paso 4 (MUY IMPORTANTE para que sea una respuesta)

    CRÍTICO PARA RESPUESTAS:
    - SIEMPRE incluye "thread_id" en create_gmail_draft para que sea una respuesta, no un email nuevo
    - El "to" debe ser el email del remitente original
    - El "subject" debe empezar con "Re:" si no lo tiene ya

    IMPORTANTE: 
    - NUNCA uses message_id hardcodeados como '1' o '2' 
    - SIEMPRE obtén los IDs reales de los mensajes primero
    - Sin thread_id, el borrador será un email nuevo, no una respuesta
    - Si no encuentras thread_id, informa el problema pero intenta crear el borrador igual
    
    Si encuentras errores, explica qué información falta y por qué."""

# Crear agente
memory = MemorySaver()
agent_executor = create_react_agent(
    model,
    tools,
    prompt=system_prompt,
    checkpointer=memory,
)

def process_latest_email():
    try:
        # Configurar el estado inicial
        config = {"configurable": {"thread_id": "gmail-processing"}}
        
        response = agent_executor.invoke(
            {"messages": [("human", "Procesa el email más reciente en la bandeja de entrada y genera un borrador de respuesta profesional.")]},
             config=config
        )
        return response['messages'][-1].content
    except Exception as e:
        print(f"Error al procesar email: {str(e)}")
        return f"Error {str(e)}"
    
# Ejecutar
if __name__ == "__main__":
    result = process_latest_email()
    print("\n" + "="*50)
    print("RESULTADO:")
    print("="*50)
    print(result)