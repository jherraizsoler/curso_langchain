from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import GmailToolkit
import os

# Configurar el directorio de trabajo
original_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configurar el toolkit de Gmail
gmail_toolkit = GmailToolkit()
tools = gmail_toolkit.get_tools()

# Configurar modelo del agente que soporte tool calling
model = init_chat_model("openai:gpt-4o", temperature=0)

# Prompt de agente que define su comportamiento
prompt  = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente de email profesional. Para procesar emails sigue EXACTAMENTE estos pasos:

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
    
    Si encuentras errores, explica qué información falta y por qué."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


#  Crear agente
agent = create_tool_calling_agent(model, tools, prompt)

# Crear executor del agente
agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10  # Limitar iteraciones para evitar bucles *loops
    )

def process_latest_email():
    try:
        response = agent_executor.invoke({
            "input": "Procesa el email más reciente en la bandeja de entrada y genera un borrador de respuesta profesional."
        })
        return response['output']
    except Exception as e:
        print(f"Error al procesar el email: {str(e)}")
        return f"Error {str(e)}"
    
# Ejecutar
if __name__ == '__main__':
    result = process_latest_email()
    print("\n" + "="*50)
    print("RESULTADO:")
    print("="*50)
    print(result)