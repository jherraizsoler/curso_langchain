from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Configuración del modelo
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Configuración del prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
# Configuración de la cadena
chain = prompt | llm

# Configuración de la memoria
store = {}

# Función para obtener el historial de mensajes
def get_session_history(session_id):
    """          Docstring
    Returns the history of messages for the given session_id.

    The history is stored in a dictionary where the key is the session_id
    and the value is a list of HumanMessage and AIMessage objects.

    If the session_id is not found in the dictionary, an empty list is
    created and returned.

    :param session_id: the session_id to retrieve the history for
    :type session_id: str
    :return: the history of messages for the given session_id
    :rtype: List[Union[HumanMessage, AIMessage]]
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Cadena con memoria automática por sesión
chain_with_memory = RunnableWithMessageHistory(
    chain, 
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
        
)

print("Chat en terminal (escribe 'salir' para terminar)\n")

session_id = "session_terminal"


# Bucle de conversación
while True:
    try:
        user_input = input("Tú: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nHasta luego!")
        break

    if not user_input:
        continue
    if user_input.lower() in {"salir", "exit", "quit"}:
        print("Hasta luego!")
        break

    respuesta = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    print("Asistente:", respuesta.content)