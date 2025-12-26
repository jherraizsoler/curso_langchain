from langgraph.graph import MessagesState, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage,AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Configuración del modelo
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

workflow = StateGraph(state_schema=MessagesState)

def chatbot_node(state):
    """Nodo que procesa mensajes y genera respuestas."""
    system_prompt = "Eres un asistente amigable que recuerda conversaciones previstas."
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return  {"messages": [response]}


workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")

# Compilar el grafo
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def chat(message, thread_id="sesion_terminal"):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=message)]}, config)
    return result["messages"][-1].content

if __name__ == "__main__":
                        
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
        
        respuesta = chat(user_input, session_id)
        print("Asistente:", respuesta)