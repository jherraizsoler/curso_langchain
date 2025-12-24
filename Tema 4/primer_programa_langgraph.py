from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Definir el esquema del estado 
class State(TypedDict):
    texto_original: str
    texto_mayus: str
    longitud: int
    
# 2. Crear el grafo de estado
graph = StateGraph(State)

# 3. Definir las funciones de los nodos
def poner_mayusculas(state):
    texto = state['texto_original']
    return {"texto_mayus": texto.upper()}

def contar_caracteres(state):
    texto = state['texto_mayus']
    return {"longitud":len(texto)}

# 4. Añadir los nodos al grafo
graph.add_node("Mayus", poner_mayusculas)
graph.add_node("Contar", contar_caracteres)

# 5. Conectar los nodos en secuencia
graph.add_edge(START, "Mayus")
graph.add_edge("Mayus", "Contar")
graph.add_edge("Contar", END)


# 6. Compilar el grafo
compiled_graph = graph.compile()

# 7. Invocar el grafo con un estado inicial
estado_inicial = {"texto_original": "Hola mundo"}
resultado = compiled_graph.invoke(estado_inicial)
print(resultado)