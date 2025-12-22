#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="Tema 3\\chroma_db"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"

resultados = retriever.invoke(consulta)

print("Top documentos mas similares a la consulta: \n")

for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")