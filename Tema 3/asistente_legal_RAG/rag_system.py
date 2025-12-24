from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever 
import  streamlit as st
from config import *
from prompts import *

@st.cache_resource
def initialize_rag_system():
    
    # Vector Store
    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDING_MODEL),
        persist_directory=CHROMA_DB_PATH
    )
    
    # Modelos
    llm_queries = ChatOpenAI(model=QUERY_MODEL, temperature=0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0)
    
    # Retriever MMR (Maximal Marginal Relevance)
    base_retriever = vector_store.as_retriever(
        search_type= SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )
    
    # Retriever adicional con similarity para comparar
    similarity_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs= {"k": SEARCH_K}
    )
    
    
    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)
    
    
    # MultiQueryRetriever con prompt personalizado
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )
    
    # Ensembre Retriever que combinar MMR y Similarity
    if ENABLE_HYBRID_SEARCH:
        ensenble_retriever = EnsembleRetriever(
            retrievers = [mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3], # Mayor peso a MMR
            similarity_threshold = SIMILARITY_THRESHOLD
        )
        final_retriever = ensenble_retriever
    else:
        final_retriever = mmr_multi_retriever
    
    
    prompt = PromptTemplate.from_template(RAG_TEMPLATE)
    
    # Función para formatear y preprocesar los documentos recuperados
    def format_docs(docs):
        formatted  = []
        
        for i, doc in enumerate(docs,1):
            header = f"[Fragmento {i}]"
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Página: {doc.metadata['page']}"
            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")
        
        return "\n\n".join(formatted)
    
    rag_chain=(
        {
            "context": final_retriever | format_docs,
            "question": RunnablePassthrough()
        } 
        | prompt 
        | llm_generation 
        | StrOutputParser()
    )
    return rag_chain, mmr_multi_retriever


def query_rag(question):
    try:
        rag_chain, retriever  = initialize_rag_system()
        
        # Obtener respuesta
        response = rag_chain.invoke(question)
        
        # Obtener documentos para mostrarlos
        docs = retriever.invoke(question)
        
        # Formatear los documentos para mostrar
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K],1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get('source', 'No especificada').split("\\")[-1],
                "pagina": doc.metadata.get('page','No especificada')
            }
            docs_info.append(doc_info)
            
        return response, docs_info
    except Exception as e:
        error_msg = f"Error al procesar la consulta: {str(e)}"
        return error_msg, []
    
def get_retriever_info():
    """Obtiene información sobre la configuración del retriever"""
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }