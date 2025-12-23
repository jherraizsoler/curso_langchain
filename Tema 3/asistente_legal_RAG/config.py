# Configuración de modelos
EMBEDING_MODEL = "text-embedding-3-large"

QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"

# Configuración del vector stores
CHROMA_DB_PATH = "Tema 3\\chroma_db"

# Configuración del retriever
SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K =  2
