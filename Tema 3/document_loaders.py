from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

#loaderPDF = PyPDFLoader("Tema 3\\curriculum_JorgeHerraizSoler.pdf")

# pagesPDF = loaderPDF.load()

#for i, page in enumerate(pagesPDF):
#    print(f"=== Pagina PDF {i+1} ===")
#    print(f"Contenido: {page.page_content}")
#    print(f"Metadatos: {page.metadata}")
    
    
loaderWeb = WebBaseLoader("https://jherraizsoler.github.io/portfolio/")

pagesWeb = loaderWeb.load()

print(pagesWeb)