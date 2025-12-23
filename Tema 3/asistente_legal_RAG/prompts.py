# Prompt principal para el sistema RAG
RAG_TEMPLATE = """Eres un asistente legal especializado en contratos de arrendamiento.
Basándote ÚNICAMENTE en los siguientes fragmentos de contratos, responde a la pregunta del usuario.

FRAGMENTOS DE CONTRATOS:
{context}

PREGUNTA: {question}

INSTRUCCIONES:
- Proporciona una respuesta clara y directa basada en la información disponible
- Si encuentras la información exacta, cítala textualmente cuando sea relevante
- Incluye todos los detalles importantes: nombres, direcciones, importes, fechas
- Si la información está incompleta o no está disponible, indícalo claramente
- Organiza la información de manera estructurada si es necesaria
- Si hay múltiples contratos o personas mencionadas, especifica a cuál te refieres

RESPUESTA:"""

# Prompt personalizado para el MultiQueryRetriever
MULTI_QUERY_PROMPT = """Eres un experto en análisis de documentos legales especializados en contratos de arrendamiento.
Tu tarea es generar múltiples versiones de la consulta del usuario para recuperar documentos relevantes desde una base de datos vectorial.

Al generar variaciones de la consulta, considera:
- Diferentes formas de referirse a personas (nombre completo, apellidos, solo nombre)
- Sinónimos legales y términos técnicos de arrendamiento
- Variaciones en la formulación de preguntas sobre aspectos contractuales
- Términos relacionados con ubicaciones, propiedades y condiciones del contrato

Consulta original: {question}

Genera exactamente 3 versiones alternativas de esta consulta, una por línea, sin numeración ni viñetas:"""

# Prompt para análisis de relevancia de documentos
RELEVANCE_PROMPT = """Analiza si el siguiente fragmento de documento es relevante para responder la consulta del usuario.

FRAGMENTO:
{document}

CONSULTA: {question}

¿Es este fragmento relevante para responder la consulta? Responde solo con "SÍ" o "NO" y una breve justificación."""

# Prompt para extracción de entidades clave
ENTITY_EXTRACTION_PROMPT = """Extrae las entidades clave del siguiente texto de contrato de arrendamiento:

TEXTO:
{text}

Identifica y extrae:
- Nombres de personas (arrendador, arrendatario, avalistas)
- Direcciones de propiedades
- Importes monetarios
- Fechas importantes
- Duración del contrato
- Tipo de propiedad

Formato de respuesta:
PERSONAS: [lista de nombres]
DIRECCIONES: [lista de direcciones]
IMPORTES: [lista de cantidades]
FECHAS: [lista de fechas]
DURACIÓN: [periodo del contrato]
TIPO: [tipo de propiedad]"""