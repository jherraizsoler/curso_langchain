# 📄 AI CV Analyzer & Evaluator

Este sistema es una herramienta avanzada de **Recursos Humanos** que utiliza Inteligencia Artificial para automatizar la evaluación de currículums. Está diseñado para analizar archivos PDF, comparar el perfil del candidato con una descripción de puesto específica y generar informes detallados con métricas de ajuste.



## 🛠️ Tecnologías Utilizadas

* **LangChain:** Framework para la orquestación de la lógica de IA y cadenas de prompts.
* **Google Gemini 1.5 Flash / OpenAI GPT-4o:** Modelos de lenguaje avanzados para el análisis semántico.
* **Streamlit:** Interfaz de usuario interactiva y profesional.
* **Pydantic:** Validación de esquemas y estructuración de la salida del modelo.
* **PyPDF2:** Extracción de texto desde archivos PDF.
* **ReportLab:** Generación dinámica de reportes en formato PDF.

---

## ⚙️ Configuración de Variables y Rutas

Antes de ejecutar la aplicación, es necesario configurar las credenciales de acceso a los modelos de lenguaje.

### 1. Variables de Entorno (API Key)
El proyecto admite tanto OpenAI como Google Gemini. Crea un archivo `.env` en la raíz del proyecto o configura tus variables de entorno:

#### **Opción A: Google Gemini (Configuración por defecto)**
Para usar Gemini, necesitas una clave de [Google AI Studio](https://aistudio.google.com/):
```python
GOOGLE_API_KEY = "tu_api_key_de_google_aqui"
```

#### **Opción B: OpenAI (Configuración por defecto)**
Para usar ChatGPT, necesitas una clave de [OpenAI Platform](https://platform.openai.com/docs/overview/):
```python
# En archivo .env o configuración de sistema
OPENAI_API_KEY = "tu_api_key_de_openai_aqui"
```
> [!IMPORTANT]
> Nota: Para alternar entre proveedores, debes modificar las líneas correspondientes en services/cv_evaluator.py.

### 2. Directorio de Resultados
Los informes generados (JSON y PDF) se guardan automáticamente en la siguiente ruta:
```python
DIRECTORIO_GUARDADO = r".\resultados_cv"
```

## 📁 Estructura del Proyecto

- **app.py**: Punto de entrada que lanza la interfaz de Streamlit.
- **models/cv_model.py**: Define el modelo de datos `AnalisisCV` (nombre, experiencia, habilidades, fortalezas, áreas de mejora y porcentaje de ajuste).
- **services/cv_evaluator.py**: Configura el modelo base (Gemini/OpenAI) y la cadena de evaluación estructurada.
- **services/pdf_processor.py**: Gestiona la extracción y limpieza de texto de los archivos PDF.
- **prompts/cv_prompts.py**: Contiene el `SISTEMA_PROMPT` y `ANALISIS_PROMPT` especializados en perfiles IT Junior y Graduate.
- **ui/streamlit_ui.py**: Lógica de la interfaz, procesamiento de archivos y generación de documentos ReportLab.

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/jherraizsoler/curso_langchain.git
cd "Tema 2/cv_analyzer"
```

### 2. Instalar dependencias

```bash
pip install streamlit langchain-openai langchain-google-genai PyPDF2 reportlab python-dotenv
```

### 3. Ejecutar la aplicación: Desde la carpeta raíz del curso, ejecuta:
```PowerShell
(venv) PS C:\Users\...\curso_langchain> streamlit run ".\Tema 2\cv_analyzer\app.py"
```
---

## 📊 Criterios de Evaluación para Perfiles Junior

El sistema evalúa a los candidatos basándose en un algoritmo de pesos configurado en el prompt:

- **Experiencia y Prácticas Relevantes (40%)**: Valora la aplicación real de tecnología en entornos de empresa.
- **Habilidades Técnicas (30%)**: Nivel de conocimiento del stack solicitado (C#, .NET, Angular, SQL Server).
- **Formación y Educación (20%)**: Títulos académicos como Grado Superior (DAM) o certificaciones.
- **Soft Skills y Potencial (10%)**: Capacidad de crecimiento, trabajo en equipo y adaptabilidad.

---

## ✨ Notas de Uso

- **Fecha de Evaluación**: El sistema utiliza la fecha actual (`{fecha_formato_cadena}`) para calcular la vigencia de la experiencia laboral.
- **Calidad del PDF**: Asegúrate de que el PDF sea legible por texto. Si el archivo es una imagen escaneada (OCR no procesado), el extractor podría no recuperar información.
