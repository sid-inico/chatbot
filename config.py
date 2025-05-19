# Modelos
LLM_MODEL = "llama3.1"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Rutas
VECTOR_DB_PATH = "./prueba"
VECTOR_DB_COLLECTION = "discapacidad_prueba"
FT_DATA_FILE = "json/train.jsonl"
CONTEXT_FILE = "json/context.jsonl"

# Umbrales
EMBEDDING_THRESHOLD = 0.85
MIN_CONFIDENCE = 0.7
FEEDBACK_PROBABILITY = 0.2

# Mensajes
BOT_NAME = "AI-Nico"
WELCOME_MESSAGE = "¡Hola! Soy tu asistente sobre discapacidad. ¿En qué puedo ayudarte?"
GOODBYE_MESSAGE = "¡Hasta luego! Recuerda que estoy aquí para ayudarte."
ERROR_MESSAGE = "Lo siento, ocurrió un error. Por favor intenta nuevamente."
OFF_TOPIC_RESPONSE = "Lo siento, solo puedo ayudar con temas de discapacidad."
SYSTEM_PROMPT = "Eres un asistente experto en discapacidad y legislación española..."

# Comandos
EXIT_COMMANDS = ["salir", "adios", "exit", "quit"]

# Versión
VERSION = "1.0.0"

# Logs
FEEDBACK_LOG_PATH = "logs/respuestas_mejorables.log"
HISTORY_LOG_PATH = "logs/chat.log"