from config import FEEDBACK_LOG_PATH

def pedir_feedback(user_input, response):
    print()
    feedback = input("¿Esta respuesta fue útil? (sí/no): ").strip().lower()
    if feedback == "no":
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write("Consulta: " + user_input + "\n")
            f.write("Respuesta: " + response + "\n\n")