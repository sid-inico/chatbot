from config import HISTORY_LOG_PATH

def guardar_intercambio(user_input, response):
    with open(HISTORY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("Tú: " + user_input + "\n")
        f.write("Nico: " + response + "\n\n")