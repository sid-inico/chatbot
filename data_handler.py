import json
import logging
from typing import List, Dict, Tuple

class FineTuningDataHandler:
    # Carga los datos de entrenamiento desde JSONL
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        data = []
        logger = logging.getLogger(__name__)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error en línea: {line}. Error: {e}")
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {file_path}")
        return data

    # Prepara pares pregunta-respuesta para embeddings
    @staticmethod
    def prepare_training_data(data: List[Dict]) -> List[Tuple[str, str]]:
        return [
            (item["messages"][1]["content"], item["messages"][2]["content"])
            for item in data if len(item["messages"]) >= 3
        ]