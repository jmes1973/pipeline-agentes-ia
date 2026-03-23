import json
import base64
from pathlib import Path
import openai

CACHE_PATH = Path("data/json/corrections_cache.json")

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_cache(software_name: str) -> dict:
    """Carga correcciones conocidas para este software."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        return cache.get(software_name, {})
    return {}


def save_cache(software_name: str, corrections: dict):
    """Guarda nuevas correcciones en el cache."""
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
    if software_name not in cache:
        cache[software_name] = {}
    cache[software_name].update(corrections)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def verify_element(image_path: str, element: str) -> bool:
    """
    Usa moondream para verificar si un elemento UI
    realmente existe en la imagen.
    """
    image_data = encode_image(image_path)

    response = client.chat.completions.create(
        model="moondream:latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f'Is there a button, label or menu item with the exact text "{element}" visible in this image? Answer only: yes or no.'
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=5
    )

    answer = response.choices[0].message.content.strip().lower()
    return "yes" in answer


def get_suspicious_elements(visible_text: list) -> list:
    """
    Identifica elementos sospechosos de ser alucinaciones.
    Criterios: términos comerciales genéricos que no suelen
    aparecer en software de gestión o educativo.
    """
    suspicious_keywords = [
        "free trial", "buy now", "purchase", "subscribe",
        "upgrade", "premium", "pricing", "checkout",
        "prueba gratuita", "comprar", "suscribir"
    ]
    suspicious = []
    for element in visible_text:
        element_lower = element.lower()
        for keyword in suspicious_keywords:
            if keyword in element_lower:
                suspicious.append(element)
                break
    return suspicious


def validate_frame(frame: dict, software_name: str) -> dict:
    """
    Valida los elementos UI de un frame usando cache
    y verificación con moondream.
    """
    cache = load_cache(software_name)
    visible_text = frame["visual_analysis"].get("visible_text", [])

    if not visible_text:
        return frame

    cleaned_text = []
    new_corrections = {}

    for element in visible_text:
        # 1. Verificar cache primero
        if element in cache:
            corrected = cache[element]
            if corrected is None:
                print(f"    Cache: eliminando '{element}'")
                new_corrections[element] = None
                continue
            else:
                print(f"    Cache: corrigiendo '{element}' → '{corrected}'")
                cleaned_text.append(corrected)
                continue

        # 2. Verificar si es sospechoso
        suspicious = get_suspicious_elements([element])
        if suspicious:
            print(f"    Verificando con moondream: '{element}'")
            exists = verify_element(frame["file_path"], element)
            if exists:
                print(f"      Confirmado: existe")
                cleaned_text.append(element)
            else:
                print(f"      Eliminado: no existe en imagen")
                new_corrections[element] = None
        else:
            cleaned_text.append(element)

    # Guardar nuevas correcciones en cache
    if new_corrections:
        save_cache(software_name, new_corrections)

    frame["visual_analysis"]["visible_text"] = cleaned_text
    return frame


def run(frames: list, software_name: str) -> list:
    """
    Valida todos los frames y retorna la lista limpia.
    """
    print(f"Validando elementos UI en {len(frames)} frames...")
    corrections_applied = 0

    for i, frame in enumerate(frames):
        original_count = len(
            frame["visual_analysis"].get("visible_text", [])
        )
        frame = validate_frame(frame, software_name)
        new_count = len(
            frame["visual_analysis"].get("visible_text", [])
        )
        diff = original_count - new_count
        if diff > 0:
            corrections_applied += diff
            print(f"  Frame {i+1}: {diff} elemento(s) corregido(s)")

    print(f"Total correcciones: {corrections_applied}")
    return frames