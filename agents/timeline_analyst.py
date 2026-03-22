import json
import base64
from pathlib import Path
from datetime import datetime
import openai

DATA_DIR = Path("data")
JSON_DIR  = DATA_DIR / "json"

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_frame(image_path: str, software_name: str, language: str = "es") -> dict:
    image_data = encode_image(image_path)
    lang_instruction = "en Spanish" if language == "es" else "in English"

    def ask(question: str) -> str:
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
                            "text": question
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()

    screen   = ask(f"What screen or window is shown? Answer {lang_instruction} in one sentence.")
    elements = ask(f"List all buttons, fields, menus and labels visible. Answer {lang_instruction}.")
    cursor   = ask(f"Where is the mouse cursor? Describe its position {lang_instruction}.")
    ui_state = ask(f"What is the user doing or about to do? Answer {lang_instruction}.")

    return {
        "screen_name": screen,
        "visible_text": elements,
        "cursor_position": cursor,
        "ui_state": ui_state
    }


def build_timeline(frames: list, meta: dict) -> list:
    software_name = meta.get("software_name", "software")
    language = meta.get("language", "es")
    language_instruction = "en español" if language == "es" else "in English"

    frames_description = ""
    for i, frame in enumerate(frames):
        frames_description += f"""
Frame {i+1}:
  - Timestamp: {frame['timestamp_label']} ({frame['timestamp_seconds']}s)
  - Archivo: {frame['frame_id']}
  - Descripción visual: {json.dumps(frame['visual_analysis'], ensure_ascii=False)}
"""

    system_prompt = f"""Eres un analista experto en documentación de procedimientos de software.
Tu tarea es analizar frames de un tutorial de {software_name} y construir una timeline detallada.
Responde ÚNICAMENTE con JSON válido {language_instruction}, sin texto adicional, sin markdown."""

    user_prompt = f"""Analiza estos {len(frames)} frames del tutorial y construye una timeline detallada.

{frames_description}

Genera un JSON con esta estructura exacta:
{{
    "segments": [
        {{
            "segment_id": 1,
            "start_seconds": 0,
            "end_seconds": 2,
            "frame_ref": "frame_0001",
            "software_name": "{software_name}",
            "screen_name": "nombre de la pantalla",
            "action": "descripción detallada de la acción que ocurre",
            "visual_elements": ["elementos visibles relevantes"],
            "text_visible_on_screen": ["textos importantes visibles"],
            "user_action": "acción específica del usuario (clic, escritura, etc)",
            "user_intent": "intención o objetivo del usuario en este momento",
            "cursor_position": "posición del cursor",
            "ui_state_before": "estado de la interfaz antes de la acción",
            "ui_state_after": "estado de la interfaz después de la acción",
            "is_transition": false,
            "navigation_menu": ["opciones del menú si son visibles"]
        }}
    ]
}}

Importante:
- Agrupa frames consecutivos si muestran la misma acción
- Marca is_transition como true si hay cambio de pantalla
- Sé específico con los textos visibles en pantalla
- Describe la intención del usuario con claridad"""

    response = client.chat.completions.create(
        model="qwen3:14b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0,
        max_tokens=4096
    )

    raw = response.choices[0].message.content.strip()

    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif part.strip().startswith("{"):
                raw = part.strip()
                break

    return json.loads(raw)


def run(meta: dict) -> dict:
    frames_index_path = JSON_DIR / "frames_index.json"
    with open(frames_index_path, encoding="utf-8") as f:
        frames_index = json.load(f)

    frames = frames_index["frames"]
    software_name = meta.get("software_name", "software")
    language = meta.get("language", "es")

    print(f"Analizando {len(frames)} frames con moondream...")

    for i, frame in enumerate(frames):
        print(f"  Frame {i+1}/{len(frames)}: {frame['frame_id']}")
        frame["visual_analysis"] = describe_frame(
            frame["file_path"],
            software_name,
            language
        )

    print("Construyendo timeline con qwen3:14b...")

    timeline_data = build_timeline(frames, meta)

    timeline_raw = {
        "generated_at": datetime.now().isoformat(),
        "video_path": frames_index["video_path"],
        "software_name": software_name,
        "language": language,
        "total_segments": len(timeline_data["segments"]),
        "segments": timeline_data["segments"]
    }

    output_path = JSON_DIR / "timeline_raw.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(timeline_raw, f, indent=2, ensure_ascii=False)

    print(f"Guardado: {output_path}")
    return timeline_raw


if __name__ == "__main__":
    with open("meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    result = run(meta)
    print(f"\nOK — {result['total_segments']} segmentos en timeline")
    print("\nPrimer segmento:")
    print(json.dumps(result["segments"][0], indent=2, ensure_ascii=False))