import json
import base64
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents import ocr_validator
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
    lang_instruction = "in Spanish" if language == "es" else "in English"

    prompt = f"""Analyze this screenshot of {software_name} carefully.
Respond ONLY with a JSON object {lang_instruction}, no markdown, no explanation:
{{
    "screen_name": "exact name of this screen or dialog (max 5 words)",
    "visible_text": ["every button label", "every field label", "every menu item you can read"],
    "cursor_position": "what UI element is the cursor on, or null if not visible",
    "ui_state": "one sentence starting with a verb describing what the user is doing"
}}
Only include UI elements you can actually see. Do not invent elements."""

    response = client.chat.completions.create(
        model="llava:latest",
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
                        "text": prompt
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=400
    )

    raw = response.choices[0].message.content.strip()

    # Limpiar markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif part.startswith("{"):
                raw = part
                break

    # Extraer JSON por posición de llaves
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    # Filtrar caracteres extraños
    cleaned = "".join(c for c in raw if ord(c) < 1000 or c in "áéíóúüñÁÉÍÓÚÜÑ¿¡")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "screen_name":     "No determinado",
            "visible_text":    [],
            "cursor_position": None,
            "ui_state":        "No determinado"
        }

def validate_visual_analysis(frames: list, meta: dict) -> list:
    """
    Usa qwen3 para validar y filtrar alucinaciones en las
    descripciones visuales de llava, comparando consistencia
    entre frames consecutivos.
    """
    software = meta.get("software_name", "software")
    language = meta.get("language", "es")
    lang_instruction = "en español" if language == "es" else "in English"

    analyses = []
    for f in frames:
        analyses.append({
            "frame_id": f["frame_id"],
            "timestamp": f["timestamp_label"],
            "analysis": f["visual_analysis"]
        })

    system_prompt = f"""Eres un validador experto de descripciones visuales de interfaces de software.
Tu tarea es revisar descripciones de frames consecutivos de un tutorial de {software} y:
1. Eliminar elementos de UI que probablemente sean alucinaciones (aparecen en un solo frame sin consistencia)
2. Corregir nombres de elementos que no tienen sentido en el contexto del software
3. Mantener solo elementos que sean consistentes o claramente visibles
Responde ÚNICAMENTE con JSON válido {lang_instruction}, sin texto adicional."""

    user_prompt = f"""Valida estas descripciones visuales de {len(frames)} frames de {software}.

{json.dumps(analyses, indent=2, ensure_ascii=False)}

Para cada frame, devuelve una versión limpia y validada. Elimina elementos inconsistentes o que parezcan alucinaciones.

Responde SOLO con este JSON:
{{
    "validated_frames": [
        {{
            "frame_id": "frame_0001",
            "screen_name": "nombre validado de la pantalla",
            "visible_text": ["solo textos realmente visibles"],
            "cursor_position": "posición validada",
            "ui_state": "estado validado",
            "confidence": "high/medium/low"
        }}
    ],
    "detected_software_language": "en/es/pt/fr",
    "common_ui_elements": ["elementos que aparecen en múltiples frames"]
}}"""

    response = client.chat.completions.create(
        model="qwen3:14b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0,
        max_tokens=4096,
        extra_body={"think": False}
    )

    raw = response.choices[0].message.content.strip()

    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif part.startswith("{"):
                raw = part
                break

    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    validation = json.loads(raw)

    # Actualizar frames con análisis validado
    validated_map = {
        v["frame_id"]: v
        for v in validation["validated_frames"]
    }

    for frame in frames:
        if frame["frame_id"] in validated_map:
            frame["visual_analysis"] = validated_map[frame["frame_id"]]
            frame["confidence"] = validated_map[frame["frame_id"]].get("confidence", "medium")

    # Guardar idioma detectado para el orquestador
    detected_lang = validation.get("detected_software_language", "en")
    meta["language_ui"] = detected_lang

    print(f"  Idioma del software detectado: {detected_lang}")
    print(f"  Elementos comunes: {', '.join(validation.get('common_ui_elements', []))}")

    return frames

def build_timeline(frames: list, meta: dict) -> dict:
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
Responde ÚNICAMENTE con JSON válido {language_instruction}, sin texto adicional, sin markdown, sin explicaciones."""

    user_prompt = f"""Analiza estos {len(frames)} frames del tutorial y construye una timeline detallada.

{frames_description}

Responde SOLO con este JSON, sin ningún texto antes ni después:
{{
    "segments": [
        {{
            "segment_id": 1,
            "start_seconds": 0,
            "end_seconds": 2,
            "frame_ref": "frame_0001",
            "software_name": "{software_name}",
            "screen_name": "nombre exacto de la pantalla",
            "action": "descripción detallada de la acción",
            "visual_elements": ["elementos visibles"],
            "text_visible_on_screen": ["textos visibles"],
            "user_action": "acción específica del usuario",
            "user_intent": "intención del usuario",
            "cursor_position": "posición del cursor",
            "ui_state_before": "estado antes",
            "ui_state_after": "estado después",
            "is_transition": false,
            "navigation_menu": ["opciones del menú"]
        }}
    ]
}}"""

    response = client.chat.completions.create(
        model="qwen3:14b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0,
        max_tokens=4096,
        extra_body={"think": False}
    )

    raw = response.choices[0].message.content.strip()

    # Limpiar thinking tags
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    # Limpiar markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif part.startswith("{"):
                raw = part
                break

    # Extraer JSON si hay texto antes o después
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error parseando JSON: {e}")
        print(f"Raw response: {raw[:500]}")
        raise

def run(meta: dict, frames_index_override: dict = None) -> dict:
    if frames_index_override:
        frames_index = frames_index_override
    else:
        frames_index_path = JSON_DIR / "frames_index.json"
        with open(frames_index_path, encoding="utf-8") as f:
            frames_index = json.load(f)

    frames = frames_index["frames"]
    software_name = meta.get("software_name", "software")
    language = meta.get("language", "es")

    print(f"Analizando {len(frames)} frames con llava...")

    for i, frame in enumerate(frames):
        print(f"  Frame {i+1}/{len(frames)}: {frame['frame_id']}")
        frame["visual_analysis"] = describe_frame(
            frame["file_path"],
            software_name,
            language
        )

    print(f"Analizando {len(frames)} frames con llava...")

    for i, frame in enumerate(frames):
        print(f"  Frame {i+1}/{len(frames)}: {frame['frame_id']}")
        frame["visual_analysis"] = describe_frame(
            frame["file_path"],
            software_name,
            language
        )

    # Validación OCR con moondream
    print("Validando OCR con moondream...")
    frames = ocr_validator.run(frames, software_name)

    # Validación semántica con qwen3
    print("Validando análisis visual con qwen3...")
    frames = validate_visual_analysis(frames, meta)

    # NUEVO — validación automática
    print("Validando análisis visual con qwen3...")
    frames = validate_visual_analysis(frames, meta)

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