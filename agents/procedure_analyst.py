import json
from pathlib import Path
from datetime import datetime
import openai

DATA_DIR = Path("data")
JSON_DIR  = DATA_DIR / "json"

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def build_procedure(timeline_raw: dict, meta: dict) -> dict:
    language    = meta.get("language", "es")
    software    = meta.get("software_name", "software")
    doc_title   = meta.get("document_title", "Procedimiento")

    lang_instruction = "en español" if language == "es" else "in English"

    segments_text = json.dumps(
        timeline_raw["segments"],
        indent=2,
        ensure_ascii=False
    )

    system_prompt = f"""Eres un redactor experto en manuales de usuario y documentación técnica de software.
Tu tarea es convertir una timeline de acciones en pasos de procedimiento claros y precisos.
Escribe {lang_instruction}, en tono profesional e instructivo.
Responde ÚNICAMENTE con JSON válido, sin texto adicional, sin markdown."""

    user_prompt = f"""Convierte esta timeline del software {software} en pasos de procedimiento para el documento "{doc_title}".

TIMELINE:
{segments_text}

Genera un JSON con esta estructura exacta:
{{
    "procedure_title": "{doc_title}",
    "software_name": "{software}",
    "objective": "objetivo general del procedimiento en una oración",
    "prerequisites": [
        "requisito previo 1",
        "requisito previo 2"
    ],
    "steps": [
        {{
            "step_number": 1,
            "title": "título corto del paso",
            "instruction": "instrucción completa y clara de qué debe hacer el usuario",
            "detail": "explicación adicional o contexto del paso si es necesario",
            "ui_reference": "elemento de interfaz específico donde ocurre la acción",
            "expected_result": "qué debe ver o pasar después de ejecutar este paso",
            "screenshot_ref": "frame_0001",
            "is_critical": true
        }}
    ],
    "total_steps": 0,
    "estimated_time_minutes": 0,
    "notes": "observaciones generales del procedimiento"
}}

Reglas importantes:
- Cada paso debe ser una acción específica y ejecutable
- Las instrucciones deben comenzar con un verbo en imperativo (Haga clic, Ingrese, Seleccione, etc.)
- ui_reference debe nombrar exactamente el elemento de interfaz (botón, campo, menú)
- is_critical es true si el paso es obligatorio para completar el procedimiento
- Agrupa acciones muy pequeñas en un solo paso cuando tenga sentido
- El campo detail puede quedar vacío si el paso es suficientemente claro"""

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
    # Cargar timeline_raw.json
    timeline_path = JSON_DIR / "timeline_raw.json"
    with open(timeline_path, encoding="utf-8") as f:
        timeline_raw = json.load(f)

    print(f"Generando procedimiento desde {timeline_raw['total_segments']} segmentos...")

    procedure = build_procedure(timeline_raw, meta)

    # Asegurar total_steps correcto
    procedure["total_steps"] = len(procedure["steps"])

    # Construir output final
    procedure_steps = {
        "generated_at": datetime.now().isoformat(),
        "video_path":   meta.get("video_path", ""),
        "software_name": meta.get("software_name", ""),
        "language":      meta.get("language", "es"),
        "procedure":     procedure
    }

    output_path = JSON_DIR / "procedure_steps.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(procedure_steps, f, indent=2, ensure_ascii=False)

    print(f"Guardado: {output_path}")
    print(f"Pasos generados: {procedure['total_steps']}")
    return procedure_steps


if __name__ == "__main__":
    with open("meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    result = run(meta)
    print("\nPrimer paso:")
    print(json.dumps(result["procedure"]["steps"][0], indent=2, ensure_ascii=False))