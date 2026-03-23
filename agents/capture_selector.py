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


def select_captures(frames_index: dict, procedure_steps: dict, meta: dict) -> dict:
    language     = meta.get("language", "es")
    software     = meta.get("software_name", "software")
    lang_instr   = "en español" if language == "es" else "in English"

    frames_summary = []
    for f in frames_index["frames"]:
        frames_summary.append({
            "frame_id":          f["frame_id"],
            "timestamp":         f["timestamp_label"],
            "timestamp_seconds": f["timestamp_seconds"],
            "file_path":         f["file_path"]
        })

    steps_summary = []
    for s in procedure_steps["procedure"]["steps"]:
        steps_summary.append({
            "step_number":   s["step_number"],
            "title":         s["title"],
            "instruction":   s["instruction"],
            "ui_reference":  s["ui_reference"],
            "screenshot_ref": s.get("screenshot_ref", "")
        })

    system_prompt = f"""Eres un experto en documentación técnica de software.
Tu tarea es seleccionar los frames más relevantes para ilustrar cada paso de un procedimiento.
Responde ÚNICAMENTE con JSON válido {lang_instr}, sin texto adicional, sin markdown."""

    user_prompt = f"""Selecciona las capturas de pantalla más adecuadas para ilustrar el procedimiento de {software}.

FRAMES DISPONIBLES:
{json.dumps(frames_summary, indent=2, ensure_ascii=False)}

PASOS DEL PROCEDIMIENTO:
{json.dumps(steps_summary, indent=2, ensure_ascii=False)}

Reglas de selección:
- Cada paso debe tener exactamente 1 captura principal
- La captura debe mostrar el momento justo ANTES o DURANTE la acción del paso
- Prefiere frames donde el cursor esté sobre el elemento de UI relevante
- Un mismo frame puede usarse para máximo 2 pasos si son consecutivos
- Si un paso no tiene frame ideal, usa el más cercano en tiempo

Responde SOLO con este JSON:
{{
    "captures": [
        {{
            "step_number": 1,
            "step_title": "título del paso",
            "frame_id": "frame_0001",
            "file_path": "ruta al archivo",
            "timestamp": "00:00:00",
            "capture_reason": "razón por la que este frame ilustra mejor este paso",
            "highlight_element": "elemento UI que debe resaltarse en la captura",
            "caption": "texto descriptivo de la captura para el documento"
        }}
    ],
    "total_captures": 0,
    "coverage_notes": "observaciones sobre la cobertura visual del procedimiento"
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

    return json.loads(raw)


def run(meta: dict) -> dict:
    # Cargar archivos necesarios
    with open(JSON_DIR / "frames_index.json", encoding="utf-8") as f:
        frames_index = json.load(f)

    with open(JSON_DIR / "procedure_steps.json", encoding="utf-8") as f:
        procedure_steps = json.load(f)

    print(f"Seleccionando capturas para {procedure_steps['procedure']['total_steps']} pasos...")

    captures = select_captures(frames_index, procedure_steps, meta)
    captures["total_captures"] = len(captures["captures"])

    captures_plan = {
        "generated_at":  datetime.now().isoformat(),
        "video_path":    meta.get("video_path", ""),
        "software_name": meta.get("software_name", ""),
        "language":      meta.get("language", "es"),
        "captures_plan": captures
    }

    output_path = JSON_DIR / "captures_plan.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captures_plan, f, indent=2, ensure_ascii=False)

    print(f"Guardado: {output_path}")
    print(f"Capturas seleccionadas: {captures['total_captures']}")
    return captures_plan


if __name__ == "__main__":
    with open("meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    result = run(meta)
    print("\nPrimera captura:")
    print(json.dumps(result["captures_plan"]["captures"][0], indent=2, ensure_ascii=False))