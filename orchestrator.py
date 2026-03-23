import json
import subprocess
from pathlib import Path
from datetime import datetime
import openai

JSON_DIR = Path("data/json")
JSON_DIR.mkdir(parents=True, exist_ok=True)

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# ─────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────

def log(stage: str, message: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "message": message
    }
    print(f"[{stage}] {message}")
    log_path = JSON_DIR / "orchestration_log.json"
    logs = []
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            logs = json.load(f)
    logs.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def load_meta(meta_path: str = "meta.json") -> dict:
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Derivar project_name desde video_path
    if "project_name" not in meta:
        meta["project_name"] = Path(meta["video_path"]).stem

    # Intervalo por defecto
    if "frame_interval_seconds" not in meta:
        meta["frame_interval_seconds"] = 2

    return meta


def detect_language(frames_index: dict) -> str:
    """
    Detecta el idioma del software analizando los textos
    visibles en los primeros 3 frames con qwen3.
    """
    sample_frames = frames_index["frames"][:3]
    texts = []
    for frame in sample_frames:
        if "visual_analysis" in frame:
            texts.append(frame["visual_analysis"].get("visible_text", ""))

    if not texts:
        return "es"

    prompt = f"""Analiza estos textos visibles en capturas de pantalla de un software:

{chr(10).join(texts)}

Detecta el idioma principal. Responde ÚNICAMENTE con el código de idioma:
- "es" si es español
- "en" si es inglés
- "pt" si es portugués
- "fr" si es francés

Responde solo el código, nada más."""

    response = client.chat.completions.create(
        model="qwen3:14b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    raw = response.choices[0].message.content.strip()

    # Limpiar thinking tags
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    lang = raw.strip().lower().replace('"', '').replace("'", "")
    if lang not in ["es", "en", "pt", "fr"]:
        lang = "es"

    return lang


def should_use_chunks(frames_index: dict, chunk_threshold: int = 90) -> bool:
    """Determina si el video necesita procesarse en chunks."""
    return frames_index["total_frames"] > chunk_threshold


def split_into_chunks(frames: list, chunk_size: int = 30) -> list:
    """Divide la lista de frames en bloques de chunk_size."""
    return [frames[i:i+chunk_size] for i in range(0, len(frames), chunk_size)]


# ─────────────────────────────────────────
# AGENTES
# ─────────────────────────────────────────

def run_ffmpeg_extractor(meta: dict) -> dict:
    log("FFmpeg", f"Extrayendo frames de {meta['video_path']}")
    from agents import ffmpeg_extractor
    result = ffmpeg_extractor.run(meta)
    log("FFmpeg", f"{result['total_frames']} frames extraídos")
    return result


def run_timeline_analyst(meta: dict, frames_index: dict) -> dict:
    from agents import timeline_analyst

    frames = frames_index["frames"]
    total  = frames_index["total_frames"]

    if should_use_chunks(frames_index):
        log("Timeline", f"Video largo detectado ({total} frames) — procesando en chunks")
        chunks = split_into_chunks(frames, chunk_size=30)
        all_segments = []
        segment_offset = 0

        for i, chunk in enumerate(chunks):
            log("Timeline", f"Chunk {i+1}/{len(chunks)} — {len(chunk)} frames")

            chunk_index = {**frames_index, "frames": chunk, "total_frames": len(chunk)}
            chunk_result = timeline_analyst.run(meta, frames_index_override=chunk_index)

            # Ajustar segment_ids para que sean continuos
            for seg in chunk_result["segments"]:
                seg["segment_id"] += segment_offset
            all_segments.extend(chunk_result["segments"])
            segment_offset += len(chunk_result["segments"])

        timeline_raw = {
            "generated_at": datetime.now().isoformat(),
            "video_path":   frames_index["video_path"],
            "software_name": meta.get("software_name", ""),
            "language":      meta.get("language_output", "es"),
            "total_segments": len(all_segments),
            "segments":       all_segments
        }

        output_path = JSON_DIR / "timeline_raw.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(timeline_raw, f, indent=2, ensure_ascii=False)

    else:
        log("Timeline", f"Video corto ({total} frames) — procesando directo")
        timeline_raw = timeline_analyst.run(meta)

    log("Timeline", f"{timeline_raw['total_segments']} segmentos generados")
    return timeline_raw

def manual_review_checkpoint(json_path: str, stage: str) -> bool:
    print(f"\n{'='*50}")
    print(f"  CHECKPOINT DE REVISION — {stage}")
    print(f"{'='*50}")
    print(f"  Archivo: {json_path}")
    print(f"\n  Opciones:")
    print(f"  [A] Aprobar y continuar")
    print(f"  [E] Editar el archivo manualmente y luego continuar")
    print(f"  [X] Cancelar pipeline")

    while True:
        choice = input("\n  Tu elección (A/E/X): ").strip().upper()
        if choice == "A":
            log("Checkpoint", f"{stage} aprobado")
            return True
        elif choice == "E":
            print(f"\n  Abre {json_path} en VS Code, edita y guarda.")
            input("  Presiona Enter cuando hayas terminado...")
            log("Checkpoint", f"{stage} editado manualmente")
            return True
        elif choice == "X":
            log("Checkpoint", "Pipeline cancelado por el usuario")
            return False
        else:
            print("  Opción no válida. Escribe A, E o X.")

def run_agent(script: str, meta: dict) -> dict:
    """Corre un agente genérico que lee meta.json y escribe su JSON."""
    log(script, "Iniciando")
    module_map = {
        "procedure_analyst": "agents.procedure_analyst",
        "capture_selector":  "agents.capture_selector",
        "role_writer":       "agents.role_writer",
        "qa_agent":          "agents.qa_agent",
    }
    import importlib
    module = importlib.import_module(module_map[script])
    result = module.run(meta)
    log(script, "Completado")
    return result


# ─────────────────────────────────────────
# ORQUESTADOR PRINCIPAL
# ─────────────────────────────────────────

def run_pipeline(meta_path: str = "meta.json"):
    start_time = datetime.now()
    print("\n" + "="*50)
    print("  PIPELINE DE AGENTES IA")
    print("="*50)

    # 1. Cargar y completar meta.json
    meta = load_meta(meta_path)
    log("Orquestador", f"Proyecto: {meta['project_name']}")
    log("Orquestador", f"Video: {meta['video_path']}")

    # 2. Extraer frames
    frames_index = run_ffmpeg_extractor(meta)

    # 3. Detectar idioma automáticamente si no está definido
    if "language_output" not in meta:
        log("Orquestador", "Detectando idioma del software...")
        # Corremos timeline parcial para obtener textos
        # Por ahora usamos detección simple basada en nombre del software
        detected = detect_language(frames_index)
        meta["language_output"] = detected
        meta["language"] = detected
        log("Orquestador", f"Idioma detectado: {detected}")
    else:
        meta["language"] = meta["language_output"]
        log("Orquestador", f"Idioma definido manualmente: {meta['language_output']}")

    # Guardar meta enriquecido
    enriched_meta_path = JSON_DIR / "meta_enriched.json"
    with open(enriched_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 4. Correr agentes en secuencia
    try:
        timeline_raw    = run_timeline_analyst(meta, frames_index)

        # Checkpoint manual
        approved = manual_review_checkpoint(
            str(JSON_DIR / "timeline_raw.json"),
            "Timeline Analysis"
        )
        if not approved:
            print("Pipeline cancelado.")
            return

        procedure_steps = run_agent("procedure_analyst", meta)

        # Checkpoint manual
        approved = manual_review_checkpoint(
            str(JSON_DIR / "procedure_steps.json"),
            "Procedure Steps"
        )
        if not approved:
            print("Pipeline cancelado.")
            return

        # captures_plan   = run_agent("capture_selector",  meta)
        # chapter_content = run_agent("role_writer",       meta)
        # qa_report       = run_agent("qa_agent",          meta)
        # docx            = run_docx_builder(meta)

    except Exception as e:
        log("Orquestador", f"ERROR: {e}")
        raise

    # 5. Resumen final
    elapsed = (datetime.now() - start_time).seconds
    log("Orquestador", f"Pipeline completado en {elapsed}s")
    print("\n" + "="*50)
    print(f"  COMPLETADO en {elapsed} segundos")
    print(f"  Segmentos generados: {timeline_raw['total_segments']}")
    print(f"  Pasos de procedimiento: {procedure_steps['procedure']['total_steps']}")
    print(f"  Idioma: {meta['language_output']}")
    print(f"  Proyecto: {meta['project_name']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_pipeline()