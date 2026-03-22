import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
FRAMES_DIR = DATA_DIR / "frames"
JSON_DIR  = DATA_DIR / "json"

def extract_video_info(video_path: Path) -> dict:
    """Obtiene duración y fps del video via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    probe = json.loads(result.stdout)

    video_stream = next(
        s for s in probe["streams"] if s["codec_type"] == "video"
    )

    fps_raw = video_stream["r_frame_rate"]  # ej: "30/1"
    num, den = fps_raw.split("/")
    fps = round(float(num) / float(den), 2)

    duration = float(probe["format"]["duration"])

    return {
        "fps": fps,
        "duration_seconds": round(duration, 2),
        "width": video_stream["width"],
        "height": video_stream["height"],
        "codec": video_stream["codec_name"]
    }


def extract_frames(video_path: Path, interval_seconds: int = 5) -> list[dict]:
    """
    Extrae un frame cada N segundos.
    Devuelve lista de metadata por frame.
    """
    video_name = video_path.stem
    output_dir = FRAMES_DIR / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps=1/{interval_seconds}",
        "-q:v", "2",
        str(output_dir / "frame_%04d.jpg"),
        "-y"
    ]

    print(f"Extrayendo frames de: {video_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")

    # Construir índice de frames
    frames = []
    for frame_file in sorted(output_dir.glob("frame_*.jpg")):
        frame_num = int(frame_file.stem.split("_")[1])
        timestamp = (frame_num - 1) * interval_seconds

        frames.append({
            "frame_id": frame_file.stem,
            "file_path": str(frame_file),
            "timestamp_seconds": timestamp,
            "timestamp_label": format_timestamp(timestamp)
        })

    return frames


def format_timestamp(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run(meta: dict) -> dict:
    video_path = Path(meta["video_path"])

    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")

    # Info del video
    video_info = extract_video_info(video_path)
    print(f"  Duracion: {video_info['duration_seconds']}s | FPS: {video_info['fps']}")

    # Extraer frames
    interval = meta.get("frame_interval_seconds", 5)
    frames = extract_frames(video_path, interval)
    print(f"  Frames extraidos: {len(frames)}")

    # Construir frames_index.json
    frames_index = {
        "generated_at": datetime.now().isoformat(),
        "video_path": str(video_path),
        "video_info": video_info,
        "frame_interval_seconds": interval,
        "total_frames": len(frames),
        "frames": frames
    }

    # Guardar
    output_path = JSON_DIR / "frames_index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(frames_index, f, indent=2, ensure_ascii=False)

    print(f"  Guardado: {output_path}")
    return frames_index


if __name__ == "__main__":
    # Prueba directa
    with open("meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    result = run(meta)
    print(f"\nOK — {result['total_frames']} frames indexados")