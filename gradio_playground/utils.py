from __future__ import annotations

import base64
import json
import mimetypes
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr

from .config import PLAYGROUND_OUTPUT_DIR, SUMMARY_PATH, types


def load_passed_tests(summary_path: Path) -> Dict[str, Dict[str, Any]]:
    if not summary_path.exists():
        return {}
    try:
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, list):
        return {}
    return {
        entry.get("name", f"scenario_{idx}"): entry
        for idx, entry in enumerate(raw)
        if entry.get("status") == "pass"
    }


PASSED_TESTS = load_passed_tests(SUMMARY_PATH)


def get_detail(name: str, fallback: str = "") -> str:
    return PASSED_TESTS.get(name, {}).get("detail", fallback) or fallback


def get_data_value(name: str, key: str, fallback: Any = None) -> Any:
    return PASSED_TESTS.get(name, {}).get("data", {}).get(key, fallback)


def safe_json_loads(value: str) -> Tuple[Optional[Any], Optional[str]]:
    if not value or not value.strip():
        return None, None
    try:
        return json.loads(value), None
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error at position {exc.pos}: {exc.msg}"


def usage_to_dict(usage: Optional[types.UsageMetadata]) -> Dict[str, Any]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
        "thought_tokens": getattr(usage, "thoughts_token_count", None),
    }


TYPE_MAP = {
    "string": types.Type.STRING,
    "number": types.Type.NUMBER,
    "integer": types.Type.INTEGER,
    "boolean": types.Type.BOOLEAN,
    "object": types.Type.OBJECT,
    "array": types.Type.ARRAY,
}


def schema_from_json(payload: Any) -> types.Schema:
    if isinstance(payload, types.Schema):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Schema JSON must decode to an object.")

    kwargs: Dict[str, Any] = {}
    schema_type = payload.get("type")
    if schema_type:
        enum_key = str(schema_type).lower()
        if enum_key not in TYPE_MAP:
            raise ValueError(f"Unsupported schema type: {schema_type}")
        kwargs["type"] = TYPE_MAP[enum_key]
    if "description" in payload:
        kwargs["description"] = payload["description"]
    if "enum" in payload:
        kwargs["enum"] = payload["enum"]
    if "required" in payload:
        kwargs["required"] = payload["required"]
    if "nullable" in payload:
        kwargs["nullable"] = bool(payload["nullable"])
    if "format" in payload:
        kwargs["format"] = payload["format"]

    if "properties" in payload and isinstance(payload["properties"], dict):
        kwargs["properties"] = {
            key: schema_from_json(value)
            for key, value in payload["properties"].items()
        }

    if "items" in payload:
        items_value = payload["items"]
        if isinstance(items_value, list):
            kwargs["items"] = [schema_from_json(item) for item in items_value]
        else:
            kwargs["items"] = schema_from_json(items_value)

    return types.Schema(**kwargs)


def extract_code_parts(parts: List[types.Part]) -> Tuple[Optional[str], Optional[str]]:
    code = None
    result = None
    for part in parts:
        if part.executable_code and part.executable_code.code:
            code = part.executable_code.code
        if part.code_execution_result and part.code_execution_result.output:
            output = part.code_execution_result.output
            result = output.strip() if isinstance(output, str) else output
    return code, result


def ensure_bytes(blob: Union[str, bytes]) -> bytes:
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, str):
        return base64.b64decode(blob)
    raise TypeError(f"Unsupported data type: {type(blob)!r}")


def suffix_for_mime(mime: Optional[str]) -> str:
    base_mime = (mime or "").split(";")[0].strip().lower()
    suffix_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/heic": ".heic",
        "image/heif": ".heif",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
        "audio/flac": ".flac",
        "audio/pcm": ".pcm",
        "audio/x-pcm": ".pcm",
        "audio/l16": ".pcm",
        "application/octet-stream": ".bin",
    }
    if base_mime in suffix_map:
        return suffix_map[base_mime]
    guessed = mimetypes.guess_extension(base_mime or "")
    if guessed:
        return guessed
    if "/" in base_mime:
        subtype = base_mime.split("/", 1)[1].strip()
        if subtype.startswith("x-"):
            subtype = subtype[2:]
        if subtype:
            return f".{subtype}"
    return ".bin"


def write_binary_asset(data: bytes, mime: Optional[str], stem: str) -> Path:
    suffix = suffix_for_mime(mime)
    filename = PLAYGROUND_OUTPUT_DIR / f"{stem}_{int(time.time())}_{uuid.uuid4().hex[:6]}{suffix}"
    filename.write_bytes(data)
    return filename


def is_riff_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


PCM_MIME_TYPES = {
    "audio/pcm",
    "audio/x-pcm",
    "audio/l16",
    "audio/x-linearpcm",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "application/octet-stream",
}


def write_wav_from_pcm(
    data: bytes,
    *,
    stem: str,
    channels: int = 1,
    sample_rate: int = 24000,
    sample_width: int = 2,
) -> Path:
    filename = PLAYGROUND_OUTPUT_DIR / f"{stem}_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data)
    return filename


def ensure_prompt(prompt: str) -> None:
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a prompt before calling the API.")


def resolve_path(upload: Union[str, Path, Any]) -> Path:
    if isinstance(upload, Path):
        return upload
    if isinstance(upload, str):
        return Path(upload)
    if hasattr(upload, "name"):
        return Path(upload.name)
    raise ValueError("Unsupported file input type")


__all__ = [
    "PASSED_TESTS",
    "PCM_MIME_TYPES",
    "ensure_bytes",
    "ensure_prompt",
    "extract_code_parts",
    "get_data_value",
    "get_detail",
    "resolve_path",
    "safe_json_loads",
    "schema_from_json",
    "usage_to_dict",
    "write_binary_asset",
    "is_riff_wav",
    "write_wav_from_pcm",
]

