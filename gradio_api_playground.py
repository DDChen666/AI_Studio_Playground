from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise RuntimeError(
        "google-genai SDK is required. Install it via `pip install google-genai google-auth`."
    ) from exc

ROOT = Path(__file__).parent
SUMMARY_PATH = ROOT / "test_outputs" / "aistudio_tests_summary.json"
OUTPUT_DIR = ROOT / "playground_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env or the shell environment before launching the playground.")

CLIENT = genai.Client(api_key=API_KEY)


def load_passed_tests(summary_path: Path) -> Dict[str, Dict[str, Any]]:
    if not summary_path.exists():
        return {}
    try:
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, list):
        return {}
    return {entry.get("name", f"scenario_{idx}"): entry for idx, entry in enumerate(raw) if entry.get("status") == "pass"}


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
    "null": types.Type.NULL,
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
        kwargs["properties"] = {key: schema_from_json(value) for key, value in payload["properties"].items()}

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
    filename = OUTPUT_DIR / f"{stem}_{int(time.time())}_{uuid.uuid4().hex[:6]}{suffix}"
    filename.write_bytes(data)
    return filename



def is_riff_wav(data: bytes) -> bool:
    """Detect whether the payload already contains a RIFF/WAVE header."""
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
    """Wrap bare PCM data in a minimal WAV container for playback."""
    filename = OUTPUT_DIR / f"{stem}_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data)
    return filename



def ensure_prompt(prompt: str) -> None:
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a prompt before calling the API.")


def run_text_generation(
    prompt: str,
    system_instruction: str,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    ensure_prompt(prompt)
    config = types.GenerateContentConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_output_tokens=int(max_tokens),
        system_instruction=system_instruction or None,
    )
    resp = CLIENT.models.generate_content(model=model, contents=prompt, config=config)
    text = resp.text or "(no text returned)"
    usage = usage_to_dict(resp.usage_metadata)
    return text, usage


def run_generation_with_config(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    candidate_count: int,
    stop_sequences: str,
    response_mime_type: str,
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    ensure_prompt(prompt)
    stops = [line.strip() for line in stop_sequences.splitlines() if line.strip()]
    config = types.GenerateContentConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_output_tokens=int(max_tokens),
        candidate_count=int(candidate_count),
        stop_sequences=stops or None,
        response_mime_type=response_mime_type or None,
    )
    resp = CLIENT.models.generate_content(model=model, contents=prompt, config=config)
    text = resp.text or "(no text returned)"
    parsed: Optional[Dict[str, Any]] = None
    if (response_mime_type or "").lower() == "application/json":
        parsed, _ = safe_json_loads(text)
    usage = usage_to_dict(resp.usage_metadata)
    return text, parsed, usage


def run_streaming(prompt: str, model: str) -> Tuple[List[str], str]:
    ensure_prompt(prompt)
    stream = CLIENT.models.generate_content_stream(model=model, contents=prompt)
    chunks: List[str] = []
    for event in stream:
        text = getattr(event, "text", None)
        if text:
            chunks.append(text)
    combined = "".join(chunks) or "(no text returned)"
    return chunks, combined


def run_json_schema(
    prompt: str,
    schema_json: str,
    strict: bool,
    model: str,
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    ensure_prompt(prompt)
    schema_payload, error = safe_json_loads(schema_json)
    if error:
        raise gr.Error(error)
    if schema_payload is None:
        raise gr.Error("Provide a JSON schema body.")
    schema = schema_from_json(schema_payload)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        strict=bool(strict),
    )
    resp = CLIENT.models.generate_content(model=model, contents=prompt, config=config)
    text = resp.text or ""
    parsed, parse_error = safe_json_loads(text)
    if parse_error:
        raise gr.Error(f"Model did not return valid JSON: {parse_error}")
    usage = usage_to_dict(resp.usage_metadata)
    return text, parsed, usage


def run_function_calling(
    prompt: str,
    model: str,
    function_name: str,
    function_description: str,
    parameters_json: str,
    tool_response_json: str,
) -> Tuple[str, str, Dict[str, Any]]:
    ensure_prompt(prompt)
    if not function_name.strip():
        raise gr.Error("Function name is required.")
    params_payload, params_error = safe_json_loads(parameters_json or "{}")
    if params_error:
        raise gr.Error(params_error)
    if params_payload is None:
        params_payload = {"type": "object"}
    schema = schema_from_json(params_payload)
    function_decl = types.FunctionDeclaration(
        name=function_name.strip(),
        description=function_description or None,
        parameters=schema,
    )
    tool_config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[function_decl])]
    )
    resp = CLIENT.models.generate_content(
        model=model,
        contents=prompt,
        config=tool_config,
    )
    call_args: Dict[str, Any] = {}
    if resp.candidates:
        candidate = resp.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.function_call:
                    call_args = dict(part.function_call.args or {})
                    break
    args_text = json.dumps(call_args, ensure_ascii=False, indent=2)

    final_text = ""
    if tool_response_json and tool_response_json.strip():
        tool_payload, tool_error = safe_json_loads(tool_response_json)
        if tool_error:
            raise gr.Error(tool_error)
        follow = CLIENT.models.generate_content(
            model=model,
            contents=[
                types.Content(role="user", parts=[types.Part(text=prompt)]),
                types.Content(
                    role="tool",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=function_name.strip(), response=tool_payload
                            )
                        )
                    ],
                ),
            ],
            config=tool_config,
        )
        final_text = follow.text or ""
    usage = usage_to_dict(resp.usage_metadata)
    return args_text, final_text, usage


def run_google_search(prompt: str, model: str, max_results: int) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    ensure_prompt(prompt)
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        candidate_count=1,
    )
    resp = CLIENT.models.generate_content(model=model, contents=prompt, config=config)
    detail = resp.text or ""
    grounding = resp.candidates[0].grounding_metadata if resp.candidates else None
    limit = int(max_results)
    sources: List[Dict[str, Any]] = []
    if grounding and grounding.grounding_chunks:
        for chunk in grounding.grounding_chunks[:limit]:
            web = getattr(chunk, "web", None)
            sources.append(
                {
                    "title": getattr(web, "title", None),
                    "domain": getattr(web, "domain", None),
                    "uri": getattr(web, "uri", None),
                }
            )
    usage = usage_to_dict(resp.usage_metadata)
    return detail, sources, usage


def run_code_execution(prompt: str, model: str) -> Tuple[str, Optional[str], Optional[str], Dict[str, Any]]:
    ensure_prompt(prompt)
    config = types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())]
    )
    resp = CLIENT.models.generate_content(model=model, contents=prompt, config=config)
    candidate = resp.candidates[0] if resp.candidates else None
    code, output = (None, None)
    if candidate and candidate.content and candidate.content.parts:
        code, output = extract_code_parts(candidate.content.parts)
    usage = usage_to_dict(resp.usage_metadata)
    return resp.text or "", code, output, usage


def resolve_path(upload: Union[str, Path, Any]) -> Path:
    if isinstance(upload, Path):
        return upload
    if isinstance(upload, str):
        return Path(upload)
    if hasattr(upload, "name"):
        return Path(upload.name)
    raise ValueError("Unsupported file input type")


def run_files_api(
    local_file: Optional[Any],
    instructions: str,
    model: str,
    keep_remote: bool,
) -> Tuple[str, Dict[str, Any]]:
    if local_file is None:
        raise gr.Error("Upload a file to send through the Files API.")
    ensure_prompt(instructions)
    path = resolve_path(local_file)
    uploaded = CLIENT.files.upload(file=str(path))
    try:
        resp = CLIENT.models.generate_content(
            model=model,
            contents=[uploaded, instructions],
        )
        detail = resp.text or ""
        meta = {
            "uploaded_name": uploaded.name,
            "uri": uploaded.uri,
            "size_bytes": getattr(uploaded, "size_bytes", None),
            "kept_remote": bool(keep_remote),
        }
    finally:
        if not keep_remote:
            try:
                CLIENT.files.delete(name=uploaded.name)
            except ClientError:
                pass
    return detail, meta


def run_image_generation(
    prompt: str,
    model: str,
    style: str,
    width: int,
    height: int,
) -> Tuple[str, Optional[str], Optional[str]]:
    ensure_prompt(prompt)
    directives: List[str] = []
    if style.strip():
        directives.append(f"Style hint: {style.strip()}")
    directives.append(f"Target size around {int(width)}x{int(height)} pixels.")
    full_prompt = prompt.strip()
    if directives:
        full_prompt = full_prompt + "\n" + "\n".join(directives)
    cfg = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    resp = CLIENT.models.generate_content(model=model, contents=full_prompt, config=cfg)
    candidate = resp.candidates[0] if resp.candidates else None
    text_output = resp.text or ""
    image_path: Optional[Path] = None
    if candidate and candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            if part.inline_data and part.inline_data.data:
                data = ensure_bytes(part.inline_data.data)
                image_path = write_binary_asset(data, part.inline_data.mime_type, "image")
                break
    path_str = str(image_path) if image_path else None
    return text_output, path_str, path_str


def run_tts(
    text: str,
    model: str,
    voice_name: str,
) -> Tuple[str, Optional[str], Optional[str], Dict[str, Any]]:
    ensure_prompt(text)
    voice = (voice_name or "").strip()
    speech_config: Optional[types.SpeechConfig] = None
    if voice:
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        )
    cfg_kwargs: Dict[str, Any] = {"response_modalities": ["AUDIO"]}
    if speech_config is not None:
        cfg_kwargs["speech_config"] = speech_config
    cfg = types.GenerateContentConfig(**cfg_kwargs)
    resp = CLIENT.models.generate_content(model=model, contents=text, config=cfg)
    candidate = resp.candidates[0] if resp.candidates else None
    audio_path: Optional[Path] = None
    mime: Optional[str] = None
    original_mime: Optional[str] = None
    payload_type: Optional[str] = None
    byte_count: Optional[int] = None
    wrapped_pcm = False
    if candidate and candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            if part.inline_data and part.inline_data.data:
                raw_payload = part.inline_data.data
                payload_type = type(raw_payload).__name__
                data = ensure_bytes(raw_payload)
                byte_count = len(data)
                original_mime = getattr(part.inline_data, "mime_type", None)
                lower_mime = (original_mime or "").split(";")[0].strip().lower()
                if is_riff_wav(data):
                    mime = "audio/wav"
                    audio_path = write_binary_asset(data, mime, "tts")
                elif lower_mime and lower_mime not in PCM_MIME_TYPES:
                    mime = original_mime
                    audio_path = write_binary_asset(data, mime, "tts")
                else:
                    audio_path = write_wav_from_pcm(data, stem="tts")
                    mime = "audio/wav"
                    wrapped_pcm = True
                break
    message = "No audio returned"
    preview = str(audio_path) if audio_path else None
    if audio_path:
        details = []
        if original_mime and original_mime != mime:
            details.append(f"mime={original_mime}->{mime}")
        elif mime:
            details.append(f"mime={mime}")
        if payload_type:
            details.append(f"source={payload_type}")
        if byte_count is not None:
            details.append(f"bytes={byte_count}")
        if wrapped_pcm:
            details.append("wrapped=wav")
        message = f"Audio saved to {audio_path}"
        if details:
            message += f" ({'; '.join(details)})"
    usage = usage_to_dict(resp.usage_metadata)
    return message, preview, preview, usage


def run_audio_transcription(
    audio_file: Optional[Any],
    prompt: str,
    model: str,
) -> Tuple[str, Dict[str, Any]]:
    if audio_file is None:
        raise gr.Error("Upload an audio file to transcribe.")
    ensure_prompt(prompt)
    path = resolve_path(audio_file)
    uploaded = CLIENT.files.upload(file=str(path))
    try:
        resp = CLIENT.models.generate_content(
            model=model,
            contents=[uploaded, prompt],
        )
        detail = resp.text or ""
        meta = {"uploaded_name": uploaded.name}
    finally:
        try:
            CLIENT.files.delete(name=uploaded.name)
        except ClientError:
            pass
    return detail, meta


def run_thinking_mode(
    prompt: str,
    model: str,
    thinking_budget: int,
) -> Tuple[str, Dict[str, Any]]:
    ensure_prompt(prompt)
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=int(thinking_budget))
    )
    resp = CLIENT.models.generate_content(model=model, contents=prompt, config=cfg)
    usage = usage_to_dict(resp.usage_metadata)
    return resp.text or "", usage


SAMPLE_IMAGE = (ROOT / "test_outputs" / "generated_image.png")
SAMPLE_IMAGE = str(SAMPLE_IMAGE) if SAMPLE_IMAGE.exists() else None
SAMPLE_AUDIO = (ROOT / "test_outputs" / "generated_tts.wav")
SAMPLE_AUDIO = str(SAMPLE_AUDIO) if SAMPLE_AUDIO.exists() else None
SAMPLE_REPORT = (ROOT / "report.pdf")
SAMPLE_REPORT = str(SAMPLE_REPORT) if SAMPLE_REPORT.exists() else None


def build_demo() -> gr.Blocks:
    passed_names = ", ".join(sorted(PASSED_TESTS)) if PASSED_TESTS else "none"

    default_text_model = get_data_value("basic_text_generation", "model", "gemini-2.5-flash")
    default_config_model = get_data_value("generation_config", "model", "gemini-2.5-pro")
    default_stream_prompt = "條列說明 CPU、GPU 與 TPU 的差異"
    default_schema = get_detail("json_schema", "{\"type\": \"object\"}")
    default_function_schema = json.dumps(
        get_data_value(
            "function_calling",
            "requested_city",
            {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        ),
        ensure_ascii=True,
        indent=2,
    )

    with gr.Blocks(title="AI Studio API Playground", theme="soft") as demo:
        gr.Markdown(
            """
            # AI Studio API Playground
            Interactively exercise Gemini models with live responses. Scenarios come from `test_outputs/aistudio_tests_summary.json`.
            """
        )
        gr.Markdown(f"**Passing scenarios detected:** {passed_names}")

        with gr.Tabs():
            with gr.Tab("Text Generation"):
                prompt = gr.Textbox(label="Prompt", value=get_detail("basic_text_generation"), lines=6)
                system_instruction = gr.Textbox(label="System instruction", lines=3)
                model = gr.Dropdown(
                    [default_text_model, "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"],
                    label="Model",
                    value=default_text_model,
                )
                with gr.Row():
                    temperature = gr.Slider(0.0, 2.0, value=0.6, step=0.1, label="Temperature")
                    top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p")
                with gr.Row():
                    top_k = gr.Slider(1, 200, value=40, step=1, label="Top-k")
                    max_tokens = gr.Slider(16, 4096, value=1024, step=16, label="Max output tokens")
                text_btn = gr.Button("Generate")
                text_output = gr.Textbox(label="Model response", lines=10)
                text_usage = gr.JSON(label="Usage metadata")
                text_btn.click(
                    run_text_generation,
                    inputs=[prompt, system_instruction, model, temperature, top_p, top_k, max_tokens],
                    outputs=[text_output, text_usage],
                )

            with gr.Tab("Generation Config"):
                prompt_cfg = gr.Textbox(label="Prompt", lines=6, value="Explain the difference between top-p and top-k.")
                model_cfg = gr.Dropdown(
                    [default_config_model, "gemini-2.5-flash", "gemini-1.5-pro"],
                    label="Model",
                    value=default_config_model,
                )
                with gr.Row():
                    cfg_temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
                    cfg_top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                with gr.Row():
                    cfg_top_k = gr.Slider(1, 200, value=32, step=1, label="Top-k")
                    cfg_max_tokens = gr.Slider(16, 4096, value=512, step=16, label="Max output tokens")
                candidate_count = gr.Slider(1, 4, value=1, step=1, label="Candidate count")
                stop_sequences = gr.Textbox(label="Stop sequences (one per line)")
                response_mime_type = gr.Dropdown(["", "text/plain", "application/json"], label="Response MIME type", value="")
                cfg_btn = gr.Button("Generate")
                cfg_text = gr.Textbox(label="Model response", lines=10)
                cfg_json = gr.JSON(label="Parsed JSON (when applicable)")
                cfg_usage = gr.JSON(label="Usage metadata")
                cfg_btn.click(
                    run_generation_with_config,
                    inputs=[
                        prompt_cfg,
                        model_cfg,
                        cfg_temperature,
                        cfg_top_p,
                        cfg_top_k,
                        cfg_max_tokens,
                        candidate_count,
                        stop_sequences,
                        response_mime_type,
                    ],
                    outputs=[cfg_text, cfg_json, cfg_usage],
                )

            with gr.Tab("Streaming"):
                stream_prompt = gr.Textbox(label="Prompt", lines=5, value=default_stream_prompt)
                stream_model = gr.Dropdown(["gemini-2.5-flash", "gemini-2.5-pro"], label="Model", value="gemini-2.5-flash")
                stream_btn = gr.Button("Start streaming")
                stream_chunks = gr.JSON(label="Chunks")
                stream_text = gr.Textbox(label="Combined text", lines=10)
                stream_btn.click(
                    run_streaming,
                    inputs=[stream_prompt, stream_model],
                    outputs=[stream_chunks, stream_text],
                )

            with gr.Tab("JSON Schema"):
                schema_prompt = gr.Textbox(label="Prompt", lines=4, value="建立一份 CLI 待辦清單應用的 JSON 摘要")
                schema_box = gr.Textbox(label="Response schema (JSON)", value=default_schema, lines=12)
                schema_model = gr.Dropdown(["gemini-2.5-flash", "gemini-2.5-pro"], value="gemini-2.5-flash", label="Model")
                schema_strict = gr.Checkbox(label="Strict mode", value=True)
                schema_btn = gr.Button("Generate JSON")
                schema_text = gr.Textbox(label="Raw JSON", lines=10)
                schema_parsed = gr.JSON(label="Parsed JSON")
                schema_usage = gr.JSON(label="Usage metadata")
                schema_btn.click(
                    run_json_schema,
                    inputs=[schema_prompt, schema_box, schema_strict, schema_model],
                    outputs=[schema_text, schema_parsed, schema_usage],
                )

            with gr.Tab("Function Calling"):
                func_prompt = gr.Textbox(label="User prompt", lines=4, value="告訴我台北今天的天氣")
                func_model = gr.Dropdown(["gemini-2.5-flash", "gemini-2.5-pro"], value="gemini-2.5-flash", label="Model")
                func_name = gr.Textbox(label="Function name", value="get_weather")
                func_description = gr.Textbox(label="Function description", value="查詢指定城市的天氣資訊", lines=2)
                func_params = gr.Textbox(label="Function parameters (JSON Schema)", value=default_function_schema, lines=8)
                func_tool_response = gr.Textbox(label="Tool response (JSON)", lines=6, value="{\n  \"temp_c\": 27,\n  \"condition\": \"Cloudy\"\n}")
                func_btn = gr.Button("Trigger function call")
                func_args = gr.Textbox(label="Arguments emitted", lines=6)
                func_follow = gr.Textbox(label="Follow-up model response", lines=8)
                func_usage = gr.JSON(label="Usage metadata")
                func_btn.click(
                    run_function_calling,
                    inputs=[func_prompt, func_model, func_name, func_description, func_params, func_tool_response],
                    outputs=[func_args, func_follow, func_usage],
                )

            with gr.Tab("Google Search"):
                search_prompt = gr.Textbox(label="Prompt", lines=3, value="Who won Euro 2024?")
                search_model = gr.Dropdown(["gemini-2.5-flash", "gemini-2.5-pro"], value="gemini-2.5-flash", label="Model")
                max_results = gr.Slider(1, 10, value=5, step=1, label="Max sources")
                search_btn = gr.Button("Run search")
                search_text = gr.Textbox(label="Model response", lines=8)
                search_sources = gr.JSON(label="Grounded sources")
                search_usage = gr.JSON(label="Usage metadata")
                search_btn.click(
                    run_google_search,
                    inputs=[search_prompt, search_model, max_results],
                    outputs=[search_text, search_sources, search_usage],
                )

            with gr.Tab("Code Execution"):
                code_prompt = gr.Textbox(label="Prompt", lines=4, value="列出 1 到 10 的平方並提供三種方法")
                code_model = gr.Dropdown(["gemini-2.5-flash", "gemini-2.5-pro"], value="gemini-2.5-flash", label="Model")
                code_btn = gr.Button("Run code")
                code_response = gr.Textbox(label="Model response", lines=6)
                code_generated = gr.Code(label="Generated code", language="python")
                code_output = gr.Textbox(label="Execution output", lines=6)
                code_usage = gr.JSON(label="Usage metadata")
                code_btn.click(
                    run_code_execution,
                    inputs=[code_prompt, code_model],
                    outputs=[code_response, code_generated, code_output, code_usage],
                )

            with gr.Tab("Files API"):
                file_input = gr.File(label="Upload file", value=SAMPLE_REPORT)
                file_instructions = gr.Textbox(label="Prompt", lines=4, value="請從附件中整理出三個重點")
                file_model = gr.Dropdown(["gemini-2.5-pro", "gemini-2.5-flash"], value="gemini-2.5-pro", label="Model")
                keep_remote = gr.Checkbox(label="Keep remote copy", value=False)
                file_btn = gr.Button("Send to model")
                file_response = gr.Textbox(label="Model response", lines=8)
                file_meta = gr.JSON(label="File metadata")
                file_btn.click(
                    run_files_api,
                    inputs=[file_input, file_instructions, file_model, keep_remote],
                    outputs=[file_response, file_meta],
                )

            with gr.Tab("Image Generation"):
                image_prompt = gr.Textbox(label="Prompt", lines=4, value="Watercolor reading nook with warm morning light")
                image_model = gr.Dropdown(["gemini-2.0-flash-preview-image-generation"], value="gemini-2.0-flash-preview-image-generation", label="Model")
                style_hint = gr.Textbox(label="Style hint", value="watercolor")
                with gr.Row():
                    width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                    height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                image_btn = gr.Button("Generate image")
                image_text = gr.Textbox(label="Model response", lines=6)
                image_preview = gr.Image(label="Preview")
                image_file = gr.File(label="Download image")
                image_btn.click(
                    run_image_generation,
                    inputs=[image_prompt, image_model, style_hint, width, height],
                    outputs=[image_text, image_preview, image_file],
                )

            with gr.Tab("Text to Speech"):
                tts_text = gr.Textbox(label="Text", lines=4, value="歡迎來到 AI Studio Playground！")
                tts_model = gr.Dropdown(["gemini-2.5-flash-preview-tts"], value="gemini-2.5-flash-preview-tts", label="Model")
                tts_voice = gr.Textbox(label="Voice name", value="callirrhoe")
                tts_btn = gr.Button("Synthesize")
                tts_response = gr.Textbox(label="Model response", lines=4)
                tts_audio = gr.Audio(label="Preview", type="filepath", interactive=False)
                tts_file = gr.File(label="Download audio")
                tts_usage = gr.JSON(label="Usage metadata")
                tts_btn.click(
                    run_tts,
                    inputs=[tts_text, tts_model, tts_voice],
                    outputs=[tts_response, tts_audio, tts_file, tts_usage],
                )



            with gr.Tab("Audio Transcription"):
                audio_input = gr.File(label="Audio file", value=SAMPLE_AUDIO)
                audio_prompt = gr.Textbox(label="Prompt", lines=3, value="請逐字轉錄這段音訊")
                audio_model = gr.Dropdown(["gemini-2.5-pro", "gemini-2.5-flash"], value="gemini-2.5-pro", label="Model")
                audio_btn = gr.Button("Transcribe")
                audio_text = gr.Textbox(label="Transcript", lines=10)
                audio_meta = gr.JSON(label="Metadata")
                audio_btn.click(
                    run_audio_transcription,
                    inputs=[audio_input, audio_prompt, audio_model],
                    outputs=[audio_text, audio_meta],
                )

            with gr.Tab("Thinking Mode"):
                thinking_prompt = gr.Textbox(label="Prompt", lines=3, value="提供兩個番茄鐘技巧")
                thinking_model = gr.Dropdown(["gemini-2.5-flash", "gemini-2.5-pro"], value="gemini-2.5-flash", label="Model")
                thinking_budget = gr.Slider(32, 2048, value=get_data_value("thinking_mode", "thought_tokens", 256) or 256, step=32, label="Thinking budget (tokens)")
                thinking_btn = gr.Button("Generate")
                thinking_text = gr.Textbox(label="Model response", lines=10)
                thinking_usage = gr.JSON(label="Usage metadata")
                thinking_btn.click(
                    run_thinking_mode,
                    inputs=[thinking_prompt, thinking_model, thinking_budget],
                    outputs=[thinking_text, thinking_usage],
                )

    return demo


def launch() -> None:
    demo = build_demo()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    launch()
