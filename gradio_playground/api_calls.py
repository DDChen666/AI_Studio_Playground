from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .config import CLIENT, ClientError, types
from .utils import (
    PCM_MIME_TYPES,
    ensure_bytes,
    ensure_prompt,
    extract_code_parts,
    resolve_path,
    safe_json_loads,
    schema_from_json,
    usage_to_dict,
    write_binary_asset,
    write_wav_from_pcm,
)
from .utils import is_riff_wav


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
    candidate = resp.candidates[0] if resp.candidates else None
    if candidate and candidate.content and candidate.content.parts:
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


def run_google_search(
    prompt: str,
    model: str,
    max_results: int,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
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


def run_code_execution(
    prompt: str,
    model: str,
) -> Tuple[str, Optional[str], Optional[str], Dict[str, Any]]:
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


__all__ = [
    "run_audio_transcription",
    "run_code_execution",
    "run_files_api",
    "run_function_calling",
    "run_generation_with_config",
    "run_google_search",
    "run_image_generation",
    "run_json_schema",
    "run_streaming",
    "run_text_generation",
    "run_thinking_mode",
    "run_tts",
]
