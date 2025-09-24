import asyncio
import base64
import mimetypes
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from gradio_playground.config import get_api_key


class BlockedTest(Exception):
    """Raised when a test cannot run due to missing capabilities or quota."""


OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
client = genai.Client(api_key=get_api_key())

results: list[Dict[str, Any]] = []


def record(name: str, status: str, detail: str = "", data: Optional[Dict[str, Any]] = None) -> None:
    entry: Dict[str, Any] = {"name": name, "status": status}
    if detail:
        entry["detail"] = detail
    if data is not None:
        entry["data"] = data
    results.append(entry)


def safe_text(value: Optional[str], limit: int = 360) -> str:
    if not value:
        return ""
    text = value.strip().replace("\r\n", "\n")
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def ensure_bytes(blob: Any) -> bytes:
    if blob is None:
        raise ValueError("No binary payload returned")
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, str):
        return base64.b64decode(blob)
    raise TypeError(f"Unsupported blob type: {type(blob)!r}")




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


def run_test(name: str, fn) -> None:
    try:
        status, detail, data = fn()
    except BlockedTest as exc:
        record(name, "blocked", detail=str(exc))
    except Exception as exc:
        record(name, "fail", detail=str(exc))
    else:
        record(name, status, detail=detail, data=data)


def first_candidate(resp: types.GenerateContentResponse):
    if not resp.candidates:
        raise AssertionError("No candidates in response")
    return resp.candidates[0]


def test_basic_text() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="給我三種適合初學者的 Python 專案點子",
    )
    return "pass", safe_text(resp.text), {"model": "gemini-2.5-flash"}


def test_generation_config() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    cfg = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
        stop_sequences=["<END>"],
        system_instruction="你是嚴謹又友善的技術助教。",
    )
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents="解釋 Top-p 與 Top-k 的差異並舉例",
        config=cfg,
    )
    usage = resp.usage_metadata
    meta = {
        "model": resp.model_version,
        "prompt_tokens": usage.prompt_token_count if usage else None,
        "candidates_tokens": usage.candidates_token_count if usage else None,
    }
    return "pass", safe_text(resp.text), meta


def test_streaming() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    stream = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents="用條列方式解釋 CPU/GPU/TPU 差異",
    )
    chunks: list[str] = []
    for event in stream:
        text = getattr(event, "text", None)
        if text:
            chunks.append(text)
    combined = "".join(chunks)
    if not combined:
        raise AssertionError("Streaming response returned no text chunks")
    return "pass", safe_text(combined), {"chunk_count": len(chunks)}


def test_json_schema() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "title": types.Schema(type=types.Type.STRING),
            "keywords": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "difficulty": types.Schema(type=types.Type.STRING, enum=["easy", "medium", "hard"]),
        },
        required=["title", "keywords", "difficulty"],
    )
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="產出一個入門級 Python 專案主題，附 5 個關鍵字與難度等級",
        config=cfg,
    )
    payload = json.loads(resp.text)
    missing = [key for key in ("title", "keywords", "difficulty") if key not in payload]
    if missing:
        raise AssertionError(f"Missing keys in JSON response: {missing}")
    if not isinstance(payload.get("keywords"), list):
        raise AssertionError("keywords field is not a list")
    return "pass", json.dumps(payload, ensure_ascii=False), None


def test_function_calling() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    weather_fn = types.FunctionDeclaration(
        name="get_weather",
        description="查詢指定城市的當前天氣（攝氏）",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"city": types.Schema(type=types.Type.STRING)},
            required=["city"],
        ),
    )
    cfg = types.GenerateContentConfig(tools=[types.Tool(function_declarations=[weather_fn])])
    prompt = "幫我查台北的天氣並建議穿著"
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=cfg,
    )
    call = None
    for part in first_candidate(resp).content.parts:
        if part.function_call:
            call = part.function_call
            break
    if call is None:
        raise AssertionError("Model did not emit a function_call part")
    args = dict(call.args or {})
    faux_weather = {"temp_c": 28, "desc": "Cloudy", "city": args.get("city", "台北")}
    follow = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text=prompt)]),
            types.Content(
                role="tool",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(name="get_weather", response=faux_weather)
                    )
                ],
            ),
        ],
        config=cfg,
    )
    return "pass", safe_text(follow.text), {"requested_city": args}


def test_google_search() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Who won Euro 2024?",
        config=cfg,
    )
    grounding = first_candidate(resp).grounding_metadata
    if grounding is None or not grounding.grounding_chunks:
        raise AssertionError("Google Search grounding metadata missing")
    detail = safe_text(resp.text)
    chunk_domains = [chunk.web.domain for chunk in grounding.grounding_chunks if getattr(chunk, "web", None)]
    return "pass", detail, {"sources": chunk_domains}


def test_url_context() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    try:
        # This mirrors the documentation example and is expected to fail in SDK 1.16.1.
        types.Tool(url_context=types.UrlContext(urls=["https://example.com/docs/faq"]))  # type: ignore[arg-type]
    except Exception as exc:
        raise BlockedTest("	ypes.UrlContext in google-genai 1.16.1 does not accept an urls field") from None
    return "pass", "URL Context tool accepted configuration unexpectedly", None


def extract_code_parts(parts: list[types.Part]) -> Tuple[Optional[str], Optional[str]]:
    code = None
    result = None
    for part in parts:
        if part.executable_code and part.executable_code.code:
            code = part.executable_code.code
        if part.code_execution_result and part.code_execution_result.output:
            result = part.code_execution_result.output
    return code, result


def test_code_execution() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    cfg = types.GenerateContentConfig(tools=[types.Tool(code_execution=types.ToolCodeExecution())])
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="請以 Python 計算前 10 個正整數平方和，並解釋方法",
        config=cfg,
    )
    code, output = extract_code_parts(first_candidate(resp).content.parts)
    if output is None:
        raise AssertionError("No code_execution_result returned")
    return "pass", safe_text(resp.text), {"code": code, "result": output.strip() if isinstance(output, str) else output}


def test_files_api() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    uploaded = client.files.upload(file="report.pdf")
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[uploaded, "請將此 PDF 濃縮成 5 點摘要"],
        )
        names = [f.name for f in client.files.list()]
        detail = safe_text(resp.text)
        return "pass", detail, {"uploaded_name": uploaded.name, "uri": uploaded.uri, "listed": uploaded.name in names}
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except ClientError as exc:
            record("files_api_cleanup", "fail", detail=str(exc))


def test_image_generation() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    cfg = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    resp = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents="A cozy reading nook with warm morning light, watercolor style",
        config=cfg,
    )
    candidate = first_candidate(resp)
    image_part = next((p for p in candidate.content.parts if p.inline_data), None)
    if image_part is None:
        raise AssertionError("No image inline_data returned")
    payload = ensure_bytes(image_part.inline_data.data)
    path = OUTPUT_DIR / "generated_image.png"
    path.write_bytes(payload)
    return "pass", safe_text(candidate.content.parts[0].text if candidate.content.parts else resp.text), {
        "file": str(path),
        "bytes": len(payload),
        "mime": image_part.inline_data.mime_type,
    }



def test_tts() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    speech = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="callirrhoe")
        )
    )
    cfg = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=speech)
    resp = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents="請用親切的語氣說：大家好，歡迎來到我們的工作室！",
        config=cfg,
    )
    audio_part = next((p for p in first_candidate(resp).content.parts if p.inline_data), None)
    if audio_part is None:
        raise AssertionError("No audio inline_data returned")
    audio_bytes = ensure_bytes(audio_part.inline_data.data)
    mime = getattr(audio_part.inline_data, "mime_type", None)
    suffix = suffix_for_mime(mime)
    payload_kind = type(audio_part.inline_data.data).__name__
    output_path = OUTPUT_DIR / f"generated_tts{suffix}"
    output_path.write_bytes(audio_bytes)
    detail = f"Saved audio to {output_path}"
    return "pass", detail, {"file": str(output_path), "bytes": len(audio_bytes), "mime": mime, "payload_type": payload_kind}

def test_audio_transcription() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    wav = client.files.upload(file="meeting.wav")
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[wav, "請先逐字轉錄，再濃縮為 3 條行動項目"],
        )
        detail = safe_text(resp.text)
        return "pass", detail, {"uploaded_name": wav.name}
    finally:
        try:
            client.files.delete(name=wav.name)
        except ClientError as exc:
            record("audio_transcription_cleanup", "fail", detail=str(exc))


def test_thinking_mode() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=256)
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="請列出兩種番茄鐘增進效率的小技巧",
        config=cfg,
    )
    usage = resp.usage_metadata
    detail = safe_text(resp.text)
    tokens = usage.thoughts_token_count if usage else None
    return "pass", detail, {"thought_tokens": tokens}


def test_context_caching() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    base_sentence = "我們的公司致力於提供環保材質的可重複使用水瓶，重視循環經濟與永續環境。"
    long_text = " ".join(base_sentence + f" 重點{idx}" for idx in range(1, 400))
    cfg = types.CreateCachedContentConfig(
        display_name="doc-cache",
        ttl="600s",
        contents=[types.Content(role="user", parts=[types.Part(text=long_text)])],
    )
    try:
        cache = client.caches.create(model="gemini-2.5-flash", config=cfg)
    except ClientError as exc:
        raise BlockedTest(f"Context caching unavailable: {exc}")
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="根據快取內容列出兩點賣點",
            config=types.GenerateContentConfig(cached_content=cache.name),
        )
        detail = safe_text(resp.text)
        return "pass", detail, {"cache_name": cache.name}
    finally:
        try:
            client.caches.delete(name=cache.name)
        except ClientError as exc:
            record("context_cache_cleanup", "fail", detail=str(exc))


def test_batch_api() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    try:
        client.batches.create(
            model="gemini-2.5-flash",
            requests=[{"contents": "請摘要這段文字"}],  # type: ignore[arg-type]
        )
    except TypeError:
        raise BlockedTest("client.batches.create requires a src pointing to GCS or BigQuery; list-based input is unsupported")
    return "pass", "Batch API accepted inline requests (unexpected)", None


def test_live_api() -> Tuple[str, str, Optional[Dict[str, Any]]]:
    async def run_live() -> str:
        async with client.aio.live.connect(model="gemini-2.5-flash-live") as session:
            await session.send_client_content(
                turns=[types.Content(role="user", parts=[types.Part(text="用一句話跟我打招呼")])]
            )
            texts: list[str] = []
            async for message in session.receive():
                server = getattr(message, "server_content", None)
                if server and server.candidates:
                    for cand in server.candidates:
                        if cand.content and cand.content.parts:
                            for part in cand.content.parts:
                                if part.text:
                                    texts.append(part.text)
                    break
            return "".join(texts)
    try:
        text = asyncio.run(run_live())
    except Exception as exc:
        raise BlockedTest(f"Live API unavailable: {exc}")
    if not text:
        raise AssertionError("Live API returned empty response")
    return "pass", safe_text(text), None


TESTS = [
    ("basic_text_generation", test_basic_text),
    ("generation_config", test_generation_config),
    ("streaming", test_streaming),
    ("json_schema", test_json_schema),
    ("function_calling", test_function_calling),
    ("google_search", test_google_search),
    ("url_context", test_url_context),
    ("code_execution", test_code_execution),
    ("files_api", test_files_api),
    ("image_generation", test_image_generation),
    ("tts_generation", test_tts),
    ("audio_transcription", test_audio_transcription),
    ("thinking_mode", test_thinking_mode),
    ("context_caching", test_context_caching),
    ("batch_api", test_batch_api),
    ("live_api", test_live_api),
]

for name, func in TESTS:
    run_test(name, func)

summary_path = OUTPUT_DIR / "aistudio_tests_summary.json"
summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(results, ensure_ascii=False, indent=2))
print("\nSummary:")
for entry in results:
    detail = entry.get("detail", "")
    snippet = detail if len(detail) <= 120 else detail[:117] + "..."
    print(f"- [{entry['status']}] {entry['name']}: {snippet}")
