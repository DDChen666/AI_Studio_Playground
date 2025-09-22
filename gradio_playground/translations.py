from __future__ import annotations

from enum import Enum
from typing import Dict


class LanguageCode(str, Enum):
    EN = "en"
    ZH = "zh"


LANGUAGE_LABELS: Dict[LanguageCode, str] = {
    LanguageCode.EN: "English",
    LanguageCode.ZH: "中文",
}

LABEL_TO_LANGUAGE = {label: code for code, label in LANGUAGE_LABELS.items()}

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "language_selector_label": {
        "en": "Language",
        "zh": "語言",
    },
    "app_title_md": {
        "en": (
            "# AI Studio API Playground\n"
            "Interactively exercise Gemini models with live responses. "
            "Scenarios come from `test_outputs/aistudio_tests_summary.json`."
        ),
        "zh": (
            "# AI Studio API Playground\n"
            "透過互動介面即時體驗 Gemini 模型，範例情境來自 `test_outputs/aistudio_tests_summary.json`。"
        ),
    },
    "passing_scenarios_md": {
        "en": "**Passing scenarios detected:** {scenarios}",
        "zh": "**偵測到的通過情境：** {scenarios}",
    },
    "no_scenarios": {
        "en": "none",
        "zh": "無",
    },
    "prompt_label": {
        "en": "Prompt",
        "zh": "提示",
    },
    "system_instruction_label": {
        "en": "System instruction",
        "zh": "系統指示",
    },
    "model_label": {
        "en": "Model",
        "zh": "模型",
    },
    "temperature_label": {
        "en": "Temperature",
        "zh": "溫度",
    },
    "top_p_label": {
        "en": "Top-p",
        "zh": "Top-p",
    },
    "top_k_label": {
        "en": "Top-k",
        "zh": "Top-k",
    },
    "max_tokens_label": {
        "en": "Max output tokens",
        "zh": "最大輸出 Token 數",
    },
    "generate_button": {
        "en": "Generate",
        "zh": "生成",
    },
    "model_response_label": {
        "en": "Model response",
        "zh": "模型回應",
    },
    "usage_metadata_label": {
        "en": "Usage metadata",
        "zh": "使用量資訊",
    },
    "candidate_count_label": {
        "en": "Candidate count",
        "zh": "候選數",
    },
    "stop_sequences_label": {
        "en": "Stop sequences (one per line)",
        "zh": "停止序列（每行一個）",
    },
    "response_mime_type_label": {
        "en": "Response MIME type",
        "zh": "回應 MIME 類型",
    },
    "parsed_json_label": {
        "en": "Parsed JSON",
        "zh": "解析後的 JSON",
    },
    "parsed_json_optional_label": {
        "en": "Parsed JSON (when applicable)",
        "zh": "解析後的 JSON（若適用）",
    },
    "start_streaming_button": {
        "en": "Start streaming",
        "zh": "開始串流",
    },
    "chunks_label": {
        "en": "Chunks",
        "zh": "逐段回應",
    },
    "combined_text_label": {
        "en": "Combined text",
        "zh": "彙整文本",
    },
    "schema_prompt_label": {
        "en": "Prompt",
        "zh": "提示",
    },
    "response_schema_label": {
        "en": "Response schema (JSON)",
        "zh": "回應結構（JSON）",
    },
    "strict_mode_label": {
        "en": "Strict mode",
        "zh": "嚴格模式",
    },
    "generate_json_button": {
        "en": "Generate JSON",
        "zh": "生成 JSON",
    },
    "raw_json_label": {
        "en": "Raw JSON",
        "zh": "原始 JSON",
    },
    "user_prompt_label": {
        "en": "User prompt",
        "zh": "使用者提示",
    },
    "function_name_label": {
        "en": "Function name",
        "zh": "函式名稱",
    },
    "function_description_label": {
        "en": "Function description",
        "zh": "函式描述",
    },
    "function_parameters_label": {
        "en": "Function parameters (JSON Schema)",
        "zh": "函式參數（JSON Schema）",
    },
    "tool_response_label": {
        "en": "Tool response (JSON)",
        "zh": "工具回應（JSON）",
    },
    "trigger_function_call_button": {
        "en": "Trigger function call",
        "zh": "觸發函式呼叫",
    },
    "arguments_emitted_label": {
        "en": "Arguments emitted",
        "zh": "輸出參數",
    },
    "follow_up_response_label": {
        "en": "Follow-up model response",
        "zh": "後續模型回應",
    },
    "run_search_button": {
        "en": "Run search",
        "zh": "執行搜尋",
    },
    "max_sources_label": {
        "en": "Max sources",
        "zh": "來源上限",
    },
    "grounded_sources_label": {
        "en": "Grounded sources",
        "zh": "引用來源",
    },
    "run_code_button": {
        "en": "Run code",
        "zh": "執行程式",
    },
    "generated_code_label": {
        "en": "Generated code",
        "zh": "生成程式碼",
    },
    "execution_output_label": {
        "en": "Execution output",
        "zh": "執行輸出",
    },
    "upload_file_label": {
        "en": "Upload file",
        "zh": "上傳檔案",
    },
    "keep_remote_copy_label": {
        "en": "Keep remote copy",
        "zh": "保留雲端副本",
    },
    "send_to_model_button": {
        "en": "Send to model",
        "zh": "送出給模型",
    },
    "file_metadata_label": {
        "en": "File metadata",
        "zh": "檔案後設資料",
    },
    "style_hint_label": {
        "en": "Style hint",
        "zh": "風格提示",
    },
    "width_label": {
        "en": "Width",
        "zh": "寬度",
    },
    "height_label": {
        "en": "Height",
        "zh": "高度",
    },
    "generate_image_button": {
        "en": "Generate image",
        "zh": "生成圖像",
    },
    "preview_label": {
        "en": "Preview",
        "zh": "預覽",
    },
    "download_image_label": {
        "en": "Download image",
        "zh": "下載圖像",
    },
    "text_label": {
        "en": "Text",
        "zh": "文字",
    },
    "voice_name_label": {
        "en": "Voice name",
        "zh": "語音名稱",
    },
    "synthesize_button": {
        "en": "Synthesize",
        "zh": "合成",
    },
    "download_audio_label": {
        "en": "Download audio",
        "zh": "下載音訊",
    },
    "audio_file_label": {
        "en": "Audio file",
        "zh": "音訊檔案",
    },
    "transcript_label": {
        "en": "Transcript",
        "zh": "逐字稿",
    },
    "metadata_label": {
        "en": "Metadata",
        "zh": "中繼資料",
    },
    "transcribe_button": {
        "en": "Transcribe",
        "zh": "轉錄",
    },
    "thinking_budget_label": {
        "en": "Thinking budget (tokens)",
        "zh": "思考預算（Tokens）",
    },
    "tab_text_generation": {
        "en": "Text Generation",
        "zh": "文本生成",
    },
    "tab_generation_config": {
        "en": "Generation Config",
        "zh": "生成設定",
    },
    "tab_streaming": {
        "en": "Streaming",
        "zh": "串流",
    },
    "tab_json_schema": {
        "en": "JSON Schema",
        "zh": "JSON 結構",
    },
    "tab_function_calling": {
        "en": "Function Calling",
        "zh": "函式呼叫",
    },
    "tab_google_search": {
        "en": "Google Search",
        "zh": "Google 搜尋",
    },
    "tab_code_execution": {
        "en": "Code Execution",
        "zh": "程式執行",
    },
    "tab_files_api": {
        "en": "Files API",
        "zh": "檔案 API",
    },
    "tab_image_generation": {
        "en": "Image Generation",
        "zh": "圖像生成",
    },
    "tab_text_to_speech": {
        "en": "Text to Speech",
        "zh": "文字轉語音",
    },
    "tab_audio_transcription": {
        "en": "Audio Transcription",
        "zh": "音訊轉錄",
    },
    "tab_thinking_mode": {
        "en": "Thinking Mode",
        "zh": "Thinking 模式",
    },
    "search_prompt_default": {
        "en": "Who won Euro 2024?",
        "zh": "誰拿下了 2024 年歐洲國家盃冠軍？",
    },
    "stream_prompt_default": {
        "en": "List the differences between CPU, GPU, and TPU.",
        "zh": "條列說明 CPU、GPU 與 TPU 的差異",
    },
    "schema_prompt_default": {
        "en": "Define a JSON schema for a CLI tool configuration.",
        "zh": "建立一個 CLI 工具的設定 JSON 結構。",
    },
    "function_prompt_default": {
        "en": "I want to know today's weather in Taipei.",
        "zh": "我想知道台北今天的天氣。",
    },
    "function_description_default": {
        "en": "Look up the weather for a specific city.",
        "zh": "查詢指定城市的天氣資訊。",
    },
    "code_prompt_default": {
        "en": "Write a Python script that prints the even numbers between 1 and 10.",
        "zh": "撰寫一個 Python 程式，列出 1 到 10 之間的偶數。",
    },
    "file_instructions_default": {
        "en": "Summarize three highlights from the uploaded file.",
        "zh": "請摘要檔案中的三個重點。",
    },
    "tts_text_default": {
        "en": "Welcome to the AI Studio Playground!",
        "zh": "歡迎使用 AI Studio Playground！",
    },
    "audio_prompt_default": {
        "en": "Please transcribe the key points from this meeting recording.",
        "zh": "請轉錄這段會議錄音的重點。",
    },
    "thinking_prompt_default": {
        "en": "Design an agenda for an educational workshop.",
        "zh": "請規劃一個教育工作坊的活動流程。",
    },
}


def get_text(key: str, language: LanguageCode) -> str:
    lang_key = language.value if isinstance(language, LanguageCode) else str(language)
    if key not in TRANSLATIONS:
        raise KeyError(f"Missing translation key: {key}")
    bucket = TRANSLATIONS[key]
    if lang_key not in bucket:
        raise KeyError(f"Missing translation for key '{key}' and language '{lang_key}'.")
    return bucket[lang_key]


def bilingual_tab_label(key: str) -> str:
    return f"{get_text(key, LanguageCode.EN)} / {get_text(key, LanguageCode.ZH)}"


__all__ = [
    "LanguageCode",
    "LANGUAGE_LABELS",
    "LABEL_TO_LANGUAGE",
    "TRANSLATIONS",
    "bilingual_tab_label",
    "get_text",
]
