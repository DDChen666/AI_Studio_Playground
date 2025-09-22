from __future__ import annotations

import json
from typing import Callable, Dict, List, Tuple

import gradio as gr

from .api_calls import (
    run_audio_transcription,
    run_code_execution,
    run_files_api,
    run_function_calling,
    run_generation_with_config,
    run_google_search,
    run_image_generation,
    run_json_schema,
    run_streaming,
    run_text_generation,
    run_thinking_mode,
    run_tts,
)
from .config import SAMPLE_AUDIO, SAMPLE_REPORT
from .translations import (
    LABEL_TO_LANGUAGE,
    LANGUAGE_LABELS,
    LanguageCode,
    bilingual_tab_label,
    get_text,
)
from .utils import PASSED_TESTS, get_data_value, get_detail

LanguageBinding = Tuple[
    gr.components.Component,
    Dict[str, str],
    Dict[str, Callable[[LanguageCode], Dict[str, str]]],
]


def build_demo() -> gr.Blocks:
    default_language = LanguageCode.ZH
    passed_ids = sorted(PASSED_TESTS)

    default_text_model = get_data_value("basic_text_generation", "model", "gemini-2.5-flash")
    default_config_model = get_data_value("generation_config", "model", "gemini-2.5-pro")
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

    language_bindings: List[LanguageBinding] = []

    def register(
        component: gr.components.Component,
        fields: Dict[str, str],
        formatters: Dict[str, Callable[[LanguageCode], Dict[str, str]]] | None = None,
    ) -> None:
        language_bindings.append((component, fields, formatters or {}))

    def scenarios_formatter(language: LanguageCode) -> Dict[str, str]:
        if passed_ids:
            scenarios = ", ".join(passed_ids)
        else:
            scenarios = get_text("no_scenarios", language)
        return {"scenarios": scenarios}

    with gr.Blocks(title="AI Studio API Playground", theme="soft") as demo:
        language_selector = gr.Radio(
            choices=[LANGUAGE_LABELS[LanguageCode.EN], LANGUAGE_LABELS[LanguageCode.ZH]],
            value=LANGUAGE_LABELS[default_language],
            label=get_text("language_selector_label", default_language),
        )
        register(language_selector, {"label": "language_selector_label"})

        header_md = gr.Markdown(get_text("app_title_md", default_language))
        register(header_md, {"value": "app_title_md"})

        initial_scenarios = (
            ", ".join(passed_ids)
            if passed_ids
            else get_text("no_scenarios", default_language)
        )
        scenarios_md = gr.Markdown(
            get_text("passing_scenarios_md", default_language).format(scenarios=initial_scenarios)
        )
        register(
            scenarios_md,
            {"value": "passing_scenarios_md"},
            {"value": scenarios_formatter},
        )

        with gr.Tabs():
            with gr.Tab(bilingual_tab_label("tab_text_generation")):
                prompt = gr.Textbox(
                    label=get_text("prompt_label", default_language),
                    value=get_detail("basic_text_generation"),
                    lines=6,
                )
                register(prompt, {"label": "prompt_label"})

                system_instruction = gr.Textbox(
                    label=get_text("system_instruction_label", default_language),
                    lines=3,
                )
                register(system_instruction, {"label": "system_instruction_label"})

                model = gr.Dropdown(
                    [default_text_model, "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"],
                    label=get_text("model_label", default_language),
                    value=default_text_model,
                )
                register(model, {"label": "model_label"})

                with gr.Row():
                    temperature = gr.Slider(
                        0.0,
                        2.0,
                        value=0.6,
                        step=0.1,
                        label=get_text("temperature_label", default_language),
                    )
                    register(temperature, {"label": "temperature_label"})

                    top_p = gr.Slider(
                        0.0,
                        1.0,
                        value=0.95,
                        step=0.05,
                        label=get_text("top_p_label", default_language),
                    )
                    register(top_p, {"label": "top_p_label"})

                with gr.Row():
                    top_k = gr.Slider(
                        1,
                        200,
                        value=40,
                        step=1,
                        label=get_text("top_k_label", default_language),
                    )
                    register(top_k, {"label": "top_k_label"})

                    max_tokens = gr.Slider(
                        16,
                        4096,
                        value=1024,
                        step=16,
                        label=get_text("max_tokens_label", default_language),
                    )
                    register(max_tokens, {"label": "max_tokens_label"})

                text_btn = gr.Button(get_text("generate_button", default_language))
                register(text_btn, {"value": "generate_button"})

                text_output = gr.Textbox(
                    label=get_text("model_response_label", default_language),
                    lines=10,
                )
                register(text_output, {"label": "model_response_label"})

                text_usage = gr.JSON(label=get_text("usage_metadata_label", default_language))
                register(text_usage, {"label": "usage_metadata_label"})

                text_btn.click(
                    run_text_generation,
                    inputs=[prompt, system_instruction, model, temperature, top_p, top_k, max_tokens],
                    outputs=[text_output, text_usage],
                )

            with gr.Tab(bilingual_tab_label("tab_generation_config")):
                prompt_cfg = gr.Textbox(
                    label=get_text("prompt_label", default_language),
                    value=get_text("generation_prompt_default", default_language),
                    lines=6,
                )
                register(
                    prompt_cfg,
                    {"label": "prompt_label", "value": "generation_prompt_default"},
                )

                model_cfg = gr.Dropdown(
                    [default_config_model, "gemini-2.5-flash", "gemini-1.5-pro"],
                    label=get_text("model_label", default_language),
                    value=default_config_model,
                )
                register(model_cfg, {"label": "model_label"})

                with gr.Row():
                    cfg_temperature = gr.Slider(
                        0.0,
                        2.0,
                        value=0.7,
                        step=0.1,
                        label=get_text("temperature_label", default_language),
                    )
                    register(cfg_temperature, {"label": "temperature_label"})

                    cfg_top_p = gr.Slider(
                        0.0,
                        1.0,
                        value=0.9,
                        step=0.05,
                        label=get_text("top_p_label", default_language),
                    )
                    register(cfg_top_p, {"label": "top_p_label"})

                with gr.Row():
                    cfg_top_k = gr.Slider(
                        1,
                        200,
                        value=32,
                        step=1,
                        label=get_text("top_k_label", default_language),
                    )
                    register(cfg_top_k, {"label": "top_k_label"})

                    cfg_max_tokens = gr.Slider(
                        16,
                        4096,
                        value=512,
                        step=16,
                        label=get_text("max_tokens_label", default_language),
                    )
                    register(cfg_max_tokens, {"label": "max_tokens_label"})

                candidate_count = gr.Slider(
                    1,
                    4,
                    value=1,
                    step=1,
                    label=get_text("candidate_count_label", default_language),
                )
                register(candidate_count, {"label": "candidate_count_label"})

                stop_sequences = gr.Textbox(
                    label=get_text("stop_sequences_label", default_language),
                )
                register(stop_sequences, {"label": "stop_sequences_label"})

                response_mime_type = gr.Dropdown(
                    ["", "text/plain", "application/json"],
                    label=get_text("response_mime_type_label", default_language),
                    value="",
                )
                register(response_mime_type, {"label": "response_mime_type_label"})

                cfg_btn = gr.Button(get_text("generate_button", default_language))
                register(cfg_btn, {"value": "generate_button"})

                cfg_text = gr.Textbox(
                    label=get_text("model_response_label", default_language),
                    lines=10,
                )
                register(cfg_text, {"label": "model_response_label"})

                cfg_json = gr.JSON(label=get_text("parsed_json_optional_label", default_language))
                register(cfg_json, {"label": "parsed_json_optional_label"})

                cfg_usage = gr.JSON(label=get_text("usage_metadata_label", default_language))
                register(cfg_usage, {"label": "usage_metadata_label"})

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

            with gr.Tab(bilingual_tab_label("tab_streaming")):
                stream_prompt = gr.Textbox(
                    label=get_text("prompt_label", default_language),
                    value=get_text("stream_prompt_default", default_language),
                    lines=5,
                )
                register(
                    stream_prompt,
                    {"label": "prompt_label", "value": "stream_prompt_default"},
                )

                stream_model = gr.Dropdown(
                    ["gemini-2.5-flash", "gemini-2.5-pro"],
                    label=get_text("model_label", default_language),
                    value="gemini-2.5-flash",
                )
                register(stream_model, {"label": "model_label"})

                stream_btn = gr.Button(get_text("start_streaming_button", default_language))
                register(stream_btn, {"value": "start_streaming_button"})

                stream_chunks = gr.JSON(label=get_text("chunks_label", default_language))
                register(stream_chunks, {"label": "chunks_label"})

                stream_text = gr.Textbox(
                    label=get_text("combined_text_label", default_language),
                    lines=10,
                )
                register(stream_text, {"label": "combined_text_label"})

                stream_btn.click(
                    run_streaming,
                    inputs=[stream_prompt, stream_model],
                    outputs=[stream_chunks, stream_text],
                )

            with gr.Tab(bilingual_tab_label("tab_json_schema")):
                schema_prompt = gr.Textbox(
                    label=get_text("schema_prompt_label", default_language),
                    value=get_text("schema_prompt_default", default_language),
                    lines=4,
                )
                register(
                    schema_prompt,
                    {"label": "schema_prompt_label", "value": "schema_prompt_default"},
                )

                schema_box = gr.Textbox(
                    label=get_text("response_schema_label", default_language),
                    value=default_schema,
                    lines=12,
                )
                register(schema_box, {"label": "response_schema_label"})

                schema_model = gr.Dropdown(
                    ["gemini-2.5-flash", "gemini-2.5-pro"],
                    value="gemini-2.5-flash",
                    label=get_text("model_label", default_language),
                )
                register(schema_model, {"label": "model_label"})

                schema_strict = gr.Checkbox(
                    label=get_text("strict_mode_label", default_language),
                    value=True,
                )
                register(schema_strict, {"label": "strict_mode_label"})

                schema_btn = gr.Button(get_text("generate_json_button", default_language))
                register(schema_btn, {"value": "generate_json_button"})

                schema_text = gr.Textbox(
                    label=get_text("raw_json_label", default_language),
                    lines=10,
                )
                register(schema_text, {"label": "raw_json_label"})

                schema_parsed = gr.JSON(label=get_text("parsed_json_label", default_language))
                register(schema_parsed, {"label": "parsed_json_label"})

                schema_usage = gr.JSON(label=get_text("usage_metadata_label", default_language))
                register(schema_usage, {"label": "usage_metadata_label"})

                schema_btn.click(
                    run_json_schema,
                    inputs=[schema_prompt, schema_box, schema_strict, schema_model],
                    outputs=[schema_text, schema_parsed, schema_usage],
                )

            with gr.Tab(bilingual_tab_label("tab_function_calling")):
                func_prompt = gr.Textbox(
                    label=get_text("user_prompt_label", default_language),
                    value=get_text("function_prompt_default", default_language),
                    lines=4,
                )
                register(
                    func_prompt,
                    {"label": "user_prompt_label", "value": "function_prompt_default"},
                )

                func_model = gr.Dropdown(
                    ["gemini-2.5-flash", "gemini-2.5-pro"],
                    value="gemini-2.5-flash",
                    label=get_text("model_label", default_language),
                )
                register(func_model, {"label": "model_label"})

                func_name = gr.Textbox(
                    label=get_text("function_name_label", default_language),
                    value="get_weather",
                )
                register(func_name, {"label": "function_name_label"})

                func_description = gr.Textbox(
                    label=get_text("function_description_label", default_language),
                    value=get_text("function_description_default", default_language),
                    lines=2,
                )
                register(
                    func_description,
                    {"label": "function_description_label", "value": "function_description_default"},
                )

                func_params = gr.Textbox(
                    label=get_text("function_parameters_label", default_language),
                    value=default_function_schema,
                    lines=8,
                )
                register(func_params, {"label": "function_parameters_label"})

                func_tool_response = gr.Textbox(
                    label=get_text("tool_response_label", default_language),
                    lines=6,
                    value="{\n  \"temp_c\": 27,\n  \"condition\": \"Cloudy\"\n}",
                )
                register(func_tool_response, {"label": "tool_response_label"})

                func_btn = gr.Button(get_text("trigger_function_call_button", default_language))
                register(func_btn, {"value": "trigger_function_call_button"})

                func_args = gr.Textbox(
                    label=get_text("arguments_emitted_label", default_language),
                    lines=6,
                )
                register(func_args, {"label": "arguments_emitted_label"})

                func_follow = gr.Textbox(
                    label=get_text("follow_up_response_label", default_language),
                    lines=8,
                )
                register(func_follow, {"label": "follow_up_response_label"})

                func_usage = gr.JSON(label=get_text("usage_metadata_label", default_language))
                register(func_usage, {"label": "usage_metadata_label"})

                func_btn.click(
                    run_function_calling,
                    inputs=[
                        func_prompt,
                        func_model,
                        func_name,
                        func_description,
                        func_params,
                        func_tool_response,
                    ],
                    outputs=[func_args, func_follow, func_usage],
                )
            with gr.Tab(bilingual_tab_label("tab_google_search")):
                search_prompt = gr.Textbox(
                    label=get_text("prompt_label", default_language),
                    value=get_text("search_prompt_default", default_language),
                    lines=3,
                )
                register(
                    search_prompt,
                    {"label": "prompt_label", "value": "search_prompt_default"},
                )

                search_model = gr.Dropdown(
                    ["gemini-2.5-flash", "gemini-2.5-pro"],
                    value="gemini-2.5-flash",
                    label=get_text("model_label", default_language),
                )
                register(search_model, {"label": "model_label"})

                max_results = gr.Slider(
                    1,
                    10,
                    value=5,
                    step=1,
                    label=get_text("max_sources_label", default_language),
                )
                register(max_results, {"label": "max_sources_label"})

                search_btn = gr.Button(get_text("run_search_button", default_language))
                register(search_btn, {"value": "run_search_button"})

                search_text = gr.Textbox(
                    label=get_text("model_response_label", default_language),
                    lines=8,
                )
                register(search_text, {"label": "model_response_label"})

                search_sources = gr.JSON(label=get_text("grounded_sources_label", default_language))
                register(search_sources, {"label": "grounded_sources_label"})

                search_usage = gr.JSON(label=get_text("usage_metadata_label", default_language))
                register(search_usage, {"label": "usage_metadata_label"})

                search_btn.click(
                    run_google_search,
                    inputs=[search_prompt, search_model, max_results],
                    outputs=[search_text, search_sources, search_usage],
                )

            with gr.Tab(bilingual_tab_label("tab_code_execution")):
                code_prompt = gr.Textbox(
                    label=get_text("prompt_label", default_language),
                    value=get_text("code_prompt_default", default_language),
                    lines=4,
                )
                register(
                    code_prompt,
                    {"label": "prompt_label", "value": "code_prompt_default"},
                )

                code_model = gr.Dropdown(
                    ["gemini-2.5-flash", "gemini-2.5-pro"],
                    value="gemini-2.5-flash",
                    label=get_text("model_label", default_language),
                )
                register(code_model, {"label": "model_label"})

                code_btn = gr.Button(get_text("run_code_button", default_language))
                register(code_btn, {"value": "run_code_button"})

                code_response = gr.Textbox(
                    label=get_text("model_response_label", default_language),
                    lines=6,
                )
                register(code_response, {"label": "model_response_label"})

                code_generated = gr.Code(
                    label=get_text("generated_code_label", default_language),
                    language="python",
                )
                register(code_generated, {"label": "generated_code_label"})

                code_output = gr.Textbox(
                    label=get_text("execution_output_label", default_language),
                    lines=6,
                )
                register(code_output, {"label": "execution_output_label"})

                code_usage = gr.JSON(label=get_text("usage_metadata_label", default_language))
                register(code_usage, {"label": "usage_metadata_label"})

                code_btn.click(
                    run_code_execution,
                    inputs=[code_prompt, code_model],
                    outputs=[code_response, code_generated, code_output, code_usage],
                )

            with gr.Tab(bilingual_tab_label("tab_files_api")):
                file_input = gr.File(
                    label=get_text("upload_file_label", default_language),
                    value=SAMPLE_REPORT,
                )
                register(file_input, {"label": "upload_file_label"})

                file_instructions = gr.Textbox(
                    label=get_text("prompt_label", default_language),
                    value=get_text("file_instructions_default", default_language),
                    lines=4,
                )
                register(
                    file_instructions,
                    {"label": "prompt_label", "value": "file_instructions_default"},
                )

                file_model = gr.Dropdown(
                    ["gemini-2.5-pro", "gemini-2.5-flash"],
                    value="gemini-2.5-pro",
                    label=get_text("model_label", default_language),
                )
                register(file_model, {"label": "model_label"})

                keep_remote = gr.Checkbox(
                    label=get_text("keep_remote_copy_label", default_language),
                    value=False,
                )
                register(keep_remote, {"label": "keep_remote_copy_label"})

                file_btn = gr.Button(get_text("send_to_model_button", default_language))
                register(file_btn, {"value": "send_to_model_button"})

                file_response = gr.Textbox(
                    label=get_text("model_response_label", default_language),
                    lines=8,
                )
                register(file_response, {"label": "model_response_label"})

                file_meta = gr.JSON(label=get_text("file_metadata_label", default_language))
                register(file_meta, {"label": "file_metadata_label"})

                file_btn.click(
                    run_files_api,
                    inputs=[file_input, file_instructions, file_model, keep_remote],
                    outputs=[file_response, file_meta],
                )

            with gr.Tab(bilingual_tab_label("tab_image_generation")):
                image_prompt = gr.Textbox(
                    label=get_text("prompt_label", default_language),
                    value="Watercolor reading nook with warm morning light",
                    lines=4,
                )
                register(image_prompt, {"label": "prompt_label"})

                image_model = gr.Dropdown(
                    ["gemini-2.0-flash-preview-image-generation"],
                    value="gemini-2.0-flash-preview-image-generation",
                    label=get_text("model_label", default_language),
                )
                register(image_model, {"label": "model_label"})

                style_hint = gr.Textbox(
                    label=get_text("style_hint_label", default_language),
                    value="watercolor",
                )
                register(style_hint, {"label": "style_hint_label"})

                with gr.Row():
                    width = gr.Slider(
                        256,
                        2048,
                        value=1024,
                        step=64,
                        label=get_text("width_label", default_language),
                    )
                    register(width, {"label": "width_label"})

                    height = gr.Slider(
                        256,
                        2048,
                        value=1024,
                        step=64,
                        label=get_text("height_label", default_language),
                    )
                    register(height, {"label": "height_label"})

                image_btn = gr.Button(get_text("generate_image_button", default_language))
                register(image_btn, {"value": "generate_image_button"})

                image_text = gr.Textbox(
                    label=get_text("model_response_label", default_language),
                    lines=6,
                )
                register(image_text, {"label": "model_response_label"})

                image_preview = gr.Image(label=get_text("preview_label", default_language))
                register(image_preview, {"label": "preview_label"})

                image_file = gr.File(label=get_text("download_image_label", default_language))
                register(image_file, {"label": "download_image_label"})

                image_btn.click(
                    run_image_generation,
                    inputs=[image_prompt, image_model, style_hint, width, height],
                    outputs=[image_text, image_preview, image_file],
                )

            with gr.Tab(bilingual_tab_label("tab_text_to_speech")):
                tts_text = gr.Textbox(
                    label=get_text("text_label", default_language),
                    value=get_text("tts_text_default", default_language),
                    lines=4,
                )
                register(
                    tts_text,
                    {"label": "text_label", "value": "tts_text_default"},
                )

                tts_model = gr.Dropdown(
                    ["gemini-2.5-flash-preview-tts"],
                    value="gemini-2.5-flash-preview-tts",
                    label=get_text("model_label", default_language),
                )
                register(tts_model, {"label": "model_label"})

                tts_voice = gr.Textbox(
                    label=get_text("voice_name_label", default_language),
                    value="callirrhoe",
                )
                register(tts_voice, {"label": "voice_name_label"})

                tts_btn = gr.Button(get_text("synthesize_button", default_language))
                register(tts_btn, {"value": "synthesize_button"})

                tts_response = gr.Textbox(
                    label=get_text("model_response_label", default_language),
                    lines=4,
                )
                register(tts_response, {"label": "model_response_label"})

                tts_audio = gr.Audio(
                    label=get_text("preview_label", default_language),
                    type="filepath",
                    interactive=False,
                )
                register(tts_audio, {"label": "preview_label"})

                tts_file = gr.File(label=get_text("download_audio_label", default_language))
                register(tts_file, {"label": "download_audio_label"})

                tts_usage = gr.JSON(label=get_text("usage_metadata_label", default_language))
                register(tts_usage, {"label": "usage_metadata_label"})

                tts_btn.click(
                    run_tts,
                    inputs=[tts_text, tts_model, tts_voice],
                    outputs=[tts_response, tts_audio, tts_file, tts_usage],
                )
            with gr.Tab(bilingual_tab_label("tab_audio_transcription")):
                audio_input = gr.File(
                    label=get_text("audio_file_label", default_language),
                    value=SAMPLE_AUDIO,
                )
                register(audio_input, {"label": "audio_file_label"})

                audio_prompt = gr.Textbox(
                    label=get_text("prompt_label", default_language),
                    value=get_text("audio_prompt_default", default_language),
                    lines=3,
                )
                register(
                    audio_prompt,
                    {"label": "prompt_label", "value": "audio_prompt_default"},
                )

                audio_model = gr.Dropdown(
                    ["gemini-2.5-pro", "gemini-2.5-flash"],
                    value="gemini-2.5-pro",
                    label=get_text("model_label", default_language),
                )
                register(audio_model, {"label": "model_label"})

                audio_btn = gr.Button(get_text("transcribe_button", default_language))
                register(audio_btn, {"value": "transcribe_button"})

                audio_text = gr.Textbox(
                    label=get_text("transcript_label", default_language),
                    lines=10,
                )
                register(audio_text, {"label": "transcript_label"})

                audio_meta = gr.JSON(label=get_text("metadata_label", default_language))
                register(audio_meta, {"label": "metadata_label"})

                audio_btn.click(
                    run_audio_transcription,
                    inputs=[audio_input, audio_prompt, audio_model],
                    outputs=[audio_text, audio_meta],
                )

            with gr.Tab(bilingual_tab_label("tab_thinking_mode")):
                thinking_prompt = gr.Textbox(
                    label=get_text("prompt_label", default_language),
                    value=get_text("thinking_prompt_default", default_language),
                    lines=3,
                )
                register(
                    thinking_prompt,
                    {"label": "prompt_label", "value": "thinking_prompt_default"},
                )

                thinking_model = gr.Dropdown(
                    ["gemini-2.5-flash", "gemini-2.5-pro"],
                    value="gemini-2.5-flash",
                    label=get_text("model_label", default_language),
                )
                register(thinking_model, {"label": "model_label"})

                default_budget = get_data_value("thinking_mode", "thought_tokens", 256) or 256
                thinking_budget = gr.Slider(
                    32,
                    2048,
                    value=default_budget,
                    step=32,
                    label=get_text("thinking_budget_label", default_language),
                )
                register(thinking_budget, {"label": "thinking_budget_label"})

                thinking_btn = gr.Button(get_text("generate_button", default_language))
                register(thinking_btn, {"value": "generate_button"})

                thinking_text = gr.Textbox(
                    label=get_text("model_response_label", default_language),
                    lines=10,
                )
                register(thinking_text, {"label": "model_response_label"})

                thinking_usage = gr.JSON(label=get_text("usage_metadata_label", default_language))
                register(thinking_usage, {"label": "usage_metadata_label"})

                thinking_btn.click(
                    run_thinking_mode,
                    inputs=[thinking_prompt, thinking_model, thinking_budget],
                    outputs=[thinking_text, thinking_usage],
                )

        components = [component for component, _, _ in language_bindings]

        def apply_language(selection: str) -> List[gr.update]:
            language = LABEL_TO_LANGUAGE.get(selection, default_language)
            updates: List[gr.update] = []
            for component, fields, formatters in language_bindings:
                update_kwargs: Dict[str, str] = {}
                for field, key in fields.items():
                    value = get_text(key, language)
                    formatter = formatters.get(field)
                    if callable(formatter):
                        extra = formatter(language)
                        value = value.format(**extra)
                    update_kwargs[field] = value
                updates.append(gr.update(**update_kwargs))
            return updates

        language_selector.change(
            fn=apply_language,
            inputs=language_selector,
            outputs=components,
        )

    return demo


__all__ = ["build_demo"]
