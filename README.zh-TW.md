---
title: AI Studio API Playground
emoji: 🎛️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# AI Studio API Playground

繁體中文 | [English](README.md)

## 專案簡介
AI Studio API Playground 是一套模組化的 Gradio 應用程式，協助您以互動方式
體驗 Google Gemini 模型。專案提供可重用的後端工具、支援中英雙語的使用者介面，
並以清晰的設定界線打造適合正式環境的實驗平台。

## 主要特色
- **中英雙語切換：** 介面內建語言切換器，並以 `translations_map.json`
  做為所有字詞的集中管理。
- **模組化架構：** `gradio_playground/` 套件內分層實作 UI、API 邏輯、
  工具函式與設定管理。
- **情境範例：** 自動讀取 `test_outputs/aistudio_tests_summary.json`，
  將測試案例結果帶入範例欄位。
- **資產管理：** 產生的檔案統一儲存於 `playground_outputs/`，並提供 PCM
  音訊轉換為 WAV 的工具函式。

## 專案結構
```
gradio_playground/
├── __init__.py
├── api_calls.py        # Gemini API 呼叫邏輯
├── config.py           # 環境變數載入與客戶端初始化
├── main.py             # 應用程式進入點
├── translations.py     # 雙語文字載入工具
├── translations_map.json
├── ui.py               # Gradio Blocks 介面 (build_demo)
├── utils.py            # 共用工具函式
└── .env                # 本機環境變數範本（請勿提交正式金鑰）
```
其他重要檔案：
- `gradio_api_playground.py`：向下相容的啟動腳本，匯入模組化實作。
- `tests/run_aistudio_tests.py`：自動化回歸測試腳本。

## 系統需求
- Python 3.10 以上（以 CPython 3.13.x 驗證）。
- 具有目標模型存取權的 Gemini API 金鑰。
- 用於安裝套件的 `pip`。

## 快速上手
1. **取得並進入專案。**
   ```bash
   git clone <repo-url>
   cd AI_Studio_Playground
   ```
2. **建立並啟用虛擬環境。**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 請使用 .venv\Scripts\Activate.ps1
   ```
3. **安裝依賴套件。**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
#### 4. **設定環境變數**

本應用程式需要 Google Gemini API 金鑰才能運作。

**A) 本地開發設定：**

1.  在專案的根目錄下，手動建立一個名為 `.env` 的檔案。
2.  將您的 API 金鑰加入到這個檔案中。應用程式會自動識別 `GOOGLE_API_KEY`。

    ```
    # 在你的 .env 檔案中
    GOOGLE_API_KEY="貼上你真實的 API 金鑰"
    ```
應用程式使用 `python-dotenv` 套件，在本地端執行時會自動載入此金鑰。

**B) Hugging Face Space 部署設定：**

部署時**請勿**上傳您的 `.env` 檔案。您必須使用 Hugging Face 內建的安全儲存功能來設定金鑰：

1.  前往您 Space 的 **Settings** 分頁。
2.  找到 **Repository secrets** 區塊。
3.  點擊 **New secret** 並新增：
    *   **Name (名稱):** `GOOGLE_API_KEY`
    *   **Value (值):** 貼上您真實的 Google Gemini API 金鑰。

Space 在執行時會自動將此 Secret 載入為環境變數。

## 啟動 Gradio Playground
可使用下列任一指令啟動介面：
```bash
python -m gradio_playground.main
# 或
python gradio_api_playground.py
```
Gradio 會啟用背景佇列並開啟本機介面，所有產出檔案將寫入專案根目錄的
`playground_outputs/`。

## 自動化測試
`tests/run_aistudio_tests.py` 提供對 `test_outputs/aistudio_tests_summary.json`
情境的快速驗證：
```bash
GEMINI_API_KEY=你的真實金鑰 python tests/run_aistudio_tests.py
# 或
GOOGLE_API_KEY=你的真實金鑰 python tests/run_aistudio_tests.py
```

## 部署至 Hugging Face Spaces
1. **建立新的 Gradio Space。** 選擇「Gradio」SDK，並視需求設定公開或私人。
2. **推送專案檔案。** `app.py` 已曝光 Gradio 所需的 `demo` 物件，
   `requirements.txt` 則列出執行時需要的套件。
3. **設定密鑰。** 進入「Settings → Secrets」，新增 `GOOGLE_API_KEY`
   （或其他支援的變數名稱）並填入 Gemini API 金鑰。
4. **選擇硬體（可選）。** 免費 CPU 即可應付一般測試，需確保 Space 具有外網連線
   以便呼叫 Gemini API。
5. **觸發建置。** Spaces 會在重新啟動時依 `requirements.txt` 安裝套件，並執行
   `app.py` 載入 Gradio 應用。
6. **驗證介面。** 測試多個分頁、確認語言切換正常，以及輸出檔案是否寫入
   `playground_outputs/` 目錄。

## 在地化流程
- 所有介面字詞集中於 `gradio_playground/translations_map.json`。
- 新增元件時請先在 JSON 檔新增 key，再於 `gradio_playground/translations.py`
  透過 `get_text` 讀取。
- 在 `ui.py` 中註冊元件，確保語言切換時會同步更新文字。

## 支援
若遇到問題或希望新增功能，請於 GitHub 發 issue，並提供現象描述、預期結果
與重現步驟。
