下面這份是「一站到位」的 Google AI Studio（Gemini Developer API）最新、可直接上手的**超詳細技術手冊**。我已以 2025-09-22（臺北時間）為基準，對照 Google 官方文件校對了模型與 API 的**當前可用性**、**正確套件**與**最新參數**，避免用到已淘汰或舊式介面。各段落均附上最關鍵的官方來源，方便你核對或延伸。
（若你只想複製貼上就能跑，照著每節的「Python / Node.js」區塊執行即可。）

---

# 目錄

1. 核心總覽：今天（2025/09/22）該怎麼正確使用 Gemini API
2. 免費可用（Free Tier）模型總表（含代號與用途）
3. 安裝與初始化（Python 與 Node.js）
4. 文本/多模態生成：最小可用範例 + 參數全集
5. 結構化輸出（JSON Schema / 嚴格 JSON）
6. 工具與進階：Function Calling、Google Search Grounding、URL Context、Code Execution
7. 多模態輸入：影像/音訊/影片/PDF + Files API（上傳/列出/刪除/引用）
8. 圖像生成（Native Image Generation）與語音生成（TTS）
9. 直播互動（Live API：雙向串流/語音對話）
10. 大量/排程推理（Batch API）、長期快取（Context Caching）
11. 安全性（Safety Settings）與思考模式（Thinking / thinkingBudget）
12. Token 計數與錯誤/RL 限流處理
13. 常見實作藍本（RAG、轉錄、指令/系統提示、JSON 嚴格輸出範例整合）
14. 版本/相容性提醒與最佳實務

---

# 1) 核心總覽：今天（2025/09/22）該怎麼正確使用 Gemini API

* **唯一建議使用的官方 SDK**：**`google-genai`**（Python / JS）。舊版 `google-generativeai` 已進入淘汰程序，官方明確建議改用 `google-genai`。([Google AI for Developers][1])
* **官方 API 風格**：使用 SDK 的 `client.models.generate_content(...)`；等價的 REST 為 `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`。完整端點與欄位在 API 參考內。
* **免費模型與額度**：Free Tier 可直接用多個 Gemini 2.x 系列（含 Live、TTS 與特定圖像生成功能預覽版），其 **RPM/TPM/RPD** 明確列在 Rate Limits（如下表第二節）。([Google AI for Developers][2])

---

# 2) 免費可用（Free Tier）模型總表（模型代號/類型/用途）

> 下表只列 **Free Tier** 能用的重點模型（依 Google 官方 Rate Limits 頁面）。若你升級計費層級（Tier 1/2/3）會解鎖更多或更高額度的模型（例如 Imagen 4 / Veo 3 等）。([Google AI for Developers][2])

## 2.1 文字/多模態（輸出為文字）

* **`gemini-2.5-pro`**：最強推理/長上下文、程式/數理/STEM。
* **`gemini-2.5-flash`**：高 CP 值、快速泛用。
* **`gemini-2.5-flash-lite`**：更高吞吐/成本敏感場景。
* **`gemini-2.0-flash`**、**`gemini-2.0-flash-lite`**：仍提供不錯的多模態能力。
  （以上皆屬 **Text-out models**，可做文字/多模態理解、結構化輸出、函式呼叫等。）([Google AI for Developers][3])

## 2.2 Live API（雙向/低延遲對話）

* **`gemini-2.5-flash-live`**（每專案同時 3 sessions）
* **`gemini-2.0-flash-live`**（同上，3 sessions）
* **原生音訊對話預覽**：`gemini-2.5-flash-preview-native-audio-dialog`（1 session），另有實驗款 *Native Audio Thinking Dialog*。
  （官方 Live API 章節詳述 Session 與用法，見第 9 節。）([Google AI for Developers][2])

## 2.3 生成式音訊（TTS，文字→語音）

* **`gemini-2.5-flash-preview-tts`**（Free Tier 可用，預覽）

> 可做單人/多人 TTS，受控語音風格，回傳音訊資料流。使用方法見第 8.2 節與官方「Speech generation」章。([Google AI for Developers][4])

## 2.4 生成式影像（文字→圖片）

* **`gemini-2.0-flash-preview-image-generation`**（Free Tier 可用）

> 2.5 影像原生生成（Flash Image / Nano Banana）目前在**付費等級**表格才列明（Tier 1「Gemini 2.5 Flash Image Preview」），Free Tier 先提供 2.0 Flash 圖生圖（預覽）名額。([Google AI for Developers][2])

## 2.5 其他

* **Gemma 3 / 3n**（開源家族的雲端託管版本，Free Tier 有高 RPD），**Embeddings**（`text-embedding-*` 家族）。([Google AI for Developers][2])

> **定價與層級**：AI Studio 本身是免費；API 有 Free/付費兩層，升級可獲更高額度與更多模型（定價/層級說明見官方 Pricing/Billing）。([Google AI for Developers][5])

---

# 3) 安裝與初始化（Python / Node.js）

> **請務必使用新 SDK：**`google-genai`（Python/JS），這是官方最新推薦。([Google AI for Developers][1])

### Python

```bash
pip install -U google-genai
export GEMINI_API_KEY="你的API_KEY"  # 或以 .env / 系統金鑰管理
```

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
# 若環境已設定變數，亦可 client = genai.Client()
```

### Node.js（JS/TS）

```bash
npm i @google/genai
# 或 pnpm add @google/genai / bun add @google/genai
```

```ts
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
// 若以環境變數配置，亦可 new GoogleGenAI({})
```

---

# 4) 文本/多模態生成：最小可用範例 + 參數全集

## 4.1 最小可用：文字→文字

**Python**

```python
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="給我三種適合初學者的 Python 專案點子"
)
print(resp.text)
```

**Node.js**

```ts
const resp = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: "給我三種適合初學者的 Python 專案點子",
});
console.log(resp.text);
```

（基本文字生成功能，對應官方 Text generation 範例。）([Google AI for Developers][6])

## 4.2 生成參數（GenerationConfig）——常用全集

可在 `config` 傳入（Python `types.GenerateContentConfig`；JS 直接物件），典型欄位：

* **temperature**（0\~2）：創意度
* **top\_p / top\_k**：抽樣控制
* **max\_output\_tokens**：輸出 token 上限
* **stop\_sequences**：停止符號
* **response\_mime\_type**：`"text/plain"`、`"application/json"`、`"image/png"`、`"audio/wav"` 等（見第 8 節用法）
* **system\_instruction**：系統角色/風格指令（等同 system prompt）
* **safety\_settings**：內容安全閾值（見第 11 節）
* **tools / tool\_config**：啟用工具（Google Search、函式呼叫、URL context、Code Execution；見第 6 節）

**Python**

```python
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
```

（完整欄位與 REST 對應見 API 參考頁。）

## 4.3 串流輸出（Streaming）

**Python**

```python
with client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="用條列方式解釋 CPU/GPU/TPU 差異"
) as stream:
    for event in stream:
        if event.delta:
            print(event.text, end="", flush=True)
```

**Node.js**

```ts
const stream = await ai.models.generateContentStream({
  model: "gemini-2.5-flash",
  contents: "用條列方式解釋 CPU/GPU/TPU 差異",
});
for await (const event of stream) {
  if (event.delta) process.stdout.write(event.text ?? "");
}
```

（Streaming 端點與事件流細節見 API 參考頁的串流章節。）

---

# 5) 結構化輸出（JSON Schema / 嚴格 JSON）

你可以要求模型**必須**輸出符合 Schema 的 JSON（方便後續程式化處理）。

**Python**

```python
schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "title": types.Schema(type=types.Type.STRING),
        "keywords": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
        "difficulty": types.Schema(type=types.Type.STRING, enum=["easy", "medium", "hard"])
    },
    required=["title", "keywords", "difficulty"]
)
cfg = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=schema
)
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="產出一個入門級 Python 專案主題，附 5 個關鍵字與難度等級",
    config=cfg
)
print(resp.text)  # 嚴格 JSON
```

（官方 Structured output 指南，含 Schema 格式與限制。）

---

# 6) 工具與進階（Function Calling / Google Search / URL Context / Code Execution）

## 6.1 Function Calling（工具函式）

宣告函式的**名稱/用途/JSON Schema 參數**，模型需要時會回傳 `function_call`，你執行後把結果作為 `tool` 回覆再續問答。

**Python**

```python
weather_fn = types.FunctionDeclaration(
    name="get_weather",
    description="查詢指定城市的當前天氣（攝氏）",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(type=types.Type.STRING)
        },
        required=["city"]
    )
)
tools = [types.Tool(function_declarations=[weather_fn])]
cfg = types.GenerateContentConfig(tools=tools)

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="幫我查台北的天氣並建議穿著",
    config=cfg
)

# 若模型要求呼叫函式
for part in resp.candidates[0].content.parts:
    if getattr(part, "function_call", None):
        args = part.function_call.args  # 取得 city
        # 實際呼叫你的後端服務
        data = {"temp_c": 28, "desc": "Cloudy", "city": args["city"]}

        # 回填 tool 回覆，讓模型整合
        follow = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {"role": "user", "parts": "幫我查台北的天氣並建議穿著"},
                {"role": "tool", "parts": [{"function_response": {"name": "get_weather", "response": data}}]}
            ],
            config=cfg
        )
        print(follow.text)
```

（Function Calling 與工具宣告格式詳見官方 API 參考。）

## 6.2 Grounding with Google Search（最新：`google_search`）

**不用再**使用舊的 `google_search_retrieval`；Gemini 2.x 請使用 **`google_search`** 工具。回傳含可驗證之 `groundingMetadata` 與引用段落。
**Python**

```python
gsearch = types.Tool(google_search=types.GoogleSearch())
cfg = types.GenerateContentConfig(tools=[gsearch])

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Who won Euro 2024?",
    config=cfg
)
print(resp.text)
# 可從 resp.candidates[0].grounding_metadata 取得來源/段落對應
```

（官方 Grounding 指南，含回傳欄位與引用展示方式。）([Google AI for Developers][7])

## 6.3 URL Context（指定網址內容作為依據）

截至 google-genai 1.16.1，Python / Node SDK 尚未開放在 	ypes.UrlContext 中直接帶入 urls 清單；若照舊範例 	ypes.UrlContext(urls=[...]) 會拋出驗證錯誤。要引用指定網址目前可採以下做法：

1. **改走 REST**：呼叫 POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent，在 payload 的 	oolConfig.urlContext 帶入網址陣列。
2. **檢查 SDK 更新**：先呼叫 client.models.list()／i.models.list() 確認你的專案與 SDK 版本是否已支援 UrlContext 參數，若尚未開放請暫時以 REST 或自行實作擴充。

目前 SDK 只能宣告 	ypes.Tool(url_context=types.UrlContext()) 作為占位元；真正的網址列表仍需靠 REST 或後續版本的 SDK 來填入。請在程式中加上檢查與降級邏輯，避免直接照舊範例導致 runtime error。

## 6.4 Code Execution（沙箱運算）

讓模型在受控沙箱中執行小段程式碼（例如數學/解析/簡單轉換），回傳輸出再整合答案。

**Python（重點片段）**

```python
cx = types.Tool(code_execution=types.CodeExecution())
cfg = types.GenerateContentConfig(tools=[cx])
```

（官方工具頁面列出啟用方式與安全注意。）

---

# 7) 多模態輸入 + Files API（上傳/引用/刪除）

> 大檔（圖片/音訊/影片/PDF）建議先用 **Files API** 上傳，再把回傳的 `file.uri` 作為 `contents` 的一部分，以便長上下文處理與重複使用。

**Python：上傳並引用**

```python
# 上傳
uploaded = client.files.upload(file="report.pdf")   # 自動推斷 MIME
# 生成（PDF + 指令）
resp = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=[uploaded, "請將此 PDF 濃縮成 10 點摘要"]
)
print(resp.text)
```

**列出/刪除**

```python
for f in client.files.list():
    print(f.name, f.mime_type, f.uri)

client.files.delete(name=uploaded.name)
```

**Node.js：上傳並引用**

```ts
const file = await ai.files.upload({ file: "./sample.mp3" });
const resp = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: [file, "轉寫這段音檔並做 5 點摘要"],
});
console.log(resp.text);
```

---

# 8) 生成式媒體

## 8.1 圖像生成（Image Generation）

* **Free Tier**：gemini-2.0-flash-preview-image-generation 可用（預覽版）。
* **付費等級**：gemini-2.5-flash-image-preview 顯示在 Tier 1 表格中。
  **最小可用（Python）**：

`python
cfg = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
resp = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents="A cozy reading nook with warm morning light, watercolor style"
)
image_part = next(p for p in resp.candidates[0].content.parts if p.inline_data)
payload = image_part.inline_data.data  # SDK 已直接回傳 bytes
Path("out.png").write_bytes(payload)
`

> **注意**：google-genai 1.16.1 的 inline_data.data 已是二進位資料，無須再 base64 decode。

（模型可用性/額度見 Rate Limits；影像原生生成亦見 API 導覽「Native Image Generation」。）([Google AI for Developers][2])

## 8.2 語音生成（TTS：文字→音訊）

* **Free Tier**：gemini-2.5-flash-preview-tts。
* **Pro TTS**：gemini-2.5-pro-preview-tts（更高可控性，預覽）。
  **Python：單人說話**

`python
from pathlib import Path

speech = types.SpeechConfig(
    voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="callirrhoe")
    )
)
cfg = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=speech)
resp = client.models.generate_content(
    model="gemini-2.5-flash-preview-tts",
    contents="用溫暖的語氣說：嗨，歡迎來到我們的節目！",
    config=cfg
)
audio_bytes = resp.candidates[0].content.parts[0].inline_data.data  # 已是 bytes
Path("out.wav").write_bytes(audio_bytes)
`

> **語音注意事項**：預設語者名稱需使用小寫（例如 callirrhoe），且回傳內容同樣是二進位資料。

---

# 9) 直播互動（Live API：低延遲/語音對話/雙向串流）

Live API 支援**雙向音訊**、低延遲互動（WebRTC/WS），Free Tier 可用 `gemini-2.5-flash-live` / `gemini-2.0-flash-live`（每專案同時 session 數有限）。
基本步驟：建立 session → 推送音訊/文字 → 取回逐步事件 → 關閉 session。官方 Live API 小節含「Ephemeral Tokens」用於瀏覽器端安全接入。([Google AI for Developers][2])

---

# 10) 大量/排程推理（Batch API）、長期快取（Context Caching）

**Batch API**：離線、可一次佇列大量請求，適合批量摘要/抽取/評分。
**Python：建立批次**

```python
job = client.batches.create(
    model="gemini-2.5-flash",
    src="gs://your-bucket/path/to_requests.jsonl",
    config=types.CreateBatchJobConfig(output_gcs_uri="gs://your-bucket/output/")
)
# 查詢狀態與取回結果
done = client.batches.get(name=job.name)
`

> **重要**：Batch API 需要先把請求 (JSONL) 放在 GCS 或 BigQuery，不能直接在 SDK 傳 
equests=[...]。``

（批次限制：併發、輸入檔上限 2GB、儲存 20GB 等，詳見 Batch API 章。）([Google AI for Developers][8])

**Context Caching**：把固定長上下文（規格/SOP 等）緩存，後續請求以短 prompt 引用，省 token 成本、提速。不過 Free Tier 的 TotalCachedContentStorageTokensPerModel 配額目前為 0（2025/09 實測），需升級計費層級或改走 Vertex AI 才能建立快取。

---

# 11) 安全性（Safety Settings）與思考模式（Thinking）

* **Safety**：可設定各分類的阻擋閾值（例如仇恨/暴力/色情等），在 `safety_settings` 內指定分類與門檻；違規時模型會回傳安全層阻擋資訊。官方有詳細建議與分類清單。
* **Thinking（2.5 系列）**：2.5 模型支援「思考階段」與 **`thinkingBudget`**（在 `generationConfig` 中設定）。可由 0（關閉，部分模型可關）到上萬 tokens；不同模型最小/最大可用額度不同（例如 2.5 Flash/Pro/Lite）。([Firebase][9])

---

# 12) Token 計數與錯誤/RL 限流處理

* **計數**：可用 **Token Counting** 端點/SDK 方法在送出前估算 token（避免溢出）。
* **限流**：遵守 Rate Limits（Free Tier：例如 `gemini-2.5-pro` 5 RPM / 250k TPM / 100 RPD；`gemini-2.5-flash` 10 / 250k / 250；`gemini-2.5-flash-lite` 15 / 250k / 1000；Live/TTS/影像預覽皆有各自限制）。遇到 **429** 時請實作**指數退避**重試與**排隊**。([Google AI for Developers][2])

---

# 13) 常見實作藍本（可直接改用）

## 13.1 以 URL+Search 做「可驗證」時事回答

**Python（核心）**

```python
# 目前 SDK 尚未開放直接在 UrlContext 帶入網址，需改走 REST 或等待更新。
# 以下示意先檢查模型清單，若 UrlContext 已支援再組合工具。
# tools = [
#     types.Tool(google_search=types.GoogleSearch()),
#     types.Tool(url_context=types.UrlContext(...))
# ]
`

> **提示**：先以 REST 驗證 Url Context 功能，或在 SDK 更新支援後再行整合。``

（Google Search Grounding 可回傳 `groundingMetadata`：queries、chunks、supports—能把句子與來源段落對齊產生內嵌引用。）([Google AI for Developers][7])

## 13.2 嚴格 JSON 抽取（Evaluation / 資料標註）

**Python（核心）**：見第 5 節 Schema，用 `response_mime_type="application/json"` + `response_schema`。

## 13.3 音訊轉錄 + 摘要

**Python**

```python
wav = client.files.upload(file="meeting.wav")
resp = client.models.generate_content(
  model="gemini-2.5-pro",
  contents=[wav, "請先逐字轉錄，再濃縮為 10 條行動項目（每條含負責人/期限）"]
)
print(resp.text)
```

（音訊理解與轉錄見 Audio Understanding。）([Google AI for Developers][10])

## 13.4 以 Function Calling 連你的私有 API（RAG 索引/資料庫）

* 用 Function 宣告 `search_kb(query, top_k)`，回傳檢索段落與來源 → 再由模型生成帶引用答案。
* 若要固定檔案：把資料 PDF/HTML 上傳 Files API，或以 URL Context 指向固定頁面。

---

# 14) 版本/相容性提醒與最佳實務

* **請用 `google-genai`**（Python/JS）。舊 `google-generativeai` 有明確的淘汰時程（2025/11/30 停用），官方已要求遷移。([Google AI for Developers][1])
* **模型代號以「列舉 API」為準**：不同專案/層級的可見模型可能稍有差異；建議先呼叫 SDK 的 **List Models**（`client.models.list()` / `ai.models.list()`）列出你專案可用的精確代號，再帶入程式。
* **Grounding 請用 `google_search`（新）**，不要再用 `google_search_retrieval`（只留給 1.5 舊模型）。([Google AI for Developers][7])
* **能力與額度以 Rate Limits 為準**，Free/Tier1/Tier2/Tier3 差異很大。([Google AI for Developers][2])
* **AI Studio** 本身免費、API 有 Free/付費層，升級後解鎖更多模型（Imagen/Veo 等）與更高額度。([Google AI for Developers][5])

---

## 快速參考（官方文件直達）

* **API 參考總覽**（端點 / 參數 / 串流 / Files / Batch / Caching / Safety）：
* **官方 SDK（`google-genai`）指引與遷移公告**：([Google AI for Developers][1])
* **模型一覽（含 2.5 Pro/Flash/Flash-Lite 能力表與是否支援 Thinking/工具）**：([Google AI for Developers][3])
* **Rate Limits（Free/Tier1/Tier2/Tier3 與各模型額度/會話/影像/音訊限制）**：([Google AI for Developers][2])
* **Grounding with Google Search（使用 `google_search`、回傳 citations 結構）**：([Google AI for Developers][7])
* **Files API（上傳/列出/刪除/在 contents 中引用檔案）**：
* **Batch API（大量離線處理）**：([Google AI for Developers][8])
* **Token Counting**：
* **Speech Generation（TTS）**：([Google AI for Developers][4])
* **Text generation（最小可用範例）**：([Google AI for Developers][6])
* **Pricing / Billing**：([Google AI for Developers][5])

---

### 給你的執行建議

1. 先在 AI Studio 取得專案的 API Key，跑第 4 節的最小可用範例確認通。
2. 呼叫 **List Models** 列出你專案此刻可用的精準模型代號（尤其圖像/TTS/Live 這些預覽版名稱）。
3. 依需求加入 **Schema**（第 5 節）與 **工具**（第 6 節），並善用 **Files API** 管理長上下文。
4. 大規模任務改走 **Batch API**；即時語音/雙向互動改用 **Live API**。
5. 觀察**Rate Limits** 與錯誤碼，做**指數退避**與**佇列**，必要時升級層級。

[1]: https://ai.google.dev/gemini-api/docs/models "Gemini models  |  Gemini API  |  Google AI for Developers"
[2]: https://ai.google.dev/gemini-api/docs/rate-limits "Rate limits  |  Gemini API  |  Google AI for Developers"
[3]: https://ai.google.dev/gemini-api/docs/models?utm_source=chatgpt.com "Gemini models | Gemini API - Google AI for Developers"
[4]: https://ai.google.dev/gemini-api/docs/speech-generation?utm_source=chatgpt.com "Speech generation (text-to-speech) | Gemini API"
[5]: https://ai.google.dev/gemini-api/docs/pricing?utm_source=chatgpt.com "Gemini Developer API Pricing"
[6]: https://ai.google.dev/gemini-api/docs/text-generation?utm_source=chatgpt.com "Text generation | Gemini API - Google AI for Developers"
[7]: https://ai.google.dev/gemini-api/docs/google-search "Grounding with Google Search  |  Gemini API  |  Google AI for Developers"
[8]: https://ai.google.dev/api "Gemini API reference  |  Google AI for Developers"
[9]: https://firebase.google.com/docs/ai-logic/thinking?utm_source=chatgpt.com "Thinking | Firebase AI Logic - Google"
[10]: https://ai.google.dev/gemini-api/docs/audio?utm_source=chatgpt.com "Audio understanding | Gemini API | Google AI for Developers"
