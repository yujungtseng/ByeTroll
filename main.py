import json
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. 配置 (Configuration) ---

# 從 Hugging Face Hub 載入模型的路徑
MODEL_PATH = "yujungtseng/ByeTroll-Classifier"

# --- 2. 初始化 FastAPI 應用 ---

# 這裡是修正後的 FastAPI 初始化，使用了關鍵字參數 'title'
app = FastAPI(
    title="YouTube Troll Classifier API",
    description="一個 API 用於分析 YouTube 留言並將其分類為 1-5 個酸民等級。",
    version="0.1.0"
)

# --- 3. 配置 CORS (跨來源資源共用) ---
# 允許所有來源，這對於 Chrome 擴充元件或任何前端開發都很方便
# 在正式生產環境中，可以考慮將 "*" 換成你前端的具體網域，以策安全
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源
    allow_credentials=True,
    allow_methods=["GET", "POST"], # 只允許需要的 HTTP 方法
    allow_headers=["*"], # 允許所有標頭
)

# --- 4. Pydantic 模型 (資料驗證) ---

class CommentInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    original_text: str
    predicted_label: int  # 1-5 的等級
    predicted_class_name: str  # 例如 "等級一：無攻擊性"
    probabilities: dict  # 各等級的機率

# --- 5. 全局變數 (用於儲存模型) ---
# 這些變數會在應用啟動時載入，避免每次請求都重新載入
model = None
tokenizer = None
device = None

# --- 6. 事件處理：應用啟動時從 Hugging Face 載入模型 ---

@app.on_event("startup")
async def load_model_and_tokenizer():
    global model, tokenizer, device
    print(f"正在從 Hugging Face Hub ({MODEL_PATH}) 載入模型和分詞器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        
        print(f"成功從 Hugging Face 載入模型：{MODEL_PATH}")

        # 檢查是否有可用的 GPU (在 Render 的 CPU 環境下，這會選擇 CPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("使用 GPU 進行推斷。")
        else:
            device = torch.device("cpu")
            print("使用 CPU 進行推斷。")
        
        model.to(device)
        model.eval()  # 設定為評估模式
        print("模型和分詞器準備就緒！")
    except Exception as e:
        print(f"從 Hugging Face 載入模型或分詞器時發生錯誤: {e}")
        raise Exception(f"無法從 Hugging Face 載入模型: {str(e)}")

# --- 7. 輔助函式 ---

def get_class_name(label_id_zero_indexed: int) -> str:
    # 根據模型的標籤配置進行映射
    # 模型配置中的 LABEL_0 到 LABEL_4
    names = {
        0: "無攻擊性/中性",
        1: "輕微不友善/抱怨",
        2: "諷刺/引戰/地圖砲嫌疑",
        3: "人身攻擊/歧視/仇恨言論",
        4: "暴力威脅/嚴重霸凌"
    }
    return f"等級{label_id_zero_indexed + 1}：{names.get(label_id_zero_indexed, '未知等級')}"

# --- 8. API 端點 (Endpoints) ---

@app.get("/", summary="檢查 API 狀態")
async def read_root():
    """
    根路徑，用於檢查 API 服務是否正常運行。
    """
    return {"message": "歡迎使用 YouTube 酸民留言分類器 API！服務運行中。"}

@app.post("/predict/", response_model=PredictionOutput, summary="預測留言的酸民等級")
async def predict_comment_level(comment_input: CommentInput):
    """
    接收一段文字留言，並返回其酸民等級預測結果。

    - **text**: 要分析的留言內容。
    """
    global model, tokenizer, device

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型尚未載入或載入失敗，服務暫時不可用。")

    text_to_analyze = comment_input.text
    if not text_to_analyze or not text_to_analyze.strip():
        raise HTTPException(status_code=400, detail="輸入文字不能為空。")

    try:
        # 1. Tokenize 輸入文字
        inputs = tokenizer(text_to_analyze, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # 2. 將輸入移至與模型相同的設備
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 3. 進行預測 (在推斷時不需要計算梯度)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 4. 計算機率 (Softmax)
        probabilities = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # 5. 取得預測的類別 (0-4)
        predicted_class_id = int(np.argmax(probabilities))
        
        # 6. 準備回應內容
        prob_dict = {get_class_name(i): round(float(p), 4) for i, p in enumerate(probabilities)}

        response_data = {
            "original_text": text_to_analyze,
            "predicted_label": predicted_class_id + 1,
            "predicted_class_name": get_class_name(predicted_class_id),
            "probabilities": prob_dict
        }
        
        # 使用 JSONResponse 並設定 ensure_ascii=False 來正確處理中文字元
        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"預測時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"內部伺服器錯誤: {str(e)}")


# --- 9. (可選) 本地開發時的啟動方式 ---
if __name__ == "__main__":
    import uvicorn
    print("正在啟動本地開發伺服器...")
    uvicorn.run(app, host="127.0.0.1", port=8000)