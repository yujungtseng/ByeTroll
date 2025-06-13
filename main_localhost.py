from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import torch
import numpy as np
import os
from torch.serialization import safe_globals
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

# --- 配置 ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model")

# 檢查模型路徑是否存在
if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
    print(f"Error: Model path {MODEL_PATH} does not exist or is empty. Please ensure the model is trained and saved.")
    exit(1)

# --- 初始化 FastAPI 應用 ---
app = FastAPI(
    title="YouTube Troll Classifier API",
    description="一個 API 用於分析 YouTube 留言並將其分類為 1-5 個酸民等級。",
    version="0.1.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.youtube.com", "http://127.0.0.1:8001", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Origin", "Authorization"],
    expose_headers=["Content-Type", "Accept", "Origin", "Authorization"],
    max_age=3600,
)

# 自定義 JSONResponse 以處理中文編碼
class CustomJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

# --- 全局變數，用於儲存載入的模型和分詞器 ---
# 這些變數會在應用啟動時載入，以避免每次請求都重新載入
model = None
tokenizer = None
device = None

# --- Pydantic 模型，用於請求和回應的資料驗證 ---
class CommentInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    original_text: str
    predicted_label: int # 1-5 的等級
    predicted_class_name: str # 例如 "等級一：無攻擊性"
    probabilities: dict # 各等級的機率 (可選)

# --- 事件處理：應用啟動時載入模型 ---
@app.on_event("startup")
async def load_model_and_tokenizer():
    global model, tokenizer, device
    print(f"正在從 {MODEL_PATH} 載入模型和分詞器...")
    try:
        if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
            raise Exception(f"模型路徑 {MODEL_PATH} 不存在或為空。請確保已訓練並儲存模型。")

        print(f"正在嘗試載入模型，路徑：{MODEL_PATH}")
        print(f"目錄內容：{os.listdir(MODEL_PATH)}")
        
        # Load tokenizer from model directory
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load model from model directory
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True,  # Only use local files
            from_tf=False,  # Ensure we're loading PyTorch weights
            ignore_mismatched_sizes=True  # Ignore size mismatches
        )
        
        print(f"成功載入模型：{MODEL_PATH}")

        # 檢查是否有可用的 GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("使用 GPU 進行推斷。")
        elif torch.backends.mps.is_available(): # 適用於 Apple Silicon (M1/M2)
             device = torch.device("mps")
             print("使用 Apple MPS 進行推斷。")
        else:
            device = torch.device("cpu")
            print("使用 CPU 進行推斷。")
        
        model.to(device)
        model.eval() # 設定為評估模式
        print("模型和分詞器載入成功！")
    except Exception as e:
        print(f"載入模型或分詞器時發生錯誤: {e}")
        raise Exception(f"無法載入模型或分詞器: {str(e)}")

def get_class_name(label_id_zero_indexed):
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

# --- API 端點：用於接收留言並返回預測結果 ---
@app.post("/predict/", response_model=PredictionOutput)
async def predict_comment_level(comment_input: CommentInput):
    global model, tokenizer, device

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型尚未載入或載入失敗，服務暫時不可用。")

    # 確保輸入文字是 UTF-8 編碼
    text_to_analyze = comment_input.text.encode('utf-8').decode('utf-8')
    if not text_to_analyze.strip():
        raise HTTPException(status_code=400, detail="輸入文字不能為空。")

    try:
        # 1. Tokenize 輸入文字
        inputs = tokenizer(text_to_analyze, return_tensors="pt", truncation=True, padding="max_length", max_length=128) # 與訓練時一致
        
        # 2. 將輸入移至與模型相同的設備
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 3. 進行預測
        with torch.no_grad(): # 在推斷時不需要計算梯度
            outputs = model(**inputs)
            logits = outputs.logits

        # 4. 計算機率 (Softmax)
        probabilities = torch.softmax(logits, dim=-1).squeeze().tolist() # squeeze() 移除 batch 維度

        # 5. 取得預測的類別 (0-4)
        predicted_class_id_zero_indexed = torch.argmax(logits, dim=-1).item()
        
        # 6. 轉換為 1-5 的等級
        predicted_label_one_indexed = predicted_class_id_zero_indexed + 1

        # 7. 準備機率字典 (可選)
        prob_dict = {get_class_name(i): round(p, 4) for i, p in enumerate(probabilities)}

        # 確保返回的文本是正確的 UTF-8 編碼
        return {
            "original_text": text_to_analyze,
            "predicted_label": predicted_label_one_indexed,
            "predicted_class_name": get_class_name(predicted_class_id_zero_indexed),
            "probabilities": prob_dict
        }

    except Exception as e:
        # 記錄更詳細的錯誤日誌會更好
        print(f"預測時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"內部伺服器錯誤: {str(e)}")
    except Exception as e:
        # 記錄更詳細的錯誤日誌會更好
        print(f"預測時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"內部伺服器錯誤: {str(e)}")

# --- (可選) 一個根端點，用於檢查服務是否運行 ---
@app.get("/")
async def read_root():
    return {"message": "歡迎使用 YouTube 酸民留言分類器 API！請訪問 /docs 查看 API 文件。"}

# --- (可選) 如果您想直接用 python main.py 執行 (主要用於開發) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting development server. Please ensure the 'final_model' directory is in the same directory as this script, or modify MODEL_PATH.")
    uvicorn.run(app, host="127.0.0.1", port=8000)