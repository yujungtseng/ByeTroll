from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# 確保 transformers 已安裝
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

# --- 配置 (非常重要！) ---
# 將 MODEL_PATH 改成你在 Hugging Face 上的儲存庫 ID
# 格式是 "你的使用者名稱/你的模型名稱"
MODEL_PATH = "yujungtseng/ByeTroll-Classifier" 

# ... (初始化 FastAPI 應用和 CORS 的程式碼保持不變) ...
app = FastAPI(...)
app.add_middleware(...)

# ... (Pydantic 模型和 CustomJSONResponse 保持不變) ...

# --- 全局變數 ---
model = None
tokenizer = None
device = None

# --- 事件處理：應用啟動時從 Hugging Face 載入模型 ---
@app.on_event("startup")
async def load_model_and_tokenizer():
    global model, tokenizer, device
    print(f"正在從 Hugging Face Hub ({MODEL_PATH}) 載入模型和分詞器...")
    try:
        # from_pretrained 會自動從 Hub 下載並快取模型
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        
        print(f"成功從 Hugging Face 載入模型：{MODEL_PATH}")

        # 檢查是否有可用的 GPU (在 Render 的 CPU 環境下，這會選擇 CPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("使用 GPU 進行推斷。")
        elif torch.backends.mps.is_available():
             device = torch.device("mps")
             print("使用 Apple MPS 進行推斷。")
        else:
            device = torch.device("cpu")
            print("使用 CPU 進行推斷。")
        
        model.to(device)
        model.eval() # 設定為評估模式
        print("模型和分詞器準備就緒！")
    except Exception as e:
        print(f"從 Hugging Face 載入模型或分詞器時發生錯誤: {e}")
        # 這裡可以加上更詳細的錯誤處理，例如檢查網路連線或模型名稱是否正確
        raise Exception(f"無法從 Hugging Face 載入模型: {str(e)}")

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