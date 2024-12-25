# Game-Revenue-Prediction
預測每週玩家的180日累積儲值金額，以利行銷跟營運決策
Predict the 180-day cumulative revenue of weekly players to support marketing and operational decisions.

### 專案簡介 | Project Overview
本專案使用遊戲玩家前 7 天的行為資訊來預測 180 天的累積儲值金額，並比較三種模型（LASSO、LightGBM 和 XGBoost）的預測成效。
最終選擇 LASSO 模型以滾動式的方式進行每月最新玩家行為的預測，並達成中位數百分比誤差約 9% 的準確度。
This project leverages the first 7 days of player behavior data to predict 180-day cumulative revenue. Three models—LASSO, LightGBM, and XGBoost—are compared, with LASSO selected for its accuracy and reliability. 
Rolling monthly predictions achieve a median percentage error of approximately 9%.

---

### 主要目的 | Main Purpose
每週行銷成本是否回本，以及營運策略的調整是否能有效提升玩家的儲值行為。
To evaluate whether weekly marketing costs yield a return on investment and assess the impact of operational strategies on player spending behavior.

---

### 輸出結果 | Output Results
1. **model.pkl**
   已訓練完成的模型，可透過 predict.py 進行預測，也可以在 MLFlow 上建立備份（檔案未包含於 GitHub，因模型屬於敏感資料）。
   The trained model file, used for predictions via predict.py. It can also be backed up on MLFlow (not included in GitHub due to sensitivity concerns).
2. **encoders.pkl**
   用於將類別數據轉換為數值的編碼標準，基於訓練數據生成，供測試數據的類別數據轉換使用。
   Encoders for converting categorical data into numerical values. These are generated from training data and applied to transform categorical data in testing.
3. **features.csv**  
   列出各特徵的重要性，能直觀呈現對模型影響較大的特徵。
   Lists feature importance, providing a clear view of which features have the most significant impact on the model.
   ![features](images/features.png)

---

### 各檔案描述 | File Descriptions
