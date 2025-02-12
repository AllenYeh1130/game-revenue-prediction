# Game-Revenue-Prediction
預測每週玩家的180日累積儲值金額，以利行銷跟營運決策  
Predict the 180-day cumulative revenue of weekly players to support marketing and operational decisions.

![result_plot](images/result_plot.png)

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
   已訓練完成的模型，可透過 predict.py 進行預測（檔案未包含，因模型屬於敏感資料）。  
   The trained model file, used for predictions via predict.py. It can also be backed up on MLFlow (not included in GitHub due to sensitivity concerns).
2. **encoders.pkl**  
   用於將類別數據轉換為數值的編碼標準，基於訓練數據生成，供測試數據的類別數據轉換使用。  
   Encoders for converting categorical data into numerical values. These are generated from training data and applied to transform categorical data in testing.
3. **model_result.txt**  
   模型交叉驗證結果，包含 MAPE、RMSE 以及最佳的 alpha 值。  
   The model cross-validation results, including MAPE, RMSE, and the best alpha.  
   ![model_result](images/model_result.png)
4. **features.csv**  
   列出各特徵的重要性，能直觀呈現對模型影響較大的特徵。  
   Lists feature importance, providing a clear view of which features have the most significant impact on the model.  
   ![features](images/features.png)
5. **predict_result.csv**  
   預測各用戶營收的結果。
   Predicted revenue results for each user.  
   ![predict_result](images/predict_result.png)

---

### 各檔案描述 | File Descriptions
- **data_processing.py**  
  玩家遊戲行為資料前處理，包含類別、整合及每日數據，並新增斜率特徵，用於觀察玩家遊戲行為隨時間變化的趨勢（如熱情增加或減少的指標）  
  Handles preprocessing of player behavior data, including categorical conversion, data integration, and generating daily metrics.
  It also creates slope-based features to observe trends such as increased or decreased engagement over time.
  
- **insert_data.py**  
  更新過去 180 天內的新玩家行為數據，並寫入 MySQL 資料庫，提升後續模型訓練與測試的效率。  
  Updates and inserts new player behavior data from the past 180 days into a MySQL database, improving the efficiency of future model training and testing.

- **modeling.py**  
  建立 LASSO 模型並上傳至 MLflow：
  1. 將數據集劃分為訓練集（train）和測試集（test），並將類別特徵轉換為數值格式，匯出 encoders.pkl 以供後續測試資料使用。
  2. 使用 LASSO 進行交叉驗證（CV），執行 50 次以尋找穩定且最佳的參數。
  3. 基於最佳參數進行建模，測試並記錄模型效能與參數細節。
  4. 將訓練結果、最佳參數與模型匯出，並上傳至 MLflow 便於後續分析與版本管理。
  
  Build the LASSO model and upload to MLflow:
  1. Split the dataset into training and testing sets, convert categorical features into numerical values, and export encoders.pkl for use with future test data.
  2. Perform cross-validation (CV) using LASSO for 50 iterations to identify stable and optimal parameters.
  3. Build the model with the optimal parameters, test it, and document performance metrics and parameter details.
  4. Export the training results, optimal parameters, and the model, then upload them to MLflow for further analysis and version management.

- **predit.py**  
  使用預測數據（CSV 格式）與訓練好的模型（PKL 格式），進行預測並輸出處理後的結果。  
  Utilizes prediction data (in CSV format) and a trained model (in PKL format) to generate predictions and process the output results.
  
- **lgbm_test.py 、 xgboost_test.py**  
  驗證 LightGBM 跟 XGBoost 模型，包含以下步驟：
  1. 驗證 min-max 正規化與 log 轉換的效果。
  2. 將類別數據轉換為數值。
  3. 使用 Optuna 尋找最佳化參數組合。
  4. 評估模型的特徵重要性。
  
  Validates LightGBM and XGBoost model, including:
   1. Testing the effectiveness of min-max normalization and log transformation.
   2. Converting categorical data into numerical values.
   3. Using Optuna to find the optimal parameter set.
   4. Assessing feature importance for the model.
