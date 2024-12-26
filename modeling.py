#%% package
# 建立LASSO模型、將需要的資訊傳到MLflow上
project_name = 'game_ltv_modeling'

### 載入套件
import os
import sys
import json
import pandas as pd
import socket
import datetime as dt
import time
import sqlalchemy
from category_encoders import TargetEncoder
from category_encoders import MEstimateEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
import numpy as np
from data_for_predict import predict_data
import shutil
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import yagmail
from sklearn.model_selection import train_test_split


#%% config
# ===== config =====
def config_read(file_name):
    # Read config file
    if sys.platform.startswith('linux'):
        file_path = r'linux 設定檔路徑'  # linux文件路径
    elif sys.platform.startswith('win'):
        file_path = r'Windows 設定檔路徑'
    else:
        print("無法判斷程式執行的作業系統")

    file_path = os.path.join(file_path, file_name) #完整設定檔路徑
    #讀入json
    with open(file_path, 'r') as file:
        config = json.load(file)
    
    config = pd.json_normalize(config).stack().reset_index(level=0, drop=True) #刪除多的一層索引
    config = config.to_dict()
    return config

config = config_read(file_name = 'all_setting.json')
API_config = config_read(file_name = 'API_setting.json')
game_token = API_config['game.token']

# ===== 判斷ip =====
# 獲得目前主機IP，要判斷是正式機還是測試機
def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
    s.close()
    return ip_address

ip_address = get_ip_address()


#%% DB connect
# ===== DB connect =====
# AI正式機資料庫連線
DATABASE = 'game'

# AI正式機資料庫連線
if ip_address == '正式機':
    # AI正式機
    USER = config['DB.formal.user']
    PASSWORD = config['DB.formal.pass']
    HOST = config['DB.formal.host']
    PORT = config['DB.formal.port']
elif ip_address == '測試機':
    # AI測試機內網
    USER = config['DB.test.user']
    PASSWORD = config['DB.test.pass']
    HOST = config['DB.test.host']
    PORT = config['DB.test.port']
else:
    # 本地端
    USER = config['DB.local.user']
    PASSWORD = config['DB.local.pass']
    HOST = config['DB.local.host']
    PORT = config['DB.local.port']
    
engine_stmt = 'mysql+pymysql://%s:%s@%s:%s/%s' % (USER, PASSWORD,HOST,PORT,DATABASE)
engine = sqlalchemy.create_engine(engine_stmt, echo=False)


#%% MLflow 設定
# ===== MLflow 設定 =====
experimen_name = 'experimen_name'
model_name = 'model_name'

# 設置tracking server uri、帳號密碼
# 以ip位置判斷MLflow該連正式還是測試
if ip_address == '正式機':         # AI正式機
    trackingServerUri = config['mlflow.mlflow_formal.trackingServerUri']
    os.environ["MLFLOW_TRACKING_USERNAME"] = config['mlflow.mlflow_formal.user_name']
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config['mlflow.mlflow_formal.password']
else:                                      # 非正式機就連到測試的MLflow
    trackingServerUri = config['mlflow.mlflow_test.trackingServerUri']
    os.environ["MLFLOW_TRACKING_USERNAME"] = config['mlflow.mlflow_test.user_name']
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config['mlflow.mlflow_test.password']
mlflow.set_tracking_uri(trackingServerUri)

# MLflow的experiment
mlflow.set_experiment(experimen_name)

# wait_until_ready函式
def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name = model_name,
      version = model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)


#%% 設定日期
# ===== 設定日期 =====
try:
    arg = sys.argv
    if len(arg) > 1:
        start_date = arg[1]
        end_date = arg[2]
    else:
        # 沒設定日期，預設抓6個月前的的資料來建模，根據預測的日期
        start_date = (pd.date_range(end=(pd.to_datetime(dt.date.today()) - dt.timedelta(days=8)).date(),periods=11, freq="MS")[0]).date()
        end_date = (pd.date_range(end=(pd.to_datetime(dt.date.today()) - dt.timedelta(days=8)).date(),periods=7, freq="MS")[0]+dt.timedelta(-1)).date()
    
    pred_month = int(pd.date_range(start=(  end_date ) ,periods=7, freq="MS")[6].strftime("%Y%m"))
    print(f'建模資料日期： {start_date} ~ {end_date}, 預測月份 : {pred_month}')
    
    
    #%% 建立模型
    # ===== 建立模型 =====
    # 建立資料
    train_data = pd.read_sql(f"SELECT * FROM game_info where register_date_utc8 between '{start_date}' and '{end_date}'", engine)
    # 刪除不需要的特徵，缺值補零
    train = train_data.copy().drop(['account_id','register_date','register_date_utc8'
                                    #,'diamond_consume','diamond_consume_avg','battle','diamond_sticker','diamond_sticker_avg'
                                    ], axis=1).fillna(0)
    # Train 類別轉數值
    # 平台、第一次職業，轉Target_revenue
    target_encoder = TargetEncoder(cols=['channel','first_player'],smoothing=0, min_samples_leaf=0)
    target_encoder_data = target_encoder.fit_transform(train[['channel','first_player']], train['revenue_180']).round(5)
    # 國家，轉MEstimate-Encoder
    MEstimate_encoder = MEstimateEncoder(cols=["country"], m=10.0)
    MEstimate_encoder_data = MEstimate_encoder.fit_transform(train[['country']], train['revenue_180']).round(5)
    # 合併encoding結果 
    train_encoded = pd.concat([target_encoder_data, MEstimate_encoder_data], axis=1)
    # 更新 Train 的類別資料
    train.update(train_encoded)
    # 類別型態轉成數值
    # train_type = train.dtypes
    train['channel'] = train['channel'].astype(float)
    train['country'] = train['country'].astype(float)
    train['first_player'] = train['first_player'].astype(float)
    # 分X跟Y
    y_train = train['revenue_180']
    X_train = train.drop(['revenue_180','revenue_120','revenue_90','revenue_60'], axis=1)
    
    # 保存類別轉數值文件，predict的資料會需要轉換
    import joblib
    # 創建一個字典來儲存encoder
    encoder_objects = {
        'target_encoder': target_encoder,
        'MEstimate_encoder': MEstimate_encoder
    }
    # 保存字典到一個.pkl 文件中
    joblib.dump(encoder_objects, 'encoders.pkl')

    # ===== Cross Validation 找最佳參數 =====
    print('模型跑50次CV找最佳參數中')
    best_alphas = []
    for i in range(1,51):
        cv = KFold(n_splits=10, shuffle=True)
        # 建立 LassoCV 模型，指定 alpha 的範圍和交叉驗證的次數
        lasso_cv = LassoCV(alphas=np.logspace(-4, 2, 100), cv=cv, verbose=False)
        # 訓練模型
        lasso_cv.fit(X_train, y_train)
        # 獲得最佳的 alpha 值並添加到列表中
        best_alpha = lasso_cv.alpha_
        best_alphas.append(best_alpha)
    # 計算平均的 alpha 值
    best_alpha = np.mean(best_alphas)
    print("Average alpha:", best_alpha)
    # 使用最佳 alpha 值建立最終模型
    model = Lasso(alpha=best_alpha, random_state=42)
    
    
    #%% 上報Mlflow
    # ===== 上報Mlflow =====
    # 創建temp資料夾
    path = os.path.join(os.getcwd(), "temp")
    os.mkdir(path)
    
    with mlflow.start_run():
        ### 使用最佳參數建模
        model = Lasso(alpha=best_alpha, random_state=42)
        
        print('交叉驗證算出10-fold結果')
        # 定義交叉驗證折數
        n_splits = 10
        cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        # 初始化空列表來存儲每個折數的評估結果
        y_mapes = []
        y_rmses = []
        # Cross Validation
        for fold, (train_index, test_index) in enumerate(cv.split(X_train)):
            # print(f"Fold {fold + 1}/{n_splits}")
            # 創建CV的train, test
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
            # 訓練CV模型
            model.fit(X_train_fold, y_train_fold)
            # 在CV驗證集上進行預測
            y_pred_fold = model.predict(X_val_fold)
            # mape結果 (整個fold)
            y_mape = round(abs(y_pred_fold.sum()-y_val_fold.sum())*100/y_val_fold.sum(),4)
            y_mapes.append(y_mape)
            # rmse結果 (各帳號算出)
            y_mse  = (y_val_fold-y_pred_fold) ** 2
            y_mse = np.mean(y_mse)
            y_rmse = np.sqrt(y_mse)
            y_rmses.append(y_rmse)
        cv_mape_result = round(np.mean(y_mapes),2)
        cv_rmse_result = round(np.mean(y_rmses),2)
        
        model.fit(X_train, y_train)
        
        ### 計算模型指標
        train_pred = model.predict(X_train)
        # Full_train_MAPE = (train_pred.sum() - y_train.sum()) / y_train.sum()
        model_start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        model_end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
        CV_MAPE = cv_mape_result
        CV_RMSE = cv_rmse_result
        Alpha = best_alpha
        # LASSO有feature selection的功能，過度收斂的特徵會被拔除，這邊篩選還有參數的數據就是選擇的特徵
        feature_num = len(X_train.columns[model.coef_ != 0])
        features = X_train.columns[model.coef_ != 0]
        coefficients = model.coef_[model.coef_ != 0]
    
        ### MLflow寫入模型、metrics
        mlflow.sklearn.log_model(model, "model", registered_model_name= model_name, serialization_format='pickle')
        mlflow.log_metric("start_date", model_start_date)
        mlflow.log_metric("end_date", model_end_date)
        mlflow.log_metric("pred_month", pred_month)
        mlflow.log_metric("CV_MAPE", CV_MAPE)
        mlflow.log_metric("CV_RMSE", CV_RMSE)
        mlflow.log_metric("Alpha",Alpha)
        mlflow.log_metric("feature_num", feature_num)
        
        # 匯出訓練結果 (CSV)
        train_result_df = pd.DataFrame()
        train_result_df['model_start_date'] = [model_start_date]
        train_result_df['model_end_date'] = [model_end_date]
        train_result_df['CV_MAPE'] = [CV_MAPE]
        train_result_df['CV_RMSE'] = [CV_RMSE]
        train_result_df['LASSO_best_Alpha'] = [Alpha]
        train_result_df.to_csv('model_result.csv', index=False)
            
        # temp資料夾中創all9fun_package資料夾
        os.mkdir(os.path.join(path,"all9fun_package"))
        
        # 產生測試用資料(供Levin使用)
        test_data = predict_data(path = os.path.join(os.getcwd(), 'test.csv'), mode = 'test')
        
        # 把檔案存進all9fun_package資料夾中
        for file in ["data_processing.py", "data_for_predict.py", "modeling.py", "predict.py", "All9funData.py", "test.csv", "features.csv", "encoders.pkl"]:
            if os.path.exists(os.path.join(os.getcwd(), file)):
                shutil.copy(os.path.join(os.getcwd(), file), os.path.join(path, "all9fun_package", file))
         
        # 把temp資料的檔案(包括all9fun_package資料夾)丟到 ./mount/mlflow_store/...
        mlflow.log_artifacts(path)
    
    print('更新Mlflow上的資訊')
    # 抓取MLflow上的版本資訊
    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    
    # 找出最新版本，跑wait_until_ready
    new_model_version = max([int(model_version_info.version) for model_version_info in model_version_infos])
    wait_until_ready(model_name, new_model_version)
    
    # Update最新版本
    client.update_model_version(
    name=model_name,
    version=new_model_version,
    description="This model version is new version."
    )
    
    # 把最新版本的模型的階段改成Production
    client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production"
    )
    
    mlflow.end_run()
    if os.path.exists(path):
        shutil.rmtree(path) 
    
#%% Exception寄信
except Exception as e:
    
    import traceback
    # 產生完整的錯誤訊息
    error_class = e.__class__.__name__ #取得錯誤類型
    detail = e.args[0] #取得詳細內容
    tb = sys.exc_info()[2] #取得Call Stack
    lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
    fileName = lastCallStack[0] #取得發生的檔案名稱
    lineNum = lastCallStack[1] #取得發生的行號
    # funcName = lastCallStack[2] #取得發生的函數名稱
    errMsg = f"File \"{fileName}\", line {lineNum} \n錯誤訊息：\n [{error_class}] {detail}"
    
    # 開始寄信
    receive = config['mail.to'] # 收信者
    # receive = 'chialun@all9fun.com' # 收信者
    sub = f"{project_name} 程式錯誤" # 信件主旨
    content = f"{project_name} 程式錯誤，錯誤訊息：\n {errMsg}" # 信件內容
    
    yag = yagmail.SMTP(user = config['mail.from'], password = config['mail.from_password']) # 寄件者
    yag.send(to=receive, subject=sub,
             contents= content)


