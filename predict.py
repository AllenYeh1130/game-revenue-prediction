#%%
# 丟進預測用資料(csv)與模型(pkl)，進行預測並處理預測出的結果
project_name = 'game_ltv_predict'

import pandas as pd
import joblib
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# data_path = 'test.csv'
# model_path = 'model.pkl'

#%%
def predict_fun(model_path, data_path):
    
    # 讀取data
    data = pd.read_csv(data_path)
    
    # 讀取model
    model = joblib.load(model_path)
            
    # 留下帳號
    data[['account_id','register_date_utc8']] = data[['account_id','register_date_utc8']].astype(str)
    acc_id = data[['account_id','register_date_utc8']]
    
    # 預測結果
    data = data.drop(['account_id','register_date_utc8'], axis=1)
    pred = model.predict(data)
    
    # 整合資料
    result = pd.DataFrame({'#account_id' : acc_id['account_id']
                           , 'ts_replace' :  acc_id['register_date_utc8'].apply(lambda date_str: int(datetime.strptime(date_str + ' 22:00:00', "%Y-%m-%d %H:%M:%S").timestamp()))*1000  
                           , 'pred' : pred
                           })
    result['#event_id'] = result['#account_id'].astype(str) + '_' + result['ts_replace'].astype(str)

    # report 
    result = result.to_dict('records')
    
    ### 測試特定ID派發獎勵用，False表示不測試
    flag = False
    if flag:
        id_test_list = ['204621','46896','1564370','1374041','1312388','abcd1234aa']   #要測試的id
        result = result.iloc[:len(id_test_list), :] #只保留對應ID數量的列
        result['#account_id'] = id_test_list   #把原ID替換為要測試的ID
    
    # 輸出為MLflow需要的格式
    lose_report = dict()
    lose_report["result"] = result
    lose_report["#event_name"] = 'ai_pred_180revenue'
    lose_report["#type"] = 'track_update'
    lose_report["ts"] = int(str(time.time()).split(".")[0])*1000
        
    return lose_report