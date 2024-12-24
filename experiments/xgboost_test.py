# -*- coding: utf-8 -*-
"""
# This script contains experimental code for validating XGboost models 
# during the early stages of the project. It was not used in production.
"""

# 變更當前工作目錄
import os
new_directory = r"xgboost_path"
os.chdir(new_directory)

#%%
import pandas as pd
import datetime as dt
import sqlalchemy
from category_encoders import TargetEncoder
from category_encoders import MEstimateEncoder
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
def get_month(x): return dt.datetime(x.year,x.month,1)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


##### 參數設定 #####
# 資料庫連線
HOST = "HOST"
DATABASE = "DATABASE"
USER = "USER"
PASSWORD = "PASSWORD"
engine_stmt = 'mysql+pymysql://%s:%s@%s:1234/%s' % (USER, PASSWORD,HOST,DATABASE)
engine = sqlalchemy.create_engine(engine_stmt, echo=False)

# 要預測的起始日、結束日
start_date = pd.to_datetime('2022-08-01')
end_date = pd.to_datetime('2023-01-15')

# 預測標的
revenue = 'revenue_180'

# 結果初始化
train_results = []
test_results = []

# 分要預測的各週起始日、結束日
week_start_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
week_end_dates = week_start_dates + pd.DateOffset(days=6)

# 模型月份名稱
loop_month = get_month(pd.to_datetime(start_date)).date()+dt.timedelta(-1)


##### 分週評估預測成效 #####
for start, end in zip(week_start_dates, week_end_dates):
    print("Week Start:", start.date(), "Week End:", end.date())
    # 訓練月份
    n_week = pd.to_datetime(start).date()
    n_month = get_month(pd.to_datetime(start)).date()
    # 4個月訓練資料 (例: 2022/3/1~2022/6/30)
    train_start_date = (pd.date_range(end=n_week,periods=11, freq="MS")[0]).date()
    train_end_date = (pd.date_range(end=n_week,periods=11, freq="MS")[4]+dt.timedelta(-1)).date()
    # 1週當測試資料 (例: 2023/1/9~2023/1/15)
    test_start_date = start.date()
    test_end_date = end.date()
    # 模型月份名稱
    loop_month = get_month(pd.to_datetime(start_date)).date()+dt.timedelta(-1)


    ##### 建立 Train 資料 #####
    # 如果該迴圈(預測週)共用同個模型就不重新建立模型
    if  loop_month != n_month :
        print(f"訓練執行日期區間：{train_start_date} - {train_end_date}")
        # 建立資料
        new_train_data = pd.read_sql("""SELECT * FROM game_info where register_date between '%s' and '%s'""" %(train_start_date, train_end_date), engine)
        train = new_train_data.copy().drop(['account_id','register_date'
                                        #,'diamond_consume','diamond_consume_avg','battle','diamond_sticker','diamond_sticker_avg'
                                        ], axis=1).fillna(0)
        # Train 類別轉數值
        # 平台、第一次職業，轉Target_revenue
        target_encoder = TargetEncoder(cols=['channel','first_player'],smoothing=0, min_samples_leaf=0)
        target_encoder_data = target_encoder.fit_transform(train[['channel','first_player']], train[revenue]).round(5)
        # 國家，轉MEstimate-Encoder
        MEstimate_encoder = MEstimateEncoder(cols=["country"], m=10.0)
        MEstimate_encoder_data = MEstimate_encoder.fit_transform(train[['country']], train[revenue]).round(5)
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
        y_train = train[revenue]
        X_train = train.drop(['revenue_180','revenue_120','revenue_90','revenue_60'], axis=1)
        # log_transforamtion轉換
        # y_train = np.log(train[['revenue']]+1)
        # train[[revenue]].sum()
        # train[['pay_amount_7days']].sum()
        # y_train.to_csv('y_train.csv', index=False)
        # X_train.to_csv('X_train.csv', index=False)
        
        # 統計訓練資料7天跟180天的倍率
        statistic_ratio = (train[revenue].sum())/(train['pay_amount_7days'].sum())


    ##### 建立 Test 資料 #####
    # 建立資料
    print(f"測試執行日期區間：{test_start_date} - {test_end_date}")
    new_test_data = pd.read_sql("""SELECT * FROM game_info where register_date between '%s' and '%s'""" %(test_start_date, test_end_date), engine)
    test = new_test_data.copy().drop(['account_id','register_date'
                                  #,'diamond_consume','diamond_consume_avg','battle','diamond_sticker','diamond_sticker_avg'
                                  ], axis=1).fillna(0)
    # Test 類別轉數值 encoding，沿用train的encoder
    target_encoder_data = target_encoder.transform(test[['channel','first_player']])
    MEstimate_encoder_data = MEstimate_encoder.transform(test[['country']])
    # 合併 encoding 結果
    test_encoded = pd.concat([target_encoder_data, MEstimate_encoder_data], axis=1)
    # 更新 Test 的類別資料
    test.update(test_encoded)
    # 類別型態轉成數值
    # test_type = test.dtypes
    test['channel'] = test['channel'].astype(float)
    test['country'] = test['country'].astype(float)
    test['first_player'] = test['first_player'].astype(float)
    # 分X跟Y
    y_test = test[revenue]
    X_test = test.drop(['revenue_180','revenue_120','revenue_90','revenue_60'], axis=1)
    # log_transforamtion轉換
    # y_test = np.log(test[['revenue']]+1)
    # test[['revenue_180']].sum()
    # test[['pay_amount_7days']].sum()
    # X_test.to_csv('X_test.csv', index=False)
    # y_test.to_csv('y_test.csv', index=False)
    
    
    '''
    ##### MinMax正規化 #####
    from sklearn.preprocessing import MinMaxScaler
    from decimal import Decimal
    def float_to_decimal(value):
        return Decimal(str(value)).quantize(Decimal('0.00000'))
    
    # 儲存原本的欄位名稱
    column_names = X_train.columns.tolist() 
    # 創建MinMaxScaler物件
    scaler = MinMaxScaler()
    # X_test轉換成decimal
    X_test = X_test.applymap(float_to_decimal)
    # 將數據進行最小-最大規範化，並使用pd.DataFrame保留欄位名稱
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=column_names)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=column_names)
    '''


    ##### optuna找最佳參數CV版本 #####
    np.random.seed(42) 
    
    def objective(trial, X, y):
        
        param = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0)
        }
        
        '''
        param = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=100),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.1, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.2, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.2, 1.0),
            'random_state': 42
        }
        '''
        
        # Cross Validation
        # cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
        cv = KFold(n_splits=10, shuffle=False)
        mse_result = []
        for train_index, test_index in cv.split(X_train, y_train):  

            # 拆訓練組、測試組
            X_train_opt, X_test_opt = X.iloc[train_index], X.iloc[test_index]
            y_train_opt, y_test_opt = y.iloc[train_index], y.iloc[test_index]
            
            # Fit the model
            optuna_model = xgb.XGBRegressor(**param)
            optuna_model.fit(X_train_opt, y_train_opt)
        
            # 使用model預測的結果
            pred_test = optuna_model.predict(X_test_opt)
            # print(pred_test)
            
            # Evaluate predictions
            mse_score = mean_squared_error(y_test_opt, pred_test)
            mse_result.append(mse_score)
            
        return np.mean(mse_result)

    # 根據參數範圍找最佳參數解，並建立模型
    # Create the study
    if  loop_month != n_month :
        study = optuna.create_study(direction='minimize', study_name='regression')
        func = lambda trial: objective(trial, X = X_train, y = y_train)
        study.optimize(func, n_trials=50, n_jobs=5)       # n_trials為要跑的試驗次數
    model = xgb.XGBRegressor(**study.best_params)


    ##### Cross Validation #####
    # 定義交叉驗證折數
    n_splits = 10
    cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    # 初始化空列表來存儲每個折數的評估結果
    y_ae_results = []
    y_results = []
    for fold, (train_index, test_index) in enumerate(cv.split(X_train)):
        # print(f"Fold {fold + 1}/{n_splits}")
        # 創建CV的train, test
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        # 訓練CV模型
        model.fit(X_train_fold, y_train_fold)
        # 在CV驗證集上進行預測
        y_pred_fold = model.predict(X_val_fold)
        y_as_result = abs(y_pred_fold.mean()-y_val_fold.mean())
        y_result = round(abs(y_pred_fold.sum()-y_val_fold.sum())*100/y_val_fold.sum(),4)
        # print(y_as_result)
        y_ae_results.append(y_as_result)
        y_results.append(y_result)
    mean_cv_result = round(np.mean(y_results),2)
    mean_cv_ae_result = round(np.mean(y_ae_results),2)
    
    # 建立模型
    model.fit(X_train, y_train)


    ##### Train 結果 #####
    if  loop_month != n_month :
        train_pred = model.predict(X_train)
        train_df = pd.DataFrame()
        train_df['true_train_ltv'] = [y_train.mean()]
        train_df['true_train_revenue'] = [y_train.sum()]
        train_df['pred_train_ltv'] = [train_pred.mean()]
        train_df['pred_train_revenue'] = [train_pred.sum()]
        train_df['ape']  = mape(train_df['true_train_ltv'], train_df['pred_train_ltv'])
        train_df['cv_train'] = mean_cv_result
        # 紀錄參數
        best_params = pd.Series(study.best_params)
        train_df['best_params'] = best_params.to_string()
        # print結果
        print('Train result, %s ~ %s, CV誤差:%s%%, FuLL train誤差:%s%%'%(train_start_date,train_end_date, mean_cv_result,round(mape(train_df['true_train_ltv'], train_df['pred_train_ltv']),2)))
        # 紀錄結果
        train_results.append(train_df)


    ##### Test 結果 #####
    test_pred = model.predict(X_test)
    test_df = pd.DataFrame()
    test_df['true_test_revenue'] = [y_test.sum()]
    test_df['true_test_ltv'] = [y_test.mean()]
    test_df['control_test_revenue'] = (test['pay_amount_7days'].sum()*statistic_ratio)
    test_df['control_test_ltv'] = (test['pay_amount_7days'].sum()*statistic_ratio)/len(test)
    test_df['control_ape']  = mape(test_df['true_test_ltv'], test_df['control_test_ltv'])
    test_df['pred_test_revenue'] = [test_pred.sum()]
    test_df['pred_test_ltv'] = [test_pred.mean()]
    test_df['pred_ape']  = mape(test_df['true_test_ltv'], test_df['pred_test_ltv'])
    test_df['cv_train'] = mean_cv_result
    # 紀錄參數
    best_params = pd.Series(study.best_params)
    test_df['best_params'] = best_params.to_string()
    # print結果
    print('Test result, %s ~ %s, 誤差:%s%%'%(test_start_date,test_end_date,round(mape(test_df['true_test_ltv'], test_df['pred_test_ltv']),2)))
    # 紀錄結果
    test_results.append(test_df)
    
    # 已經執行train的月份紀錄在loop_month
    loop_month = n_month


##### 紀錄結果 #####
train_result = pd.concat(train_results, ignore_index=True)
train_result.to_csv(f'train_result_{train_start_date}_{train_end_date}.csv', index=False)

test_result = pd.concat(test_results, ignore_index=True)
test_result.to_csv(f'test_result_{test_start_date}_{test_end_date}.csv', index=False)
    

##### xgboost的特徵重要性 #####
import matplotlib.pyplot as plt
# 獲取特徵重要性
feature_importance = model.feature_importances_
# 創建特徵名稱列表
feature_names = X_train.columns
# 將特徵重要性和特徵名稱組成一個DataFrame
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
# 按照特徵重要性降序排序
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# 顯示特徵重要性排名
print(feature_importance_df)
# 可視化特徵重要性
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
# 特徵csv匯出
feature_importance_df.to_csv(f'feature_importance_df_{test_start_date}_{test_end_date}.csv', index=False)