#%% package
# 呼叫data_processing模組，處理資料並寫進mysql DB
project_name = 'game_ltv_insert_data'

### 載入套件
import os
import sys
import json
import pandas as pd
import socket
import datetime as dt
import sqlalchemy
from data_processing import data_processing
import yagmail
import warnings
warnings.filterwarnings("ignore")

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

#%% 設定日期
try:
    # ===== 設定日期 =====
    ltv_days = 180
    
    arg = sys.argv
    if len(arg) > 1:
        start_date = arg[1]
        end_date = arg[2]
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
    else:
        # 沒設定日期，預設跑180天內的數據更新還沒滿180日的數據還有剛滿7日數據的新註冊帳號
        start_date = (pd.to_datetime(dt.date.today()) - dt.timedelta(days=181)).date()
        end_date = (pd.to_datetime(dt.date.today()) - dt.timedelta(days=8)).date()
        
    print(f"日期區間：{start_date} ~ {end_date}、預測LTV天數:{ltv_days}")

    # 計算每個迴圈的起始日期和終止日期
    date_diff = (end_date-start_date).days
    loop_size = 30
    num_loops = (date_diff // loop_size)+1
    
    #%% 下載遊戲API資料寫入DB
    # 下載遊戲API資料處理後寫入DB，每次迴圈最多更新30天
    for i in range(0,num_loops):
        # i = 0
        loop_range = i*loop_size
        loop_start_date = start_date + dt.timedelta(days=loop_range)
        if (loop_start_date + dt.timedelta(days=loop_size) >= end_date + dt.timedelta(days=-1)):
            loop_end_date = end_date
        else :
            loop_end_date = loop_start_date + dt.timedelta(days=loop_size - 1)
        print(f"正在執行日期區間：{loop_start_date} ~ {loop_end_date}")
        
        ### 算時間起始
        import time
        start_time = time.time()
        
        # 建立資料
        # data_processing 會將每個帳號的前7天做數據統整，轉成特徵X跟累積營收Y撈下來
        data_fn = data_processing(token = game_token, start_date = loop_start_date, end_date = loop_end_date, ltv_days = ltv_days)
        data = data_fn.merge_data()
        
        # 資料寫入、迴圈處理每個batch、每個batch處理1000筆資料
        table_name = 'game_info'
        batch_size = 1000
        
        if len(data) > 0:
            print(f"開始寫入表 '{table_name}'")
            
            # DUPLICATE_KEY_UPDATE_SQL，若主鍵重覆就會覆蓋更新資料
            DUPLICATE_KEY_UPDATE_SQL = " ON DUPLICATE KEY UPDATE " + \
                ", ".join(["%s = VALUES(%s)" %(i, i) for i in data.columns])
            
            # columns
            col_names = ", ".join(data.columns)
            
            for i in range(0, len(data), batch_size):    #0, 1000, 2000
                
                batch_data = data[i:i+batch_size]        #只取需要的df，超過行數也不會報錯  
                
                temp_list = []
                for index, row in batch_data.iterrows():
                    temp_list.append(str(tuple(row)))    #組成(a,b,c)格式，放入list
                
                insert_values = ', '.join(temp_list)
        
                insert_sql = f"INSERT {table_name} ({col_names}) " \
                    f"VALUES {insert_values} {DUPLICATE_KEY_UPDATE_SQL}"
                    
                insert_sql = insert_sql.replace("nan", "NULL") # None 轉 NULL
                    
                engine.execute(insert_sql)
                print(f"數據插入成功，從{i+1}到{i+len(batch_data)}")
                
        else:
            print("dataframe為空，不執行插入DB")

        print(f"DB寫入完成，共寫入{len(data)}筆")
        
        ### 時間結算
        end_time = time.time()
        print(f"程式執行時間: {end_time - start_time} 秒")    
    
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
    sub = f"{project_name} 程式錯誤" # 信件主旨
    content = f"{project_name} 程式錯誤，錯誤訊息：\n {errMsg}" # 信件內容
    
    yag = yagmail.SMTP(user = config['mail.from'], password = config['mail.from_password']) # 寄件者
    yag.send(to=receive, subject=sub,
             contents= content)
