import datetime as dt
import requests
import urllib
import pandas as pd
from io import StringIO
import numpy as np

class data_processing (object):

    # 定義變數 (註冊開始時間、註冊結束時間、欲預測ltv天數)
    def __init__(self, token, start_date, end_date, ltv_days):
        # self.token
        self.token = token
        # 時間變數
        self.start_date = start_date
        self.end_date = end_date
        self.start_time = pd.to_datetime(start_date) + dt.timedelta(hours=13)
        self.end_time = (pd.to_datetime(end_date) + dt.timedelta(days=1) + dt.timedelta(hours=13) - dt.timedelta(milliseconds=1)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # 7日資料最後時間點
        self.end_time_data = (pd.to_datetime(end_date) + dt.timedelta(days=7) + dt.timedelta(hours=13) - dt.timedelta(milliseconds=1)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # 4點當做一天開始起始、結束
        self.start_time_reset = pd.to_datetime(start_date) + dt.timedelta(hours=17)
        self.end_time_data_reset = (pd.to_datetime(end_date) + dt.timedelta(days=7) + dt.timedelta(hours=17) - dt.timedelta(milliseconds=1)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # 預測營收日
        self.ltv_days = ltv_days
        # 預測營收日最後時間
        self.end_time_data_ltv = (pd.to_datetime(end_date) + dt.timedelta(days=ltv_days) + dt.timedelta(hours=13) - dt.timedelta(milliseconds=1)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"build data start !! start:{start_date} , end:{end_date}, target days:{ltv_days}")

    # 每日平均
    def calculate_day_mean(g, login_days, value, column_name):
        avg_data = g.groupby('account_id')[value].agg('sum').reset_index(name='value')
        avg_data = pd.merge(avg_data, login_days, on='account_id', how='inner')
        avg_data[column_name] = round(avg_data['value']/avg_data['day_count'],2)
        avg_data = avg_data.drop(['value','day_count'], axis=1)
        return avg_data
    
    # 斜率
    def calculate_slope(g, x_col, y_col):
        return ((g[y_col] - g[y_col].shift()) / (g[x_col] - g[x_col].shift())).mean()
    
    # 正負斜率次數總和
    def calculate_slope_count(g, x_col, y_col):
        slope = (g[y_col] - g[y_col].shift()) / (g[x_col] - g[x_col].shift())
        slope = np.where(slope > 0, 1, np.where(slope < 0, -1, 0))
        return slope.sum()
    
    # TA資料
    def merge_data(self):
        ###### 平台、國家 ######
        # 如果玩家屬性沒資料看註冊資料，沒有的話看登入資料
        print("channel_country_data start !!")
        SQL = '''
        SELECT user.account_id, register_date, register_date_utc8, COALESCE(channel,register.package,login.package,'other') channel, COALESCE(country,register.country_code,login.country_code,'other') country
        FROM (
            --$part_date
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date, Date(register_time) register_date_utc8, initial_package channel, initial_country_code country FROM ta.user_info
            where "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
        ) user LEFT JOIN (
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, package, "#country_code" country_code 
            FROM  game_data
            where "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND ("$part_event" IN ('register'))
        ) register on user.account_id = register.account_id LEFT JOIN (
            -- 第一次登入
            SELECT * FROM (
                SELECT "#account_id" account_id, date_add('hour', -13, "#event_time") as event_time, package, "#country_code" country_code 
                , row_number() over(partition by "#account_id" order by "#event_time" ASC) num
                FROM  game_data
                where "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND ("$part_event" IN ('login'))
            )
            WHERE num = 1
        ) login on user.account_id = login.account_id''' %(self.start_time, self.end_time, self.start_time, self.end_time, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        channel_country_data = pd.read_csv(StringIO(s))
        # 特例處理
        channel_country_data.loc[channel_country_data['country'] == '419', 'country'] = 'other'
        channel_country_data.loc[channel_country_data['country'] == 'None', 'country'] = 'other'
        channel_country_data = channel_country_data.fillna('other')
        
        
        ###### 第一次選擇職業 ######
        print("first_player_data start !!")
        SQL = '''
        SELECT account_id, COALESCE(first_player,'no_player') first_player FROM (
            --$part_date
            SELECT "#account_id" account_id, "#user_id" user_id FROM ta.user_info
            where "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
        ) register LEFT JOIN (
            SELECT "#user_id" user_id, tag_value as first_player FROM cluster WHERE "cluster_name"='contract_first_plyer'
        ) first_player on register.user_id = first_player.user_id''' %(self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        first_player_data = pd.read_csv(StringIO(s))
        # first_player_data.groupby('first_player').count()
        
        
        ###### 登入天數 ######
        print("login_day_data start !!")
        SQL = '''
        SELECT register.account_id, register_date, login_date, date_diff('day', DATE(register_date), DATE(login_date))+1 as day FROM (
            --$part_date
            SELECT DISTINCT "#account_id" as account_id, Date(date_add('hour', -13, "#event_time")) as login_date
            FROM  game_data
            WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('login'))
        ) login INNER JOIN (
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
            where "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
        ) register on login.account_id = register.account_id
        where date_diff('day', DATE(register_date), DATE(login_date))+1 <= 7
        order by register.account_id, date_diff('day', DATE(register_date), DATE(login_date))+1 ASC''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        login_day_data = pd.read_csv(StringIO(s))
        # 各天是否登入
        login_day = pd.get_dummies(login_day_data['day'])
        login_day = pd.concat([login_day_data['account_id'],login_day], axis=1)
        login_day.columns = ['account_id','login_day1', 'login_day2', 'login_day3', 'login_day4', 'login_day5', 'login_day6', 'login_day7']
        login_day = login_day.groupby('account_id').agg(sum).reset_index()
        # 登入幾天
        login_days = login_day_data.groupby('account_id')['day'].agg(day_count =('count')).reset_index()
        # 最大連續登入、diff算出鄰近數值差異、ne(1)把1變成False、cumsum分組看數量，只有連續的會累加所以選max
        login_max_consecutive_days = login_day_data.groupby('account_id')['day'].apply(lambda x: x.diff().fillna(1).ne(1).cumsum().value_counts().max()).reset_index()
        # 彙總
        login_day = pd.merge(login_day, login_days, on='account_id', how='outer')\
                    .merge(login_max_consecutive_days, on='account_id', how='outer')
        
        
        ###### 登入時長 ######
        print("online_time_data start !!")
        SQL = '''
        SELECT *, row_number() over(partition by account_id ORDER BY day ASC) num FROM (
            SELECT register.account_id, register_date, login_date, date_diff('day', DATE(register_date), DATE(login_date))+1 as day, online_time FROM (
            --$part_date
                SELECT "#account_id" as account_id, Date(date_add('hour', -13, "#event_time")) as login_date, sum("online_time") online_time FROM  game_data
                WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('logout'))
                group by "#account_id", Date(date_add('hour', -13, "#event_time"))
            ) login INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                where "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register on login.account_id = register.account_id
            where date_diff('day', DATE(register_date), DATE(login_date))+1 <= 7
            order by register.account_id, login_date ASC
        )''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        online_time_data = pd.read_csv(StringIO(s))
        # 各天上線時長
        online_time = online_time_data.pivot_table(index='account_id', columns='day', values='online_time').fillna(0)
        online_time.columns = ['onlinte_time_day1', 'onlinte_time_day2', 'onlinte_time_day3', 'onlinte_time_day4', 'onlinte_time_day5', 'onlinte_time_day6', 'onlinte_time_day7']
        online_time = online_time.reset_index()
        # 平均日上線時長
        online_time_avg = round(data_processing.calculate_day_mean(online_time_data, login_days, value='online_time', column_name = 'online_time_avg'),0)
        # 上線時長斜率
        online_time_slope = round(online_time_data.groupby('account_id').apply(data_processing.calculate_slope, x_col='num', y_col='online_time').fillna(0).reset_index(name='online_time_slope'),0)
        # 上線時長正負斜率次數總和
        online_time_slope_count = online_time_data.groupby('account_id').apply(data_processing.calculate_slope_count, x_col='num', y_col='online_time').fillna(0).reset_index(name='online_time_slope_count')
        # 彙總
        online_time = pd.merge(online_time, online_time_avg, on='account_id', how='outer')\
                        .merge(online_time_slope, on='account_id', how='outer')\
                        .merge(online_time_slope_count, on='account_id', how='outer')
        
        
        ###### 第一次登入時間 ######
        # 第一天的數據可能會受到安裝時間影響第一次上線時間較晚，所以排除第一天的資料
        # 註冊以UTC-5的0:00當一天開始，第一次登入時間以UTC-5的4:00當一天開始
        print("first_online_data start !!")
        SQL = '''
        SELECT *, row_number() over(partition by account_id ORDER BY login_date ASC) num FROM (
            SELECT register.account_id, login_date, first_online_hour FROM (
                --$part_date
                SELECT account_id, login_date, hour(min(online_time)) as first_online_hour FROM (
                    SELECT "#account_id" as account_id, Date(date_add('hour', -17, "#event_time")) as login_date, date_add('hour', -13, "#event_time") online_time FROM  game_data
                    WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('login'))
                )
                GROUP BY account_id, login_date
            ) firstlogin INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                where "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register on firstlogin.account_id = register.account_id
            where date_diff('day', DATE(register_date), DATE(login_date))+1 between 2 and 7
        order by register.account_id, login_date
        )''' %(self.start_time_reset, self.end_time_data_reset, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        first_online_data = pd.read_csv(StringIO(s))
        # 第一次登入斜率
        first_online_slope = round(first_online_data.groupby('account_id').apply(data_processing.calculate_slope, x_col='num', y_col='first_online_hour').fillna(0).reset_index(name='first_online_slope'),2)
        # 第一次登入正負斜率次數總和
        first_online_slope_count = first_online_data.groupby('account_id').apply(data_processing.calculate_slope_count, x_col='num', y_col='first_online_hour').fillna(0).reset_index(name='first_online_slope_count')
        # 彙總
        first_online = pd.merge(first_online_slope, first_online_slope_count, on='account_id', how='outer')
        
        
        ###### 訓練等級 ######
        print("max_level_data start !!")
        SQL = '''
        SELECT *, row_number() over(partition by account_id ORDER BY event_date ASC) num FROM (
            SELECT register.account_id, event_date, level_training FROM (
                --$part_date
                SELECT account_id, event_date, max(level_training) as level_training FROM (
                    SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, level_training FROM  game_data
                    where "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('logout'))
                ) 
                GROUP BY account_id, event_date
            ) level INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                where "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register on level.account_id = register.account_id
            where date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
            order by register.account_id, event_date
        )''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        max_level_data = pd.read_csv(StringIO(s))
        # 最大訓練等級
        max_level = max_level_data.groupby('account_id')['level_training'].max().reset_index()
        # 訓練等級斜率
        max_level_slope = round(max_level_data.groupby('account_id').apply(data_processing.calculate_slope, x_col='num', y_col='level_training').fillna(0).reset_index(name='max_level_slope'),2)
        # 順練等級正負斜率次數總和
        max_level_slope_count = max_level_data.groupby('account_id').apply(data_processing.calculate_slope_count, x_col='num', y_col='level_training').fillna(0).reset_index(name='max_level_slope_count')
        # 彙總
        max_level = pd.merge(max_level, max_level_slope, on='account_id', how='outer')\
                        .merge(max_level_slope_count, on='account_id', how='outer')
        
        
        ###### 技能升級 ######
        print("talent_data start !!")
        SQL = '''
        SELECT * FROM (
            SELECT talent.*, row_number () over (partition by talent.account_id, skill_id order by event_time desc) as num  FROM (
                --$part_date
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, "#event_time" event_time, skill_id, player_skill_after
                FROM  game_data
                WHERE ("#event_time" BETWEEN CAST('%s' AS timestamp) AND CAST('%s' AS timestamp))
                AND "$part_event" = 'development_talent'
            ) talent INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                where "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register on talent.account_id = register.account_id
            WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
        )
        WHERE num = 1
        order by account_id, event_date''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        talent_data = pd.read_csv(StringIO(s))
        # 技能升級總次數
        talent_sum = talent_data.groupby('account_id')['player_skill_after'].sum().reset_index(name='talent_sum')
        # 達到3等技能總數 (7天玩家的平均2.73)
        talent_max_count = talent_data[talent_data['player_skill_after'] == 3].groupby('account_id')['skill_id'].count().reset_index(name='talent_max_count')
        # 彙總
        talent = pd.merge(talent_sum, talent_max_count, on='account_id', how='outer')
        
        
        ###### 最大球員評價、突破、評價*突破  ######
        print("max_player_rating_evolve start !!")
        SQL = '''
        --玩家的球員最大階級，取player_name可以跟突破數據串接
        with contract_new_player as 
            (
                --$part_date
                --突破數據的player_id是player_name，這邊將player_name轉成player_id方便串接
                SELECT "#account_id", player_name as player_id, player_type
                , CASE WHEN player_rating = 500 AND player_type = '球星' then 600 else player_rating END player_rating
                FROM  game_data a JOIN (
                    SELECT "player_id@player_id" player_id, "player_id@player_type" player_type FROM ta_dim.dim_18_0_98333
                ) b on a.player_id = b.player_id 
                WHERE ("#event_time" BETWEEN CAST('%s' AS timestamp) AND CAST('%s' AS timestamp))
                AND "$part_event" = 'contract_new_player'
            )
            ,max_player_id AS (   
            -- 篩選max_rating對應的player_id，這邊若1個ID有2個以上同評價球員都會篩出來
                SELECT "#account_id", player_id, player_rating
                FROM contract_new_player
                WHERE ("#account_id", player_rating) in (
                    SELECT "#account_id", max(player_rating) FROM contract_new_player GROUP BY "#account_id"
                )
            )
        --最大階級球員最大突破
        SELECT player_evolve.account_id, max(player_rating) max_rating, max(player_evolve_after) max_evolve
        , max(player_rating)*max(player_evolve_after) rating_evolve_score FROM (
            --$part_date
            SELECT a."#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, b.player_rating, player_evolve_after
            FROM  game_data a INNER JOIN max_player_id b ON a."#account_id" = b."#account_id" AND a.player_id = b.player_id
            WHERE ("#event_time" BETWEEN CAST('%s' AS timestamp) AND CAST('%s' AS timestamp))
            AND "$part_event" = 'development_evolve'
        ) player_evolve INNER JOIN (
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
            where "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
        ) register on player_evolve.account_id = register.account_id
        where date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
        group by player_evolve.account_id''' %(self.start_time, self.end_time_data, self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        # 彙總
        max_player_rating_evolve = pd.read_csv(StringIO(s))
        
        
        ###### 任務 ###### 
        # 註冊以UTC-5的0:00當一天開始，第一次登入時間以UTC-5的4:00當一天開始
        print("quest_data start !!")
        SQL = '''
        SELECT *, row_number() over(partition by account_id ORDER BY login_date ASC) num FROM (
            SELECT register.account_id, login_date, daily_quest_num FROM (
                --$part_date
                SELECT account_id, login_date, count(daily_quest_id) as daily_quest_num FROM (
                    SELECT "#account_id" as account_id, Date(date_add('hour', -17, "#event_time")) as login_date, daily_quest_id FROM  game_data
                    WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('daily_quest'))
                ) 
                GROUP BY account_id, login_date
            ) daily_quest INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                where "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register on daily_quest.account_id = register.account_id
            where date_diff('day', DATE(register_date), DATE(login_date))+1 <= 7
            order by register.account_id, login_date
        )''' %(self.start_time_reset, self.end_time_data_reset, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        quest_data = pd.read_csv(StringIO(s))
        # 日任務完成總數
        quest = quest_data.groupby('account_id')['daily_quest_num'].sum().reset_index(name = 'quest')
        # 日任務登入斜率
        quest_slope = round(quest_data.groupby('account_id').apply(data_processing.calculate_slope, x_col='num', y_col='daily_quest_num').fillna(0).reset_index(name='quest_slope'),2)
        # 日任務正負斜率次數總和
        quest_slope_count = quest_data.groupby('account_id').apply(data_processing.calculate_slope_count, x_col='num', y_col='daily_quest_num').fillna(0).reset_index(name='quest_slope_count')
        # 彙總
        quest = pd.merge(quest, quest_slope, on='account_id', how='outer')\
                        .merge(quest_slope_count, on='account_id', how='outer')
        
        
        ###### 第一次儲值天數、金額 ######
        print("first_pay start !!")
        SQL = '''
        SELECT register.account_id, date_diff('day', DATE(register_date), DATE(event_date))+1 first_pay_day, pay_amount first_pay_amount FROM (
            --$part_date
            SELECT "#account_id" as account_id, Date(date_add('hour', -13, "#event_time")) as event_date, date_add('hour', -13, "#event_time") event_time, pay_amount
            , row_number() over(PARTITION BY "#account_id" ORDER BY "#event_time") pay_num FROM  game_data
            WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('order_finish'))
        ) payamount INNER JOIN (
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
            WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
        ) register ON payamount.account_id = register.account_id
        WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
        AND pay_num = 1''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        # 彙總
        first_pay = pd.read_csv(StringIO(s))
        
        
        ###### 儲值 ######
        print("pay_data start !!")
        SQL = '''
        SELECT register.account_id, event_date, pay_amount, pay_count, row_number () over(PARTITION BY register.account_id ORDER BY event_date) num FROM (
            --$part_date
            SELECT "#account_id" as account_id, Date(date_add('hour', -13, "#event_time")) as event_date, round(sum(pay_amount),2) pay_amount, count(pay_amount) pay_count FROM  game_data
            WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('order_finish'))
            GROUP BY "#account_id", Date(date_add('hour', -13, "#event_time"))
        ) payamount INNER JOIN (
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
            WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
        ) register ON payamount.account_id = register.account_id
        WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
        ORDER BY register.account_id, event_date''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        pay_data = pd.read_csv(StringIO(s))
        # 7日總付費金額
        pay_amount_7days = pay_data.groupby('account_id')['pay_amount'].sum().reset_index(name='pay_amount_7days')
        # 7日總儲值次數
        pay_count_7days = pay_data.groupby('account_id')['pay_count'].sum().reset_index(name='pay_count_7days')
        # 平均每日儲值
        pay_amount_avg = data_processing.calculate_day_mean(pay_data, login_days, value='pay_amount', column_name = 'pay_amount_avg')
        # 儲值金額斜率
        pay_amount_slope = round(pay_data.groupby('account_id').apply(data_processing.calculate_slope, x_col='num', y_col='pay_amount'),2).fillna(0).reset_index(name='pay_amount_slope')
        # 儲值金額正負斜率加總
        pay_amount_slope_count = pay_data.groupby('account_id').apply(data_processing.calculate_slope_count, x_col='num', y_col='pay_amount').fillna(0).reset_index(name='pay_amount_slope_count')
        # 彙總
        pay = pd.merge(pay_amount_7days, pay_count_7days, on='account_id', how='outer')\
                .merge(pay_amount_avg, on='account_id', how='outer')\
                .merge(pay_amount_slope, on='account_id', how='outer')\
                .merge(pay_amount_slope_count, on='account_id', how='outer')
        
        
        ###### IAA營收 ######
        # IAA營收數據使用iaa_income_to_id時區是UTC+8，註冊時間是UTC-5的話會有前13小時的IAA沒算到
        print("iaa_data start !!")
        SQL = '''
        SELECT register.account_id, iaa.event_date, date_diff('day', DATE(register_date), DATE(iaa.event_date))+1 iaa_day, iaa_freq, round(iaa_income,4) iaa_income, round((iaa_income/iaa_freq),6) as income_per_click
        , row_number () over(PARTITION BY register.account_id ORDER BY iaa.event_date) num FROM (
            -- $part_date
            SELECT "#account_id" as account_id, Date("#event_time") as event_date, sum(iaa_freq) iaa_freq, sum(round(iaa_income_predict,6)) iaa_income
            FROM  game_data
            -- event_time的時分不影響結果，原始數據都統一計算到23:00:00
            WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) 
            AND ("$part_event" IN ('iaa_income_to_id'))
            GROUP BY "#account_id", Date("#event_time")
        ) iaa INNER JOIN (
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
            WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
        ) register ON iaa.account_id = register.account_id
        WHERE date_diff('day', DATE(register_date), DATE(iaa.event_date))+1 <= 7
        ORDER BY register.account_id, iaa.event_date''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        iaa_data = pd.read_csv(StringIO(s))
        # 首點廣告日，0的狀況當作是第一天
        iaa_first_day = iaa_data.groupby('account_id')['iaa_day'].min().reset_index(name='iaa_first_day')
        iaa_first_day = iaa_first_day.replace(0, 1)
        # 7日總廣告點擊次數
        iaa_count_7days = iaa_data.groupby('account_id')['iaa_freq'].sum().reset_index(name='iaa_count_7days')
        # 7日總廣告金額
        iaa_income_7days = round(iaa_data.groupby('account_id')['iaa_income'].sum().reset_index(name='iaa_income_7days'),4)
        # 平均每次點擊營收
        iaa_income_per_click = round(iaa_data.groupby('account_id')['income_per_click'].min().reset_index(name='iaa_income_per_click'),6)
        # 平均每日廣告點擊次數，不算入第0天
        iaa_click_avg = data_processing.calculate_day_mean(iaa_data[iaa_data['iaa_day']>0], login_days, value='iaa_freq', column_name = 'iaa_click_avg')
        # 廣告點擊斜率，不算入第0天
        iaa_click_slope = round(iaa_data[iaa_data['iaa_day']>0].groupby('account_id').apply(data_processing.calculate_slope, x_col='num', y_col='iaa_freq'),2).fillna(0).reset_index(name='iaa_click_slope')
        # 廣告點擊正負斜率加總，不算入第0天
        iaa_click_slope_count = iaa_data[iaa_data['iaa_day']>0].groupby('account_id').apply(data_processing.calculate_slope_count, x_col='num', y_col='iaa_freq').fillna(0).reset_index(name='iaa_click_slope_count')
        # 彙總
        iaa = pd.merge(iaa_first_day, iaa_count_7days, on='account_id', how='outer')\
                .merge(iaa_income_7days, on='account_id', how='outer')\
                .merge(iaa_income_per_click, on='account_id', how='outer')\
                .merge(iaa_click_avg, on='account_id', how='outer')\
                .merge(iaa_click_slope, on='account_id', how='outer')\
                .merge(iaa_click_slope_count, on='account_id', how='outer')
        
        
        
        ###### 物品消耗 ######
        ###### 鑽石消耗 ######
        print("diamond_consume_data start !!")
        SQL = '''
        SELECT *, row_number() over(PARTITION BY account_id ORDER BY event_date) num FROM (
            SELECT diamond.account_id, event_date, sum(diamond_consume) diamond_consume FROM (
                --$part_date
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, diamond_consume_amount diamond_consume FROM  game_data
                WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('diamond_consume'))
                AND diamond_consume_amount > 0
            ) diamond INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register ON diamond.account_id = register.account_id
            WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
            GROUP BY diamond.account_id, event_date
        )''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        diamond_consume_data = pd.read_csv(StringIO(s))
        # 總鑽石消耗
        diamond_consume = diamond_consume_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_consume')
        # 平均每日鑽石消耗
        diamond_consume_avg = data_processing.calculate_day_mean(diamond_consume_data, login_days, value='diamond_consume', column_name = 'diamond_consume_avg')
        # 鑽石消耗斜率
        diamond_consume_slope = round(diamond_consume_data.groupby('account_id').apply(data_processing.calculate_slope, x_col='num', y_col='diamond_consume'),2).fillna(0).reset_index(name='diamond_consume_slope')
        # 鑽石消耗正負斜率次數
        diamond_consume_slope_count = diamond_consume_data.groupby('account_id').apply(data_processing.calculate_slope_count, x_col='num', y_col='diamond_consume').fillna(0).reset_index(name='diamond_consume_slope_count')
        # 彙總
        diamond_consume = pd.merge(diamond_consume, diamond_consume_avg, on='account_id', how='outer')\
                .merge(diamond_consume_slope, on='account_id', how='outer')\
                .merge(diamond_consume_slope_count, on='account_id', how='outer')
        
        
        ###### 綁鑽消耗 ######
        print("sapphire_consume_data start !!")
        SQL = '''
        SELECT *, row_number() over(PARTITION BY account_id ORDER BY event_date) num FROM (
            SELECT diamond.account_id, event_date, sum(sapphire_consume) sapphire_consume FROM (
                --$part_date
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, sapphire_consume_amount sapphire_consume FROM  game_data
                WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('diamond_consume'))
                AND sapphire_consume_amount > 0
            ) diamond INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register ON diamond.account_id = register.account_id
            WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
            GROUP BY diamond.account_id, event_date
        )''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        sapphire_consume_data = pd.read_csv(StringIO(s))
        # 總綁鑽消耗
        sapphire_consume = sapphire_consume_data.groupby('account_id')['sapphire_consume'].sum().reset_index(name='sapphire_consume')
        # 平均每日綁鑽消耗
        sapphire_consume_avg = data_processing.calculate_day_mean(sapphire_consume_data, login_days, value='sapphire_consume', column_name = 'sapphire_consume_avg')
        # 綁鑽消耗斜率
        sapphire_consume_slope = round(sapphire_consume_data.groupby('account_id').apply(data_processing.calculate_slope, x_col='num', y_col='sapphire_consume').fillna(0).reset_index(name='sapphire_consume_slope'),2)
        # 綁鑽消耗正負斜率次數
        sapphire_consume_slope_count = sapphire_consume_data.groupby('account_id').apply(data_processing.calculate_slope_count, x_col='num', y_col='sapphire_consume').fillna(0).reset_index(name='sapphire_consume_slope_count')
        # 彙總
        sapphire_consume = pd.merge(sapphire_consume, sapphire_consume_avg, on='account_id', how='outer')\
                .merge(sapphire_consume_slope, on='account_id', how='outer')\
                .merge(sapphire_consume_slope_count, on='account_id', how='outer')
        
        
        ###### 戰令、日常戰令、月卡購買 ######
        print("diamond_battle_pass_data start !!")
        SQL = '''
        SELECT battle_pass.account_id, event_date, date_diff('day', DATE(register_date), DATE(battle_pass.event_date))+1 buy_day, diamond_consume
        , CASE WHEN reason = '月卡商店购买' THEN 'diamond_monthly_card'
        WHEN reason = '战令通行证充值' THEN 'diamond_battle_pass'
        WHEN reason = '日常战令购买' THEN 'diamond_daily_pass' ELSE reason END reason FROM (
            --$part_date
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, date_add('hour', -13, "#event_time") event_time
            , reason, diamond_consume_amount diamond_consume
            , row_number() over(PARTITION BY "#account_id", reason ORDER BY "#event_time") buy_order FROM  game_data
            WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('diamond_consume'))
            -- 排除綁鑽
            AND diamond_consume_amount > 0
            AND reason in ('月卡商店购买','战令通行证充值','日常战令购买')
        ) battle_pass INNER JOIN (
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
            WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
        ) register ON battle_pass.account_id = register.account_id
        WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
        -- 戰令、月卡、日常戰令排除2次購買特別狀況
        AND buy_order = 1
        ORDER BY register.account_id''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        diamond_battle_pass_data = pd.read_csv(StringIO(s))
        # 鑽石總消耗
        diamond_battle_pass = diamond_battle_pass_data.pivot_table(index='account_id', columns='reason', values='diamond_consume').fillna(0)
        # 月卡第幾天購買
        # diamond_monthly_card_buy_day = diamond_battle_pass_data[diamond_battle_pass_data['reason']=='diamond_monthly_card'][['account_id','buy_day']].rename(columns = {'buy_day': 'diamond_monthly_card_buy_day'})
        
        
        ###### 解鎖技能槽 ######
        print("diamond_skill_slot start !!")
        SQL = '''
        SELECT skill.account_id, sum(diamond_consume) diamond_skill_slot FROM (
            --$part_date
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, reason, diamond_consume_amount diamond_consume FROM  game_data
            WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('diamond_consume'))
            -- 排除綁鑽
            AND diamond_consume_amount > 0
            AND reason in ('解锁技能槽')
        ) skill INNER JOIN (
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
            WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
        ) register ON skill.account_id = register.account_id
        WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
        GROUP BY skill.account_id''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        diamond_skill_slot = pd.read_csv(StringIO(s))
        
        
        ###### 球星球員招募(金星、銀星、球員金幣) ######
        print("diamond_gold_silver start !!")
        # 球星球員招募(金星、銀星、球員金幣)
        def diamond_gold_silver_fun(item1, item2):
            SQL = '''
            SELECT *, row_number() over(PARTITION BY account_id ORDER BY event_date) num FROM (
                SELECT diamond.account_id, event_date, sum(diamond_consume) diamond_consume FROM (
                    --$part_date
                    SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, diamond_consume_amount diamond_consume FROM  game_data
                    WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('diamond_consume'))
                    AND diamond_consume_amount > 0
                    AND (item_gain like '%%%s%%' or item_gain like '%%%s%%')
                ) diamond INNER JOIN (
                    SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                    WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                    AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
                ) register ON diamond.account_id = register.account_id
                WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
                GROUP BY diamond.account_id, event_date
            )''' %(self.start_time, self.end_time_data, item1, item2, self.start_time, self.end_time)
            r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                          headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                          data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
            s=str(r.content,'utf-8')
            diamond_gold_silver = pd.read_csv(StringIO(s))
            return diamond_gold_silver
        # 金星、銀星資料、總量、每日
        diamond_player_star_data = diamond_gold_silver_fun('金星','银星')
        diamond_player_star = diamond_player_star_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_player_star')
        diamond_player_star_avg = data_processing.calculate_day_mean(diamond_player_star_data, login_days, value='diamond_consume', column_name = 'diamond_player_star_avg')
        # 球員金幣資料、總量、每日
        diamond_player_coin_data =  diamond_gold_silver_fun('幸运金币','球员金币')
        diamond_player_coin = diamond_player_coin_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_player_coin')
        diamond_player_coin_avg = data_processing.calculate_day_mean(diamond_player_coin_data, login_days, value='diamond_consume', column_name = 'diamond_player_coin_avg')
        # 彙總
        diamond_player_star_coin = pd.merge(diamond_player_star, diamond_player_star_avg, on='account_id', how='outer')\
                .merge(diamond_player_coin, on='account_id', how='outer')\
                .merge(diamond_player_coin_avg, on='account_id', how='outer')
                
        
        ###### 其他鑽石項目(鑽石轉盤,購買球員,永久限購商店,簽約球員商店,購買服裝,購買戰令經驗,球員突破商店,花式商店,購買服裝屬性,貼紙商店,兌換金幣) ######
        print("diamond_other_data start !!")
        SQL = '''
        SELECT *, row_number() over(PARTITION BY account_id, reason ORDER BY event_date) num FROM (
            SELECT diamond.account_id, event_date, reason, sum(diamond_consume) diamond_consume FROM (
                --$part_date
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, reason, diamond_consume_amount diamond_consume FROM  game_data
                WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('diamond_consume'))
                AND diamond_consume_amount > 0
                AND reason in ('绑钻转盘','购买球员','永久限购商店购买','球员签约商店购买','购买服装','战令商店经验购买','球员突破商店购买','花式动作商店购买','时装选择属性','精英卡牌商店','兑换金币')
            ) diamond INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register ON diamond.account_id = register.account_id
            WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
            GROUP BY diamond.account_id, event_date, reason
        )''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        diamond_other_data = pd.read_csv(StringIO(s))
        # 鑽石轉盤資料、總量、每日
        diamond_turntable_data =  diamond_other_data[diamond_other_data['reason']=='绑钻转盘']
        diamond_turntable = diamond_turntable_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_turntable')
        diamond_turntable_avg = data_processing.calculate_day_mean(diamond_turntable_data, login_days, value='diamond_consume', column_name = 'diamond_turntable_avg')
        # 購買球員資料、總量、每日
        diamond_buy_player_data =  diamond_other_data[diamond_other_data['reason']=='购买球员']
        diamond_buy_player = diamond_buy_player_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_buy_player')
        diamond_buy_player_avg = data_processing.calculate_day_mean(diamond_buy_player_data, login_days, value='diamond_consume', column_name = 'diamond_buy_player_avg')
        # 永久限購商店資料、總量、每日
        diamond_limit_shop_data =  diamond_other_data[diamond_other_data['reason']=='永久限购商店购买']
        diamond_limit_shop = diamond_limit_shop_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_limit_shop')
        diamond_limit_shop_avg = data_processing.calculate_day_mean(diamond_limit_shop_data, login_days, value='diamond_consume', column_name = 'diamond_limit_shop_avg')
        # 簽約球員商店資料、總量、每日
        diamond_contract_player_shop_data =  diamond_other_data[diamond_other_data['reason']=='球员签约商店购买']
        diamond_contract_player_shop = diamond_contract_player_shop_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_contract_player_shop')
        diamond_contract_player_shop_avg = data_processing.calculate_day_mean(diamond_contract_player_shop_data, login_days, value='diamond_consume', column_name = 'diamond_contract_player_shop_avg')
        # 購買服裝資料、總量、每日
        diamond_clothes_data =  diamond_other_data[diamond_other_data['reason']=='购买服装']
        diamond_clothes = diamond_clothes_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_clothes')
        diamond_clothes_avg = data_processing.calculate_day_mean(diamond_clothes_data, login_days, value='diamond_consume', column_name = 'diamond_clothes_avg')
        # 購買戰令經驗資料、總量、每日
        diamond_battle_pass_exp_data =  diamond_other_data[diamond_other_data['reason']=='战令商店经验购买']
        diamond_battle_pass_exp = diamond_battle_pass_exp_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_battle_pass_exp')
        diamond_battle_pass_exp_avg = data_processing.calculate_day_mean(diamond_battle_pass_exp_data, login_days, value='diamond_consume', column_name = 'diamond_battle_pass_exp_avg')
        # 球員突破商店資料、總量、每日
        diamond_evolve_shop_data =  diamond_other_data[diamond_other_data['reason']=='球员突破商店购买']
        diamond_evolve_shop = diamond_evolve_shop_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_evolve_shop')
        diamond_evolve_shop_avg = data_processing.calculate_day_mean(diamond_evolve_shop_data, login_days, value='diamond_consume', column_name = 'diamond_evolve_shop_avg')
        # 花式商店資料、總量、每日
        diamond_fancy_data =  diamond_other_data[diamond_other_data['reason']=='花式动作商店购买']
        diamond_fancy = diamond_fancy_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_fancy')
        diamond_fancy_avg = data_processing.calculate_day_mean(diamond_fancy_data, login_days, value='diamond_consume', column_name = 'diamond_fancy_avg')
        # 購買服裝屬性資料、總量、每日
        diamond_clothes_status_data =  diamond_other_data[diamond_other_data['reason']=='时装选择属性']
        diamond_clothes_status = diamond_clothes_status_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_clothes_status')
        diamond_clothes_status_avg = data_processing.calculate_day_mean(diamond_clothes_status_data, login_days, value='diamond_consume', column_name = 'diamond_clothes_status_avg')
        # 貼紙商店資料、總量、每日
        diamond_sticker_data =  diamond_other_data[diamond_other_data['reason']=='精英卡牌商店']
        diamond_sticker = diamond_sticker_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_sticker')
        diamond_sticker_avg = data_processing.calculate_day_mean(diamond_sticker_data, login_days, value='diamond_consume', column_name = 'diamond_sticker_avg')
        # 兌換金幣資料、總量、每日
        diamond_coin_data =  diamond_other_data[diamond_other_data['reason']=='兑换金币']
        diamond_coin = diamond_coin_data.groupby('account_id')['diamond_consume'].sum().reset_index(name='diamond_coin')
        diamond_coin_avg = data_processing.calculate_day_mean(diamond_coin_data, login_days, value='diamond_consume', column_name = 'diamond_coin_avg')
        # 彙總
        diamond_consume_detail = pd.merge(diamond_turntable, diamond_turntable_avg, on='account_id', how='outer')\
                .merge(diamond_buy_player, on='account_id', how='outer')\
                .merge(diamond_buy_player_avg, on='account_id', how='outer')\
                .merge(diamond_limit_shop, on='account_id', how='outer')\
                .merge(diamond_limit_shop_avg, on='account_id', how='outer')\
                .merge(diamond_contract_player_shop, on='account_id', how='outer')\
                .merge(diamond_contract_player_shop_avg, on='account_id', how='outer')\
                .merge(diamond_clothes, on='account_id', how='outer')\
                .merge(diamond_clothes_avg, on='account_id', how='outer')\
                .merge(diamond_battle_pass_exp, on='account_id', how='outer')\
                .merge(diamond_battle_pass_exp_avg, on='account_id', how='outer')\
                .merge(diamond_evolve_shop, on='account_id', how='outer')\
                .merge(diamond_evolve_shop_avg, on='account_id', how='outer')\
                .merge(diamond_fancy, on='account_id', how='outer')\
                .merge(diamond_fancy_avg, on='account_id', how='outer')\
                .merge(diamond_clothes_status, on='account_id', how='outer')\
                .merge(diamond_clothes_status_avg, on='account_id', how='outer')\
                .merge(diamond_sticker, on='account_id', how='outer')\
                .merge(diamond_sticker_avg, on='account_id', how='outer')\
                .merge(diamond_coin, on='account_id', how='outer')\
                .merge(diamond_coin_avg, on='account_id', how='outer')
        
        
        ###### 對戰(rank+unrank) ###### 使用game_end當一場對戰，對戰結束才有詳細數據(勝利、PING)
        print("battle_data start !!")
        SQL = '''
        SELECT *, row_number() over(PARTITION BY account_id ORDER BY event_date) num FROM (
            SELECT battle.account_id, event_date, count(1) battle
            , SUM(CASE WHEN game_result='勝利' THEN 1 ELSE 0 END) battle_win
            , SUM(CASE WHEN part_event='game_ranked_end' THEN 1 ELSE 0 END) battle_rank
            , AVG(player_ping) battle_ping FROM (
                --$part_date
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, "$part_event" part_event, game_result, player_ping FROM  game_data
                WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('game_ranked_end','game_unranked_end'))
            ) battle INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register ON battle.account_id = register.account_id
            WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
            GROUP BY battle.account_id, event_date
        )
        ORDER BY account_id, event_date''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        battle_data = pd.read_csv(StringIO(s))
        # 總隊戰次數
        battle = battle_data.groupby('account_id')['battle'].sum().reset_index()
        # 每日平均對戰次數
        battle_avg = data_processing.calculate_day_mean(battle_data, login_days, value='battle', column_name = 'battle_avg')
        # 對戰次數斜率
        battle_slope = round(battle_data.groupby('account_id').apply(data_processing.calculate_slope, x_col='num', y_col='battle').fillna(0).reset_index(name='battle_slope'),2)
        # 對戰次數正負斜率次數
        battle_slope_count = battle_data.groupby('account_id').apply(data_processing.calculate_slope_count, x_col='num', y_col='battle').fillna(0).reset_index(name='battle_slope_count')
        # 對戰總勝率
        battle_win_ratio = battle_data.groupby('account_id').agg(sum_battle_win=('battle_win','sum'), sum_battle=('battle','sum'))
        battle_win_ratio['battle_win_ratio'] = round(battle_win_ratio['sum_battle_win']/battle_win_ratio['sum_battle'],4).replace([np.inf, -np.inf],0)
        battle_win_ratio = battle_win_ratio.reset_index().drop(['sum_battle', 'sum_battle_win'], axis=1)
        # 牌位對戰佔比
        battle_rank_ratio = battle_data.groupby('account_id').agg(sum_battle_rank=('battle_rank','sum'), sum_battle=('battle','sum'))
        battle_rank_ratio['battle_rank_ratio'] = round(battle_rank_ratio['sum_battle_rank']/battle_rank_ratio['sum_battle'],4).replace([np.inf, -np.inf],0)
        battle_rank_ratio = battle_rank_ratio.reset_index().drop(['sum_battle_rank', 'sum_battle'], axis=1)
        # 牌位對戰佔比(類別)，70%以上就是牌位玩家
        battle_rank_ratio['battle_rank_class'] = np.where(battle_rank_ratio['battle_rank_ratio']>0.7,1,0)
        # 平均ping值
        battle_ping = round(battle_data.groupby('account_id')['battle_ping'].mean(),0).reset_index()
        # 彙總
        battle = pd.merge(battle, battle_avg, on='account_id', how='outer')\
                .merge(battle_slope, on='account_id', how='outer')\
                .merge(battle_slope_count, on='account_id', how='outer')\
                .merge(battle_win_ratio, on='account_id', how='outer')\
                .merge(battle_rank_ratio, on='account_id', how='outer')\
                .merge(battle_ping, on='account_id', how='outer')
        
        
        ###### 對戰間隔(rank+unrank) ###### 使用game_start判斷對戰間隔
        print("battle_between_data start !!")
        SQL = '''
        --$part_date
        SELECT battle.* FROM (
            SELECT "#account_id" account_id, DATE(date_add('hour', -13, "#event_time")) event_date, date_add('hour', -13, "#event_time") event_time,
                    -- 根據LAG跟PARTITION得知下一場的資訊，計算戰鬥開始時間
                   DATE_DIFF('second', LAG("#event_time") OVER (PARTITION BY "#account_id" ORDER BY "#event_time"), "#event_time") AS before_battle_second
             FROM  game_data
            WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('game_ranked_start','game_unranked_start'))
        ) battle INNER JOIN (
            SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
            WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
            AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
        ) register ON battle.account_id = register.account_id
        WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
        -- 因為第一場沒有間隔所以排除
        AND before_battle_second IS NOT NULL
        ''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        battle_between_data = pd.read_csv(StringIO(s))
        # 對戰間隔少於5分鐘內場次比例，數據是由start到start所以加多180秒當作戰鬥時間
        battle_between = battle_between_data[battle_between_data['before_battle_second']<=480].groupby('account_id')['before_battle_second'].count().reset_index(name = 'quick_between')
        all_battle_between = battle_between_data.groupby('account_id')['before_battle_second'].count().reset_index(name = 'all_between')
        battle_between = pd.merge(all_battle_between, battle_between, on='account_id', how='left').fillna(0)
        battle_between['battle_between_ratio'] =  round(battle_between['quick_between']/battle_between['all_between'],4).replace([np.inf, -np.inf],0)
        battle_between = battle_between.drop(['all_between'], axis=1)
        
        
        ###### 真人對戰(rank) ###### AI判斷只有rank數據有所以只計算rank數據，使用game_end來判定一場對戰
        print("battle_real_data start !!")
        SQL = '''
        SELECT *, row_number() over(PARTITION BY account_id ORDER BY event_time) num FROM (
            SELECT battle.account_id, event_date, event_time, CASE WHEN game_result='勝利' THEN 1 ELSE 0 END battle_win, battle_real
            -- 判斷評價大於0的才是真人並計算敵我雙方平均後差異
            , (a1 + a2 + a3) / cardinality(FILTER(ARRAY[a1, a2, a3], x -> x > 0)) - (h1 + h2 + h3) / cardinality(FILTER(ARRAY[h1, h2, h3], x -> x > 0)) as power_diff_avg
            FROM (
                --$part_date
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "#event_time")) as event_date, date_add('hour', -13, "#event_time") event_time
                , "$part_event" part_event, game_result
                , CASE WHEN cardinality(filter(split("player_id_home",chr(0009)),x -> length(x) = 14))+cardinality(filter(split("player_id_away",chr(0009)),x -> length(x) = 14)) > 0 THEN 0 ELSE 1 END battle_real
                -- ,  cardinality(filter(split("player_id_home",chr(0009)),x -> length(x) in (8,9)))+cardinality(filter(split("player_id_away",chr(0009)),x -> length(x) in (8,9))) new_ai
                , "power_home_player01" as h1, "power_home_player02" as h2, "power_home_player03" as h3
                , "power_away_player01" as a1, "power_away_player02" as a2, "power_away_player03" as a3
                FROM  game_data
                WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('game_ranked_end'))
            ) battle INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register ON battle.account_id = register.account_id
            WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= 7
        )
        ORDER BY account_id, event_time''' %(self.start_time, self.end_time_data, self.start_time, self.end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                      headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                      data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        battle_real_data = pd.read_csv(StringIO(s))
        # 是否有真人對戰
        battle_real = battle_real_data[battle_real_data['battle_real']==1].groupby('account_id')['battle_real'].min().reset_index()
        # 真人對戰場次
        battle_real_count = battle_real_data[battle_real_data['battle_real']==1].groupby('account_id')['battle_real'].sum().reset_index(name = 'battle_real_count')
        # 真人對戰佔總對戰比例
        battle_total = battle_real_data.groupby('account_id')['num'].max().reset_index(name='battle_total')
        battle_real_ratio = pd.merge(battle_total, battle_real_count, on='account_id', how='left').fillna(0)
        battle_real_ratio['battle_real_ratio'] = round(battle_real_ratio['battle_real_count']/battle_real_ratio['battle_total'],4)
        battle_real_ratio = battle_real_ratio.drop(['battle_real_count','battle_total'], axis=1)
        # 真人對戰總勝率
        battle_real_win_ratio = battle_real_data[battle_real_data['battle_real']==1].groupby('account_id').agg(sum_battle_win=('battle_win','sum'), sum_battle=('battle_real','sum'))
        battle_real_win_ratio['battle_real_win_ratio'] = round(battle_real_win_ratio['sum_battle_win']/battle_real_win_ratio['sum_battle'],4).replace([np.inf, -np.inf],0)
        battle_real_win_ratio = battle_real_win_ratio.reset_index().drop(['sum_battle', 'sum_battle_win'], axis=1)
        # 對戰評價差異
        battle_power_diff = round(battle_real_data.groupby('account_id')['power_diff_avg'].mean().reset_index(name='battle_power_diff'),0)
        # 真人對戰評價差異
        battle_real_power_diff = round(battle_real_data[battle_real_data['battle_real']==1].groupby('account_id')['power_diff_avg'].mean().reset_index(name='battle_real_power_diff'),0)
        # 彙總
        battle_real = pd.merge(battle_real, battle_real_count, on='account_id', how='outer')\
                .merge(battle_real_ratio, on='account_id', how='outer')\
                .merge(battle_real_win_ratio, on='account_id', how='outer')\
                .merge(battle_power_diff, on='account_id', how='outer')\
                .merge(battle_real_power_diff, on='account_id', how='outer')
        
        
        ###### 欲預測付費(180日) ###### 
        print("pay_amount start !!")
        def pay_amount_fn (days):
            SQL = '''
            SELECT register.account_id, sum(pay_amount) pay_amount FROM (
                --$part_date
                SELECT "#account_id" as account_id, Date(date_add('hour', -13, "#event_time")) as event_date, round(sum(pay_amount),2) pay_amount FROM  game_data
                WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) AND ("$part_event" IN ('order_finish'))
                GROUP BY "#account_id", Date(date_add('hour', -13, "#event_time"))
            ) payamount INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register ON payamount.account_id = register.account_id
            WHERE date_diff('day', DATE(register_date), DATE(event_date))+1 <= %s
            GROUP BY register.account_id
            ORDER BY register.account_id''' %(self.start_time, self.end_time_data_ltv, self.start_time, self.end_time, days)
            r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                          headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                          data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
            s=str(r.content,'utf-8')
            pay_amount = pd.read_csv(StringIO(s))
            return pay_amount
        pay_amount_180 = pay_amount_fn(180).rename(columns={'pay_amount': 'pay_amount_180'})
        pay_amount_120 = pay_amount_fn(120).rename(columns={'pay_amount': 'pay_amount_120'})
        pay_amount_90 = pay_amount_fn(90).rename(columns={'pay_amount': 'pay_amount_90'})
        pay_amount_60 = pay_amount_fn(60).rename(columns={'pay_amount': 'pay_amount_60'})
        # 彙總
        pay_amount = pd.merge(pay_amount_180, pay_amount_120, on='account_id', how='outer')\
                .merge(pay_amount_90, on='account_id', how='outer')\
                .merge(pay_amount_60, on='account_id', how='outer')
        
        
        ###### 欲預測IAA營收(180日) ###### 
        print("iaa_income start !!")
        def iaa_income_fn (days):
            SQL = '''
            SELECT register.account_id, round(sum(iaa_income),4) iaa_income FROM (
                -- $part_date
                SELECT "#account_id" as account_id, Date("#event_time") as event_date, sum(round(iaa_income_predict,6)) iaa_income
                FROM  game_data
                WHERE "#event_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp) 
                AND "$part_event" IN ('iaa_income_to_id')
                GROUP BY "#account_id", Date("#event_time")
            ) iaa INNER JOIN (
                SELECT "#account_id" account_id, Date(date_add('hour', -13, "register_time")) register_date FROM ta.user_info
                WHERE "register_time" BETWEEN CAST('%s' as timestamp) AND CAST('%s' as timestamp)
                AND (NOT("#user_id" IN (SELECT "#user_id" FROM cluster WHERE (cluster_name IN ('exclude_cluster)))))
            ) register ON iaa.account_id = register.account_id
            WHERE date_diff('day', DATE(register_date), DATE(iaa.event_date))+1 <= %s
            GROUP BY register.account_id
            ORDER BY register.account_id
            ''' %(self.start_time, self.end_time_data_ltv, self.start_time, self.end_time, days)
            r = requests.post(url = 'http://API_address/querySql?token='+ self.token,  
                          headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                          data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
            s=str(r.content,'utf-8')
            iaa_income = pd.read_csv(StringIO(s))
            return iaa_income
        iaa_income_180 = iaa_income_fn(180).rename(columns={'iaa_income': 'iaa_income_180'})
        iaa_income_120 = iaa_income_fn(120).rename(columns={'iaa_income': 'iaa_income_120'})
        iaa_income_90 = iaa_income_fn(90).rename(columns={'iaa_income': 'iaa_income_90'})
        iaa_income_60 = iaa_income_fn(60).rename(columns={'iaa_income': 'iaa_income_60'})
        # 彙總
        iaa_income = pd.merge(iaa_income_180, iaa_income_120, on='account_id', how='outer')\
                .merge(iaa_income_90, on='account_id', how='outer')\
                .merge(iaa_income_60, on='account_id', how='outer')
                
        print("合併儲值+IAA")
        # 欲預測付費+欲預測IAA營收
        revenue =  pd.merge(pay_amount, iaa_income, on="account_id", how="outer")
        revenue = revenue.fillna(0)
        revenue["revenue_180"] = revenue["pay_amount_180"] + revenue["iaa_income_180"]
        revenue["revenue_120"] = revenue["pay_amount_120"] + revenue["iaa_income_120"]
        revenue["revenue_90"] = revenue["pay_amount_90"] + revenue["iaa_income_90"]
        revenue["revenue_60"] = revenue["pay_amount_60"] + revenue["iaa_income_60"]
        revenue = revenue[["account_id", "revenue_180","revenue_120","revenue_90","revenue_60"]]

    
        ###### 合併所有特徵 ######
        print("merge_data start !!")
        merge_data = pd.merge(channel_country_data, first_player_data, on='account_id', how='outer')\
            .merge(login_day, on='account_id', how='outer')\
            .merge(online_time, on='account_id', how='outer')\
            .merge(first_online, on='account_id', how='outer')\
            .merge(max_level, on='account_id', how='outer')\
            .merge(talent, on='account_id', how='outer')\
            .merge(max_player_rating_evolve, on='account_id', how='outer')\
            .merge(quest, on='account_id', how='outer')\
            .merge(first_pay, on='account_id', how='outer')\
            .merge(pay, on='account_id', how='outer')\
            .merge(iaa, on='account_id', how='outer')\
            .merge(diamond_consume, on='account_id', how='outer')\
            .merge(sapphire_consume, on='account_id', how='outer')\
            .merge(diamond_battle_pass, on='account_id', how='outer')\
            .merge(diamond_skill_slot, on='account_id', how='outer')\
            .merge(diamond_player_star_coin, on='account_id', how='outer')\
            .merge(diamond_consume_detail, on='account_id', how='outer')\
            .merge(battle, on='account_id', how='outer')\
            .merge(battle_between, on='account_id', how='outer')\
            .merge(battle_real, on='account_id', how='outer')\
            .merge(revenue, on='account_id', how='outer')
        return merge_data