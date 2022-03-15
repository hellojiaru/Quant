# region import
import os
import csv
import glob
import json
from cv2 import threshold
import tqdm
import datetime
import jqdatasdk
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
# from jqdatasdk.alpha101 import *
from db_content import MySql, SqlServer
# ID是申请时所填写的手机号; Password为聚宽官网登录密码, 新申请用户默认为手机号后6位
jqdatasdk.auth('18616758695', 'Abcd5678!')
# jqdatasdk.auth('18688758429', 'Cqy758429')
# jqdatasdk.auth('18616758695', '758695')
# jqdatasdk.auth('18582302358', 'Ljj95271234')
# jqdatasdk.auth('15277367685', 'Ljj9527123')
# jqdatasdk.auth('15207145035', 'Zxc123456')
# jqdatasdk.auth('15527806369', 'Zxc123456')
# jqdatasdk.auth('18738700420', 'Ztf700420')
# jqdatasdk.auth('18617015614', 'Lyz015614')
# jqdatasdk.auth('15652338313', 'Cy338313')
# jqdatasdk.auth('17727440723', 'Cpy440723')
# jqdatasdk.auth('13163743627', 'Cpy1743627')
# jqdatasdk.auth('18025464723 ', 'Cpy2512753')
# endregion


# region variable
norm_threshold = 1
three_classify_threshold = 0.005
exclude_columns = ['date', 'code']

backtrader_fields = ['open','close','low','high','volume','money','factor','high_limit','low_limit','paused']
# 开盘价, 收盘价, 最低价, 最高价, 成交股票数, 前复权, 涨停价, 跌停价, 平均价, 前一天收盘价, 是否停牌
get_price_fields =  ['open','close','low','high','volume','money','factor','high_limit','low_limit','avg','pre_close','paused']
# 时间, 当前价（不复权）, 累计成交量（股）, 累计成交额（元）
daily_auctions_fields = ['time', 'current', 'volume', 'money']
# 日期, 融资余额(元）, 融资买入额（元）, 融资偿还额（元）, 融券余量（股）, 融券卖出量（股）,融券偿还量（股）, 融资融券余额（元）
fin_values_fields = ['date', 'fin_value', 'fin_buy_value', 'fin_refund_value', 'sec_value', 'sec_sell_value', 'sec_refund_value', 'fin_sec_value']
# 日期, 涨跌幅(%), 主力净额(万), 主力净占比(%), 超大单净额(万), 超大单净占比(%), 大单净额(万), 大单净占比(%), 中单净额(万), 中单净占比(%), 小单净额(万), 小单净占比(%)
money_flow_fields = ['date', 'change_pct', 'net_amount_main', 'net_pct_main', 'net_amount_xl', 'net_pct_xl', 'net_amount_l', 'net_pct_l', 'net_amount_m', 'net_pct_m', 'net_amount_s', 'net_pct_s']

black_list = ['accounts_payable_turnover_rate',
              'current_asset_turnover_rate',
              'non_current_asset_ratio',
              'cash_to_current_liability',
              'rnoa_ttm',
              'DEGM_8y',
              'OperatingCycle',
              'margin_stability',
              'roic_ttm',
              'financial_liability',
              'book_leverage',
              'DEGM',
              'inventory_turnover_rate',
              'net_working_capital',
              'interest_carry_current_liability',
              'goods_service_cash_to_operating_revenue_ttm',
              'quick_ratio',
              'maximum_margin',
              'sales_growth',
              'financial_expense_rate',
              'net_operate_cash_flow_to_total_current_liability',
              'long_debt_to_working_capital_ratio',
              'GMI',
              'total_profit_to_cost_ratio',
              'current_ratio',
              'financial_expense_ttm',
              'admin_expense_rate',
              'gross_income_ratio',
              'account_receivable_turnover_days',
              'operating_liability',
              'asset_turnover_ttm',
              'goods_sale_and_service_render_cash_ttm',
              'long_term_debt_to_asset_ratio',
              'super_quick_ratio',
              'inventory_turnover_days',
              'market_leverage',
              'account_receivable_turnover_rate'
              ]

# endregion


# region export data

def export_all_trade_days(export_dir='datasets'):
    df = pd.DataFrame(jqdatasdk.get_all_trade_days())
    df.rename(columns={0:'date'}, inplace=True)
    df.to_csv(f'{export_dir}\\trade_days.csv', index=False)


def get_stock_feature_data(stock, start_date, end_date, all_factor=False):
    ''' 获取股票特征数据 '''
    
    # 股票每日交易数据
    daily_price = jqdatasdk.get_price(stock, fields=get_price_fields, start_date=start_date, end_date=end_date, skip_paused=False)
    if len(daily_price) == 0:
        return pd.DataFrame()
    daily_price.reset_index(inplace=True)
    daily_price.rename(columns={'index':'date'}, inplace=True)
    # 是否ST股
    is_st = jqdatasdk.get_extras('is_st', stock, start_date=start_date, end_date=end_date)
    is_st.reset_index(inplace=True)
    is_st.rename(columns={stock:'is_st'}, inplace=True)
    is_st.rename(columns={'index':'date'}, inplace=True)
    
    df = pd.merge(daily_price, is_st, how='inner', on='date')
    
    # 集合竞价数据
    daily_auctions = jqdatasdk.get_call_auction(stock, fields=daily_auctions_fields, start_date=start_date, end_date=end_date)
    daily_auctions.drop(['code'], axis=1, inplace=True)
    daily_auctions.rename(columns={'time':'date'}, inplace=True)
    daily_auctions.rename(columns={'current':'auctions_current'}, inplace=True)
    daily_auctions.rename(columns={'volume':'auctions_volume'}, inplace=True)
    daily_auctions.rename(columns={'money':'auctions_money'}, inplace=True)
    daily_auctions["date"] = [np.datetime64(d.strftime('%Y-%m-%d')) for d in daily_auctions["date"]]
    
    df = pd.merge(df, daily_auctions, how='left', on='date')

    # 股票融资数据
    fin_values = jqdatasdk.get_mtss(stock, fields=fin_values_fields, start_date=start_date, end_date=end_date)
    
    df = pd.merge(df, fin_values, how='left', on='date')
    
    # 股票资金流向数据
    money_flow = jqdatasdk.get_money_flow(stock, fields=money_flow_fields, start_date=start_date, end_date=end_date)
    
    df = pd.merge(df, money_flow, how='left', on='date')
    
    # 股票财务数据
    v_tabel = jqdatasdk.valuation
    q = jqdatasdk.query(v_tabel.capitalization,                 # 总股本(万股)
                        v_tabel.circulating_cap,                # 流通股本(万股)
                        v_tabel.market_cap,                     # 总市值(亿元)
                        v_tabel.circulating_market_cap,         # 流通市值(亿元)
                        v_tabel.turnover_ratio,                 # 换手率(%)
                        v_tabel.pe_ratio,                       # 市盈率(PE, TTM)
                        v_tabel.pe_ratio_lyr,                   # 市盈率(PE)
                        v_tabel.pb_ratio,                       # 市净率(PB)
                        v_tabel.ps_ratio,                       # 市销率(PS, TTM)
                        v_tabel.pcf_ratio,                      # 市现率(PCF, 现金净流量TTM)
                        jqdatasdk.indicator.eps,                # 每股收益EPS(元)
                        jqdatasdk.balance.cash_equivalents,     # 货币资金(元)
                        jqdatasdk.income.total_operating_revenue # 营业总收入(元)
                        # v_tabel.turnover_ratio,
                        # v_tabel.market_cap,
                        # jqdatasdk.indicator.eps
                        ).filter(v_tabel.code.in_([stock]))
    fundamentals = jqdatasdk.get_fundamentals_continuously(q, end_date=end_date, count=len(df), panel=False)
    fundamentals.drop(['code'], axis=1, inplace=True)
    fundamentals.rename(columns={'day':'date'}, inplace=True)
    fundamentals["date"] = [np.datetime64(d) for d in fundamentals["date"]]
    
    df = pd.merge(df, fundamentals, how='left', on='date')
    
    # 股票因子数据
    factors_info = jqdatasdk.get_all_factors()
    
    if all_factor:
        for _factor in factors_info['factor']:
            factor_data = jqdatasdk.get_factor_values(stock, factors=_factor, start_date=start_date, end_date=end_date)
            # 判断因子数据是否为null
            if not factor_data[_factor].isnull().all().iloc[0]:
                df_factor = factor_data[_factor]
                df_factor.reset_index(inplace=True)
                df_factor.rename(columns={'index':'date'}, inplace=True)
                df_factor.rename(columns={stock:_factor}, inplace=True)
                df = pd.merge(df, df_factor, how='left', on='date')
    else:
        for f in ['technical', 'risk']: #'risk'
            factors = list(factors_info.loc[[True if item == f else False for item in factors_info['category']]]['factor'])
            factor_data = jqdatasdk.get_factor_values(stock, factors=factors, start_date=start_date, end_date=end_date)
            for _factors in factor_data:
                if not factor_data[_factors].isnull().all().iloc[0]:
                    df_factor = factor_data[_factors]
                    df_factor.reset_index(inplace=True)
                    df_factor.rename(columns={stock:_factors}, inplace=True)
                    df_factor.rename(columns={'index':'date'}, inplace=True)
                    df = pd.merge(df, df_factor, how='left', on='date')
    
    df["code"] = str(str(stock).split('.')[0])
    return df


def export_stocks_to_csv(export_dir, stock_index='000300.XSHG', release_date='2010-01-01', all_factor=False):
    ''' 导出指定指数下的股票特征数据 '''
    
    stocks = jqdatasdk.get_index_stocks(stock_index)
    for stock in tqdm.tqdm(stocks):
        print(f'======== {stock} =========')
        securities = [item[:-4] for item in os.listdir(export_dir)]
        if stock in securities:
            continue
        
        # 股票基本信息
        stock_info = jqdatasdk.get_security_info(stock)
        start_date = stock_info.start_date.strftime('%Y-%m-%d')
        if start_date < release_date :
            start_date = release_date
        end_date = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        df = get_stock_feature_data(stock, start_date, end_date, all_factor)
        df.to_csv(f'{export_dir}\{stock}.csv', index=False)


def update_latest_stocks(stock_path=r'000300\*.csv', all_factor=False):
    ''' 更新已有的股票数据 '''
    
    trade_days = pd.read_csv(r'datasets\\trade_days.csv')['date']
    for path in tqdm.tqdm(glob.glob(stock_path)):
        stock = path.split('\\')[1][:-4]
        print(f'======== {stock} =========')
        
        old_df = pd.read_csv(path)
        if old_df.iloc[-1].hasnans:
            old_df.drop([len(old_df) - 1], inplace=True)
            old_df.to_csv(path, index=False)
        
        exists_latest_date = pd.read_csv(path)['date'].max()
        end_date = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        if not trade_days.loc[trade_days <= end_date].iloc[-1] > exists_latest_date: continue
        
        latest_date = datetime.datetime.strptime(exists_latest_date, '%Y-%m-%d')
        start_date = (latest_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
        new_df = get_stock_feature_data(stock, start_date, end_date, all_factor)
        if len(new_df) == 0: continue
        
        new_df['date'] = [d.strftime('%Y-%m-%d') for d in new_df['date']]
        with open(path, mode='a', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerows(new_df.values)


def export_backtrader_to_csv(stock_index='000300.XSHG', release_date='2005-01-01', export_dir='backtrader'):
    ''' 导出回测数据 '''
    
    stocks = jqdatasdk.get_index_stocks(stock_index)
    for stock in tqdm.tqdm(stocks):
        print(f'======== {stock} =========')
        securities = [item[:-4] for item in os.listdir(export_dir)]
        if stock in securities:
            continue
        
        # 股票基本信息
        stock_info = jqdatasdk.get_security_info(stock)
        start_date = stock_info.start_date.strftime('%Y-%m-%d')
        if start_date < release_date :
            start_date = release_date
        end_date = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        
        # 股票每日交易数据
        daily_price = jqdatasdk.get_price(stock, fields=backtrader_fields, start_date=start_date, end_date=end_date, skip_paused=False)
        daily_price.reset_index(inplace=True)
        daily_price.rename(columns={'index':'date'}, inplace=True)
    
        daily_price.to_csv(f'{export_dir}\{stock}.csv', index=False)


def update_latest_backtrader(backtrader_path=r'backtrader\*.csv'):
    ''' 更新回测数据 '''
    
    for path in tqdm.tqdm(glob.glob(backtrader_path)):
        stock = path.split('\\')[1][:-4]
        print(f'======== {stock} =========')
        
        latest_date = datetime.datetime.strptime(pd.read_csv(path)['date'].max(), '%Y-%m-%d')
        start_date = (latest_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # 股票每日交易数据
        df = jqdatasdk.get_price(stock, fields=backtrader_fields, start_date=start_date, end_date=end_date, skip_paused=False)
        if len(df) == 0: continue
        df.reset_index(inplace=True)
        df.rename(columns={'index':'date'}, inplace=True)
        df['date'] = [d.strftime('%Y-%m-%d') for d in df['date']]
        
        with open(path, mode='a', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerows(df.values)


# region 获取股票每日数据合并到因子数据
# def add_open_close():
#     for path in tqdm.tqdm(glob.glob('factors/*.csv')):
#         if path[8:-4] == '000002.XSHE':
#             continue
#         df = pd.read_csv(path)
#         start_date = df['date'].min().replace('/', '-')
#         end_date = df['date'].max().replace('/', '-')
#         daily_price = jqdatasdk.get_price(path[8:-4], fields=['open', 'close', 'low', 'high', 'volume', 'money'], start_date=start_date, end_date=end_date, skip_paused=False)
#         daily_price.reset_index(inplace=True)
#         daily_price.rename(columns={'index':'date'}, inplace=True)
#         daily_price['date'] = [d.strftime('%Y-%m-%d') for d in daily_price['date']]
#         df = pd.merge(daily_price, df, how='inner', on='date')
#         df.to_csv(r'factors\{0}.csv'.format(path[8:-4]), index=False)
# endregion


# region 导出股票因子训练数据
# def export_factors_train_data(factor_path='factors/*.csv', root_dir='000300'):
#     ''' 导出股票因子训练数据 '''
#     for path in tqdm.tqdm(glob.glob(factor_path)):
#         stock = path.split('\\')[1][:-4]
#         print(f'======== {stock} =========')
#         factor_df = pd.read_csv(path)
#         stock_df = pd.read_csv(f'{root_dir}/{stock}.csv')
#         exclude_columns = factor_df.columns.to_list()
#         exclude_columns.remove('date')
        
#         stock_df = stock_df[[col for col in stock_df.columns.to_list() if col not in exclude_columns]]
#         df = pd.merge(factor_df, stock_df, how='left', on='date')
        
#         df.to_csv(f'datasets\\factors_datas\\{stock}.csv', index=False)
# endregion


def export_alpha101_to_csv(start_date='2010-01-01', export_dir='alpha'):
    ''' 导出alpha101的因子数据 '''
  
    for i in range(1, 102):
        f_num = str(i)
        if len(str(i)) != 3:
            f_num = ('0' * (3 - len(str(i)))) + str(i)
            
        dir = f'{export_dir}/alpha_{f_num}'
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        while start_date < datetime.datetime.now().strftime('%Y-%m-%d'):
            try:
    
                data = eval(f"alpha_{f_num}('{start_date}', '000300.XSHG')")
            
                if len(data) == 0: 
                    start_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                    continue
                
                data.to_csv(f'{dir}/{start_date}.csv')
        
                start_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
            except:
                start_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                print('An exception occurred')
        
        start_date = start_date

# endregion


# region processing data

def _standard_scaler(series):
    ''' 标准化 '''
    
    return (series - series.mean()) / series.std()


def _min_max_scaler(series):
    ''' 最大最小化 '''
    
    return (series - series.min()) / (series.max() - series.min())


def get_data_frame(csv_path, data_slice=None, label_encoder=False):
    df = pd.DataFrame()
    file_path = glob.glob(csv_path)
    if data_slice is not None:
        file_path = file_path[data_slice]
    for path in tqdm.tqdm(file_path):
        _df = pd.read_csv(path)
        
        # 分股票归一化
        _df['close_slope'] = _min_max_scaler(_df['close'])
        
        # 记录股票代码
        _df['code'] = path.split('\\')[1][:-4]
        
        # 处理时间日期特征
        day = 24*60*60
        year = (365.2425) * day
        timestamp_s = pd.to_datetime(_df['date'], format='%Y.%m.%d').map(pd.Timestamp.timestamp)
        _df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        _df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        _df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        _df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        
        # 补全NAN缺失数据
        is_null = dict(_df.isnull().any())
        for col in is_null:
            if is_null[col]: _df[col].fillna(method="bfill", inplace=True)
            
        # 处理bool值类型            
        _df['is_st'].loc[_df['is_st'] == True] = 1.0
        _df['is_st'].loc[_df['is_st'] != 1] = 0.0
        _df['is_st'] = _df['is_st'].astype('int')
        
        df = pd.concat([df, _df])
    
    df.reset_index(drop=True, inplace=True)
    
    # 补全NAN缺失数据为0
    is_null = dict(df.isnull().any())
    for col in is_null:
        if is_null[col]: df[col].fillna(0, inplace=True)
    
    # 标准化
    for col in df.columns.to_list():
        if col in exclude_columns: continue
        if df[col].min() < -norm_threshold:
            df[col] = _standard_scaler(df[col])
        elif df[col].max() > norm_threshold:
            df[col] = _min_max_scaler(df[col])
    
    if label_encoder:
        df['label_code'] = LabelEncoder().fit_transform(df['code'])
    
    print(df.info())
    print(df.head())
    print(df.tail())
    print(df.describe())
    print(df.isnull().any())
    return df


def build_lstm_data(df, time_steps, future_steps, classify_num, threshold_slope=None):
    x, y = [], []
    for i in range(len(df)):
        if i + time_steps + future_steps >= len(df): break
        # 计算未来涨跌情况
        y_future = df.iloc[i + time_steps - 1: i + time_steps - 1 + future_steps]['close_slope'].to_numpy()[:, np.newaxis]
        x_future = np.arange(0, len(y_future))[:, np.newaxis]
        slope = np.squeeze(LinearRegression().fit(x_future, y_future).coef_)
        if threshold_slope is not None and (-threshold_slope < slope < threshold_slope): continue
        if classify_num == 1:
            label = slope
        elif classify_num == 2:
            label = 0 if slope <= 0 else 1
        else:
            label = 0 if slope <= -three_classify_threshold else 2 if slope >= three_classify_threshold else 1
        # ax = sns.regplot(x_future, y_future, x_ci=None)
        # ax.set_title(slope)
        # plt.show()
        
        x.append(df.iloc[i: i + time_steps][[col for col in df.columns if col not in exclude_columns]].to_numpy())
        y.append(label)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return x, y


def export_train_valid_data_to_tfrecord(csv_path, source_path, test_start_date, file_name, time_steps=30, future_steps=10, classify_num=2):
    ''' 导出tfrecord类型的股票预训练数据 '''
    import tensorflow as tf
    
    if not os.path.exists(csv_path):
        df = get_data_frame(source_path, data_slice=None, label_encoder=False)
        df.to_csv(csv_path, index=False)
    
    df = pd.read_csv(csv_path)
    
    train_writer = tf.io.TFRecordWriter(rf'datasets\tfreocrd\{file_name}_train.tfrecord')
    valid_writer = tf.io.TFRecordWriter(rf'datasets\tfreocrd\{file_name}_valid.tfrecord')

    for _, _df in tqdm.tqdm(df.groupby('code')): 
        
        if 'to' in test_start_date:
            start, between, end = test_start_date.split('to')
            train_df = _df.loc[(_df['date'] >= start) & (_df['date'] < between)]
            test_df = _df.loc[(_df['date'] >= between) & (_df['date'] < end)]
        else:
            train_df = _df.loc[_df['date'] <= test_start_date]
            test_df = _df.loc[_df['date'] > test_start_date]

        _x_train, _y_train = build_lstm_data(train_df, time_steps, future_steps, classify_num)
        _x_test, _y_test = build_lstm_data(test_df, time_steps, future_steps, classify_num)
        
        for i in range(len(_x_train)):
            feature = {
                "x_train": tf.train.Feature(float_list=tf.train.FloatList(value=list(_x_train[i].flatten()))),
                "y_train": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(_y_train[i])]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            train_writer.write(example.SerializeToString())
            
        for j in range(len(_x_test)):
            feature = {
                "x_test": tf.train.Feature(float_list=tf.train.FloatList(value=list(_x_test[j].flatten()))),
                "y_test": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(_y_test[j])]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            valid_writer.write(example.SerializeToString())

    train_writer.close()
    valid_writer.close()


def read_train_valid_data_for_tfrecord(file_name, batch_size, feature_shape):
    import tensorflow as tf
    
    train_data = tf.data.TFRecordDataset(rf'datasets\tfreocrd\{file_name}_train.tfrecord')
    valid_data = tf.data.TFRecordDataset(rf'datasets\tfreocrd\{file_name}_valid.tfrecord')

    train_features = {
        'x_train': tf.io.FixedLenFeature(feature_shape, tf.float32),
        'y_train': tf.io.FixedLenFeature([1], tf.int64),
    }
    valid_features = {
        'x_test': tf.io.FixedLenFeature(feature_shape, tf.float32),
        'y_test': tf.io.FixedLenFeature([1], tf.int64),
    }

    train_data = train_data.map(lambda x: tf.io.parse_single_example(x, train_features))
    valid_data = valid_data.map(lambda x: tf.io.parse_single_example(x, valid_features))

    train_data = train_data.map(lambda x: (x['x_train'], x['y_train']))
    valid_data = valid_data.map(lambda x: (x['x_test'], x['y_test']))

    train_data = train_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    valid_data = valid_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_data, valid_data


def write_test_data_to_tfrecord(start_date, end_date, file_name, time_steps, future_steps, csv_path, stock_path):
    import tensorflow as tf
    
    writer = tf.io.TFRecordWriter(rf'datasets\tfreocrd\{file_name}_test.tfrecord')

    stocks_df = pd.read_csv(csv_path)
    stocks = [path.split('\\')[1][:-4] for path in glob.glob(stock_path)]
    trade_days = pd.read_csv(r'datasets\\trade_days.csv')['date']
    start_index = trade_days.loc[trade_days >= start_date].index[0]
    start_date = trade_days.iloc[start_index]
    
    while start_date < end_date :
        date = start_date
        print(f'date: {date} \t end_date: {end_date}')
        
        keys, values, labels = [], [], []
        for stock in stocks:
            _df = stocks_df.loc[stocks_df['code'] == stock]
            _df.reset_index(drop=True, inplace=True)
            indexs = _df.loc[_df['date'] == date].index
            if len(indexs) == 0 or (indexs[0]+future_steps+1) >= len(_df) : continue
            row_index = indexs[0]
            stock_features = _df.iloc[row_index-time_steps:row_index]
            slope_features = _df.iloc[row_index:row_index+future_steps]
            if len(stock_features) == 0 or len(slope_features) == 0 : continue
            y = slope_features['close'].values[:, np.newaxis]
            x = np.arange(0, len(y))[:, np.newaxis]
            keys.append(stock)
            values.append(stock_features.drop(exclude_columns, axis=1).values)
            labels.append(float(np.squeeze(LinearRegression().fit(x, y).coef_)))
    
        start_index += 1
        start_date = trade_days.iloc[start_index]

        features = tf.train.Features(feature = {
            'date': tf.train.Feature(bytes_list=tf.train.BytesList(value=[date.encode('utf-8')])),
            'keys': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item.encode('utf-8') for item in keys])),
            'values': tf.train.Feature(float_list=tf.train.FloatList(value=np.array(values, dtype=np.float32).flatten())),
            'labels': tf.train.Feature(float_list=tf.train.FloatList(value=labels))
        })
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    writer.close()


def read_test_data_for_tfrecord(file_name):
    import tensorflow as tf
    
    test_data = tf.data.TFRecordDataset(rf'datasets\tfreocrd\{file_name}_test.tfrecord')
    test_features = {
        'date': tf.io.FixedLenFeature([], tf.string),
        'keys': tf.io.VarLenFeature(tf.string),
        'values': tf.io.VarLenFeature(tf.float32),
        'labels': tf.io.VarLenFeature(tf.float32)
    }
    test_data = test_data.map(lambda x: tf.io.parse_single_example(x, test_features))
    # test_data = test_data.map(lambda x: (x['date'], x['keys'], x['values'], x['labels']))
    return test_data


def write_stocks_probability_using_test_data_for_tfrecord(file_name, time_steps, model, model_name, use_db='sqlserver'):
    mysql = MySql()
    sqlserver = SqlServer()

    def save_sotck_probability_to_mysql(dt, sotck_prob, m):
        json_str = json.dumps(sotck_prob)
        sql = "INSERT INTO quant(Date, Stock_Prob, Model) VALUES('{}','{}','{}')".format(dt, json_str, m)
        mysql.execute_sql(sql)
        
    def is_mysql_record(dt, m):
        sql = "select * from quant where Model='{0}' and Date='{1}'".format(m, dt)
        result = mysql.get_sql(sql)
        return True if len(result) > 0 else False
    
    def save_sotck_probability_to_sqlserver(dt, sotck_prob, m):
        json_str = json.dumps(sotck_prob)
        sql = "insert into Simulated_Transaction values('{}','{}','{}')".format(dt, json_str, m)
        sqlserver.execute_sql(sql)
    
    def is_sqlserver_record(dt, m):
        sql = "select * from Simulated_Transaction where Model='{0}' and Date='{1}'".format(m, dt)
        result = sqlserver.get_sql(sql)
        return True if len(result) > 0 else False
    
    test_data = read_test_data_for_tfrecord(file_name)
    
    for item in tqdm.tqdm(test_data):
        date = item['date'].numpy().decode('utf-8')
        stocks = [i.decode('utf-8') for i in item['keys'].values.numpy()]
        values = item['values'].values.numpy().reshape(-1, time_steps, 313)
        labels = item['labels'].values.numpy()
    
        if use_db == 'sqlserver':
            if is_sqlserver_record(date, model_name): continue
        else:
            if is_mysql_record(date, model_name): continue
    
        probs = model.predict(values)
            
        new_probs = []        
        for i, label in enumerate(np.argmax(probs, axis=-1)):
            if label == 0:
                new_probs.append(1-probs[i][label])
            else:
                new_probs.append(probs[i][label])
        probs = new_probs
        
        prob_dict = {}
        for i in range(len(probs)):
            prob_dict[list(stocks)[i]] = float(probs[i])
        
        if use_db == 'sqlserver':
            save_sotck_probability_to_sqlserver(date, prob_dict, model_name)
        else:
            save_sotck_probability_to_mysql(date, prob_dict, model_name)


def summary_stocks_probability_using_test_data_for_tfrecord(file_name, time_steps, model, writer_path):
    date_list, probs_list, slope_list = [], [], []
    
    test_data = read_test_data_for_tfrecord(file_name)
    for item in test_data:
        date = item['date'].numpy().decode('utf-8')
        stocks = [i.decode('utf-8') for i in item['keys'].values.numpy()]
        values = item['values'].values.numpy().reshape(-1, time_steps, 313)
        labels = item['labels'].values.numpy()
    
        probs = model.predict(values)
        new_probs = []        
        for i, label in enumerate(np.argmax(probs, axis=-1)):
            if label == 0:
                new_probs.append(1-probs[i][label])
            else:
                new_probs.append(probs[i][label])
        probs = new_probs

        prob_dict, slope_dict = {}, {}
        for i in range(len(probs)):
            prob_dict[list(stocks)[i]] = float(probs[i])
            slope_dict[list(stocks)[i]] = float(labels[i])

        date_list.append(date)
        probs_list.append(prob_dict)
        slope_list.append(slope_dict)
    
    rows = []
    thresholds = [(0.1,0.9), (0.2,0.8), (0.3,0.7), (0.4,0.6)]
    for lt, gt in thresholds:
        total_num, total_right_num = 0, 0
        total_lt_num, total_lt_right_num = 0, 0
        total_gt_num, total_gt_right_num = 0, 0
        for date, prob, slope in zip(date_list, probs_list, slope_list):
            
            threshold_probs = {k: v for k, v in prob.items() if v > gt or v < lt}
            threshold_slopes = {k: v for k, v in slope.items() if k in threshold_probs.keys()}
            
            total_num += len(threshold_probs)
            total_right_num += len([True for p, s in zip(threshold_probs.values(), threshold_slopes.values()) if (p > gt and s > 0) or (p < lt and s < 0)])
            
            total_lt_num += len([True for p, s in zip(threshold_probs.values(), threshold_slopes.values()) if p < lt])
            total_lt_right_num += len([True for p, s in zip(threshold_probs.values(), threshold_slopes.values()) if p < lt and s < 0])
            
            total_gt_num += len([True for p, s in zip(threshold_probs.values(), threshold_slopes.values()) if p > gt])
            total_gt_right_num += len([True for p, s in zip(threshold_probs.values(), threshold_slopes.values()) if p > gt and s > 0])
        
        rows.append({'threshold': f'{lt}_{gt}', 'total_num':total_num, 'total_right_num':total_right_num, 
                     'total_lt_num':total_lt_num, 'total_lt_right_num':total_lt_right_num,
                     'total_gt_num':total_gt_num, 'total_gt_right_num':total_gt_right_num
                    })
        
    with open(writer_path, mode='x', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(['threshold', 'total_num', 'total_right_num', 'total_acc',
                            'total_lt_num', 'total_lt_right_num', 'total_lt_acc', 
                            'total_gt_num', 'total_gt_right_num', 'total_gt_acc'])
        for row in rows:
            csv_write.writerow([row['threshold'], row['total_num'], row['total_right_num'], 0 if row['total_num'] == 0 else row['total_right_num'] / row['total_num'],
                                row['total_lt_num'], row['total_lt_right_num'],  0 if row['total_lt_num'] == 0 else row['total_lt_right_num'] / row['total_lt_num'],
                                row['total_gt_num'], row['total_gt_right_num'],  0 if row['total_gt_num'] == 0 else row['total_gt_right_num'] / row['total_gt_num']
                                ])


def write_stocks_probability_to_db(time_steps=30, model_dir={'v1':'weight\\000300_v1', 'v2':'weight\\000300_v2'}, stock_path=r'000300\*.csv', root_dir='datasets\simulated_transaction'):
    ''' 写入模型概率 '''
    
    try:
        mysql = MySql()
        # sqlserver = SqlServer()

        def save_sotck_probability_to_mysql(dt, sotck_prob, m):
            json_str = json.dumps(sotck_prob)
            sql = "INSERT INTO quant(Date, Stock_Prob, Model) VALUES('{}','{}','{}')".format(dt, json_str, m)
            mysql.execute_sql(sql)
            
        # def save_sotck_probability_to_sqlserver(dt, sotck_prob, m):
        #     json_str = json.dumps(sotck_prob)
        #     sql = "insert into Simulated_Transaction() values('{}','{}','{}')".format(dt, json_str, m)
        #     sqlserver.execute_sql(sql)

        date = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        trade_days = pd.read_csv(r'datasets\\trade_days.csv')['date'].to_list()
        if date not in trade_days:
            return 'market closed.'
        
        today_stocks = f'{root_dir}\{date}.csv'
        latest_all_stocks = f'{root_dir}\latest.csv'
        
        if not os.path.exists(today_stocks):
            update_latest_stocks()
            get_data_frame(stock_path, label_encoder=True).to_csv(latest_all_stocks, index=False)
            stocks = [path.split('\\')[1][:-4] for path in glob.glob(stock_path)]
            
            today_stocks_df = pd.DataFrame()
            df = pd.read_csv(latest_all_stocks)
            for stock in stocks:
                _df = df.loc[df['code'] == stock]
                _df.reset_index(drop=True, inplace=True)    
                row_index = _df.loc[_df['date'] == date].index[0]
                stock_features = _df.iloc[row_index-time_steps:row_index]
                today_stocks_df = pd.concat([today_stocks_df, stock_features])
            today_stocks_df.to_csv(today_stocks, index=False)
            
        df = pd.read_csv(today_stocks)
        for m in model_dir:
            model = get_trained_model_by_name(m, time_steps, model_dir)
            prob_dict, keys, values = {}, [], []
            for code, features in df.groupby('code'):
                keys.append(code)
                features = features.drop(exclude_columns, axis=1)
                values.append(features.values)
            values = np.array(values, dtype=np.float32)
            probs = model.predict(values)
            for i in range(len(probs)):
                prob_dict[list(keys)[i]] = float(probs[i])
            save_sotck_probability_to_mysql(date, prob_dict, m)
            # save_sotck_probability_to_sqlserver(date, prob_dict, m)
                
        return 'success.'
    except Exception as e:
        return e.message

# endregion


def get_single_finetune_data(df, time_steps=30, future_steps=10, test_size=0.2, classify_num=2, test_date=None):
    ''' 获取单只股票训练的数据 '''
    train_df, test_df = None, None
    
    if test_date is not None:
        train_df = df.loc[df['date'] < test_date]
        test_df = df.loc[df['date'] > test_date]
    else:
        train_df = df.iloc[:(len(df)-int(len(df)*test_size))]
        test_df = df.iloc[-(int(len(df)*test_size)):]

    test_start_date = test_df['date'].min()
    test_end_date = test_df['date'].max()
        
    x_train, y_train = build_lstm_data(train_df, time_steps, future_steps, classify_num)
    
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    return x_train, y_train, test_start_date, test_end_date


def get_lstm_train_test_data(csv_path, source_path='000300/*.csv', data_slice=None, time_steps=30, future_steps=10, test_size=0.2, classify_num=2, test_date=None, threshold_slope=None):
    ''' 获取股票预训练数据 '''
    x_train, x_test, y_train, y_test = [], [], [], []
    
    if not os.path.exists(csv_path):
        df = get_data_frame(source_path, data_slice=data_slice, label_encoder=True)
        df.to_csv(csv_path, index=False)
    
    df = pd.read_csv(csv_path)
    
    for _, _df in tqdm.tqdm(df.groupby('code')): 
        train_df, test_df = None, None
        if test_date is not None:
            train_df = _df.loc[_df['date'] <= test_date]
            test_df = _df.loc[_df['date'] > test_date]
        else:
            train_df = _df.iloc[:(len(_df)-int(len(_df)*test_size))]
            test_df = _df.iloc[-(int(len(_df)*test_size)):]
            
        _x_train, _y_train = build_lstm_data(train_df, time_steps, future_steps, classify_num, threshold_slope)
        _x_test, _y_test = build_lstm_data(test_df, time_steps, future_steps, classify_num, threshold_slope)
        
        x_train.extend(_x_train)
        x_test.extend(_x_test)
        y_train.extend(_y_train)
        y_test.extend(_y_test)
    
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    return x_train, x_test, y_train, y_test


def get_lstm_train_data_by_date(csv_path, start_date, end_date, source_path='000300/*.csv', data_slice=None, time_steps=30, future_steps=10, classify_num=2, threshold_slope=None):
    ''' 获取股票预训练数据 '''

    x, y = [], []

    if not os.path.exists(csv_path):
        df = get_data_frame(source_path, data_slice=data_slice, label_encoder=True)
        df.to_csv(csv_path, index=False)
    
    df = pd.read_csv(csv_path)
    
    df = df.loc[(df['date'] >= start_date) & (df['date'] < end_date)]
    
    for _, _df in tqdm.tqdm(df.groupby('code')):
        _x, _y = build_lstm_data(_df, time_steps, future_steps, classify_num, threshold_slope)
        x.extend(_x)
        y.extend(_y)
    
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return x, y


def get_trained_model_by_name(name, time_steps, model_config):
    import tensorflow as tf
    
    if (name == 'v1') or (name == 'v2'):
        model = tf.keras.Sequential([
            tf.keras.layers.Convolution1D(64, 3, activation='relu', input_shape=(time_steps, 81)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4), loss='binary_crossentropy', metrics=['acc'])
        model.load_weights(f'{model_config[name]}\lstm.ckpt')
        
    elif (name == 'v10') or (name == 'v11') or (name == 'v12') or (name == 'v12_1') or (name == 'v12_2') or (name == 'v13') or (name == 'v14'):
        model = tf.keras.Sequential([
            tf.keras.layers.Convolution1D(128, 3, padding='same', activation='relu', input_shape=(time_steps, 81)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4), loss='binary_crossentropy', metrics=['acc'])
        model.load_weights(f'{model_config[name]}\lstm.ckpt')
    
    elif (name == 'v15') or (name == 'v16') or (name == 'v17') or (name == 'v18'):
        model = tf.keras.Sequential([
            tf.keras.layers.Convolution1D(64, 3, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(16, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4), loss='binary_crossentropy', metrics=['acc'])
        model.load_weights(f'{model_config[name]}\lstm.ckpt')
        
    elif (name == 'v19'):
        model = tf.keras.Sequential([
            tf.keras.layers.Convolution1D(64, 3, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(16, dropout=0.2),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='huber', metrics=['mae'])
        model.load_weights(f'{model_config[name]}\lstm.ckpt')
        
    elif (name == 'v20') or (name == 'v21') or (name == 'v23'):
        model = tf.keras.Sequential([
            tf.keras.layers.Convolution1D(64, 3, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(16, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=['acc'])
        model.load_weights(f'{model_config[name]}\lstm.ckpt')
        
    elif (name == "v22"):
        model = tf.keras.Sequential([
            tf.keras.layers.Convolution1D(64, 3, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=['acc'])
        model.load_weights(f'{model_config[name]}\lstm.ckpt')
    
    elif (name == "v24"):
        model = tf.keras.Sequential([
            tf.keras.layers.Convolution1D(128, 3, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Convolution1D(64, 3, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=['acc'])
        model.load_weights(f'{model_config[name]}\lstm.ckpt')
    
    elif (name == "v25"):
        model = tf.keras.Sequential([
            tf.keras.layers.Convolution1D(128, 3, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Convolution1D(64, 3, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy')
        model.load_weights(f'{model_config[name]}\lstm.ckpt')
        
    elif (name == "tfrecord"):
        model = tf.keras.Sequential([
            tf.keras.layers.Convolution1D(128, 3, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Convolution1D(64, 3, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=['acc'])
        model.load_weights(f'{model_config[name]}\lstm.ckpt')
    
    return model


def write_history_stocks_probability_to_db(start_date, end_date, model_dir, time_steps=30, csv_name=r'datasets\historical\latest.csv', stock_path=r'000300\*.csv', use_db='sqlserver', model=None):
    ''' 写入模型概率 '''
    
    mysql = MySql()
    sqlserver = SqlServer()

    def save_sotck_probability_to_mysql(dt, sotck_prob, m):
        json_str = json.dumps(sotck_prob)
        sql = "INSERT INTO quant(Date, Stock_Prob, Model) VALUES('{}','{}','{}')".format(dt, json_str, m)
        mysql.execute_sql(sql)
        
    def is_mysql_record(dt, m):
        sql = "select * from quant where Model='{0}' and Date='{1}'".format(m, dt)
        result = mysql.get_sql(sql)
        return True if len(result) > 0 else False
    
    def save_sotck_probability_to_sqlserver(dt, sotck_prob, m):
        json_str = json.dumps(sotck_prob)
        sql = "insert into Simulated_Transaction values('{}','{}','{}')".format(dt, json_str, m)
        sqlserver.execute_sql(sql)
    
    def is_sqlserver_record(dt, m):
        sql = "select * from Simulated_Transaction where Model='{0}' and Date='{1}'".format(m, dt)
        result = sqlserver.get_sql(sql)
        return True if len(result) > 0 else False
    
    stocks_df = pd.read_csv(csv_name)
    stocks = [path.split('\\')[1][:-4] for path in glob.glob(stock_path)]
    trade_days = pd.read_csv(r'datasets\\trade_days.csv')['date']
    start_index = trade_days.loc[trade_days >= start_date].index[0]
    start_date = trade_days.iloc[start_index]
    
    while(start_date < end_date):
        
        date = start_date
        print(f'date: {date} \t end_date: {end_date}')
        
        df = pd.DataFrame()
        for stock in stocks:
            _df = stocks_df.loc[stocks_df['code'] == stock]
            _df.reset_index(drop=True, inplace=True)
            indexs = _df.loc[_df['date'] == date].index
            if len(indexs) == 0 : continue
            row_index = indexs[0]
            stock_features = _df.iloc[row_index-time_steps:row_index]
            df = pd.concat([df, stock_features])

        for m in model_dir:
            if use_db == 'sqlserver':
                if (is_sqlserver_record(date, m)): continue
            else:
                if (is_mysql_record(date, m)): continue
            
            if model is None:
                model = get_trained_model_by_name(m, time_steps, model_dir)
            prob_dict, keys, values = {}, [], []
            for code, features in df.groupby('code'):
                keys.append(code)
                features = features.drop(exclude_columns, axis=1)
                values.append(features.values)
            values = np.array(values, dtype=np.float32)
            probs = model.predict(values)
            
            new_probs = []        
            for i, label in enumerate(np.argmax(probs, axis=-1)):
                if label == 0:
                    new_probs.append(1-probs[i][label])
                else:
                    new_probs.append(probs[i][label])
            
            probs = new_probs
            
            for i in range(len(probs)):
                prob_dict[list(keys)[i]] = float(probs[i])
                
            if use_db == 'sqlserver':
                save_sotck_probability_to_sqlserver(date, prob_dict, m)
            else:
                save_sotck_probability_to_mysql(date, prob_dict, m)
    
        start_index += 1
        start_date = trade_days.iloc[start_index]


def export_statistical_model_accuracy(
    start_date=None, end_date=None, level='day', future_steps=10,
    m=['v1', 'v2', 'v10', 'v11', 'v12', 'v12_1', 'v12_2', 'v22'], 
    avg_m=['v1', 'v2'], stock_path=r'000300\*.csv', writer_dir=r'statistical_model_accuracy4.csv'
    ):
    db = SqlServer()
    stocks_dict = {path.split('\\')[1][:-4]: pd.read_csv(path)[['date', 'close']] for path in glob.glob(stock_path)}
    trade_days = pd.read_csv(r'datasets\trade_days.csv')
    index = trade_days.loc[trade_days['date'] == datetime.datetime.now().strftime('%Y-%m-%d')].index[0]
    date = trade_days.iloc[index-10]['date']
    
    m_dicts = {}
    for _m in m:
        if start_date is not None:
            sql = "select * from Simulated_Transaction where Model='{}' and Date >= '{}' and Date < '{}' Order By Date".format(_m, start_date, end_date)
        else:
            sql = "select * from Simulated_Transaction where Model='{}' Order By Date".format(_m)
        data = db.get_sql(sql)
        data = [item for item in data if item['Date'].strftime('%Y-%m-%d') < date]
        m_dicts[_m] = data
    
    if len(avg_m) >= 2:
        avg_probs = []
        for i, d in enumerate([d['Date'] for d in m_dicts[m[0]]]):
            stock_probs = []
            for key in m_dicts:
                if key in avg_m:
                    stock_probs.append(json.loads(m_dicts[key][i]['Stock_Prob']))
            values = []
            for i in range(len(stock_probs)):
                values.append(np.array(list(stock_probs[i].values()))[np.newaxis,:])
            keys = stock_probs[0].keys()
            mean_values = np.mean(np.concatenate(values), axis=0)
            avg_probs.append({'Date': d, 'Stock_Prob': json.dumps({k: v for k, v in zip(keys, mean_values)})})
            
        m_dicts['avg'] = avg_probs
    
    result = []
    for m_key in m_dicts:
        
        date_temp = {}
        m_1_9_total_num, m_1_9_right_num, m_1_9_lt_right_num, m_1_9_lt_total_num, m_1_9_gt_right_num, m_1_9_gt_total_num = 0, 0, 0, 0, 0, 0
        m_2_8_total_num, m_2_8_right_num, m_2_8_lt_right_num, m_2_8_lt_total_num, m_2_8_gt_right_num, m_2_8_gt_total_num = 0, 0, 0, 0, 0, 0
        m_3_7_total_num, m_3_7_right_num, m_3_7_lt_right_num, m_3_7_lt_total_num, m_3_7_gt_right_num, m_3_7_gt_total_num = 0, 0, 0, 0, 0, 0
        m_4_6_total_num, m_4_6_right_num, m_4_6_lt_right_num, m_4_6_lt_total_num, m_4_6_gt_right_num, m_4_6_gt_total_num = 0, 0, 0, 0, 0, 0
        
        for d_item in tqdm.tqdm(m_dicts[m_key]):
            
            date = d_item['Date'].strftime('%Y-%m-%d')
            probs, slopes = {}, {}
            
            all_stock_probs = json.loads(d_item['Stock_Prob'])
            for s_key in all_stock_probs:
                y = stocks_dict[s_key].loc[stocks_dict[s_key]['date'] >= date]['close'].values[:future_steps][:, np.newaxis]
                x = np.arange(0, len(y))[:, np.newaxis]
                probs[s_key] = all_stock_probs[s_key]
                slopes[s_key] = float(np.squeeze(LinearRegression().fit(x, y).coef_))
            
            ### greater == 0.9 and less == 0.1 ####
            _1_9_probs = {k: p for k, p in probs.items() if p > 0.9 or p < 0.1}
            _1_9_slopes = {k: s for k, s in slopes.items() if k in _1_9_probs.keys()}
            _1_9_right = [True for p, s in zip(_1_9_probs.values(), _1_9_slopes.values()) if (p > 0.9 and s > 0) or (p < 0.1 and s < 0)]
            
            _1_9_count = len(_1_9_probs)
            _1_9_acc = 0 if _1_9_count == 0 else len(_1_9_right) / _1_9_count
            
            m_1_9_total_num += _1_9_count
            m_1_9_right_num += len(_1_9_right)
            
            m_1_9_lt_right_num += len([True for p, s in zip(_1_9_probs.values(), _1_9_slopes.values()) if p < 0.1 and s < 0])
            m_1_9_lt_total_num += len([True for p, s in zip(_1_9_probs.values(), _1_9_slopes.values()) if p < 0.1])
            
            m_1_9_gt_right_num += len([True for p, s in zip(_1_9_probs.values(), _1_9_slopes.values()) if p > 0.9 and s > 0])
            m_1_9_gt_total_num += len([True for p, s in zip(_1_9_probs.values(), _1_9_slopes.values()) if p > 0.9])
            
            ### greater == 0.8 and less == 0.2 ####
            _2_8_probs = {k: p for k, p in probs.items() if p > 0.8 or p < 0.2}
            _2_8_slopes = {k: s for k, s in slopes.items() if k in _2_8_probs.keys()}
            _2_8_right = [True for p, s in zip(_2_8_probs.values(), _2_8_slopes.values()) if (p > 0.8 and s > 0) or (p < 0.2 and s < 0)]
            
            _2_8_count = len(_2_8_probs)
            _2_8_acc = 0 if _2_8_count == 0 else len(_2_8_right) / _2_8_count
            
            m_2_8_total_num += len(_2_8_probs)
            m_2_8_right_num += len(_2_8_right)

            m_2_8_lt_right_num += len([True for p, s in zip(_2_8_probs.values(), _2_8_slopes.values()) if p < 0.2 and s < 0])
            m_2_8_lt_total_num += len([True for p, s in zip(_2_8_probs.values(), _2_8_slopes.values()) if p < 0.2])
            
            m_2_8_gt_right_num += len([True for p, s in zip(_2_8_probs.values(), _2_8_slopes.values()) if p > 0.8 and s > 0])
            m_2_8_gt_total_num += len([True for p, s in zip(_2_8_probs.values(), _2_8_slopes.values()) if p > 0.8])
            
            ### greater == 0.7 and less == 0.3 ####
            _3_7_probs = {k: p for k, p in probs.items() if p > 0.7 or p < 0.3}
            _3_7_slopes = {k: s for k, s in slopes.items() if k in _3_7_probs.keys()}
            _3_7_right = [True for p, s in zip(_3_7_probs.values(), _3_7_slopes.values()) if (p > 0.7 and s > 0) or (p < 0.3 and s < 0)]
            
            _3_7_count = len(_3_7_probs)
            _3_7_acc = 0 if _3_7_count == 0 else len(_3_7_right) / _3_7_count
            
            m_3_7_total_num += len(_3_7_probs)
            m_3_7_right_num += len(_3_7_right)

            m_3_7_lt_right_num += len([True for p, s in zip(_3_7_probs.values(), _3_7_slopes.values()) if p < 0.3 and s < 0])
            m_3_7_lt_total_num += len([True for p, s in zip(_3_7_probs.values(), _3_7_slopes.values()) if p < 0.3])
            
            m_3_7_gt_right_num += len([True for p, s in zip(_3_7_probs.values(), _3_7_slopes.values()) if p > 0.7 and s > 0])
            m_3_7_gt_total_num += len([True for p, s in zip(_3_7_probs.values(), _3_7_slopes.values()) if p > 0.7])

            ### greater == 0.6 and less == 0.4 ####
            _4_6_probs = {k: p for k, p in probs.items() if p > 0.6 or p < 0.4}
            _4_6_slopes = {k: s for k, s in slopes.items() if k in _4_6_probs.keys()}
            _4_6_right = [True for p, s in zip(_4_6_probs.values(), _4_6_slopes.values()) if (p > 0.6 and s > 0) or (p < 0.4 and s < 0)]
            
            _4_6_count = len(_4_6_probs)
            _4_6_acc = 0 if _4_6_count == 0 else len(_4_6_right) / _4_6_count
            
            m_4_6_total_num += len(_4_6_probs)
            m_4_6_right_num += len(_4_6_right)

            m_4_6_lt_right_num += len([True for p, s in zip(_4_6_probs.values(), _4_6_slopes.values()) if p < 0.4 and s < 0])
            m_4_6_lt_total_num += len([True for p, s in zip(_4_6_probs.values(), _4_6_slopes.values()) if p < 0.4])
            
            m_4_6_gt_right_num += len([True for p, s in zip(_4_6_probs.values(), _4_6_slopes.values()) if p > 0.6 and s > 0])
            m_4_6_gt_total_num += len([True for p, s in zip(_4_6_probs.values(), _4_6_slopes.values()) if p > 0.6])
            
            date_temp[date] = {
                    'probs': probs, 'slopes': slopes, 
                    'd_1-9_count': _1_9_count, 'd_1-9_acc': _1_9_acc,
                    'd_2-8_count': _2_8_count, 'd_2-8_acc': _2_8_acc,
                    'd_3-7_count': _3_7_count, 'd_3-7_acc': _3_7_acc,
                    'd_4-6_count': _4_6_count, 'd_4-6_acc': _4_6_acc,
                    }
            
        result.append({
            m_key: date_temp,
            'm_1_9_acc': 0 if m_1_9_total_num == 0 else m_1_9_right_num / m_1_9_total_num,
            'm_1_9_right_num': m_1_9_right_num,
            'm_1_9_lt_acc': 0 if m_1_9_lt_total_num == 0 else m_1_9_lt_right_num / m_1_9_lt_total_num,
            'm_1_9_lt_right_num': m_1_9_lt_right_num,
            'm_1_9_gt_acc': 0 if m_1_9_gt_total_num == 0 else m_1_9_gt_right_num / m_1_9_gt_total_num,
            'm_1_9_gt_right_num': m_1_9_gt_right_num,
            
            'm_2_8_acc': 0 if m_2_8_total_num == 0 else m_2_8_right_num / m_2_8_total_num,
            'm_2_8_right_num': m_2_8_right_num,
            'm_2_8_lt_acc': 0 if m_2_8_lt_total_num == 0 else m_2_8_lt_right_num / m_2_8_lt_total_num,
            'm_2_8_lt_right_num': m_2_8_lt_right_num,
            'm_2_8_gt_acc': 0 if m_2_8_gt_total_num == 0 else m_2_8_gt_right_num / m_2_8_gt_total_num,
            'm_2_8_gt_right_num': m_2_8_gt_right_num,
            
            'm_3_7_acc': 0 if m_3_7_total_num == 0 else m_3_7_right_num / m_3_7_total_num,
            'm_3_7_right_num': m_3_7_right_num,
            'm_3_7_lt_acc': 0 if m_3_7_lt_total_num == 0 else m_3_7_lt_right_num / m_3_7_lt_total_num,
            'm_3_7_lt_right_num': m_3_7_lt_right_num,
            'm_3_7_gt_acc': 0 if m_3_7_gt_total_num == 0 else m_3_7_gt_right_num / m_3_7_gt_total_num,
            'm_3_7_gt_right_num': m_3_7_gt_right_num,
            
            'm_4_6_acc': 0 if m_4_6_total_num == 0 else m_4_6_right_num / m_4_6_total_num,
            'm_4_6_right_num': m_4_6_right_num,
            'm_4_6_lt_acc': 0 if m_4_6_lt_total_num == 0 else m_4_6_lt_right_num / m_4_6_lt_total_num,
            'm_4_6_lt_right_num': m_4_6_lt_right_num,
            'm_4_6_gt_acc': 0 if m_4_6_gt_total_num == 0 else m_4_6_gt_right_num / m_4_6_gt_total_num,
            'm_4_6_gt_right_num': m_4_6_gt_right_num
            })
    
    if os.path.exists(writer_dir): os.remove(writer_dir)
    if level == 'day':
        with open(writer_dir, mode='x', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(['model', 'date', 'code', 'probs', 'slopes', 
                                'd_1-9_count', 'd_1-9_acc', 
                                'd_2-8_count', 'd_2-8_acc', 
                                'd_3-7_count', 'd_3-7_acc', 
                                'd_4-6_count', 'd_4-6_acc',
                                'm_1-9_acc',
                                'm_2-8_acc',
                                'm_3-7_acc',
                                'm_4-6_acc',
                                ])
            for item in result:
                arr_key = list(item.keys())[0]
                for date in item[arr_key]:
                    probs = item[arr_key][date]['probs']
                    slopes = item[arr_key][date]['slopes']
                    for (p_k, p_v), s_v in zip(probs.items(), slopes.values()):
                        csv_write.writerow([arr_key, date, str(p_k), str(p_v), str(s_v), 
                                            str(item[arr_key][date]['d_1-9_count']), 
                                            str(item[arr_key][date]['d_1-9_acc']),
                                            str(item[arr_key][date]['d_2-8_count']), 
                                            str(item[arr_key][date]['d_2-8_acc']),
                                            str(item[arr_key][date]['d_3-7_count']), 
                                            str(item[arr_key][date]['d_3-7_acc']),
                                            str(item[arr_key][date]['d_4-6_count']), 
                                            str(item[arr_key][date]['d_4-6_acc']),
                                            str(item['m_1-9_acc']),
                                            str(item['m_2-8_acc']),
                                            str(item['m_3-7_acc']),
                                            str(item['m_4-6_acc']),
                                            ])
    else:
        with open(writer_dir, mode='x', newline='') as f:
                    csv_write = csv.writer(f)
                    csv_write.writerow(['model', 'year', 
                                        'm_1_9_acc', 'm_1_9_right_num', 'm_1_9_lt_acc', 'm_1_9_lt_right_num', 'm_1_9_gt_acc', 'm_1_9_gt_right_num',
                                        'm_2_8_acc', 'm_2_8_right_num', 'm_2_8_lt_acc', 'm_2_8_lt_right_num', 'm_2_8_gt_acc', 'm_2_8_gt_right_num',
                                        'm_3_7_acc', 'm_3_7_right_num', 'm_3_7_lt_acc', 'm_3_7_lt_right_num', 'm_3_7_gt_acc', 'm_3_7_gt_right_num',
                                        'm_4_6_acc', 'm_4_6_right_num', 'm_4_6_lt_acc', 'm_4_6_lt_right_num', 'm_4_6_gt_acc', 'm_4_6_gt_right_num'])
                    for item in result:
                        arr_key = list(item.keys())[0]
                        csv_write.writerow([arr_key, start_date.split('-')[0],
                                            
                                            str(item['m_1_9_acc']),
                                            str(item['m_1_9_right_num']),
                                            str(item['m_1_9_lt_acc']),
                                            str(item['m_1_9_lt_right_num']),
                                            str(item['m_1_9_gt_acc']),
                                            str(item['m_1_9_gt_right_num']),
                                            
                                            str(item['m_2_8_acc']),
                                            str(item['m_2_8_right_num']),
                                            str(item['m_2_8_lt_acc']),
                                            str(item['m_2_8_lt_right_num']),
                                            str(item['m_2_8_gt_acc']),
                                            str(item['m_2_8_gt_right_num']),
                                            
                                            str(item['m_3_7_acc']),
                                            str(item['m_3_7_right_num']),
                                            str(item['m_3_7_lt_acc']),
                                            str(item['m_3_7_lt_right_num']),
                                            str(item['m_3_7_gt_acc']),
                                            str(item['m_3_7_gt_right_num']),
                                            
                                            str(item['m_4_6_acc']),
                                            str(item['m_4_6_right_num']),
                                            str(item['m_4_6_lt_acc']),
                                            str(item['m_4_6_lt_right_num']),
                                            str(item['m_4_6_gt_acc']),
                                            str(item['m_4_6_gt_right_num'])
                                            ])



if __name__ == '__main__':
    
    # import tensorflow as tf
    
    # model = tf.keras.models.load_model('E:\Quantitative\keras_model')
    
    # write_stocks_probability_using_test_data_for_tfrecord(
    #     file_name='000300f_2019', 
    #     time_steps=60, 
    #     model=model, 
    #     model_name='transformer_v2',
    #     write_db=True,
    #     use_db='mysql'
    #     )
    
    # summary_stocks_probability_using_test_data_for_tfrecord(file_name='000300f_2021', time_steps=60, model=model, writer_path='testtttttttttttttttt.csv')
    
    
    # export_train_valid_data_to_tfrecord(r'datasets\000300f.csv', r'000300_all_factors/*.csv', '2018-12-31', '000300f_v2_60_10', time_steps=60)
    # export_train_valid_data_to_tfrecord(r'datasets\000300f.csv', r'000300_all_factors/*.csv', '2016-01-01to2019-01-01to2020-01-01', '000300f_16to19to20_30_10')
    
    
    # update_latest_stocks(r'000300_all_factors\*.csv', True)
    
    # get_data_frame(r'000300_all_factors/*.csv', label_encoder=True).to_csv(r'datasets\000300f.csv', index=False)
    
    # write_test_data_to_tfrecord('2019-01-01', '2020-01-01', '000300f_2019_30_10', 30, 10, csv_path=r'datasets\000300f.csv', stock_path=r'000300_all_factors\*.csv')
    # write_test_data_to_tfrecord('2020-01-01', '2021-01-01', '000300f_2020_30_10', 30, 10, csv_path=r'datasets\000300f.csv', stock_path=r'000300_all_factors\*.csv')
    # write_test_data_to_tfrecord('2021-01-01', '2022-01-01', '000300f_2021_30_10', 30, 10, csv_path=r'datasets\000300f.csv', stock_path=r'000300_all_factors\*.csv')
    
    # aa = read_test_data_for_tfrecord('testttttttt', 128)
    # file_name = 'testttttttt'
    
    
    
    
    # test_data = tf.data.TFRecordDataset(r'datasets\tfreocrd\000300f_2021_test.tfrecord')

    # test_features = {
    #     'date': tf.io.FixedLenFeature([], tf.string),
    #     'keys': tf.io.VarLenFeature(tf.string),
    #     'values': tf.io.VarLenFeature(tf.float32),
    #     'labels': tf.io.VarLenFeature(tf.float32)
    # }

    # test_data = test_data.map(lambda x: tf.io.parse_single_example(x, test_features))
    
    # for i in test_data:
    #     print(i['date'].numpy().decode('utf-8'))
    #     print([i.decode('utf-8') for i in i['keys'].values.numpy()])
    #     print(i['values'].values.numpy().reshape(-1, 60, 313))
    #     print(i['labels'].values.numpy())

    # import tensorflow as tf
    
    # model = tf.keras.models.load_model('E:\Quantitative\keras_model')

    # write_history_stocks_probability_to_db('2019-01-01', '2022-03-10', time_steps=60, model_dir={'transformer':''}, csv_name=r'datasets\historical\f_latest.csv', stock_path=r'000300_all_factors/*.csv', model=model)
    # export_statistical_model_accuracy('2019-01-01', '2020-01-01', m=['transformer'], avg_m=[], level='year', future_steps=10, stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_transformer_2019.csv')
    # export_statistical_model_accuracy('2020-01-01', '2021-01-01', m=['transformer'], avg_m=[], level='year', future_steps=10, stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_transformer_2020.csv')
    # export_statistical_model_accuracy('2021-01-01', '2022-01-01', m=['transformer'], avg_m=[], level='year', future_steps=10, stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_transformer_2021.csv')
    
    
    # export_train_valid_data_to_tfrecord(r'datasets/399905f_v3_tfrecord.csv', r'399905_all_factors/*.csv', '2018-12-31', '399905f_v3')
    
    # train_data, valid_data = read_train_valid_data_for_tfrecord('399905f_v2',128, [30, 312])
    
    # stock_obj = {}

    # for stock in tqdm.tqdm(os.listdir('000300_all_factors')):
    #     key = stock[:-4]
    #     stock_obj[key] = jqdatasdk.get_security_info(key).display_name

    # for stock in tqdm.tqdm(jqdatasdk.get_index_stocks('399905.XSHE')):
    #     stock_obj[stock] = jqdatasdk.get_security_info(stock).display_name
        
    # b = json.dumps(stock_obj, ensure_ascii=False)
    # f = open('test.json','w', encoding='utf-8')
    # f.write(b)
    # f.close()

    # write_stocks_probability_to_db()
    
    # export_train_valid_data_to_tfrecord(r'datasets\399905f_tfrecord.csv', r'399905_all_factors\*.csv', '2020-12-31', '399905f')
        
    # read_train_data_for_tfrecord('000300f', 128)
    
    # import tensorflow as tf
    
    # n1 = np.random.normal(size=(10, 60, 300))
    # n2 = np.random.randint(0, 2, size=(10, 60, 1))
    # n3 = np.random.normal(size=(5, 60, 300))
    # n4 = np.random.randint(0, 2, size=(5, 60, 1))
    
    # feature = {
    #     "x_train": tf.train.Feature(bytes_list=tf.train.BytesList(value=[n1.tobytes()])),
    #     "x_test": tf.train.Feature(bytes_list=tf.train.BytesList(value=[n2.tobytes()])),
    #     "y_train": tf.train.Feature(bytes_list=tf.train.BytesList(value=[n3.tobytes()])),
    #     "y_test": tf.train.Feature(bytes_list=tf.train.BytesList(value=[n4.tobytes()]))
    # }
    # example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # writer = tf.io.TFRecordWriter("test.tfrecord")
    # writer.write(example_proto.SerializeToString())
    # writer.close()
    
    # raw_dataset = tf.data.TFRecordDataset("test.tfrecord")
    
    # for raw_record in raw_dataset.take(1):
    #     example = tf.train.Example()
    #     example.ParseFromString(raw_record.numpy())
    #     print(example)
        
    # result = {}
    # # example.features.feature is the dictionary
    # for key, feature in example.features.feature.items():
    #     # The values are the Feature objects which contain a `kind` which contains:
    #     # one of three fields: bytes_list, float_list, int64_list
    #     kind = feature.WhichOneof('kind')
    #     result[key] = np.array(getattr(feature, kind).value)

    # print(result)
    
    # import random
    
    # objs = {'a':1,'b':2,'c':3,'d':4,'e':5}
    
    # print(random.sample(objs.keys(), 2))
    
    # export_stocks_to_csv('399905_all_factors', stock_index='399905.XSHE', all_factor=True)
    
    # after_code_changed中调用  inout_cash
    # count = jqdatasdk.get_query_count()
    
    # slope_arr = []
    # time_steps = 30
    # future_steps = 10
    # df = get_data_frame('000300_/*.csv')
    # for _, _df in tqdm.tqdm(df.groupby('code')):
    #     for i in range(len(_df)):
    #         if i + time_steps + future_steps >= len(_df): break
    #         y_future = _df.iloc[i + time_steps - 1: i + time_steps - 1 + future_steps]['close_slope'].to_numpy()[:, np.newaxis]
    #         x_future = np.arange(0, len(y_future))[:, np.newaxis]
    #         slope = np.squeeze(LinearRegression().fit(x_future, y_future).coef_)
    #         slope_arr.append(float(slope))
    #         # ax = sns.regplot(x_future, y_future, x_ci=None)
    #         # ax.set_title(slope)
    #         # plt.show()
    # sns.displot(slope_arr, kde=True)
    # plt.show()
    
    # df = jqdatasdk.get_index_weights(index_id="000300.XSHG", date="2021-01-01")
    # df.to_csv('get_index_weights2.csv', encoding='utf_8_sig')
    
    # df = get_data_frame('000300/*.csv', label_encoder=True)
    # df.to_csv('datasets/000300_v10.csv', index=False)
    # write_stocks_probability_to_db()
    
    # stock_index='000300.XSHG'
    # start_date = '2005-04-08'
    # end_date = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
    # # 股票每日交易数据
    # daily_price = jqdatasdk.get_price(stock_index, fields=backtrader_fields, start_date=start_date, end_date=end_date, skip_paused=False)
    # daily_price.reset_index(inplace=True)
    # daily_price.rename(columns={'index':'date'}, inplace=True)
    # daily_price.to_csv(f'datasets\{stock_index}_index.csv', index=False)
    # factors_info = jqdatasdk.get_all_factors()
    # write_history_stocks_probability_to_db('2019-01-01', '2022-01-18', model_dir={
    #     'v1':'weight\\000300_v1', 
    #     'v2':'weight\\000300_v2',
    #     }, use_db='mysql')
    # export_statistical_model_accuracy('2019-01-01', '2020-01-01', m=['v15'], avg_m=[], writer_dir='results/model_accuracy_v15_2019.csv')
    # export_statistical_model_accuracy('2020-01-01', '2021-01-01', m=['v15'], avg_m=[], writer_dir='results/model_accuracy_v15_2020.csv')
    # export_statistical_model_accuracy('2021-01-01', '2022-01-01', m=['v15'], avg_m=[], writer_dir='results/model_accuracy_v15_2021.csv')
    # export_statistical_model_accuracy(m=['v15'], avg_m=[], writer_dir='results/model_accuracy_v15_all.csv')
    # export_statistical_model_accuracy('2019-01-01', '2020-01-01', m=['v16'], avg_m=[], writer_dir='results/model_accuracy_v16_2019.csv')
    # export_statistical_model_accuracy('2020-01-01', '2021-01-01', m=['v16'], avg_m=[], writer_dir='results/model_accuracy_v16_2020.csv')
    # export_statistical_model_accuracy('2021-01-01', '2022-01-01', m=['v16'], avg_m=[], writer_dir='results/model_accuracy_v16_2021.csv')r
    # export_statistical_model_accuracy(m=['v16'], avg_m=[], writer_dir='results/model_accuracy_v16_all.csv')
    
    # get_data_frame(r'000300_all_factors/*.csv', label_encoder=True).to_csv(r'datasets\historical\f_latest.csv', index=False)
    # write_history_stocks_probability_to_db('2019-01-01', '2022-01-10',  model_dir={'v22':r'weight\000300_v22'}, csv_name=r'f_latest.csv', stock_path=r'000300_all_factors/*.csv')
    # export_statistical_model_accuracy('2019-01-01', '2020-01-01', m=['v22'], avg_m=[], stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v22_2019.csv')
    # export_statistical_model_accuracy('2020-01-01', '2021-01-01', m=['v22'], avg_m=[], stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v22_2020.csv')
    # export_statistical_model_accuracy('2021-01-01', '2022-01-01', m=['v22'], avg_m=[], stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v22_2021.csv')
    # export_statistical_model_accuracy(m=['v22'], avg_m=[], stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v22_all.csv')
    
    # get_data_frame(r'000300_all_factors/*.csv', label_encoder=True).to_csv(r'datasets\historical\f_latest.csv', index=False)
    # write_history_stocks_probability_to_db('2019-01-01', '2022-01-10',  model_dir={'v22':r'weight\000300_v22'}, csv_name=r'f_latest.csv', stock_path=r'000300_all_factors/*.csv')
    # export_statistical_model_accuracy('2019-01-01', '2020-01-01', m=['v22'], avg_m=[], stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v22_2019.csv')
    # export_statistical_model_accuracy('2020-01-01', '2021-01-01', m=['v22'], avg_m=[], stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v22_2020.csv')
    # export_statistical_model_accuracy('2021-01-01', '2022-01-01', m=['v22'], avg_m=[], stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v22_2021.csv')
    # export_statistical_model_accuracy(m=['v22'], avg_m=[], stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v22_all.csv')
    
    # get_data_frame(r'000300_all_factors/*.csv', label_encoder=True).to_csv(r'datasets\historical\f_latest.csv', index=False)
    # write_history_stocks_probability_to_db('2019-01-01', '2022-01-01', model_dir={'v21':r'weight\000300_v21'})
    # export_statistical_model_accuracy('2019-01-01', '2020-01-01', m=['v21'], avg_m=[], level='year', future_steps=3, writer_dir=r'results/model_accuracy_v21_2019.csv')
    # export_statistical_model_accuracy('2020-01-01', '2021-01-01', m=['v21'], avg_m=[], level='year', future_steps=3, writer_dir=r'results/model_accuracy_v21_2020.csv')
    # export_statistical_model_accuracy('2021-01-01', '2022-01-01', m=['v21'], avg_m=[], level='year', future_steps=3, writer_dir=r'results/model_accuracy_v21_2021.csv')
    
    # write_history_stocks_probability_to_db('2019-01-01', '2022-01-01', time_steps=30, model_dir={'v23':r'weight\000300_v23'})
    # export_statistical_model_accuracy('2019-01-01', '2020-01-01', m=['v23'], avg_m=[], level='year', future_steps=5, writer_dir=r'results/model_accuracy_v23_2019.csv')
    # export_statistical_model_accuracy('2020-01-01', '2021-01-01', m=['v23'], avg_m=[], level='year', future_steps=5, writer_dir=r'results/model_accuracy_v23_2020.csv')
    # export_statistical_model_accuracy('2021-01-01', '2022-01-01', m=['v23'], avg_m=[], level='year', future_steps=5, writer_dir=r'results/model_accuracy_v23_2021.csv')
    # get_data_frame(r'000300_all_factors/*.csv', label_encoder=True).to_csv(r'datasets\historical\f_latest.csv', index=False)
    
    # get_data_frame(r'000300_all_factors/*.csv', label_encoder=True).to_csv(r'datasets\historical\f_latest.csv', index=False)
    # write_history_stocks_probability_to_db('2019-01-01', '2022-01-01', time_steps=60, model_dir={'v24':r'weight\000300_v24'}, csv_name=r'f_latest.csv', stock_path=r'000300_all_factors/*.csv')
    # export_statistical_model_accuracy('2019-01-01', '2020-01-01', m=['v24'], avg_m=[], level='year', future_steps=10, stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v24_2019.csv')
    # export_statistical_model_accuracy('2020-01-01', '2021-01-01', m=['v24'], avg_m=[], level='year', future_steps=10, stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v24_2020.csv')
    # export_statistical_model_accuracy('2021-01-01', '2022-01-01', m=['v24'], avg_m=[], level='year', future_steps=10, stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v24_2021.csv')
    
    # get_data_frame(r'000300_all_factors/*.csv', label_encoder=True).to_csv(r'datasets\historical\f_latest.csv', index=False)
    # write_history_stocks_probability_to_db('2021-01-01', '2022-01-01', time_steps=60, model_dir={'v25':r'weight\000300_v25'}, csv_name=r'f_latest.csv', stock_path=r'000300_all_factors/*.csv')
    # export_statistical_model_accuracy('2021-01-01', '2022-01-01', m=['v25'], avg_m=[], level='year', future_steps=10, stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_v25_2021.csv')
    
    # write_history_stocks_probability_to_db('2022-01-01', '2022-02-24', time_steps=60, model_dir={'v24':r'weight\000300_v24'}, csv_name=r'f_latest.csv', stock_path=r'000300_all_factors/*.csv')
    
    # get_data_frame(r'000300_all_factors/*.csv', label_encoder=True).to_csv(r'datasets\historical\f_latest.csv', index=False)
    # write_history_stocks_probability_to_db('2019-01-01', '2022-01-01', time_steps=60, model_dir={'tfrecord':r'weight\tfrecord'}, csv_name=r'datasets\historical\f_latest.csv', stock_path=r'000300_all_factors/*.csv')
    # export_statistical_model_accuracy('2019-01-01', '2020-01-01', m=['tfrecord'], avg_m=[], level='year', future_steps=5, stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_tfrecord_2019.csv')
    # export_statistical_model_accuracy('2020-01-01', '2021-01-01', m=['tfrecord'], avg_m=[], level='year', future_steps=5, stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_tfrecord_2020.csv')
    # export_statistical_model_accuracy('2021-01-01', '2022-01-01', m=['tfrecord'], avg_m=[], level='year', future_steps=5, stock_path=r'000300_all_factors/*.csv', writer_dir=r'results/model_accuracy_tfrecord_2021.csv')
    
    print("====complete====")
