import os
import csv
import glob
import tqdm
import copy
import json
import datetime
import numpy as np
import pandas as pd
import backtrader as bt
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from db_content import SqlServer
from data_processing import get_single_finetune_data, get_data_frame, exclude_columns


class SingleStockStrategy(bt.Strategy):
    ''' 单股票交易策略类 '''
    params = (
        ('model', None),
        ('df', None),
        ('less', None),
        ('greater', None),
        ('time_steps', None),
    )

    def __init__(self):
        self.order = None
        self.dataclose = self.datas[0].close
        self.model = self.params.model
        self.df = self.params.df
        self.less = self.params.less
        self.greater = self.params.greater
        self.time_steps = self.params.time_steps
        # self.sma5 = bt.indicators.SimpleMovingAverage(self.datas[0], period=5)
        # self.sma10 = bt.indicators.SimpleMovingAverage(self.datas[0], period=10)

    def log(self, txt, dt=None):
        ''' 此策略的日志记录功能'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        '''通知订单中的任何状态更改'''
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('Buy Executed, Price: %.2f, Cost: %.2f, Comm %.2f' % (
                    order.executed.price, order.executed.value, order.executed.comm))
            else:  # Sell
                self.log('Sell Executed, Price: %.2f, Cost: %.2f, Comm %.2f' % (
                    order.executed.price, order.executed.value, order.executed.comm))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        '''通知任何开仓/更新/平仓交易'''
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def _get_sotck_probability(self):
        date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        index = self.df.loc[self.df['date'] == date].index[0]
        x = self.df.iloc[index-self.time_steps:index]
        x.drop(exclude_columns, axis=1, inplace=True)
        x = np.expand_dims(x, axis=0)
        y = np.squeeze(self.model.predict(x))
        self.log(f'y：{y}')
        return y

    def next(self):
        '''制定交易策略'''
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # region 双均线止盈止亏策略
        # if not self.position:
        #     if self.sma5 > self.sma10:
        #         self.log('BUY CREATE, %.2f' % self.dataclose[0])
        #         self.buy()
        # else:
        #     if self.sma5 < self.sma10 and (self.dataclose[0] >= self.position.price * 1.25 or self.dataclose[0] <= self.position.price * 0.9):
        #         self.log('SELL CREATE, %.2f' % self.dataclose[0])
        #         self.sell()
        # endregion

        # region 模型三分类策略
        # y = self._get_y()
        # y = np.argmax(y)
        # if not self.position:
        #     if y == 2:
        #         self.log('BUY CREATE, %.2f' % self.dataclose[0])
        #         self.buy()
        # else:
        #     if y == 0:
        #         self.log('SELL CREATE, %.2f' % self.dataclose[0])
        #         self.sell()
        # endregion

        # region 模型止盈止亏策略
        y = self._get_sotck_probability()
        if not self.position:  # 判断是否持仓
            if y > self.greater:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()
        else:  # 如何模型判断未来会盈利, 并且已盈利25%; 或者模型判单未来会亏损, 并且已亏损10%
            if y > self.greater:
                return

            if y < self.less:
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.sell()
                return

            # self.dataclose[0] >= self.position.price * 1.25 or
            if self.dataclose[0] <= self.position.price * 0.9:
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.sell()
        # endregion


class MultiStockStrategy(bt.Strategy):
    ''' 多股票交易策略类 '''
    params = (
        ('model', None),
        ('df', None),
        ('less', None),
        ('greater', None),
        ('time_steps', None),
        ('save_stock_prob', None),
        ('save_model_name', None),
    )

    def __init__(self):
        self.model = self.params.model
        self.df = self.params.df
        self.less = self.params.less
        self.greater = self.params.greater
        self.time_steps = self.params.time_steps
        self.save_stock_prob = self.params.save_stock_prob
        self.save_model_name = self.params.save_model_name
        self.holding_stock = []
        self.holding_stock_num = 10
        self.trade_status = {0: 'Created', 1: 'Open', 2: 'Closed'}
        self.cash_line = {}
        self.value_line = {}
        if self.save_stock_prob:
            self.db = SqlServer()

    def log(self, txt, dt=None):
        ''' 此策略的日志记录功能'''
        dt = dt or self.datetime.date()
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        '''通知订单中的任何状态更改'''
        # self.log(f'[ORDER {order.data._name} {order.ref}] Status {order.getstatusname()}')
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                buy_or_sell = 'Buy'
            else:  # Sell
                buy_or_sell = 'Sell'
                self.holding_stock.remove(order.data._name)
            self.log(f'[ORDER {order.data._name} {order.ref}] {buy_or_sell} Executed, Size: %.2f, Price: %.2f, Cost: %.2f, Comm %.2f' % (
                order.executed.size, order.executed.price, order.executed.value, order.executed.comm))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(
                f'[ORDER {order.data._name} {order.ref}] Order Canceled/Margin/Rejected')
            self.holding_stock.remove(order.data._name)

    def notify_trade(self, trade):
        '''通知任何开仓/更新/平仓交易'''
        # self.log(f'[TRADE {trade.data._name} {trade.ref}] Status {self.trade_status[trade.status]}')
        if not trade.isclosed:
            return
        self.log(f'[TRADE {trade.data._name} {trade.ref}] Closed: PnL Gross %.2f, Net %.2f' % (
            trade.pnl, trade.pnlcomm))

    def notify_cashvalue(self, cash, value):
        ''' 通知经纪人当前现金和投资组合 '''
        dt = self.datetime.date()
        self.cash_line[dt] = cash
        self.value_line[dt] = value
        self.log(f'[CASH VALUE] Cash %.2f, Value %.2f' % (cash, value))

    def save_sotck_probability_to_db(self, sotck_prob):
        date = self.datetime.date().isoformat()
        json_str = json.dumps(sotck_prob)
        sql = "insert into Sotck_Probability values('{}','{}','{}')".format(
            date, json_str, self.save_model_name)
        self.db.execute_sql(sql)

    def _get_sotck_probability(self):
        prob_dict = {}
        feature_arr = []
        dt = self.datetime.date().strftime('%Y-%m-%d')
        for stock in self.dnames:
            df = self.df[stock]
            row_index = df.loc[df['date'] == dt].index[0]
            features = df.iloc[row_index-self.time_steps:row_index]
            features.drop(exclude_columns, axis=1, inplace=True)
            feature_arr.append(features.to_numpy())
        feature_arr = np.array(feature_arr, dtype=np.float32)
        probs = self.model.predict(feature_arr)
        for i in range(len(probs)):
            prob_dict[list(self.dnames.keys())[i]] = float(probs[i])
        return prob_dict

    def _print_holding_stock(self):
        self.log(f'==== 当前持仓详情[{len(self.holding_stock)}]====')
        for i in range(len(self.holding_stock)):
            stock = self.holding_stock[i]
            data = self.getdatabyname(stock)
            pos = self.broker.getposition(data)
            value = self.broker.get_value([data])
            self.log(f'{stock}: %.2f %.2f %.2f' % (pos.price, pos.size, value))

    def next(self):
        '''制定交易策略'''
        buy, sell = 0, 0
        self.log('==== Transaction Start ====')
        self._print_holding_stock()
        rise_fall_prob = self._get_sotck_probability()
        if self.save_stock_prob:
            self.save_sotck_probability_to_db(rise_fall_prob)

        ### 卖出 ###
        _holding = copy.deepcopy(self.holding_stock)
        for i in range(len(_holding)):
            _threshold = rise_fall_prob[_holding[i]]
            if _threshold < self.less and self.getpositionbyname(_holding[i]):
                self.order_target_value(data=self.getdatabyname(
                    _holding[i]), target=0)  # Create Sell
                self.log(f'[NEXT {_holding[i]}] Create Sell, Size: %.2f, Open_Price: %.2f, Close_Price: %.2f' % (
                    0, self.getdatabyname(_holding[i]).open[0], self.getdatabyname(_holding[i]).close[0]))
                sell += 1

        ### 买入 ###
        if len(self.holding_stock) < self.holding_stock_num:
            buy_num = self.holding_stock_num - len(self.holding_stock)
            buy_size = (self.broker.cash - 10000) / buy_num
            sorted_tuple = sorted(rise_fall_prob.items(),
                                  key=lambda kv: (kv[1], kv[0]), reverse=True)
            for stock, _threshold in sorted_tuple:
                if _threshold > self.greater and not self.getpositionbyname(stock):
                    self.order_target_value(data=self.getdatabyname(
                        stock), target=buy_size)  # Create Buy
                    self.log(f'[NEXT {stock}] Create Buy, Size: %.2f, Open_Price: %.2f, Close_Price: %.2f' % (
                        buy_size, self.getdatabyname(stock).open[0], self.getdatabyname(stock).close[0]))
                    self.holding_stock.append(stock)
                    buy += 1
                if len(self.holding_stock) == self.holding_stock_num:
                    break

        if not buy and not sell:
            self.log('Nothing To Do')
        else:
            self.log(f'Buy: {buy} Orders,  Sell: {sell} Orders')
        self.log('==== Transaction End ====')


class Run_Backtrader(object):
    def __init__(self):
        super().__init__()
        ### basic config ###
        self.stocks = []
        self.pre_train = True
        self.fine_tune = False
        self.multi_stock = True
        self.save_stock_prob = False
        self.save_model_name = 'v8'
        self.start_date = '2019-01-01'
        self.end_date = '2021-10-28'
        self.polt_path = r'plots\plot_multi_stock\\'
        self.stocks_path = r'000300\*.csv'
        self.csv_columns = ['stock', 'start_date', 'end_date', 'avg_annual_return',
                            'max_draw_down', 'max_money_down', 'sharpe_ratio', 'sharpe_ratio']
        ### backtrader config ####
        self.less_than = 0.2
        self.greater_than = 0.8
        self.bt_feed_dir = r'backtrader\\'
        self.bt_cash = 100000
        self.bt_commission = 0.003
        ### model config ###
        self.classify_num = 2
        self.epochs = 300
        self.time_steps = 30
        self.batch_size = 32
        self.learning_rate = 3e-4
        self.input_shape = (self.time_steps, 12)
        self.pre_stocks_path = r'datasets\000300_v8.csv'
        self.pre_weights_path = r'weight\000300_v8\lstm.ckpt'
        ### init ###
        if self.pre_train:
            self.model = self._bulid_model()
            self.model.load_weights(self.pre_weights_path)
        if self.pre_train and not self.multi_stock:
            self.pre_df = pd.read_csv(self.pre_stocks_path)
        self._check_plot_dir()

    def _run_backtrader(self):
        result = {}
        # Create a cerebro entity
        cerebro = bt.Cerebro()
        # cerebro = bt.Cerebro(cheat_on_open=True)
        # cerebro.broker.set_coo(True)
        # Add a strategy
        if self.multi_stock:
            cerebro.addstrategy(MultiStockStrategy, model=self.model, df=self.df, less=self.less_than,
                                greater=self.greater_than, time_steps=self.time_steps, save_stock_prob=self.save_stock_prob)
        else:
            cerebro.addstrategy(SingleStockStrategy, model=self.model, df=self.df, less=self.less_than,
                                greater=self.greater_than, time_steps=self.time_steps, save_model_name=self.save_model_name)
        # Create a Data Feed
        start = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(self.end_date, '%Y-%m-%d')
        for stock in self.stocks:
            data = bt.feeds.GenericCSVData(
                dataname=f'{self.bt_feed_dir}{stock}.csv',
                dtformat=('%Y-%m-%d'),
                fromdate=datetime.datetime(start.year, start.month, start.day),
                todate=datetime.datetime(end.year, end.month, end.day),
                datetime=0, open=1, close=2, low=3, high=4, volume=5, openinterest=-1)
            # Add the Data Feed to Cerebro
            cerebro.adddata(data)
        # Set our desired cash start
        cerebro.broker.setcash(self.bt_cash)
        # Add a AllInSizer sizer according to the stake
        if self.multi_stock:
            cerebro.addsizer(bt.sizers.FixedSize, stake=100)
        else:
            cerebro.addsizer(bt.sizers.AllInSizer)
        # Set the commission - 0.1% ... divide by 100 to remove the %
        cerebro.broker.setcommission(commission=self.bt_commission)
        # cerebro.broker.set_checksubmit(checksubmit=False)
        # 一些指标
        cerebro.addanalyzer(bt.analyzers.AnnualReturn,
                            _name='annual_return')  # 年利润率
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='draw_down')  # 回测
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe_ratio_a')
        # Print out the starting conditions
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        # Run over everything
        s = cerebro.run()[0]
        # Print out the final result
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        if not self.multi_stock:
            fig = cerebro.plot(show=False)[0][0]
        else:
            fig = self._plot_mulit_stock(s)
        result["figs"] = fig
        result["annual_return"] = s.analyzers.annual_return.get_analysis()
        result["draw_down"] = s.analyzers.draw_down.get_analysis()
        result["sharpe_ratio"] = s.analyzers.sharpe_ratio.get_analysis()
        result["sharpe_ratio_a"] = s.analyzers.sharpe_ratio_a.get_analysis()
        return result

    def _bulid_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Convolution1D(
                128, 3, padding='same', activation='relu', input_shape=self.input_shape),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(
            lr=self.learning_rate), loss='binary_crossentropy', metrics=['acc'])
        return model

    def _train_model(self):
        x_train, y_train, _, _ = get_single_finetune_data(
            self.df, classify_num=self.classify_num)
        self.model.fit(x_train, y_train, epochs=self.epochs,
                       batch_size=self.batch_size)

    def _save_result(self, result):
        if self.multi_stock:
            result['figs'].savefig(f'{self.polt_path}multi_stock.png')
        else:
            # 保存图片
            result['figs'].savefig(f'{self.polt_path}{self.stocks[0]}.png')
            # 写入回撤数据
            avg_annual_return = np.mean(list(result['annual_return'].values()))
            max_draw_down = result['draw_down']['max']['drawdown']
            max_money_down = result['draw_down']['max']['moneydown']
            sharpe_ratio = result['sharpe_ratio']['sharperatio']
            sharpe_ratio_a = result['sharpe_ratio_a']['sharperatio']
            with open(f'{self.polt_path}back_traded.csv', mode='a', newline='') as f:
                _writer = csv.writer(f)
                _writer.writerow([self.stocks[0], self.start_date, self.end_date, avg_annual_return,
                                 max_draw_down, max_money_down, sharpe_ratio, sharpe_ratio_a])

    def _check_plot_dir(self, file_name='back_traded.csv'):
        if not os.path.exists(self.polt_path):
            os.makedirs(self.polt_path)
            with open(f'{self.polt_path}\\{file_name}', 'w', newline='') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(self.csv_columns)

    def _check_dataframe(self, df):
        skip_stocks, _dict = [], {}
        for stock, _df in df.groupby('code'):
            index = _df.loc[_df['date'] >= self.start_date].index
            assert len(index) >= self.time_steps
            start_date = df.iloc[index[0]-self.time_steps]['date']
            _dict[stock] = start_date

        # 以最多的股票时间为回测开始时间
        if len(set(_dict.values())) != 1:
            date = max(_dict.values(), key=list(
                _dict.values()).count)  # 筛选出票数最多的时间
            stocks = [k for k, v in _dict.items() if v == date]  # 根据时间找到股票代码
            skip_stocks = list(set(_dict.keys()).difference(
                set(stocks)))  # 求两个list的差集
        return skip_stocks

    def _plot_mulit_stock(self, result):
        ax = sns.lineplot(x=result.value_line.keys(),
                          y=result.value_line.values())
        ax.set(xlabel='date', ylabel='values')
        plt.get_current_fig_manager().full_screen_toggle()
        plt.pause(1)
        plt.close()
        return ax.figure

    def _get_single_df(self, stock, path):
        if self.pre_train:
            df = self.pre_df.loc[self.pre_df['code'] == int(stock[:6])]
            df.reset_index(drop=True, inplace=True)
        else:
            df = get_data_frame(path)
        return df

    def _get_multi_df(self, data_slice):
        df_dict = {}
        df = pd.read_csv(self.pre_stocks_path)
        stocks = sorted(list(set(df['code'])))
        if data_slice is not None:
            stocks = stocks[data_slice]
        skip_stocks = self._check_dataframe(df)
        if len(skip_stocks) > 0:
            stocks = [s for s in stocks if s not in skip_stocks]
        for _, _df in df.groupby('code'):
            index = _df.loc[_df['date'] >= self.start_date].index[0]
            start_date = df.iloc[index-self.time_steps]['date']
            break
        df = df.loc[(df['code'].isin(stocks)) & (
            (df['date'] >= start_date) & (df['date'] <= self.end_date))]
        for code, _df in df.groupby('code'):
            temp_df = copy.deepcopy(_df)
            temp_df.reset_index(drop=True, inplace=True)
            df_dict[code] = temp_df
        return stocks, df_dict

    def run_single_stock(self, data_slice):
        stocks = glob.glob(self.stocks_path)
        if data_slice is not None:
            stocks = stocks[data_slice]
        plotted = [item[:-4] for item in os.listdir(self.polt_path)]
        for path in tqdm.tqdm(stocks):
            stock = path.split('\\')[1][:-4]
            if stock in plotted:
                continue
            self.stocks = [stock]
            self.df = self._get_single_df(stock, path)
            if not self.pre_train:
                self.model = self._bulid_model()
            if self.fine_tune:
                self._train_model()
            result = self._run_backtrader()
            self._save_result(result)
            if self.pre_train and self.fine_tune:
                self.model.load_weights(self.pre_weights_path)

    def run_multi_sotck(self, data_slice):
        if not self.pre_train and self.multi_stock:
            return
        self.stocks, self.df = self._get_multi_df(data_slice)
        result = self._run_backtrader()
        self._save_result(result)

    def run(self, data_slice=None):
        if self.multi_stock:
            self.run_multi_sotck(data_slice)
        else:
            self.run_single_stock(data_slice)


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import pandas as pd
    import json
    import random
    # run_func_demo
    import jqdatasdk
    jqdatasdk.auth('18616758695', 'Abcd5678!')
    from rqalpha import run_func
    from rqalpha import api as rqalpha_api

    sqlserver = SqlServer()

    def init(context):
        '''初始化方法, 会在程序启动的时候执行'''
        rqalpha_api.logger.info("init")
        # 在context中保存全局变量
        context.model = model_version
        context.less = 0.1
        context.greater = 0.9
        context.holding_stock = []
        context.holding_stock_num = 10
        context.latest_00300 = jqdatasdk.get_index_stocks('000300.XSHG')
        rqalpha_api.update_universe(context.latest_00300)

    def before_trading(context):
        '''会在每天策略交易开始前调用'''
        rqalpha_api.logger.info(
            f'==== 开盘前持仓详情[{len(context.holding_stock)}]====')
        for i in range(len(context.holding_stock)):
            stock = context.holding_stock[i]
            pos = context.portfolio.positions[stock]
            name = rqalpha_api.instruments(stock).symbol
            print(f'{stock}: {name} %.2f %.2f' % (pos.quantity, pos.pnl))
        rqalpha_api.logger.info('==== 开盘前持仓详情 End ====')

    def handle_bar(context, bar_dict):
        '''bar数据更新时会自动触发调用'''
        buy, sell = 0, 0
        print('==== market_open Start ====')

        date = context.now.strftime('%Y-%m-%d')
        sql = "select * from Simulated_Transaction where Model='{0}' and Date='{1}'".format(context.model, date)
        result = sqlserver.get_sql(sql)
        if len(result) == 0:
            return

        stock_probs = json.loads(result[0]['Stock_Prob'])

        holding = copy.deepcopy(context.holding_stock)
        for i in range(len(holding)):
            stock = holding[i]
            if stock in context.portfolio.positions and stock_probs[stock] < context.less:
                _order = rqalpha_api.order_target_value(stock, 0)  # Create Sell
                if _order:  # Sell Successful
                    context.holding_stock.remove(stock)
                    name = rqalpha_api.instruments(stock).symbol
                    day_open = rqalpha_api.history_bars(stock, 1, '1d', 'open')[0]
                    print(f'[{stock} | {name}] Create Sell, Size: %.2f, Open_Price: %.2f' % (0, day_open))
                    sell += 1

        if len(context.holding_stock) < context.holding_stock_num:
            buy_num = context.holding_stock_num - len(context.holding_stock)
            buy_size = context.portfolio.cash / buy_num
            filter_stock = {k: v for k, v in stock_probs.items() if k in context.latest_00300 and k not in context.portfolio.positions and v > context.greater}
            buy_stocks = random.sample(filter_stock.keys(), len(filter_stock) if buy_num > len(filter_stock) else buy_num)
            for stock in buy_stocks:
                _order = rqalpha_api.order_value(stock, buy_size)  # Create Buy
                if _order:  # Buy Successful
                    context.holding_stock.append(stock)
                    name = rqalpha_api.instruments(stock).symbol
                    day_open = rqalpha_api.history_bars(stock, 1, '1d', 'open')[0]
                    print(f'[{stock} | {name}] Create Buy, Size: %.2f, Open_Price: %.2f' % (buy_size, day_open))
                    buy += 1

        if not buy and not sell:
            print('Nothing To Do')
        else:
            print(f'Buy: {buy} Orders,  Sell: {sell} Orders')
        print('==== market_open End ====')

    def after_trading(context):
        '''会在每天交易结束后调用'''
        rqalpha_api.logger.info(f'==== 收盘后打印收益价值 ====')
        rqalpha_api.logger.info(context.portfolio.total_value)
        rqalpha_api.logger.info(f'==== 收盘后打印收益价值 End ====')


    model_version = 'transformer'

    alpha, beta, sharpe, annualized_returns = [], [], [], []

    for i in range(50):

        config = {
            "base": {
                "start_date": "2021-01-01",
                "end_date": "2022-01-01",
                "benchmark": "000300.XSHG",
                "accounts": {"stock": 100000}
            },
            "extra": {
                "log_level": "verbose",
            },
            "mod": {
                "sys_analyser": {"enabled": True, "plot": False}
            },
        } # 'plot_save_file': rf'rqalpha/{model_version}/{i}.png', "output_file": "result.pkl"

        # 您可以指定您要传递的参数
        summary = run_func(init=init, handle_bar=handle_bar, config=config)['sys_analyser']['summary'] #  before_trading=before_trading,after_trading=after_trading,

        alpha.append(summary['alpha'])
        beta.append(summary['beta'])
        sharpe.append(summary['sharpe'])
        annualized_returns.append(summary['annualized_returns'])

    print(np.mean(alpha), np.mean(beta), np.mean(sharpe), np.mean(annualized_returns))

        # 如果你的函数命名是按照 API 规范来，则可以直接按照以下方式来运行
        # run_func(**globals())

        # df = pd.read_pickle('result.pkl')

        # print(df['summary'])

    # Run_Backtrader().run()
    print('===== complete ======')
