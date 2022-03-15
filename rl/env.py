import gym
import glob
import numpy as np
import pandas as pd


class StockEnv(gym.Env):
    def __init__(self,start_date, end_date, amount=100000, holding_num=10, time_steps=30, single_stock=None, allin=True):
        self.size = 100
        self.amount = amount
        self.time_steps = time_steps
        self.holding_num = holding_num
        self.start_date = start_date
        self.end_date = end_date
        self.single_stock = single_stock
        self.allin = allin
        self.trade_days = self.load_trade_days()
        self.df = self.load_stock_data()
        self.bt_df = self.load_bt_data()
        action_shape = (1,) if self.single_stock is not None else (len(self.df),)
        observation1_shape = (self.time_steps, 81) if self.single_stock is not None else (len(self.df), self.time_steps, 81)
        observation2_shape = (1*4,) if self.single_stock is not None else (len(self.df) * 4,)
        self.action_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=action_shape, dtype=np.float32)
        self.observation1_space = gym.spaces.Box(low=-1, high=1, shape=observation1_shape, dtype=np.float32)
        self.observation2_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=observation2_shape, dtype=np.float32)
    
    def reset(self):
        self.cash = self.amount
        self.value = self.amount
        self.holding_stocks = []
        self.date = self.start_date
        self.date_index = iter(range(self.time_steps, len(self.trade_days), 1))
        self.positions = self.load_positions()
        return self.get_observation()

    def step(self, action):
        self.value = self.update_value()
        if self.allin:
            for stock, money in zip(self.df.keys(), [action]):
                if action == 0:
                    self.sell_all(stock)
                else:
                    self.buy_all(stock)
        else:
            for stock, money in zip(self.df.keys(), action * self.amount):
                if money > 0 :
                    self.buy(stock, money)
                else:
                    self.sell(stock, np.absolute(money))
        self.date, done = self.get_next_date()
        return self.get_observation(), (self.value/self.amount)-1 , done, {}

    def render(self, mode='human'):
        print('no render function')

    def close(self):
        print('no close function')
        
    def load_trade_days(self):
        df = pd.read_csv(r'datasets\\trade_days.csv')
        index = df.loc[df['date'] == self.start_date].index[0]
        start_date = df.iloc[index-self.time_steps].values[0]
        df = df.loc[(df['date'] >= start_date) & (df['date'] < self.end_date)]['date']
        df.reset_index(drop=True, inplace=True)
        return df
    
    def load_stock_data(self):
        _dict = {}
        df = pd.read_csv(r'datasets\\000300.csv')
        if self.single_stock is not None:
            df = df.loc[df['code'] == self.single_stock]
        for code, data in df.groupby('code'):
            _df = data.loc[(data['date'] >= self.trade_days.min()) & (data['date'] <= self.trade_days.max())]
            if len(_df) != len(self.trade_days): continue
            _df.reset_index(drop=True, inplace=True)
            _dict[code] = _df
        return _dict
    
    def load_bt_data(self):
        _dict = {}
        stocks = glob.glob(r'backtrader\\*.csv')
        if self.single_stock is not None:
            stocks = [f'backtrader\\{self.single_stock}.csv']
        for path in stocks:
            stock = path.split('\\')[1][:-4]
            if stock not in self.df.keys(): continue
            _df = pd.read_csv(path)
            _df = _df.loc[(_df['date'] >= self.trade_days.min()) & (_df['date'] <= self.trade_days.max())]
            _df.reset_index(drop=True, inplace=True)
            _dict[stock] = _df
        return _dict

    def get_next_date(self):
        date_index = self.trade_days.loc[self.trade_days == self.date].index[0]
        tomorrow_date = self.trade_days.iloc[date_index+1]
        if tomorrow_date == self.trade_days.iloc[-1]:
            return tomorrow_date, True
        return tomorrow_date, False

    def load_positions(self):
        _dict = {}
        for key in self.df:
            _dict[key] = {'price':0, 'size':0, 'cost':0}
        return _dict
    
    def get_observation(self):
        stock_data = self.get_stock_his_data()
        position_data = self.get_position_data()
        if self.single_stock is not None:
            return np.squeeze(stock_data), np.squeeze(position_data)
        return stock_data, position_data
    
    def get_stock_his_data(self):
        stock_his_data = []
        index = next(self.date_index)
        for key in self.df:
            _df = self.df[key].iloc[index-self.time_steps:index]
            _df.drop(['date', 'code'], axis=1, inplace=True)
            stock_his_data.append(_df.to_numpy())
        return np.array(stock_his_data, dtype=np.float32)
        
    def get_position_data(self):
        _arr = []
        for key in self.positions:
            _arr.append([self.positions[key]['price']/self.amount, self.positions[key]['cost']/self.amount, self.cash/self.amount, self.value/self.amount])
        return np.array(_arr, dtype=np.float32)

    def get_day_price(self, stock):
        df = self.bt_df[stock]
        return df.loc[df['date'] == self.date]['open'].values[0]

    def buy(self, stock, money=None):
        if len(self.holding_stocks) == self.holding_num: return
        if money > self.cash: return
        price = self.get_day_price(stock)
        size = int(np.floor((money/price)/self.size)) * self.size
        if size < self.size: return
        buy_money = price * size
        if self.positions[stock]['size'] == 0:
            self.positions[stock]['price'] = price
            self.positions[stock]['size'] = size
            self.positions[stock]['cost'] = buy_money
            self.holding_stocks.append(stock)
        else:
            self.positions[stock]['size'] = self.positions[stock]['size'] + size
            self.positions[stock]['cost'] = self.positions[stock]['cost'] + buy_money
            self.positions[stock]['price'] = self.positions[stock]['cost'] / self.positions[stock]['size']
        self.cash -= buy_money
            
    def sell(self, stock, money=None):
        if self.positions[stock]['size'] == 0: return
        price = self.get_day_price(stock)
        stock_value = self.positions[stock]['size'] * price
        if money > stock_value:
            size = self.positions[stock]['size']
        else:
            size = int(np.floor((money/price)/self.size)) * self.size
        if size < self.size: return
        sell_money = price * size
        self.positions[stock]['size'] = self.positions[stock]['size'] - size
        if self.positions[stock]['size'] == 0:
            self.positions[stock]['price'] = 0
            self.positions[stock]['cost'] = 0
            self.holding_stocks.remove(stock)
        else:
            self.positions[stock]['cost'] = self.positions[stock]['price'] * self.positions[stock]['size']
        self.cash += sell_money

    def buy_all(self, stock):
        if len(self.holding_stocks) == self.holding_num: return
        if self.positions[stock]['size'] > 0: return
        price = self.get_day_price(stock)
        if self.cash < (self.size * price): return
        buy_size_multiple = int(self.cash / (self.size * price))
        buy_size = buy_size_multiple * self.size
        buy_money = buy_size * price
        self.positions[stock]['price'] = price
        self.positions[stock]['size'] = buy_size
        self.positions[stock]['cost'] = buy_money
        self.holding_stocks.append(stock)
        self.cash -= buy_money

    def sell_all(self, stock):
        if self.positions[stock]['size'] == 0: return
        price = self.get_day_price(stock)
        sell_money = self.positions[stock]['size'] * price
        self.positions[stock]['price'] = 0
        self.positions[stock]['size'] = 0
        self.positions[stock]['cost'] = 0
        self.holding_stocks.remove(stock)
        self.cash += sell_money

    def update_value(self):
        value = 0
        for stock in self.holding_stocks:
            day_price = self.get_day_price(stock)
            size = self.positions[stock]['size']
            value += day_price * size
        value += self.cash
        return value


if __name__ == '__main__':
    start_date = '2015-09-30'
    end_date = '2021-10-01'
    single_stock = '000001.XSHE'
    
    # env = StockEnv(start_date, end_date, single_stock=single_stock)
    # for i_episode in range(20):
    #     observation = env.reset()
    #     while True:
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         print(env.date)
    #         if done:
    #             print('Episode finished after {} timesteps'.format(t+1))
    #             break