import os
import gym
import glob
import math
import numpy as np
import pandas as pd
from stockstats import StockDataFrame


class StockEnv(gym.Env):
    def __init__(self, start_date, end_date, initial_amount=100000, allin=True):
        self.size = 100
        self.start_date = start_date
        self.end_date = end_date
        self.initial_amount = initial_amount
        self.allin = allin
        self.trade_days = self._load_trade_days()
        self.trade_data = self._load_trade_data()
        self.stock_data = self._load_stock_data()
        self.stock_data = self._processing_feature(col=['date', 'open', 'close', 'high', 'low', 'factor', 'volume'])
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.stock_data),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=((len(self.stock_data)*6)+1,), dtype=np.float32)
    
    def reset(self):
        self.reward = 0
        self.date = self.start_date
        self.cash = self.initial_amount
        self.value = self.initial_amount
        self.date_index = iter(range(len(self.trade_days)))
        self.positions = self._load_positions()
        return self._get_observation()

    def step(self, action):
        next_day_value = self._update_value()
        self.reward = (next_day_value - self.value) * 1e-4
        self.value = next_day_value
        for stock, size in zip(self.positions.keys(), action * self.size):
            if size > 0:
                self._buy(stock, size)
            else:
                self._sell(stock, np.absolute(size))
        self.date, done = self._get_next_date()
        return self._get_observation(), self.reward , done, {}

    def render(self, mode='human'):
        print('no render function')

    def close(self):
        print('no close function')
        
    def _load_trade_days(self, file=r'datasets\\trade_days.csv'):
        df = pd.read_csv(file)
        df = df.loc[(df['date'] >= self.start_date) & (df['date'] < self.end_date)]['date']
        df.reset_index(drop=True, inplace=True)
        return df
    
    def _load_trade_data(self, dir=r'datasets\\000300\\*.csv'):
        df_dict = {}
        for path in glob.glob(dir):
            df = pd.read_csv(path)
            df = df.loc[(df['date'] >= self.trade_days.min()) & (df['date'] <= self.trade_days.max())]
            if len(df) != len(self.trade_days): continue
            df = df[['date', 'open', 'close']]
            df.reset_index(drop=True, inplace=True)
            df_dict[os.path.basename(path)[:-4]] = df
        
        return df_dict

    def _load_stock_data(self, dir=r'datasets\\000300\\'):
        df_dict = {}
        for key in self.trade_data:
            path = f'{dir}{key}.csv'
            _df = pd.read_csv(path)
            _df = _df.loc[(_df['date'] >= self.trade_days.min()) & (_df['date'] <= self.trade_days.max())]
            _df.reset_index(drop=True, inplace=True)
            df_dict[key] = _df
            
        return df_dict

    def _load_positions(self):
        _dict = {}
        for key in self.stock_data:
            _dict[key] = {'price':0, 'size':0, 'cost':0}
        return _dict
    
    def _processing_feature(self, col):
        df_dict = {}
        for key in self.stock_data:
            ### 计算特征 ###
            df = self.stock_data[key][col].copy()
            df['factor'] = df['factor'].apply(lambda x: 1 if x == 0 else x)
            df['open'] = df['open'] / df['factor']
            df['adj_close'] = df['close'] / df['factor']
            df['high'] = df['high'] / df['factor']
            df['low'] = df['low'] / df['factor']
            df = df[['date', 'open', 'adj_close', 'high', 'low', 'volume']]

            ### 计算指标 ###
            stock = StockDataFrame.retype(df.copy())
            stock['close'] = stock['adj_close']
            
            df['macd'] = stock['macd'].values
            df['rsi'] = stock['rsi_30'].values
            df['cci'] = stock['cci_30'].values
            df['dx'] = stock['dx_30'].values
        
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='pad', inplace=True)

            df_dict[key] = df
    
        return df_dict
        
    def _get_observation(self):
        observations = []
        day_index = next(self.date_index)
        for key in self.stock_data:
            day_stock = self.stock_data[key].iloc[day_index]
            observations.extend(day_stock[['adj_close', 'macd', 'rsi', 'cci', 'dx']])
            observations.append(day_stock['adj_close'] * self.positions[key]['size'])
        observations.append(self.cash)
        return np.array(observations, dtype=np.float32)
    
    def _update_value(self):
        value = 0
        for key in self.positions:
            day_price = self._get_day_price(key)
            size = self.positions[key]['size']
            value += (day_price * size)
        value += self.cash
        return value
    
    def _get_day_price(self, stock):
        df = self.trade_data[stock]
        return df.loc[df['date'] == self.date]['open'].values[0]
    
    def _buy(self, stock, size=None):
        price = self._get_day_price(stock)
        available_size = self.cash / price
        if available_size < size: return
        buy_money = size * price
        if self.positions[stock]['size'] == 0:
            self.positions[stock]['price'] = price
            self.positions[stock]['size'] = size
            self.positions[stock]['cost'] = buy_money
        else:
            self.positions[stock]['size'] += size
            self.positions[stock]['cost'] += buy_money
            self.positions[stock]['price'] = self.positions[stock]['cost'] / self.positions[stock]['size']

        self.cash -= buy_money
            
    def _sell(self, stock, size=None):
        if self.positions[stock]['size'] == 0: return
        price = self._get_day_price(stock)
        size = min(size, self.positions[stock]['size'])
        sell_money = price * size
        self.positions[stock]['size'] -= size
        if self.positions[stock]['size'] == 0:
            self.positions[stock]['price'] = 0
            self.positions[stock]['cost'] = 0
        else:
            self.positions[stock]['cost'] = self.positions[stock]['price'] * self.positions[stock]['size']
        self.cash += sell_money

    def _get_next_date(self):
        date_index = self.trade_days.loc[self.trade_days == self.date].index[0]
        tomorrow_date = self.trade_days.iloc[date_index+1]
        if tomorrow_date == self.trade_days.iloc[-1]:
            return tomorrow_date, True
        return tomorrow_date, False



if __name__ == '__main__':
    start_date = '2010-09-30'
    end_date = '2015-10-01'
    env = StockEnv(start_date, end_date)
    
    from stable_baselines3 import PPO
    
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=200000)
    # model.save("ppo_t_200000")
    
    model = PPO.load('ppo_t_200000')
    
    for _ in range(10):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            # action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            print(env.date, reward, env.value)
            if done:
                break
    