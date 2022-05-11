import matplotlib
import numpy as np
import pandas as pd
from gym.utils import seeding
from datetime import datetime

from env.BaseEnv import BaseTradeEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class StockEnvTrain(BaseTradeEnv):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self,  df, index_df, stock_dim, config, flag_days=None, time_window=0, day=0):
        BaseTradeEnv.__init__(self, stock_dim=stock_dim, index_df=index_df, time_window=time_window, config=config)
        # super(StockEnvTrain, self).__init__()
        # money = 10 , scope = 1
        assert time_window < len(df.index.unique()), 'Time window should not be longer that given dataframe'
        self.day = day + time_window
        self.time_window = time_window
        self.df = df
        self.index_df = index_df
        self.stock_dim = stock_dim
        self.config = config
        self.flag_day_trackker = 0
        self.flag_days = flag_days
        self.unique_trade_date = df.Date.unique()

        self.terminal = False
        # initalize state
        self.env_name = 'train'

        self.data = self.df.loc[self.day, :]
        self.past_data = self.df.loc[self.day - self.time_window: self.day, ['adjcp','volume']]

        self.index = self.index_df.loc[self.day, :]

        self.state = self._get_observation(initial=True)



        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [self.config.INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        # self.reset()
        self._seed()

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:

            self.grade = 1
            #TODO: give as end information
            #plt.plot(self.asset_memory, "r")
            #plt.savefig("results/account_value_train.png")
            #plt.close()

            # print("end_total_asset:{}".format(end_total_asset))
            #df_total_value = pd.DataFrame(self.asset_memory)
            #df_total_value.to_csv("results/account_value_train.csv")

            #df_total_value.columns = ["account_value"]
            #df_total_value["daily_return"] = df_total_value.pct_change(1)

            return self.state, self.reward, self.terminal, {}

        else:
            #print(actions[:4])
            actions = actions * self.HMAX_NORMALIZE
            if np.isnan(actions).any():
                print(f'actions are nan, in {self.day} and actions are {actions}, state is : {self.state}')

            # actions = (actions.astype(int))

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions[: self.stock_dim])

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]

            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            today = pd.to_datetime(self.unique_trade_date[self.day + 1], format="%Y-%m-%d")
            if self.flag_days is not None:
                if today in self.flag_days:
                    self._close_all_positions()
                else:
                    for index in sell_index:
                        # print('take sell action'.format(actions[index]))
                        self._sell_stock(index, actions[index])

                    for index in buy_index:
                        # print('take buy action: {}'.format(actions[index]))
                        self._calculate_avg_bought_price(index, actions[index])
                        self._buy_stock(index, actions[index])
            else:
                for index in sell_index:
                    # print('take sell action'.format(actions[index]))
                    self._sell_stock(index, actions[index])

                for index in buy_index:
                    # print('take buy action: {}'.format(actions[index]))
                    self._calculate_avg_bought_price(index, actions[index])
                    self._buy_stock(index, actions[index])

            self.day += 1
            if self.day > len(self.df.index.unique()) * 0.5 and self.grade != 1:

                self.grade = 1
            self.data = self.df.loc[self.day, :].dropna(subset=["ticker"])
            self.past_data = self.df.loc[self.day - self.time_window: self.day, ['adjcp', 'volume']]
            self.index = self.index_df.loc[self.day, :]

            # load next state

            self.state = self._get_observation(initial=False)

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            self.end_total_asset = end_total_asset
            # TODO: change reward initial balance when flag day comes up with stocks
            self.reward = self._calculate_reward(self.asset_memory[0], begin_total_asset, end_total_asset)


            self.asset_memory.append(end_total_asset)

            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            # CHECK IF SELL INDEX IS IN STOCKS

            self.reward = self.reward * self.config.REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):

        self.asset_memory = [self.config.INITIAL_ACCOUNT_BALANCE]
        self.day = 0 + self.time_window
        self.data = self.df.loc[self.day, :]
        self.past_data = self.df.loc[self.day - self.time_window: self.day, ['adjcp', 'volume']]
        self.index = self.index_df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.flag_day_trackker = 0
        self.terminal = False
        self.rewards_memory = []
        # initiate state

        self.state = self._get_observation(initial=True)
        # iteration += 1
        return self.state

    def render(self, mode="human"):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
