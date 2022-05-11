import math

import gym
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
from gym.utils import seeding

from env.BaseEnv import BaseTradeEnv

matplotlib.use("Agg")
import pickle

import matplotlib.pyplot as plt

# shares normalization factor


class StockEnvValidation(BaseTradeEnv):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        index_df,
        stock_dim,
        config,
        flag_days=None,
        time_window=0,
        day=0,
        turbulence_threshold=140,
        iteration="",
    ):
        BaseTradeEnv.__init__(self,  stock_dim=stock_dim, index_df=index_df, time_window=time_window, config=config)
        # money = 10 , scope = 1
        assert time_window < len(df.index.unique()), 'Time window should not be longer that given dataframe'
        self.day = day + time_window
        self.time_winow = time_window
        self.df = df
        self.index_df = index_df
        self.stock_dim = stock_dim
        self.config = config
        self.flag_days = flag_days
        # action_space normalization and shape is STOCK_DIM
        self.unique_trade_date = df.Date.unique()
        self.env_name = 'val'
        self.terminal = False
        self.turbulence_threshold = 140
        # initalize state

        self.data = self.df.loc[self.day, :]
        self.past_data = self.df.loc[self.day - self.time_window: self.day, ['adjcp', 'volume']]
        self.index = self.index_df.loc[self.day, :]

        self.state = self._get_observation(initial=True)

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [self.config.INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        # self.reset()
        self._seed()

        self.iteration = iteration

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index + self.stock_dim + 1] < 0:
            self.state[index + self.stock_dim + 1] = 0

        if self.turbulence < self.turbulence_threshold:
            # State = Assets + [1-STOCKDIM]
            # Check if you have stock on the state
            if self.state[1 + index] > 0:
                if self.state[index + self.stock_dim + 1] > 0:
                    # Update shorted amount left subtract current holdings
                    # Normal action
                    # update balance
                    amount = min(abs(action), self.state[index + self.stock_dim + 1])
                    self.state[0] += (
                        self.state[index + 1]
                        * amount
                        * (1 - self.config.TRANSACTION_FEE_PERCENT)
                    )
                    # Update your holding on given stock
                    self.state[index + self.stock_dim + 1] -= amount
                    self.cost += (
                        self.state[index + 1]
                        * amount
                        * self.config.TRANSACTION_FEE_PERCENT
                    )

                    self.trades += 1

        else:
            # if turbulence goes over threshold, just clear out all positions

            #Check your holdings
            short_action = action
            # Check if you have
            if self.state[index + self.stock_dim + 1] > 0:
                # Update shorted amount left
                short_action -= self.state[index + self.stock_dim + 1]
                # update balance
                self.state[0] += (
                    self.state[index + 1]
                    * self.state[index + self.stock_dim + 1]
                    * (1 - self.config.TRANSACTION_FEE_PERCENT)
                )
                self.state[index + self.stock_dim + 1] = 0
                self.cost += (
                    self.state[index + 1]
                    * self.state[index + self.stock_dim + 1]
                    * self.config.TRANSACTION_FEE_PERCENT
                )


                self.trades += 1

            elif self.state[index + self.stock_dim + 1] < 0:
                print('Holding negative')
                self.state[index + self.stock_dim + 1] = 0

                #self._close_short(index)
                pass

    def _buy_stock(self, index, action):
        if self.state[index + self.stock_dim + 1] < 0:
            self.state[index + self.stock_dim + 1] = 0
        # perform buy action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:

            if self.state[1 + index] > 0:
                available_amount = max(self.state[0] // self.state[index + 1], 0)
                # print('available_amount:{}'.format(available_amount))

                amount = min(available_amount, action)
                # update balance
                self.state[0] -= (
                        self.state[index + 1]
                        * amount
                        * (1 + self.config.TRANSACTION_FEE_PERCENT)
                )
                self.state[index + self.stock_dim + 1] += amount
                self.cost += (
                        self.state[index + 1] * amount * self.config.TRANSACTION_FEE_PERCENT
                )

                self.trades += 1

        else:
            # if turbulence goes over threshold, just stop buying
            pass

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            ending_asset = self._calculate_total_asset()
            return self.state, self.reward, self.terminal, {'total_asset': ending_asset}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * self.HMAX_NORMALIZE

            # actions = (actions.astype(int))

            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-self.HMAX_NORMALIZE] * self.stock_dim)
            # being total asset is initial ballance and sum of price * owned stocks

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            today = pd.to_datetime(self.unique_trade_date[self.day + 1], format="%Y-%m-%d")
            if self.flag_days is not None:
                if today in self.flag_days:
                    print('Closing positions')
                    self._close_all_positions()
                else:
                    for index in sell_index:
                        # print('take sell action'.format(actions[index]))
                        self._sell_stock(index, actions[index])

                    for index in buy_index:
                        # print('take buy action: {}'.format(actions[index]))
                        self._buy_stock(index, actions[index])
            else:
                for index in sell_index:
                    # print('take sell action'.format(actions[index]))
                    self._sell_stock(index, actions[index])

                for index in buy_index:
                    # print('take buy action: {}'.format(actions[index]))
                    self._buy_stock(index, actions[index])

            self.day += 1

            self.data = self.df.loc[self.day, :].dropna(subset=["ticker"])
            self.past_data = self.df.loc[self.day - self.time_window: self.day, ['adjcp', 'volume']]
            self.index = self.index_df.loc[self.day, :]
            self.state = self._get_observation(initial=False)

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            self.reward = self._calculate_reward(self.asset_memory[0], begin_total_asset, end_total_asset)


            self.turbulence = self.data["turbulence"].values[0]
            # print(self.turbulence)
            # load next state



            # Calculate holdings for shorting cost and fee

            #self.HMAX_NORMALIZE = self.config.HMAX_NORMALIZE if end_total_asset < self.config.INITIAL_ACCOUNT_BALANCE else self.config.HMAX_NORMALIZE * (end_total_asset // self.config.INITIAL_ACCOUNT_BALANCE)

            self.asset_memory.append(end_total_asset)
            # print("end_total_asset:{}".format(end_total_asset))

            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)

            self.reward = self.reward * self.config.REWARD_SCALING

        return self.state, self.reward, self.terminal, {'end_total_asset':end_total_asset}

    def reset(self):
        self.asset_memory = [self.config.INITIAL_ACCOUNT_BALANCE]
        self.day = 0 + self.time_window
        self.data = self.df.loc[self.day, :].dropna(subset=["ticker"])
        self.past_data = self.df.loc[self.day - self.time_window: self.day, ['adjcp', 'volume']]
        self.index = self.index_df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0

        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        # initiate state

        self.state = self._get_observation(initial=True)

        return self.state

    def render(self, mode="human", close=False):
        ending_asset = self._calculate_total_asset()
        return self.state, self.trades, ending_asset

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
