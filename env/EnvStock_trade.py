import os

import matplotlib
import numpy as np
import pandas as pd
from gym.utils import seeding
import streamlit as st
from datetime import datetime
#from loguru import logger
#import plotly.express as px
from env.BaseEnv import BaseTradeEnv

matplotlib.use("Agg")


class StockEnvTrade(BaseTradeEnv):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        index_df,
        stock_dim,
        unique_trade_date,
        config,
        flag_days=None,
        time_window=0,
        day=0,
        turbulence_threshold=140,
        initial=True,
        previous_state=[],
        model_name="",
        iteration="",
            debug=False
    ):
        BaseTradeEnv.__init__(self, stock_dim=stock_dim, index_df=index_df, time_window=time_window )
        # money = 10 , scope = 1
        assert time_window < len(df.index.unique()), 'Time window should not be longer that given dataframe'
        self.date = unique_trade_date[day]
        self.unique_trade_date = df.Date.unique()
        self.day = day + time_window
        self.time_winow = time_window
        self.df = df
        self.index_df = index_df
        self.turbulanced_days = 0
        self.config = config
        self.flag_days = flag_days
        self.env_name = 'trade'
        self.debug = debug



        self.initial = initial
        self.previous_state = previous_state
        self.stock_dim = stock_dim
        # action_space normalization and shape is STOCK_DIM

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold

        # initalize state
        self.tickers = list(self.df.ticker.unique())

        self.data = self.df.loc[self.day, :]
        self.past_data = self.df.loc[self.day - self.time_window: self.day, ['adjcp', 'volume']]

        self.index = self.index_df.loc[self.day, :]

        self.state = self._get_observation(True)
        if self.debug:
            print('====DEBUG TRADING ENV====')
            print(self.state)
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0

        # memorize all the total balance change
        self.asset_memory = [self.config.INITIAL_ACCOUNT_BALANCE]
        date = self.df.iloc[self.day]["Date"].strftime("%Y-%m-%d")

        self.date_memory = [date]
        self.rewards_memory = []
        self.trade_memory = []

        # self.reset()
        self._seed()
        self.model_name = model_name
        self.iteration = iteration

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            # State = Assets + [1-STOCKDIM]
            # Check if you have stock on the state
            if self.state[1 + index] > 0 and self.state[index + self.stock_dim + 1] > 0:
                # Normal long action
                # update balance
                amount = min(abs(action), self.state[index + self.stock_dim + 1])
                #update cash
                self.state[0] += (
                    self.state[index + 1]
                    * amount
                    * (1 - self.config.TRANSACTION_FEE_PERCENT)
                )
                # Update your holding on given stock
                self.state[index + self.stock_dim + 1] -= amount
                # calculate
                self.cost += (
                    self.state[index + 1]
                    * amount
                    * self.config.TRANSACTION_FEE_PERCENT
                )
                trade_vals = {
                    "Date": self.date,
                    "ticker": self.tickers[index],
                    "amount":  -1*amount,
                    "price": self.state[index + 1],
                }
                self.trade_memory.append(trade_vals)
                if amount != 0:
                    self.trades += 1

            else:
                pass

        else:
            # if turbulence goes over threshold, just clear out all positions
            self.turbulanced_days += 1
            if self.state[index + self.stock_dim + 1] > 0:
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
                trade_vals = {
                    "Date": self.date,
                    "ticker": self.tickers[index],
                    "amount":  self.state[index + self.stock_dim + 1],
                    "price": self.state[index + 1],
                }
                self.trade_memory.append(trade_vals)

                self.trades += 1

            elif self.state[index + self.stock_dim + 1] < 0:
                print(self.turbulence)
                print('Holding negative')
                print(self.state[: 1 + self.stock_dim *2])
                #self._close_short(index)
                pass

        holdings = self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
        if any(i < 0 for i in holdings):
            print('After sell')
            print('Sold index is', index)
            print(holdings)
            print('Holdings contains negative holdings')


    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            if self.state[1 + index] > 0:
                available_amount = max(self.state[0] // self.state[index + 1],0)
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
                trade_vals = {
                    "Date": self.date,
                    "ticker": self.tickers[index],
                    "amount": amount,
                    "price": self.state[index + 1],
                }

                self.trade_memory.append(trade_vals)
                if amount != 0:
                    self.trades += 1

        else:
            # if turbulence goes over threshold, just stop buying
            self.turbulanced_days += 1

    def _log_portfolio(self):
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        print("previous_total_asset:{}".format(self.asset_memory[0]))

        print("end_total_asset:{}".format(end_total_asset))
        print(
            "total_asset_change:{}".format(
                self.state[0]
                + sum(
                    np.array(self.state[1: (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )
        )
        print(f"Total cash is: {self.state[0]}$ and total holdings in stocks are {end_total_asset - self.state[0]}$")
        print("Buy & Hold strategy with previous total asset: ", self._calculate_buy_and_hold(self.asset_memory[0]))
        print("Total Cost: ", self.cost)
        print("Sum of rewards ", sum(self.rewards_memory))
        print("Total trades: ", self.trades)
        print("Total days in turbulance: ", self.turbulanced_days)


    def step(self, actions):

        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:

            df_total_value = pd.DataFrame(self.asset_memory)
            #date = self.df.iloc[self.day]["Date"].strftime("%Y-%m-%d")
            #st.write('RL Model Performance')
            d = {"account_value": self.asset_memory}
            path = f"results/account_value_trade_main_{self.model_name}.csv"
            if os.path.exists(path):
                print("Saving to ", path)

                df_total_value = pd.DataFrame(d)
                df = pd.read_csv(path)
                df = df.append(df_total_value, ignore_index=True)
                df.to_csv(path, index=False)
                st.line_chart(df)

            else:

                df_total_value = pd.DataFrame(d)
                df_total_value.to_csv(path, index=False)

            """df_total_value_csv = pd.DataFrame(d, columns=["account_value", "date"])
            df_total_value_csv.to_csv(
                "results/account_value_trade_{}_{}.csv".format(
                    self.model_name, self.iteration
                )
            )"""

            self._log_portfolio()

            df_total_value.columns = ["account_value"]
            df_total_value["daily_return"] = df_total_value.pct_change(1)
            sharpe = (
                (4 ** 0.5)
                * df_total_value["daily_return"].mean()
                / (df_total_value["daily_return"].std() + 0.0001)
            )
            print("Sharpe: ", sharpe)

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv(
                "results/account_rewards_trade_{}.csv".format(
                    self.model_name, self.iteration
                )
            )

            df_trades = pd.DataFrame(self.trade_memory)
            #df_rewards = pd.DataFrame({'Date':self.date_memory, 'Rewards':self.rewards_memory})
            reward_path = f'results/rewards_memory_{self.model_name}.csv'
            trade_path = f'results/trade_memory_{self.model_name}.csv'
            if os.path.exists(trade_path):
                trade_mem = pd.read_csv(trade_path)
                trade_mem = trade_mem.append(df_trades, ignore_index=True)
                trade_mem.to_csv(trade_path, index=False)
            else:
                df_trades.to_csv(trade_path, index=False)

            """if os.path.exists(reward_path):
                reward_mem = pd.read_csv(reward_path)
                reward_mem = reward_mem.append(df_rewards, ignore_index=True)
                reward_mem.to_csv(reward_path, index=False)
            else:
                df_rewards.to_csv(reward_path, index=False)"""



            return self.state, self.reward, self.terminal, {}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * self.HMAX_NORMALIZE
            if self.debug:
                print('====DEBUG====')
                print('Printing actions from the agent ',actions)
            # actions = (actions.astype(int))

            if self.turbulence >= self.turbulence_threshold:

                actions = np.array([-self.HMAX_NORMALIZE] * self.stock_dim)


            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]


            # TODO: If it is flag day dont buy or sell at that day
            self.date = pd.to_datetime(self.unique_trade_date[self.day + 1], format="%Y-%m-%d")
            if self.flag_days is not None:

                #TODO: ADD day increment here
                if pd.to_datetime(self.date,  format="%Y-%m-%d") in self.flag_days:
                    print("=" * 30)
                    print(f"Previously traded stocks are {self.df.loc[self.day, 'ticker'].unique()}")
                    print(f"====Closing position changing stocks at day {self.date}====")
                    print(f"New stocks are {self.df.loc[self.day + 1, 'ticker'].unique()}")
                    self._close_all_positions()
                    print(f"Current dollar is {self.state[0]}")
                    print("=" * 30)
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

            self.data = self.df.loc[self.day, :]
            self.past_data = self.df.loc[self.day - self.time_window: self.day, ['adjcp', 'volume']]
            self.index = self.index_df.loc[self.day, :]

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            holdings = self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
            if self.debug:
                print('===DEBUG TRADING ENV===')
                if any(i < 0 for i  in holdings):
                    print('Holdings contains negative holdings')
                    print('Actions before negative holdings were', actions)
                    print('Holdings are ', holdings)
                    print('For given period the prices were', self.state[1: (self.stock_dim + 1)])
            if any(i < 0 for i in holdings):
                print(self.day)
                print('Holdings contains negative holdings')
            self.reward = self._calculate_reward(self.asset_memory[0], begin_total_asset, end_total_asset)


            # print(self.data['turbulence'])

            self.turbulence = self.data["turbulence"].values[0]

            # load next state
            # print("stock_shares:{}".format(self.state[29:]))

            self.state = self._get_observation(initial=False)


            self.asset_memory.append(end_total_asset)

            self.date_memory.append(self.date)

            # print("end_total_asset:{}".format(end_total_asset))


            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)

            self.reward = self.reward * self.config.REWARD_SCALING
            #For tensor callback
            #self.total_asset = end_total_asset

        return self.state, self.reward, self.terminal, {}

    def reset(self):

        if self.initial:
            self.asset_memory = [self.config.INITIAL_ACCOUNT_BALANCE]
            self.day = 0 + self.time_window
            self.date = pd.to_datetime(self.unique_trade_date[self.day], format="%Y-%m-%d")
            self.date_memory = [self.date]
            self.data = self.df.loc[self.day, :]
            self.past_data = self.df.loc[self.day - self.time_window: self.day, ['adjcp', 'volume']]
            self.index = self.index_df.loc[self.day, :]
            self.turbulanced_days = 0
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=self.iteration
            self.rewards_memory = []
            # initiate state

            self.state = self._get_observation(initial=True)
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.previous_state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]
            # self.asset_memory = [self.previous_state[0]]
            self.day = 0  + self.time_window
            self.date = self.df.iloc[self.day]["Date"].strftime("%Y-%m-%d")

            self.date_memory = [self.date]
            self.data = self.df.loc[self.day, :]
            self.past_data = self.df.loc[self.day - self.time_window: self.day, ['adjcp', 'volume']]
            self.index = self.index_df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=iteration
            self.rewards_memory = []


            indicators = []
            for ind in self.config.INDICATORS:
                indicators.extend(self.data[ind].values.tolist())
            if self.time_window == 0:
                self.state = (
                    [self.previous_state[0]]
                    + self.data.adjcp.values.tolist()
                    + self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]
                    + indicators
                    #+ self.index.index_close.values.tolist()
                    #+ self.index.index_macd.values.tolist()
                    #+ self.index.index_rsi.values.tolist()
                    + [self.data.month.values.tolist()[0]]
                    + [self.data.day.values.tolist()[0]]
                )
            else:
                self.state = (
                        [self.previous_state[0]]
                        + self.data.adjcp.values.tolist()
                        + self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]
                        + indicators
                        # + self.index.index_close.values.tolist()
                        # + self.index.index_macd.values.tolist()
                        # + self.index.index_rsi.values.tolist()
                        + self.past_data.adjcp.values.tolist()
                        + self.past_data.volume.values.tolist()
                        + [self.data.month.values.tolist()[0]]
                        + [self.data.day.values.tolist()[0]]
                )

            if self.debug:
                print('====DEBUG TRADE ENV====')
                print('After reset:', self.state)

        return self.state

    def render(self, mode="human", close=False):
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        return self.state, {'end_total_asset':end_total_asset}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
