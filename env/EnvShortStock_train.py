import matplotlib
import numpy as np
import pandas as pd
from gym.utils import seeding

from env.BaseEnv import BaseTradeEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class StockShortEnvTrain(BaseTradeEnv):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, index_df, stock_dim, config, day=0):
        BaseTradeEnv.__init__(self, stock_dim=stock_dim, index_df=index_df)
        # super(StockEnvTrain, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.df = df
        self.index_df = index_df
        self.stock_dim = stock_dim
        self.config = config

        self.terminal = False
        # initalize state

        self.data = self.df.loc[self.day, :]

        self.index = self.index_df.loc[self.day, :]
        self.shorted_prices = [0] * self.stock_dim

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
            plt.plot(self.asset_memory, "r")
            plt.savefig("results/account_value_train.png")
            plt.close()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            # print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv("results/account_value_train.csv")

            df_total_value.columns = ["account_value"]
            df_total_value["daily_return"] = df_total_value.pct_change(1)

            # print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * self.config.HMAX_NORMALIZE

            # actions = (actions.astype(int))

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            ) + + self._calculate_unrealized_pnl()
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions[: self.stock_dim])

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]



            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day, :].dropna(subset=["ticker"])
            self.index = self.index_df.loc[self.day, :]
            # load next state

            self.state = self._get_observation(initial=False)

            # Calculate holdings for shorting cost and fee
            total_short_holdings = sum(
                [
                    a * b
                    for a, b in zip(
                    self.shorted_prices,
                    self.state[1 + self.stock_dim * 2: self.stock_dim * 3 + 1],
                )
                ]
            )

            self.cost += total_short_holdings * self.config.SHORT_FEE

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            ) + + self._calculate_unrealized_pnl()

            self.asset_memory.append(end_total_asset)

            # self.reward = self._calculate_reward(begin_total_asset, end_total_asset)
            self.reward = end_total_asset - begin_total_asset
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            # CHECK IF SELL INDEX IS IN STOCKS
            if len(self.rewards_memory) > 10:
                # check if last 10 rewards are the same
                if set(self.rewards_memory[-10:]) == 1:
                    self.reward = -100
            else:
                self.reward = self.reward * self.config.REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):

        self.asset_memory = [self.config.INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :].dropna(subset=["ticker"])
        self.index = self.index_df.loc[self.day, :]
        self.shorted_prices = [0] * self.stock_dim
        self.cost = 0
        self.trades = 0
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
