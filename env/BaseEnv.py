import math

import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from utils.indicators import indicator_list
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# TODO: Tensorboard integration


class EnvConfig:
    INITIAL_ACCOUNT_BALANCE = 1000000
    HMAX_NORMALIZE = 100
    REWARD_SCALING = 1e-4 #5e-4
    TRANSACTION_FEE_PERCENT = 0.001
    SHORT_FEE = 0.00005
    INDICATORS = indicator_list
    OBSERVATIONS = len(indicator_list) + 2
    INDEX_OBSERVATIONS = 3
    REWARD_INTERVAL = 1
    seed = 42


class BaseTradeEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, stock_dim, index_df, time_window, config=EnvConfig):
        self.config = config
        self.time_window= time_window
        self.stock_dim = stock_dim
        self.index_df = index_df
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
        #self.index_dim = len(self.index_df.ticker.unique())

        obs_shape = 3 + self.stock_dim * self.config.OBSERVATIONS + (self.time_window * 2 * (self.stock_dim + 2)) #+ (self.index_dim * self.config.INDEX_OBSERVATIONS)

        self.HMAX_NORMALIZE = self.config.HMAX_NORMALIZE
        self.observation_space = spaces.Box(low=-100, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.trade_memory = [0] * stock_dim
        self.avg_bought_prices = [0] * stock_dim
        self.stop_loss_trigerred = 0
        # It will be updated when the asset is bought ht the price will be stop loss %n of the bought price
        self.stop_loss_prices = [0] * self.stock_dim
        self.shorted = 0

    def _get_observation(self, initial: bool):
        indicators = []

        for ind in self.config.INDICATORS:
            #TODO: get rid of this
            inds = self.data[ind].values.tolist()
            assert len(inds) == self.stock_dim, 'stock dimension does not match indicator dimension'
            indicators.extend(self.data[ind].values.tolist())
        if self.time_window == 0:
            if initial:
                state = (
                    [self.config.INITIAL_ACCOUNT_BALANCE]
                    + self.data.adjcp.values.tolist()
                    + [0] * self.stock_dim
                    + indicators
                    # + self.index.index_close.values.tolist()
                    # + self.index.index_macd.values.tolist()
                    # + self.index.index_rsi.values.tolist()
                    + [self.data.month.values.tolist()[0]]
                    + [self.data.day.values.tolist()[0]]
                )
            else:
                state = (
                    [self.state[0]]
                    + self.data.adjcp.values.tolist()
                    + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                    + indicators
                    # + self.index.index_close.values.tolist()
                    # + self.index.index_macd.values.tolist()
                    # + self.index.index_rsi.values.tolist()
                    + [self.data.month.values.tolist()[0]]
                    + [self.data.day.values.tolist()[0]]
                )
        else:
            if initial:
                state = (
                    [self.config.INITIAL_ACCOUNT_BALANCE]
                    + self.data.adjcp.values.tolist()
                    + [0] * self.stock_dim
                    + indicators
                    # + self.index.index_close.values.tolist()
                    # + self.index.index_macd.values.tolist()
                    # + self.index.index_rsi.values.tolist()
                    + self.past_data.adjcp.values.tolist()
                    + self.past_data.volume.values.tolist()
                    + [self.data.month.values.tolist()[0]]
                    + [self.data.day.values.tolist()[0]]
                )
            else:
                state = (
                    [self.state[0]]
                    + self.data.adjcp.values.tolist()
                    + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                    + indicators
                    # + self.index.index_close.values.tolist()
                    # + self.index.index_macd.values.tolist()
                    # + self.index.index_rsi.values.tolist()
                    + self.past_data.adjcp.values.tolist()
                    + self.past_data.volume.values.tolist()
                    + [self.data.month.values.tolist()[0]]
                    + [self.data.day.values.tolist()[0]]
                )

        return state

    def _sell_stock(self, index, action):
        """
        Sell and short shares
        """
        if self.state[index + self.stock_dim + 1] > 0:
            # Update shorted amount left from long holdings
            # Normal long action
            # update balance
            self.state[0] += (
                self.state[index + 1]
                * min(abs(action), self.state[index + self.stock_dim + 1])
                * (1 - self.config.TRANSACTION_FEE_PERCENT)
            )
            # Update the stock holdings
            self.state[index + self.stock_dim + 1] -= min(
                abs(action), self.state[index + self.stock_dim + 1]
            )

            #TODO: Selling below avg should be punished


            if self.state[index + self.stock_dim + 1] < 0:
                print(str(self.state[index + self.stock_dim + 1]), 'went to minus')
                print(str(self.state[index+ 1]), 'is the price')
                print('action was: ', action, 'executed amount was: ', str(min(abs(action), self.state[index + self.stock_dim + 1])))
            self.cost += (
                self.state[index + 1]
                * min(abs(action), self.state[index + self.stock_dim + 1])
                * self.config.TRANSACTION_FEE_PERCENT
            )
            self.trades += 1
        else:
            pass

    def _calculate_avg_bought_price(self, index, action):
        # action is new amount for buying
        holding = self.state[1 + self.stock_dim +index]
        cur_price = self.state[1 + index]
        prev_avg_bought_price = self.avg_bought_prices[index]
        # for buying
        self.avg_bought_prices[index] = (prev_avg_bought_price * holding) + (cur_price * action) / (holding + action)


    def _buy_stock(self, index, action):
        """
        Buy shares
        """
        # If it is tradable
        if self.state[1 + index] > 0:
            # perform buy action based on the sign of the action
            available_amount = max(self.state[0] // self.state[index + 1], 0)
            # print('available_amount:{}'.format(available_amount))

            amount = min(available_amount, action)
            # print('available_amount:{}'.format(available_amount))
            # update balance
            self.state[0] -= (
                self.state[index + 1]
                * amount
                * (1 + self.config.TRANSACTION_FEE_PERCENT)
            )

            self.state[index + self.stock_dim + 1] += amount

            #self._calculate_avg_bought_price(index, amount)

            self.cost += (
                self.state[index + 1]
                * amount
                * self.config.TRANSACTION_FEE_PERCENT
            )
            self.trades += 1

    def _close_short(self, index):
        # Calculate PnL
        # Get from shorted memory
        holding = self.state[index + self.stock_dim * 2 + 1]
        shorted_amount = self.shorted_prices[index] * holding
        current_amount = self.state[1 + index] * holding
        # 2 Close short position
        self.state[index + self.stock_dim * 2 + 1] = 0

        # if the price as shorting period is bigger the pnl will be positive
        pnl = shorted_amount - current_amount
        # Update balance
        self.state[0] += pnl * (1 + self.config.TRANSACTION_FEE_PERCENT)

        # Close shorting price
        self.shorted_prices[index] = 0
        # Update cost and trade
        self.cost += pnl * self.config.TRANSACTION_FEE_PERCENT
        self.trades += 1

    def _short_stock(self, index, action):
        # Action is positive
        # Allow shorting if no long position is held on that stock
        # 1 add the shorted price to the list of shorted_stocks _price list
        self.shorted_prices[index] = self.state[index + 1]
        # 2 update balance
        available_amount = self.state[0] // self.state[index + 1]

        self.state[0] -= (
            self.state[index + 1]
            * min(available_amount, action)
            * (1 + self.config.SHORT_FEE)
        )

        # Update holdings make it negative
        self.state[index + self.stock_dim * 2 + 1] += min(available_amount, action)

        self.cost += (
            self.state[index + 1]
            * min(available_amount, action)
            * self.config.SHORT_FEE
        )
        self.trades += 1
        self.shorted += 1

    def _calculate_unrealized_pnl(self):
        unrealized_pnl = sum(
            [
                a * b
                for a, b in zip(
                (np.array(self.state[1 + self.stock_dim : self.stock_dim * 2 + 1]))-(np.array(self.shorted_prices)),
                self.state[1 + self.stock_dim * 2: self.stock_dim * 3 + 1],
            )
            ]
        )
        return unrealized_pnl

    def _close_all_positions(self):
        for index in range(self.stock_dim):
            if self.state[index + self.stock_dim + 1] > 0:
                # Sell all available stock
                self._sell_stock(index, self.state[index + self.stock_dim + 1]) # index, action
            elif self.state[index + self.stock_dim + 1] < 0: #TODO:  if short is allowed change dimension of this code

                self._close_short(index)

    def _calculate_reward(self, initial_balance, begin_total_asset, end_total_asset, grade = 0):
        """
        Our reward function defines
        todays total asset - previous total asset) + (sharpe_ratio - 1 ) * scaler
        # TODO add auxilarry tasks
        # TODO: add cirriculum for agent to learn
        # TODO: track highest buying point punish if sold lower
        # 1. learn not to sell on loss
        # 2. Learn to generate alpha on stock
        # 3. Diversify
        # Cirriculum learning for the reward function
        :param end_total_asset:
        :return:
        """

        """df_total_value = pd.DataFrame(self.asset_memory)
        df_total_value.columns = ["account_value"]
        df_total_value["daily_return"] = df_total_value.pct_change(1)
        sharpe = (
            (252 ** 0.5)
            * df_total_value["daily_return"].mean()
            / (df_total_value["daily_return"].std() + 0.001)
        )

        if sharpe <= 0 or math.isnan(sharpe):
            reward = end_total_asset - begin_total_asset
        else:"""
        # Cirriculum
        # 1. learn to sell for profit
        # 2. learn to generate alpha
        # 3. optional learn to maximize sharpe

        if grade == 0:
            # Benchmark is buy and hold strategy
            if self.day % self.config.REWARD_INTERVAL == 0:
                #benchmark = self._alpha(initial_balance)  # returns buy and hold
                reward = end_total_asset - begin_total_asset  # + ((sharpe) * 5)
                # Add alpha to reward or substract
                #reward += (end_total_asset - benchmark) * 2
                return reward
            else:
                return 0
        elif grade == 1:
            if self.day % self.config.REWARD_INTERVAL == 0:
                benchmark = self._alpha(initial_balance)  # returns buy and hold
                reward = end_total_asset - begin_total_asset  # + ((sharpe) * 5)
                # Add alpha to reward or substract
                reward += (end_total_asset - benchmark) * 2
                return reward
            else:
                return 0

        elif grade == 2:
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            df_total_value["daily_return"] = df_total_value.pct_change(1)
            sharpe = (
                    (252 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / (df_total_value["daily_return"].std() + 0.001)
            )
            benchmark = self._alpha(initial_balance)  # returns buy and hold
            reward = end_total_asset - begin_total_asset  + ((sharpe + 1) ** 5)
            # Add alpha to reward or substract
            reward += (end_total_asset - benchmark) * 2
            return reward



    def _alpha_on_stock(self):
        raise NotImplementedError

    def _alpha(self, initial_balance) -> int:
        "buy and hold from the beggining of the window"
        # Compared to buy & hold
        account_balance = float(initial_balance) / self.stock_dim
        balances = np.array([account_balance] * self.stock_dim)
        first_prices = np.array(self.df.loc[0, 'adjcp'].replace(0,1).values.tolist())
        # TODO: check if self.day is called on the right day in step process
        dates_prices = np.array(self.df.loc[self.day,'adjcp'].replace(0,1).values.tolist()) # todays price
        pct_changes = (dates_prices - first_prices) / first_prices
        portfolio_value = sum(balances + (balances * pct_changes))

        return portfolio_value

    def calculate_alpha_on_stock(self, index):

        pass

    def _calculate_var(self):
        pass

    def _calculate_buy_and_hold(self, account_balance):
        ticker_len = len(self.df.ticker.unique())
        coin_balance = float(account_balance) / ticker_len

        portfolio_value = 0
        for t in self.df.ticker.unique():
            df = self.df[self.df.ticker == t]
            first_val = df.adjcp.iat[0]

            i = 0
            while first_val == 0:
                first_val = df.adjcp.iat[i]
                i += 1
                if i >= (len(df.index.unique()) - 2):
                    portfolio_value += coin_balance
                    print('Stock was not active', t)
                    break
            if first_val != 0:
                last_val = df.adjcp.iat[-1]
                pct_change = (last_val - first_val) / first_val
                portfolio_value += coin_balance + (coin_balance * pct_change)
                coin_balance = float(account_balance) / ticker_len

        return portfolio_value

    def stop_loss(self):
        """
        Stop loss with expiriy date
        :return:
        """

        for index in range(self.stock_dim):
            todays_low = self.df.loc[self.day, 'low'].values.tolist()[index]
            if todays_low < self.stop_loss_prices[index]:
                self.stop_loss_trigerred += 1
                return self._clear_position(index)

    def step(self, actions):
        pass

    def reset(self):
        pass

    def render(self, mode="human", close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
