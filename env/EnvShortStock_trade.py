import os

import matplotlib
import numpy as np
import pandas as pd
from gym.utils import seeding

from env.BaseEnv import BaseTradeEnv

matplotlib.use("Agg")


class StockShortEnvTrade(BaseTradeEnv):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        index_df,
        stock_dim,
        unique_trade_date,
        config,
        day=0,
        turbulence_threshold=140,
        initial=True,
        previous_state=[],
        model_name="",
        iteration="",
    ):
        BaseTradeEnv.__init__(self, stock_dim=stock_dim, index_df=index_df)
        # money = 10 , scope = 1
        self.date = unique_trade_date[day]
        self.unique_trade_date = unique_trade_date
        self.day = day
        self.df = df
        self.index_df = index_df
        self.turbulanced_days = 0
        self.config = config

        self.initial = initial
        self.previous_state = previous_state
        self.stock_dim = stock_dim
        # action_space normalization and shape is STOCK_DIM

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold

        # initalize state
        self.tickers = list(self.df.ticker.unique())
        self.data = self.df.loc[self.day, :]

        self.index = self.index_df.loc[self.day, :]

        self.shorted_prices = [0] * self.stock_dim

        self.state = self._get_observation(True)
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
            if self.state[1 + index] > 0:
                # Shorted amount
                short_action = abs(action) - self.state[index + self.stock_dim + 1]
                if self.state[index + self.stock_dim + 1] > 0:
                    # Normal long action
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
                    trade_vals = {
                        "Date": self.date,
                        "ticker": self.tickers[index],
                        "amount": -1 * amount,
                        "price": self.state[index + 1],
                    }
                    self.trade_memory.append(trade_vals)
                    self.trades += 1

                if short_action > 0:
                    self._short_stock(index, short_action)
            else:
                pass

        else:
            # if turbulence goes over threshold, just clear out all positions
            self.turbulanced_days += 1
            short_action = abs(action)
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
                trade_vals = {
                    "Date": self.date,
                    "ticker": self.tickers[index],
                    "amount": -1 * self.state[index + self.stock_dim + 1],
                    "price": self.state[index + 1],
                }
                self.trade_memory.append(trade_vals)
                self.trades += 1

            elif self.state[index + self.stock_dim*2 + 1] < 0:
                self._close_short(index)

    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            buy_action = action
            if self.state[index + self.stock_dim + 1] < 0:
                # Whats left since holdings is minus
                buy_action += self.state[index + self.stock_dim + 1]
                self._close_short(index)

            if self.state[1 + index] > 0 and buy_action > 0:

                available_amount = self.state[0] // self.state[index + 1]
                # print('available_amount:{}'.format(available_amount))

                amount = min(available_amount, buy_action)
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
                self.trades += 1

        else:

            # if turbulence goes over threshold, just stop buying
            self.turbulanced_days += 1

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:

            # plt.plot(self.asset_memory, 'r')
            # plt.savefig('results/account_value_trade_{}_{}_backtest.png'.format(self.model_name, self.iteration))
            # plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            date = self.df.iloc[self.day]["Date"].strftime("%Y-%m-%d")
            d = {"account_value": self.asset_memory, "date": self.date_memory}
            path = f"results/account_value_trade_main_{self.model_name}.csv"
            if os.path.exists(path):
                df_total_value_csv = pd.read_csv(path, index_col=0)
                df_total_value_csv.append(d, ignore_index=True)
                df_total_value_csv.to_csv(path)
            else:
                df_total_value_csv = pd.DataFrame(d, columns=["account_value", "date"])
                df_total_value_csv.to_csv(path)

            df_total_value_csv = pd.DataFrame(d, columns=["account_value", "date"])
            df_total_value_csv.to_csv(
                "results/account_value_trade_{}_{}.csv".format(
                    self.model_name, self.iteration
                )
            )

            df_trades = pd.DataFrame(self.trade_memory)
            df_trades.to_csv(
                "results/trade_history_{}_{}_backtest.csv".format(
                    self.model_name, self.iteration
                )
            )

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            print("previous_total_asset:{}".format(self.asset_memory[0]))

            print("end_total_asset:{}".format(end_total_asset))
            print(
                "total_reward:{}".format(
                    self.state[0]
                    + sum(
                        np.array(self.state[1 : (self.stock_dim + 1)])
                        * np.array(
                            self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                        )
                    )
                    - self.asset_memory[0]
                )
            )
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)
            print("total short orders: ", self.shorted)
            print("total days in turbulance: ", self.turbulanced_days)

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
                "results/account_rewards_trade_{}_{}.csv".format(
                    self.model_name, self.iteration
                )
            )

            return self.state, self.reward, self.terminal, {}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * self.config.HMAX_NORMALIZE

            # actions = (actions.astype(int))

            if self.turbulence >= self.turbulence_threshold:

                actions = np.array([-self.config.HMAX_NORMALIZE] * self.stock_dim)


            unrealized_pnl = self._calculate_unrealized_pnl()
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            ) + unrealized_pnl
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
            # Short index % Close SHort index

            # Calculate holdings for shorting cost and fee


            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1

            self.data = self.df.loc[self.day, :].dropna(subset=["ticker"])
            self.index = self.index_df.loc[self.day, :]



            # print(self.data['turbulence'])

            self.turbulence = self.data["turbulence"].values[0]

            # load next state
            # print("stock_shares:{}".format(self.state[29:]))

            self.state = self._get_observation(initial=False)

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


            unrealized_pnl = self._calculate_unrealized_pnl()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            ) + unrealized_pnl
            self.asset_memory.append(end_total_asset)
            self.date = self.df.iloc[self.day]["Date"].strftime("%Y-%m-%d")

            self.date_memory.append(self.date)

            # print("end_total_asset:{}".format(end_total_asset))

            self.reward = end_total_asset - begin_total_asset
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)

            self.reward = self.reward * self.config.REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):

        if self.initial:
            self.asset_memory = [self.config.INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.date = self.df.iloc[self.day]["Date"].strftime("%Y-%m-%d")
            self.date_memory = [self.date]

            self.shorted_prices = [0] * self.stock_dim
            self.data = self.df.loc[self.day, :].dropna(subset=["ticker"])
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
            self.day = 0
            self.date = self.df.iloc[self.day]["Date"].strftime("%Y-%m-%d")
            # TODO: get shorted_stock_prices from previous window
            self.shorted_prices = [0] * self.stock_dim
            self.date_memory = [self.date]
            self.data = self.df.loc[self.day, :].dropna(subset=["ticker"])
            self.index = self.index_df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=iteration
            self.rewards_memory = []
            # initiate state
            # self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]
            # [0]*STOCK_DIM + \

            self.state = (
                [self.previous_state[0]]
                + self.data.adjcp.values.tolist()
                + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                + self.previous_state[(self.stock_dim*2 + 1): (self.stock_dim * 3 + 1)]
                + self.data.macd.values.tolist()
                + self.data.tema.values.tolist()
                + self.data.boll.values.tolist()
                + self.data.boll_ub.values.tolist()
                + self.data.boll_lb.values.tolist()
                + self.data.wr_10.values.tolist()
                + self.data.rsi_30.values.tolist()
                + self.data.rsi_14.values.tolist()
                + self.data.cci_30.values.tolist()
                + self.data.dx_30.values.tolist()
                + self.data.vr.values.tolist()
                + self.data.atr.values.tolist()
                + self.data.dma.values.tolist()
                + self.data.volume_delta.values.tolist()
                + self.data.kdjk.values.tolist()
                + self.data.kdjd.values.tolist()
                + self.data.kdjj.values.tolist()
                + self.index.index_close.values.tolist()
                + self.index.index_macd.values.tolist()
                + self.index.index_rsi.values.tolist()
                + self.data.volume.values.tolist()
                + self.data.category.values.tolist()
            )

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
