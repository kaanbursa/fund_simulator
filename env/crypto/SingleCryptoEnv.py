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
    REWARD_SCALING = 1e-5 #5e-4
    TRANSACTION_FEE_PERCENT = 0.001
    SHORT_FEE = 0.00005
    INDICATORS = indicator_list
    OBSERVATIONS = len(indicator_list) + 2
    INDEX_OBSERVATIONS = 3
    REWARD_INTERVAL = 3
    seed = 42
    use_turbulance = False


class SingleCryptoEnv(gym.Env):
    """A crypto trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, stock_dim,  time_window, mode, config=EnvConfig):
        self.config = config
        self.time_window= time_window
        self.stock_dim = stock_dim
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
        self.grade = 1

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
                    + [self.data.time.values.tolist()[0]]
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
                    + [self.data.time.values.tolist()[0]]
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
        # convert dolar amount to quantitiy
        action = action / self.state[index + 1]
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
        # convert dolar amount to quantitiy
        action = action / self.state[index + 1]
        if self.state[1 + index] > 0:
            # perform buy action based on the sign of the action
            available_amount = max(self.state[0] // self.state[index + 1], 0)
            # print('available_amount:{}'.format(available_amount))
            if available_amount != 0:
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

    def _calculate_reward(self, initial_balance, begin_total_asset, end_total_asset):
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

        if self.grade == 0:
            # Benchmark is buy and hold strategy
            if self.day % self.config.REWARD_INTERVAL == 0:
                #benchmark = self._alpha(initial_balance)  # returns buy and hold
                reward = end_total_asset - begin_total_asset  # + ((sharpe) * 5)
                # Add alpha to reward or substract
                #reward += (end_total_asset - benchmark) * 2
                #print(f'Reward:  {reward} Begin Asset: {begin_total_asset} End Asset: {end_total_asset}')
                return reward
            else:
                return 0
        elif self.grade == 1:
            if self.day % self.config.REWARD_INTERVAL == 0:
                benchmark = self._alpha(initial_balance)  # returns buy and hold
                reward = end_total_asset - begin_total_asset  # + ((sharpe) * 5)
                # Add alpha to reward or substract
                reward += (end_total_asset - benchmark) * 2
                return reward
            else:
                return 0

        elif self.grade == 2:
            """df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            df_total_value["daily_return"] = df_total_value.pct_change(1)
            sharpe = (
                    (252 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / (df_total_value["daily_return"].std() + 0.001)
            )"""
            benchmark = self._alpha(initial_balance)  # returns buy and hold
            reward = end_total_asset - benchmark # + ((sharpe + 1) ** 5)
            # Add alpha to reward or substract
            #reward += (end_total_asset - benchmark) * 2
            return reward

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

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:

            df_total_value = pd.DataFrame(self.asset_memory)
            # date = self.df.iloc[self.day]["Date"].strftime("%Y-%m-%d")
            # st.write('RL Model Performance')
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
            # df_rewards = pd.DataFrame({'Date':self.date_memory, 'Rewards':self.rewards_memory})
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
            # print(actions[:3])
            actions = actions * self.HMAX_NORMALIZE
            if self.debug:
                print('====DEBUG====')
                print('Printing actions from the agent ', actions)
            # actions = (actions.astype(int))

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            # TODO: If it is flag day dont buy or sell at that day
            self.date = pd.to_datetime(self.unique_trade_date[self.day + 1], format="%Y-%m-%d")
            if self.flag_days is not None:

                # TODO: ADD day increment here
                if pd.to_datetime(self.date, format="%Y-%m-%d") in self.flag_days:
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

            self.state = self._get_observation(initial=False)

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            holdings = self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
            if self.debug:
                print('===DEBUG TRADING ENV===')
                if any(i < 0 for i in holdings):
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

            self.asset_memory.append(end_total_asset)

            self.date_memory.append(self.date)

            # print("end_total_asset:{}".format(end_total_asset))

            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)

            self.reward = self.reward * self.config.REWARD_SCALING
            # For tensor callback
            # self.total_asset = end_total_asset

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
                np.array(self.previous_state[1: (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]
            # self.asset_memory = [self.previous_state[0]]
            self.day = 0 + self.time_window
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
                        + self.past_data.adjcp.values.tolist()
                        + self.past_data.volume.values.tolist()
                        + [self.data.month.values.tolist()[0]]
                        + [self.data.day.values.tolist()[0]]
                )
            else:
                self.state = (
                        [self.previous_state[0]]
                        + self.data.adjcp.values.tolist()
                        + self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]
                        + indicators
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
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
