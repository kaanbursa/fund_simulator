import os
import re
import time
from datetime import datetime, timedelta

import torch
#from loguru import logger
import optuna
# RL models from stable-baselines
from stable_baselines3 import A2C, PPO, DDPG, TD3, DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

#Hyperparameter tuning
import optuna
import wandb
import streamlit as st

from config import config

from data.preprocessing import *
from env.BaseEnv import EnvConfig
from env.EnvStock_trade import StockEnvTrade
from env.EnvStock_train import StockEnvTrain
from env.EnvStock_val import StockEnvValidation
from env.OldTrade import OldStockEnvTrade
from env.OldTrain import OldStockEnvTrain
from utils.helper_training import *
from utils.pbt import sample_ppo_params, sample_sac_params

# customized env
#from utils.pbt import sample_ppo_params, optimize_ppo2
from utils.indicators import indicator_list, indicators_stock_stats

from data.Clustering import UnsupervisedLearning
from data.data_generator import *


class TrainerConfig:
    TRAINED_MODEL_DIR = "trained_models"
    rebalance_window = 63
    validation_window = 63
    pretrain_window = 365
    start_date = "2012-01-01"
    start_trade = "2016-01-02"
    end_date = datetime.now()  # datetime.strftime(datetime.now(), 'yyyy-mm-dd')
    indicator_list = indicator_list
    indicators_stock_stats = indicators_stock_stats
    timesteps = 50000
    policy_kwargs = {}
    use_turbulance = False
    clip_obs = 1
    gamma = 0.99

    #Paths
    stocks_file_path = './data/all_stocks.csv'
    all_stocks_path= 'data/all_stocks_price2.csv'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(
        self,
        model,
        policy,
        env_train,
        env_val,
        env_trade,
        config,
        model_name,
        dataset_version,
        population = 5,
        env_config=EnvConfig,
        debug = False,
        tensorboard=False,
    ):
        model_dict = {"PPO": PPO, "A2C": A2C, 'DDPG':DDPG, 'TD3':TD3, 'DQN':DQN, 'SAC':SAC}
        self.model_type = model
        self.policy = policy
        self.model = model_dict[model]
        self.dataset_version = dataset_version
        self.env_train = env_train
        self.env_val = env_val
        self.env_trade = env_trade
        self.config = config
        self.model_name = model_name
        self.total_reward = 0
        self.env_config = env_config
        self.population = population
        self.policy_kwargs = config.policy_kwargs
        self.agen_df_path = "./agents_hparams.csv"
        self.stocks_file_path = config.stocks_file_path
        self.indicators_stock_stats = indicators_stock_stats
        self.index_list = config.index_list
        self.timesteps = config.timesteps if not debug else 10
        self.stocks = load_all_stocks(config.start_date, config.all_stocks_path)
        print('Total number of stocks:  ', len(self.stocks.ticker.unique()))
        #self.tbcallback = TensorboardCallback()
        self.debug = debug
        self.tensorboard=tensorboard
        self.tensoriter = 0
        #wandb.init(project='rl-dqn', entity='kaanb', config=hparams)
        if torch.cuda.is_available():
            print('GPU available')


        if os.path.exists(self.agen_df_path):
            self.agents_df = pd.read_csv(self.agen_df_path, index_col=0)
        else:
            self.agents_df = pd.DataFrame(
                columns=[
                    "model_name",
                    "model_type",
                    "total_reward",
                    "start_date",
                    "end_date",
                    "dataset_version",
                    "hparams",
                    "net_arch",
                    "sharpe",
                    "train-start",
                    "train-end",
                    "indicators",
                    "env_hparams",
                ]
            )

    def train_model(self, env_train, hparams, timesteps=50000, load=False, model_to_load=''):
        start = time.time()
        #build_and_log_model(model=self.model, model_alias='PPO', model_name=self.model_name, config=hparams)
        model_path= f"{self.config.TRAINED_MODEL_DIR}/{self.model_name}"
        if self.tensorboard:
            model = self.model(
                self.policy,
                env_train,
                callback = WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=model_path,
                ),
                tensorboard='./tensorboard_logs'
                **hparams,
            )

        else:
            model = self.model(
                self.policy,
                env_train,
                **hparams,
            )
        if load:
            path = self.config.TRAINED_MODEL_DIR
            """names = os.listdir(path)
            model_p = [name for name in names if 'PPO' in name]
            latest = str(max([[int(s) for s in re.findall(r'[0-9]+', p)][-1] for p in model_p]))
            model_p = [name for name in model_p if latest in name][0]"""

            print("Model PPO is loading")

            model.load(path + "/" + model_to_load)
        if self.tensorboard:
            model.learn(total_timesteps=timesteps, tb_log_name=f"{self.tensoriter}")
            self.tensoriter += 1
        else:
            model.learn(total_timesteps=timesteps)
        end = time.time()
        model.save(model_path)
        print(
            "Training time ", self.model_name, ": ", (end - start) / 60, " minutes"
        )
        return model

    def prediction(self, model, last_state, iter_num, flag_days, turbulence_threshold, initial, normalize=False, time_frame=0):

        stock_dimension = len(self.dataset[self.dataset.Date == self.unique_trade_date[iter_num]].ticker.dropna().unique())

        trade_data = data_split(
            self.dataset,
            start=self.unique_trade_date[iter_num - self.config.rebalance_window],
            end=self.unique_trade_date[iter_num],
        )

        if normalize:
            env_trade = VecNormalize(DummyVecEnv(
                [
                    lambda: self.env_trade(
                        trade_data,
                        trade_data,
                        time_window=time_frame,
                        flag_days=flag_days,
                        stock_dim=stock_dimension,
                        unique_trade_date=self.unique_trade_date,
                        turbulence_threshold=turbulence_threshold,
                        initial=initial,
                        config=self.env_config,
                        previous_state=last_state,
                        model_name=self.model_name,
                        iteration=iter_num,
                        debug=self.debug
                    )
                ]
            )
            )
        else:
            env_trade = DummyVecEnv(
                [
                    lambda: self.env_trade(
                        trade_data,
                        trade_data,
                        time_window=time_frame,
                        flag_days=flag_days,
                        stock_dim=stock_dimension,
                        unique_trade_date=self.unique_trade_date,
                        turbulence_threshold=turbulence_threshold,
                        initial=initial,
                        config=self.env_config,
                        previous_state=last_state,
                        model_name=self.model_name,
                        iteration=iter_num,
                        debug=self.debug
                    )
                ]
            )
        obs_trade = env_trade.reset()

        for i in range(len(trade_data.index.unique())):

            action, _states = model.predict(obs_trade)

            obs_trade, rewards, dones, info = env_trade.step(action)

            total_reward = sum(rewards)
            self.total_reward += sum(rewards)

            if i == (len(trade_data.index.unique()) - 2):
                # print(env_test.render())

                last_state, obj = env_trade.render()
                if self.debug:
                    print('=====DEBUG=====')
                    print('Last state: '
                          , last_state)
        end_total_asset = last_state[0] + sum(
            np.array(last_state[1: (stock_dimension + 1)])
            * np.array(last_state[(stock_dimension + 1): (stock_dimension * 2 + 1)])
        )
        print("=============")
        print("Total reward for the the window is {}".format(self.total_reward))

        print("=============")

        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(
            "results/last_state_{}_{}.csv".format(self.model_name, i), index=False
        )

        obj['trade_reward'] = total_reward
        obj['end_total_asset'] = end_total_asset
        obj['date'] = self.unique_trade_date[iter_num]
        try:
            log_trade(obj)
        except:
            print('Error with wandb')

        self.last_state_ensemble = last_state

    def exploit_and_explore(
        self, hyperparam_names, perturb_factors=[1.2, 0.8]
    ):
        """Copy parameters from the better model and the hyperparameters
        and running averages from the corresponding optimizer."""
        #study = optuna.create_study()
        #trial = study.ask()

        sampler = {
            'PPO':sample_ppo_params(hyperparam_names),
            'SAC':sample_sac_params(hyperparam_names)
        }

        new_hparams = sampler[self.model_type]

        return new_hparams

    def optimize_train(self, trial):
        model_params = optimize_ppo2(trial)
        env = self.env_train
        model = PPO('MlpPolicy', env, verbose=0, nminibatches=1, **model_params)
        model.learn(10000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        return -1 * mean_reward

    def DRL_validation(self, model, test_data, test_env, test_obs):
        """
        Return sum and mean
        :param model:
        :param test_data:
        :param test_env:
        :param test_obs:
        :return:
        """
        total_rewards = []
        #asset_list = []
        for i in range(len(test_data.Date.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)
            total_rewards.append(sum(rewards))
            #asset_list.append(info['end_total_asset'])
            if i == (len(test_data.Date.unique()) - 2):

                _, trades, end_total_asset = test_env.render()

        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=5)
        print("-----------------")
        print("Total Reward: ", sum(total_rewards))
        print("Total Trades: ", trades)
        print('End total asset for validation', end_total_asset)
        print("Mean Reward:", mean_reward)
        print("STD reward:", std_reward)
        print("-----------------")
        return sum(total_rewards), end_total_asset

    def get_validation_sharpe(self, iteration) -> str:
        try:
            df_total_value = pd.read_csv(
                "results/account_value_validation_{}_{}.csv".format(self.dataset_version,iteration), index_col=0
            )

            df_total_value.columns = ["account_value_train"]

            df_total_value["daily_return"] = df_total_value.pct_change(1)
            sharpe = (
                (4 ** 0.5)
                * df_total_value["daily_return"].mean()
                / (df_total_value["daily_return"].std() + 0.00001)
            )
            return sharpe
        except:
            return "0"

    def _save_model_info(self, hparams, reward, start, end):

        model_info = {
            "model_name": self.model_name,
            "model_type": self.model,
            "total_reward": reward,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "dataset_version": self.dataset_version,
            "hparams": ",".join(str(k) + ':' + str(v) for k,v in hparams.items()),
            "indicators": ",".join(str(e) for e in self.config.indicator_list),
            "val-start": start,
            "val-end": end,
            "env_hparams": "",
            'date_training':datetime.now()
        }
        if os.path.exists(self.agen_df_path):
            df = pd.read_csv(self.agen_df_path)
            df = df.append(model_info, ignore_index=True)
            df.to_csv(self.agen_df_path, index=False)
        else:
            df = pd.DataFrame(
                columns=[
                    "model_name",
                    "model_type",
                    "total_reward",
                    "start_date",
                    "end_date",
                    "dataset_version",
                    "hparams",
                    "net_arch",
                    "sharpe",
                    "train-start",
                    "train-end",
                    "indicators",
                    "env_hparams",
                ]
            )
            df = df.append(model_info, ignore_index=True)
            df.to_csv(self.agen_df_path, index=False)

    def _get_envs(self, train, validation, stock_dimension, i , turbulence_threshold, normalize, dates_to_change=None, time_frame=0):
        if not normalize:
            env_train = DummyVecEnv(
                [
                    lambda: self.env_train(
                        train,
                        train,
                        time_window=time_frame,
                        flag_days= dates_to_change,
                        config=self.env_config,
                        stock_dim=stock_dimension,
                    )
                ]
            )

            env_val = DummyVecEnv(
                [
                    lambda: self.env_val(
                        validation,
                        validation,  # Normaly index dataframe
                        time_window=time_frame,
                        flag_days=dates_to_change,
                        stock_dim=stock_dimension,
                        config=self.env_config,
                        turbulence_threshold=turbulence_threshold,
                        iteration=i,
                    )
                ]
            )
        else:
            env_train = VecNormalize(DummyVecEnv(
                [
                    lambda: self.env_train(
                        train,
                        train,
                        time_window=time_frame,
                        flag_days=dates_to_change,
                        config=self.env_config,
                        stock_dim=stock_dimension,
                    )
                ]
            ))

            env_val = VecNormalize(DummyVecEnv(
                [
                    lambda: self.env_val(
                        validation,
                        validation,
                        time_window=time_frame,
                        flag_days=dates_to_change,
                        stock_dim=stock_dimension,
                        config=self.env_config,
                        turbulence_threshold=turbulence_threshold,
                        iteration=i,
                    )
                ]
            ))
        return env_train, env_val


    def cluster(self,
                normalize=False,
                period=365,
                number_of_clusters=5, stocks_per_cluster=4) -> None:
        date_parse = lambda x: pd.to_datetime(x)
        # TODO: Move to another folder
        # 1: Load all stocks
        start_date = self.config.start_date
        processed_path = f"datasets/{self.dataset_version}_processed.csv"
        if os.path.exists(processed_path):
            print('The dataset has been made before starting training :)')

            self.dataset = pd.read_csv(processed_path, parse_dates=['Date'], date_parser=date_parse, index_col=0)
            self.dataset = self.dataset.sort_values(['Date', 'ticker']).reset_index(drop=True)

            year_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=period)

            end_date = datetime.today()
            delta = timedelta(days=period)
            self.dates_to_change = []

            #TODO: Change the ending date
            while year_date <= datetime.strptime('2021-01-28','%Y-%m-%d'):
                # Get first date in dataframe after stocks are released for changing
                if year_date > datetime.today():
                    break

                year_date += delta

                date_to_change = self.dataset[self.dataset.Date <= year_date].Date.values[-1]
                self.dates_to_change.append(date_to_change)

            start_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=self.config.rebalance_window +
                                                                                    self.config.validation_window + period)
            self.unique_trade_date = self.dataset[(self.dataset.Date > pd.to_datetime(start_date)) & (
                    self.dataset.Date <= pd.to_datetime(datetime.now()))].Date.unique()

            self.train_with_cluster(
                self.dataset,
                normalize=normalize
            )
        else:
            print('Loading all symbols')
            total_number_tickers = stocks_per_cluster * number_of_clusters # total number of ticker for given time
            df = self.stocks

            # filter unexisting stocks with dates
            year_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.today()
            delta = timedelta(days=period)
            days = []
            self.dates_to_change = []

            sample_df = get_price('AAPL',start_date, end_date)

            while year_date <= end_date:
                if year_date > datetime.today():
                    break

                date_to_change = sample_df[sample_df.Date < year_date].Date.values[-1]
                self.dates_to_change.append(date_to_change)
                days.append(date_to_change)
                year_date += delta

            # 2: Filter existing stocks in given period
            self.dataset = pd.DataFrame()
            periods = {}
            for i,day in enumerate(days[:-2]):
                # filter data
                # 3: Apply Clustering for creating uncorelated stocks and
                st_day = pd.to_datetime(day)
                to_start_trade_date = pd.to_datetime(days[i + 1]) # For clustering period for given stocks
                period_end_date = to_start_trade_date + timedelta(days=period)
                # For naming the dataset we start from the ending of clustering period to the next year
                start_str = str(to_start_trade_date.year) + '_' + str(to_start_trade_date.month)
                end_str = str(period_end_date.year) + '_' + str(period_end_date.month)
                dataset_name = f"{self.dataset_version}_from_{start_str}_to_{end_str}" #
                if os.path.exists("./datasets/" + dataset_name + ".csv"):
                    print('Loading existing dataset')
                    new_dataset = pd.read_csv("./datasets/" + dataset_name + ".csv", parse_dates=['Date'], date_parser=date_parse, index_col=0)
                    new_dataset = new_dataset[new_dataset.Date > to_start_trade_date]
                    self.dataset = pd.concat([self.dataset, new_dataset])
                    continue
                print(f'Clustering stocks for the periord {str(days[i])} - {str(days[i + 1])}')
                # To date is when these stocks will end

                clustered_stocks, training_df = get_clustered_stocks(df, st_day, to_start_trade_date, number_of_clusters)

                # 4: pick stocks with metrics.
                # TODO: try different metrics

                stocks_to_train = pick_from_kmeans_cluster(clustered_stocks, training_df, n_from_each=stocks_per_cluster)

                assert len(set(stocks_to_train)) == total_number_tickers, f'Total number of ticker per period does not match the training stock size {len(set(stocks_to_train))} : {total_number_tickers}'
                print('Stocks to train during these days are: ', stocks_to_train, 'Adding indicators for next period')
                # 5: add indicators to selected stocks which has been picked from the previous year


                start_day_for_indicators = to_start_trade_date - timedelta(days=180)

                periods[i] = stocks_to_train

                new_dataset, index_df = add_indicators_for_stocks(name=dataset_name,
                                                                  ticker_list=stocks_to_train, index_list=[],
                                                                  indicators_stock_stats=self.config.indicators_stock_stats,
                                                                  start_date= start_day_for_indicators, end_date =period_end_date)
                assert len(new_dataset.ticker.unique()) == total_number_tickers, f'Failed to load some of the symbols'
                new_dataset = new_dataset[new_dataset.Date > to_start_trade_date]

                self.index_df = index_df

                self.dataset = pd.concat([self.dataset, new_dataset])
                assert 'turbulence' in self.dataset.columns, 'turbulance is not in columns'

                # 1 add new dataset dataframe to previous
                # 2 seperate dataset for training session
                # 3 setup a way to sell all previous tickers by changing last state ensemble

                #self.dates_to_change.append(days[i+1])
            # 4: Preprocess and add indicators to prices
            # 5: Train and save the model
            # Repeat 2 - 5 until periods ends

            self.dataset = self.dataset.sort_values(['Date','ticker']).reset_index()
            self.dataset.to_csv(processed_path)
            self.dataset.Date = pd.to_datetime(self.dataset.Date)
            if self.debug:
                print('======DEBUG======')
                print('INFORMATION ABOUT DATASET')
                print(self.dataset.describe())
            start_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=self.config.rebalance_window +
                                                                               self.config.validation_window + period) # Skip first period
            self.unique_trade_date = self.dataset[(self.dataset.Date > pd.to_datetime(start_date)) & (
                    self.dataset.Date <= pd.to_datetime(end_date))].Date.unique()

            self.train_with_cluster(
                self.dataset,
                normalize=normalize
            )

    def pretrain(self, pretrain_dataset, start_date, to_date, normalize):
        if isinstance(pretrain_dataset, str):
            pretrain_dataset = pd.read_csv(pretrain_dataset)
        pretrain_dataset = pretrain_dataset[pretrain_dataset.Date <= to_date]
        unique_trade_date = pretrain_dataset.Date.unique()[self.config.pretrain_window + self.config.validation_window:]

        for i in range(
            self.config.pretrain_window + self.config.validation_window,
            len(unique_trade_date),
            self.config.pretrain_window,
        ):
            print("============================================")

            # Tuning trubulence index based on historical data
            # Turbulence lookback window is one quarter

            #turbulence_threshold= 1
            stocks = pretrain_dataset[pretrain_dataset.Date == unique_trade_date[i]].ticker.dropna().unique()
            stock_dimension = len(stocks)

            print('Stocks trading at the end of this period is ', stocks)
            end = unique_trade_date[
                    i - self.config.pretrain_window - self.config.validation_window
                ]
            train = data_split(
                pretrain_dataset,
                start=start_date,
                end=end,
            )

            # val end date
            end = unique_trade_date[
                i  - self.config.validation_window
                ]

            # Dates to sell all stock for incoming new cluster


            ## validation env
            validation = data_split(
                pretrain_dataset,
                start=unique_trade_date[
                    i - self.config.pretrain_window - self.config.validation_window
                ],
                end=end,
            )

            """validation_index = data_split(
                self.index_df,
                start=self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window
                ],
                end=end,
            )"""

            env_train, env_val = self._get_envs(train, validation, stock_dimension, i,0, normalize)

            obs_val = env_val.reset()
            ############## Training and Validation starts ##############

            end_date = unique_trade_date[
                    i - self.config.pretrain_window - self.config.validation_window
                ]
            print(
                "======Model training from: ",
                start_date,
                "to ",
                end_date,
            )

            print(f"======Recurrent PPO Training for a population of {self.population}========")
            reward = 0
            first_reward = True
            winner_hparams = dict()
            hparams = self.config.hparams
            for agent in range(self.population):
                if self.population > 1: # If Population based training is being used
                    #hparams["learning_rate"] = sched_LR.value
                    model_rec_ppo = self.train_model(env_train, hparams, timesteps=self.timesteps, load=False, model_to_load=None)
                    print(
                        "======Recurrent PPO Validation from: ",
                        unique_trade_date[
                            i - self.config.pretrain_window - self.config.validation_window
                        ],
                        "to ",
                        unique_trade_date[i - self.config.pretrain_window],
                    )
                    total_reward, end_total_asset = self.DRL_validation(
                        model=model_rec_ppo,
                        test_data=validation,
                        test_env=env_val,
                        test_obs=obs_val,
                    )

                    if total_reward > reward or first_reward:
                        print(f"Agent #{agent} has better performance for the training period with total reward: {total_reward}")
                        reward = total_reward
                        winner = model_rec_ppo
                        winner_hparams = hparams
                        first_reward = False
                    hparams = self.exploit_and_explore(hyperparam_names=self.config.hparams)
                else: # Use optuna for hyperparameter tuning
                    #model_rec_ppo = self.train_model(env_train, hparams, timesteps=timesteps, load=load)
                    #study = optuna.create_study()
                    #study.optimize(self.optimize_train, n_trials=100)
                    winner = self.train_model(env_train, hparams, timesteps=self.timesteps, load=False)
                    winner_hparams = hparams
                    print('Best params, ', winner_hparams)
                    print(
                        "======Recurrent PPO Validation from: ",
                        unique_trade_date[
                            i - self.config.pretrain_window - self.config.validation_window
                            ],
                        "to ",
                        unique_trade_date[i - self.config.pretrain_window],
                    )
                    total_reward, end_total_asset = self.DRL_validation(
                        model=winner,
                        test_data=validation,
                        test_env=env_val,
                        test_obs=obs_val,
                    )


                print("Total reward at validation for Reccurent PPO", total_reward)
                sharpe_rec_ppo = get_validation_sharpe(i)
                print("Sharpe Ratio: ", sharpe_rec_ppo)
            self._save_model_info(
                winner_hparams,
                reward,
                start_date,
                end_date
            )
            self.config.hparams = winner_hparams
            model_path = f"{self.config.TRAINED_MODEL_DIR}/{self.model_name}_pretrained"
            winner.save(model_path)

        return model_path


    def train_with_cluster(self, dataset, load=False, model_to_load='', normalize: bool = False) -> None:

        """assert set(self.dataset.Date.unique()) == set(
            self.index_df.Date.unique()
        ), "Dataset sizes dont match" """
        if isinstance(dataset, str):
            date_parse = lambda x: pd.to_datetime(x)
            self.dataset_version = dataset
            self.dataset = pd.read_csv(dataset, parse_dates=['Date'], date_parser=date_parse)
            self.dataset = self.dataset[self.dataset.Date >= self.config.start_date]
        else:
            self.dataset = dataset[dataset.Date >= self.config.start_date]
        self.unique_trade_date = self.dataset[self.dataset.Date >= self.config.start_trade].Date.unique()
        timesteps = self.timesteps
        self.last_state_ensemble = []

        rec_ppo_sharpe_list = []
        model_use = []
        # based on the analysis of the in-sample data
        # turbulence_threshold = 140


        insample_turbulence = self.dataset[
            (self.dataset.Date < self.config.end_date)
            & (self.dataset.Date >= self.config.start_date)
        ]
        insample_turbulence = insample_turbulence.drop_duplicates(subset=["Date"])
        insample_turbulence_threshold = np.quantile(
            insample_turbulence.turbulence.values, 0.90
        )

        if self.debug:
            print('====DEBUG====')
            print('Total stocks for training process', len(self.dataset.ticker.unique()))

        start = time.time()
        for i in range(
            self.config.rebalance_window + self.config.validation_window,
            len(self.unique_trade_date),
            self.config.rebalance_window,
        ):
            print("============================================")
            ## initial state is empty
            if i - self.config.rebalance_window - self.config.validation_window == 0:
                # inital state
                initial = True
            else:
                # previous state
                initial = False

            # Tuning trubulence index based on historical data
            # Turbulence lookback window is one quarter

            historical_turbulence = self.dataset[
                (
                    self.dataset.Date
                    < self.unique_trade_date[
                        i - self.config.rebalance_window - self.config.validation_window
                    ]
                )
                & (
                    self.dataset.Date
                    >= (
                        self.unique_trade_date[
                            i
                            - self.config.rebalance_window
                            - self.config.validation_window
                            - 63
                        ]
                    )
                )
            ]
            historical_turbulence = historical_turbulence.drop_duplicates(
                subset=["Date"]
            )

            historical_turbulence_mean = np.mean(
                historical_turbulence.turbulence.values
            )

            if historical_turbulence_mean > insample_turbulence_threshold:
                # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
                # then we assume that the current market is volatile,
                # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
                # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
                turbulence_threshold = insample_turbulence_threshold
            else:
                # if the mean of the historical data is less than the 90% quantile of insample turbulence data
                # then we tune up the turbulence_threshold, meaning we lower the risk
                turbulence_threshold = np.quantile(
                    insample_turbulence.turbulence.values, 0.99
                )
            #turbulence_threshold= 1
            stocks = self.dataset[self.dataset.Date == self.unique_trade_date[i]].ticker.dropna().unique()
            stock_dimension = len(stocks)



            print('Stocks trading this period is ', stocks)
            end = self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window
                ]

            train = data_split(
                self.dataset,
                start=self.config.start_date,
                end=end,
            )

            # val end date
            end = self.unique_trade_date[
                i  - self.config.validation_window
                ]

            # Dates to sell all stock for incoming new cluster

            ## validation env
            validation = data_split(
                self.dataset,
                start=self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window
                ],
                end=end,
            )

            """validation_index = data_split(
                self.index_df,
                start=self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window
                ],
                end=end,
            )"""
            if not hasattr(self,'dates_to_change'):
                start_date = self.config.start_date
                period = 365
                year_date = datetime.strptime(start_date, "%Y-%m-%d")
                end_date = datetime.today()
                delta = timedelta(days=period)
                days = []
                self.dates_to_change = []

                sample_df = get_price('AAPL', start_date, end_date)

                while year_date <= end_date:
                    if year_date > datetime.today():
                        break

                    date_to_change = sample_df[sample_df.Date < year_date].Date.values[-1]
                    self.dates_to_change.append(date_to_change)
                    days.append(date_to_change)
                    year_date += delta


            env_train, env_val = self._get_envs(train, validation, stock_dimension, i, turbulence_threshold, normalize, dates_to_change=self.dates_to_change)

            obs_val = env_val.reset()

            ############## Training and Validation starts ##############

            end_date = self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window
                ]
            print(
                "======Model training from: ",
                self.config.start_date,
                "to ",
                end_date,
            )


            print(f"======Recurrent PPO Training for a population of {self.population}========")
            reward = -100
            winner_hparams = dict()
            initial_reward = True
            hparams = self.config.hparams
            seed = hparams['seed']
            for agent in range(self.population):
                # hparams["learning_rate"] = sched_LR.value
                try:
                    model_rec_ppo = self.train_model(env_train, hparams, timesteps=timesteps, load=load,
                                                     model_to_load=model_to_load)
                    # TODO: Try different seasons for validations pick the top reward
                    total_reward = 0
                    # Validation For different market conditions
                    # TODO: Change to framed conditional training for validation
                    # ====================================
                    """validation_frames = 1
                    S = len(validation) // validation_frames
                    N = int(len(validation) / S)
                    frames = [validation.iloc[i * S:(i + 1) * S].copy() for i in range(N + 1)]
                    for i,data in enumerate(frames):"""

                    _, env_val = self._get_envs(train, validation, stock_dimension, i,
                                                turbulence_threshold, normalize)
                    print(
                        f"======{self.model_name} Validation from: ",
                        validation['Date'].iloc[0],
                        "to ",
                        validation['Date'].iloc[-1],
                    )
                    obs_val = env_val.reset()
                    period_reward, end_total_asset = self.DRL_validation(
                        model=model_rec_ppo,
                        test_data=validation,
                        test_env=env_val,
                        test_obs=obs_val,
                    )
                    print(f'Reward for the period is {period_reward}')
                    total_reward += end_total_asset

                    if total_reward > reward:
                        print(
                            f"Agent #{agent} has better performance for the training period with total reward: {total_reward}")
                        reward = total_reward
                        winner = model_rec_ppo
                        winner_hparams = hparams
                    # ====================================
                    hparams = self.exploit_and_explore(hyperparam_names=self.config.hparams)
                    hparams['seed'] = seed
                    hparams['device'] = 'cuda'
                except:
                    print('Model destabilized with params: '
                          , ' Creating new params')
                    hparams = self.exploit_and_explore(hyperparam_names=self.config.hparams)

                    hparams['seed'] = seed
            else:  # Use optuna for hyperparameter tuning
                # model_rec_ppo = self.train_model(env_train, hparams, timesteps=timesteps, load=load)
                # study = optuna.create_study()
                # study.optimize(self.optimize_train, n_trials=100)
                obs_val = env_val.reset()
                hparams = self.config.hparams
                winner = self.train_model(env_train, hparams, timesteps=timesteps, load=load)
                winner_hparams = hparams

                print(
                    "======Recurrent PPO Validation from: ",
                    self.unique_trade_date[
                        i - self.config.rebalance_window - self.config.validation_window
                        ],
                    "to ",
                    self.unique_trade_date[i - self.config.rebalance_window],
                )
                total_reward, end_total_asset = self.DRL_validation(
                    model=winner,
                    test_data=validation,
                    test_env=env_val,
                    test_obs=obs_val,
                )
                print("Total reward at validation for Reccurent PPO", total_reward)
            self._save_model_info(
                winner_hparams,
                reward,
                self.config.start_date,
                end_date
            )
            self.config.hparams = winner_hparams
            # ppo_sharpe_list.append(sharpe_ppo)
            #rec_ppo_sharpe_list.append(sharpe_rec_ppo)

            # Model Selection based on sharpe ratio
            # if (sharpe_ppo >= sharpe_a2c):
            model_ensemble = winner
            model_use.append("Rec_PPO")

            ############## Training and Validation ends ##############

            ############## Trading starts ##############

            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.config.rebalance_window],
                "to ",
                self.unique_trade_date[i],
                "Model is : ",
                model_use[-1],
            )
            # print("Used Model: ", model_ensemble)

            self.prediction(
                model=model_ensemble,
                flag_days = self.dates_to_change,
                last_state=self.last_state_ensemble,
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
                normalize=normalize
            )



            # print("============Trading Done============")
            ############## Trading ends ##############

        end = time.time()
        self.tensoriter = 0
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

    def train(self, dataset, timesteps=30000, time_frame=0, load=False,  model_to_load='', normalize: bool = False) -> None:
        """
        Traingin function for the model to run the strategy
        :param dataset: str | pd.dataframe for training
        :param timesteps: how many timesteps to train the model
        :param time_frame: time frame for the model to look back on past data if zero it wont look back
        :param load: bool to load a model or not
        :param model_to_load: if load is true the path to file
        :param normalize: normalize the environment
        :return: None
        """
        """assert set(self.dataset.Date.unique()) == set(
            self.index_df.Date.unique()
        ), "Dataset sizes dont match"""
        if isinstance(dataset, str):
            date_parse = lambda x: pd.to_datetime(x)
            self.dataset_version = dataset
            self.dataset = pd.read_csv(dataset, parse_dates=['Date'], date_parser=date_parse)
            self.dataset = self.dataset[self.dataset.Date >= self.config.start_date]
        else:
            self.dataset = dataset[dataset.Date >= self.config.start_date]

        self.unique_trade_date = self.dataset[self.dataset.Date >= self.config.start_trade].Date.unique()

        try:
            load_and_log(dataset=[],
                         description="Fund simulator for risk management",
                         indicator_list=self.config.indicator_list,
                         stock_list=[]
                         )
        except:
            print('Error using wandb')


        self.last_state_ensemble = []
        rec_ppo_sharpe_list = []
        model_use = []
        # based on the analysis of the in-sample data
        # turbulence_threshold = 140

        insample_turbulence = self.dataset[
            (self.dataset.Date < pd.to_datetime(self.config.end_date))
            & (self.dataset.Date >= pd.to_datetime(self.config.start_date))
        ]
        insample_turbulence = insample_turbulence.drop_duplicates(subset=["Date"])
        insample_turbulence_threshold = np.quantile(
            insample_turbulence.turbulence.values, 0.90
        )

        total_timesteps = (
            len(self.unique_trade_date) // self.config.rebalance_window
        ) * timesteps
        #sched_LR = LinearSchedule(total_timesteps, 0.0005, 0.00001)

        start = time.time()


        for i in range(
            self.config.rebalance_window + self.config.validation_window + time_frame,
            len(self.unique_trade_date),
            self.config.rebalance_window,
        ):
            print("============================================")
            ## initial state is empty
            if i - self.config.rebalance_window - self.config.validation_window - time_frame == 0:
                # inital state
                initial = True
            else:
                # previous state
                initial = False

            # Tuning trubulence index based on historical data
            # Turbulence lookback window is one quarter

            historical_turbulence = self.dataset[
                (
                    self.dataset.Date
                    < self.unique_trade_date[
                        i - self.config.rebalance_window - self.config.validation_window
                    ]
                )
                & (
                    self.dataset.Date
                    >= (
                        self.unique_trade_date[
                            i
                            - self.config.rebalance_window
                            - self.config.validation_window
                            - 63
                        ]
                    )
                )
            ]
            historical_turbulence = historical_turbulence.drop_duplicates(
                subset=["Date"]
            )

            historical_turbulence_mean = np.mean(
                historical_turbulence.turbulence.values
            )

            if historical_turbulence_mean > insample_turbulence_threshold:
                # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
                # then we assume that the current market is volatile,
                # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
                # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
                turbulence_threshold = insample_turbulence_threshold
            else:
                # if the mean of the historical data is less than the 90% quantile of insample turbulence data
                # then we tune up the turbulence_threshold, meaning we lower the risk
                turbulence_threshold = np.quantile(
                    insample_turbulence.turbulence.values, 0.99
                )
            #turbulence_threshold= 1
            stocks = self.dataset[self.dataset.Date == self.unique_trade_date[i]].ticker.dropna().unique()
            stock_dimension = len(stocks)


            train = data_split(
                self.dataset,
                start=self.config.start_date,
                end=self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window - time_frame
                ],
            )
            train_index = data_split(
                self.dataset,
                start=self.config.start_date,
                end=self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window
                ],
            )


            ## validation env
            validation = data_split(
                self.dataset,
                start=self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window - time_frame
                ],
                end=self.unique_trade_date[i - self.config.rebalance_window],
            )

            validation_index = data_split(
                self.dataset,
                start=self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window
                ],
                end=self.unique_trade_date[i - self.config.rebalance_window],
            )

            env_train, env_val = self._get_envs(train, validation, stock_dimension, i, turbulence_threshold, normalize, time_frame=time_frame)



            ############## Training and Validation starts ##############

            end_date = self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window
                ]
            print(
                "======Model training from: ",
                self.config.start_date,
                "to ",
                end_date,
            )


            print(f"======Training Agents with the population of {self.population}========")
            reward = -100
            seed = self.config.hparams['seed']
            hparams = self.config.hparams
            hparams['seed'] = seed
            winner_hparams = dict()

            for agent in range(self.population):
                if self.population > 1: # If Population based training is being used
                    #hparams["learning_rate"] = sched_LR.value
                    try:
                        model_rec_ppo = self.train_model(env_train, hparams, timesteps=timesteps, load=load, model_to_load=model_to_load)
                        #TODO: Try different seasons for validations pick the top reward
                        total_reward = 0
                        # Validation For different market conditions
                        # TODO: Change to framed conditional training for validation
                        # ====================================
                        """validation_frames = 1
                        S = len(validation) // validation_frames
                        N = int(len(validation) / S)
                        frames = [validation.iloc[i * S:(i + 1) * S].copy() for i in range(N + 1)]
                        for i,data in enumerate(frames):"""

                        _, env_val = self._get_envs(train, validation, stock_dimension, i,
                                                            turbulence_threshold, normalize, time_frame=time_frame)
                        print(
                            f"======{self.model_name} Validation from: ",
                            validation['Date'].iloc[0],
                            "to ",
                            validation['Date'].iloc[-1],
                        )
                        obs_val = env_val.reset()
                        period_reward, end_total_asset = self.DRL_validation(
                            model=model_rec_ppo,
                            test_data=validation,
                            test_env=env_val,
                            test_obs=obs_val,
                        )
                        print(f'Reward for the period is {period_reward}')
                        total_reward += end_total_asset

                        if total_reward > reward:
                            print(f"Agent #{agent} has better performance for the training period with total reward: {total_reward}")
                            reward = total_reward
                            winner = model_rec_ppo
                            winner_hparams = hparams
                        # ====================================
                        hparams = self.exploit_and_explore(hyperparam_names=self.config.hparams)
                        hparams['seed'] = seed
                        hparams['device'] = 'cuda'
                    except:
                        print('Model destabilized with params: '
                              , ' Creating new params')
                        hparams = self.exploit_and_explore(hyperparam_names=self.config.hparams)

                        hparams['seed'] = seed
                else: # Use optuna for hyperparameter tuning
                    #model_rec_ppo = self.train_model(env_train, hparams, timesteps=timesteps, load=load)
                    #study = optuna.create_study()
                    #study.optimize(self.optimize_train, n_trials=100)
                    obs_val = env_val.reset()
                    hparams = self.config.hparams
                    winner = self.train_model(env_train, hparams, timesteps=timesteps, load=load)
                    winner_hparams = hparams

                    print(
                        "======Recurrent PPO Validation from: ",
                        self.unique_trade_date[
                            i - self.config.rebalance_window - self.config.validation_window -time_frame
                            ],
                        "to ",
                        self.unique_trade_date[i - self.config.rebalance_window],
                    )
                    total_reward, end_total_asset = self.DRL_validation(
                        model=winner,
                        test_data=validation,
                        test_env=env_val,
                        test_obs=obs_val,
                    )
                    print("Total reward at validation for Reccurent PPO", total_reward)

            self._save_model_info(
                winner_hparams,
                reward,
                self.unique_trade_date[i - self.config.rebalance_window - self.config.validation_window - time_frame],
                self.unique_trade_date[i - self.config.rebalance_window]
            )
            sharpe_rec_ppo = get_validation_sharpe(i)
            print("Sharpe Ratio: ", sharpe_rec_ppo)
            print('='*80)
            print('Best params, ', winner_hparams)
            print('=' * 80)
            self.config.hparams = winner_hparams

            # ppo_sharpe_list.append(sharpe_ppo)
            rec_ppo_sharpe_list.append(sharpe_rec_ppo)

            # Model Selection based on sharpe ratio
            # if (sharpe_ppo >= sharpe_a2c):
            model_ensemble = winner
            model_use.append("Rec_PPO")

            ############## Training and Validation ends ##############

            ############## Trading starts ##############

            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.config.rebalance_window],
                "to ",
                self.unique_trade_date[i],
                "Model is : ",
                model_use[-1],
            )
            # print("Used Model: ", model_ensemble)

            self.prediction(
                model=model_ensemble,
                flag_days=None,
                last_state=self.last_state_ensemble,
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
                normalize=normalize,
                time_frame=time_frame
            )

            # print("============Trading Done============")
            ############## Trading ends ##############

        end = time.time()

        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")


def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def train_rec_PPO(env_train, model_name, hparam={}, timesteps=50000, load=False):
    """PPO model"""

    start = time.time()
    model = PPO2(
        RecurrentPPOPolicy,
        env_train,
        nminibatches=1,
        verbose=0,
        seed=42,
        tensorboard_log="./ppo_rec_trade_tensorboard/",
        **hparam,
    )

    if load:
        path = "./trained_models"
        names = os.listdir(path)

        model_p = [name for name in names if "PPO" in name]
        latest = str(
            max([[int(s) for s in re.findall(r"[0-9]+", p)][-1] for p in model_p])
        )

        model_p = [name for name in model_p if latest in name][0]

        print("Model PPO is loading")

        model.load(path + "/" + model_p)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (PPO Recurrent): ", (end - start) / 60, " minutes")
    return model


def DRL_prediction(
    df,
    index_df,
    model,
    name,
    last_state,
    iter_num,
    unique_trade_date,
    rebalance_window,
    turbulence_threshold,
    initial,
):
    """
    Make prediction based on trained model
    :param df:
    :param model:
    :param name:
    :param last_state:
    :param iter_num:
    :param unique_trade_date:
    :param rebalance_window:
    :param turbulence_threshold:
    :param initial:
    :return:
    """
    stock_dimension = len(df.ticker.dropna().unique())
    trade_data = data_split(
        df,
        start=unique_trade_date[iter_num - rebalance_window],
        end=unique_trade_date[iter_num],
    )
    trade_index_data = data_split(
        index_df,
        start=unique_trade_date[iter_num - rebalance_window],
        end=unique_trade_date[iter_num],
    )

    env_trade = DummyVecEnv(
        [
            lambda: OldStockEnvTrade(
                trade_data,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
                previous_state=last_state,
                model_name=name,
                iteration=iter_num,
            )
        ]
    )
    obs_trade = env_trade.reset()
    total_reward = 0

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)

        obs_trade, rewards, dones, info = env_trade.step(action)
        total_reward += sum(rewards)

        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())

            last_state = env_trade.render()
    print("=============")
    print("Total reward for the the window is {}".format(total_reward))
    print("=============")
    df_last_state = pd.DataFrame({"last_state": last_state})
    df_last_state.to_csv("results/last_state_{}_{}.csv".format(name, i), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs):
    """
    Return sum and mean
    :param model:
    :param test_data:
    :param test_env:
    :param test_obs:
    :return:
    """
    """total_rewards = []
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)

        test_obs, rewards, dones, info = test_env.step(action)
        total_rewards.append(sum(rewards))
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
    print("-----------------")
    print("Mean Reward:", mean_reward)
    print("STD reward:", std_reward)
    print("-----------------")"""
    return 10


def get_validation_sharpe(iteration) -> str:
    try:
        df_total_value = pd.read_csv(
            "results/account_value_validation_{}.csv".format(iteration), index_col=0
        )

        df_total_value.columns = ["account_value_train"]

        df_total_value["daily_return"] = df_total_value.pct_change(1)
        sharpe = (
            (4 ** 0.5)
            * df_total_value["daily_return"].mean()
            / (df_total_value["daily_return"].std() + 0.00001)
        )
        return sharpe
    except:
        return "0"


def run_ensemble_strategy(
    df,
    index_df,
    unique_trade_date,
    rebalance_window,
    validation_window,
    start_date,
    end_date,
    model_name,
    models,
    hparam_list,
    population=10,
    timesteps=30000,
    load=False,
) -> None:

    """assert set(df.Date.unique()) == set(
        index_df.Date.unique()
    ), "Dates of index and df sizes dont match"""
    last_state_ensemble = []

    ppo_sharpe_list = []
    rec_ppo_sharpe_list = []
    a2c_sharpe_list = []
    trpo_sharpe_list = []

    model_use = []
    # based on the analysis of the in-sample data
    # turbulence_threshold = 140

    insample_turbulence = df[(df.Date < end_date) & (df.Date >= start_date)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=["Date"])
    insample_turbulence_threshold = np.quantile(
        insample_turbulence.turbulence.values, 0.90
    )

    total_timesteps = (len(unique_trade_date) // rebalance_window) * timesteps
    sched_LR = LinearSchedule(total_timesteps, 0.0005, 0.00001)

    start = time.time()
    for i in range(
        rebalance_window + validation_window, len(unique_trade_date), rebalance_window
    ):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter

        historical_turbulence = df[
            (df.Date < unique_trade_date[i - rebalance_window - validation_window])
            & (
                df.Date
                >= (unique_trade_date[i - rebalance_window - validation_window - 63])
            )
        ]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=["Date"])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(
                insample_turbulence.turbulence.values, 0.99
            )

        stock_dimension = len(df.ticker.dropna().unique())
        train = data_split(
            df,
            start=start_date,
            end=unique_trade_date[i - rebalance_window - validation_window],
        )
        train_index = data_split(
            df,
            start=start_date,
            end=unique_trade_date[i - rebalance_window - validation_window],
        )

        env_train = DummyVecEnv(
            [
                lambda: OldStockEnvTrain(
                    train
                )
            ]
        )

        ## validation env
        validation = data_split(
            df,
            start=unique_trade_date[i - rebalance_window - validation_window],
            end=unique_trade_date[i - rebalance_window],
        )

        validation_index = data_split(
            df,
            start=unique_trade_date[i - rebalance_window - validation_window],
            end=unique_trade_date[i - rebalance_window],
        )

        env_val = DummyVecEnv(
            [
                lambda: StockEnvValidation(
                    validation,
                    validation_index,
                    stock_dim=stock_dimension,
                    config=EnvConfig,
                    turbulence_threshold=turbulence_threshold,
                    iteration=i,
                )
            ]
        )

        obs_val = env_val.reset()
        ############## Training and Validation starts ##############

        """models = []
        reward = 0
        for i in range(population):
            hparam = sample_ppo_params()
            print("======Model training from: ", start_date, "to ",
                  unique_trade_date[i - rebalance_window - validation_window])
            # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
            # print("==============Model Training===========")

            print("======Recurrent PPO Training========")
            #hparam_list['learning_rate'] = sched_LR.value
            model_rec_ppo = train_rec_PPO(env_train,
                                          model_name="Rec_PPO_100k_{}_{}".format(model_name, i),
                                          hparam=hparam, timesteps=timesteps,
                                          load=load)
            print("======Recurrent PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window],
                  "to ",
                  unique_trade_date[i - rebalance_window])
            total_reward = DRL_validation(model=model_rec_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
            if total_reward > reward:
                models.append(model_rec_ppo)
                print('Total reward at validation for Reccurent PPO', total_reward)
                sharpe_rec_ppo = get_validation_sharpe(i)
                print("PPO Sharpe Ratio: ", sharpe_rec_ppo)"""

        print(
            "======Model training from: ",
            start_date,
            "to ",
            unique_trade_date[i - rebalance_window - validation_window],
        )
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")

        print("======Recurrent PPO Training========")
        hparam_list["learning_rate"] = sched_LR.value
        model_rec_ppo = train_PPO(
            env_train,
            model_name="PPO_100k_{}_{}".format(model_name, i),
            timesteps=timesteps,

        )
        print(
            "======Recurrent PPO Validation from: ",
            unique_trade_date[i - rebalance_window - validation_window],
            "to ",
            unique_trade_date[i - rebalance_window],
        )
        total_reward = DRL_validation(
            model=model_rec_ppo,
            test_data=validation,
            test_env=env_val,
            test_obs=obs_val,
        )
        print("Total reward at validation for Reccurent PPO", total_reward)
        sharpe_rec_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_rec_ppo)

        """print("======TRPO Training========")

        model_trpo = train_trpo(env_train, model_name="TRPO_10k_{}_{}".format(model_name, i), timesteps=timesteps,load=load)
        # model_ddpg = train_TD3(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=20000)
        print("======TRPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])

        DRL_validation(model=model_trpo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_trpo = get_validation_sharpe(i)
        print("TRPO Sharpe Ratio: ", sharpe_trpo)"""

        # ppo_sharpe_list.append(sharpe_ppo)
        rec_ppo_sharpe_list.append(model_rec_ppo)
        # a2c_sharpe_list.append(sharpe_a2c)
        # trpo_sharpe_list.append(sharpe_trpo)
        # trpo_sharpe_list.append(sharpe_trpo)

        # Model Selection based on sharpe ratio
        # if (sharpe_ppo >= sharpe_a2c):
        model_ensemble = model_rec_ppo
        model_use.append("Rec_PPO")
        # else:
        #    model_ensemble = model_a2c
        #    model_use.append('A2C')
        """if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_trpo):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_trpo):
            model_ensemble = model_a2c
            model_use.append('A2C')
        else:
            model_ensemble = model_trpo
            model_use.append('TRPO')"""

        ############## Training and Validation ends ##############

        ############## Trading starts ##############

        print(
            "======Trading from: ",
            unique_trade_date[i - rebalance_window],
            "to ",
            unique_trade_date[i],
            "Model is : ",
            model_use[-1],
        )
        # print("Used Model: ", model_ensemble)

        last_state_ensemble = DRL_prediction(
            df=df,
            index_df=df,
            model=model_ensemble,
            name=model_name,
            last_state=last_state_ensemble,
            iter_num=i,
            unique_trade_date=unique_trade_date,
            rebalance_window=rebalance_window,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
        )

        # print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")