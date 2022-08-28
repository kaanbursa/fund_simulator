import torch
from agents.Base.deepagents import AgentDDPG #, #AgentSAC, AgentTD3, AgentA2C
from agents.agents.ppo import AgentPPO
from agents.agents.rec_ppo import AgentRecurrentPPO, AgentSharedRecurrentPPO
from agents.Base.run import Arguments, train_and_evaluate
from agents.wrappers.ObservationWrapper import NormalizedEnv
from torch.utils.tensorboard import SummaryWriter
from utils.pbt import sample_own_ppo_params, sample_sac_params
import time
from utils.helper_training import *
from data.preprocessing import *
from env.BaseEnv import EnvConfig
from utils.indicators import indicator_list, indicators_stock_stats

from agents.Base.config import recurrent_config
MODELS = {'shared_recurrent_ppo':AgentSharedRecurrentPPO,
          'recurrent_ppo':AgentRecurrentPPO, 'PPO':AgentPPO} # OLD MOdels do not work
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo", "a2c"]
"""MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}"""

class TrainerConfig:
    TRAINED_MODEL_DIR = "trained_models"
    rebalance_window = 63
    validation_window = 63
    pretrain_window = 365
    start_date = "2010-01-01"
    start_trade = "2016-01-02"
    end_date = datetime.now()  # datetime.strftime(datetime.now(), 'yyyy-mm-dd')
    indicator_list = indicator_list
    indicators_stock_stats = indicators_stock_stats
    timesteps = 50000
    policy_kwargs = {}
    use_turbulance = False
    normalize_env = True
    normalize_reward = True
    clip_obs = 1
    population = 1
    gamma = 0.99
    hparams  = {
        'learning_rate':0.002,
        'batch_size':8,
        'gamma':0.95,
        'seed':42,
        'net_dimension':160,
        'target_step':160,
        'eval_time_gap':10
    }
    buffer_params = {

    }

    #Paths
    stocks_file_path = './data/all_stocks.csv'
    all_stocks_path= 'data/all_stocks_price2.csv'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class DRLAgent:
    """Provides implementations for DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, model_name, model_type,
                 env_train, env_val, env_trade, data, model_kwargs, train_config= TrainerConfig, config=EnvConfig):
        self.env_train = env_train
        self.env_val = env_val
        self.env_trade = env_trade
        self.data = data
        self.end_date = train_config.end_date
        self.config = train_config
        self.env_config = config
        self.unique_trade_date = data[data.Date > self.config.start_trade].Date.unique()
        self.model_name = model_name
        self.population = train_config.population
        self.agent = MODELS[model_type](recurrent_config)

        self.dataset_version = train_config.dataset_version
        self.config.hparams = model_kwargs
        self.model_type = model_type
        if not os.path.exists("./outputs"):
            os.makedirs("./outputs")
        with open("outputs/runids.txt", "r") as f:
            self.runid = int(f.readline())

        if model_type not in MODELS:
            raise NotImplementedError("NotImplementedError")


    def build_envs(self, train_data, val_data, i):
        stock_dim = len(self.data.ticker.unique())
        env_config_train = {
            "df": train_data,
            "index_df": train_data,
            "stock_dim": stock_dim,
            "config": self.env_config,
        }

        env_config_val = {
            "df": val_data,
            "index_df": val_data,
            "stock_dim": stock_dim,
            "config": self.env_config,
            "iteration" : i,
        }

        #train_env_vec = VecNormalize(DummyVecEnv([lambda: self.env_train(**env_config_train)]))
        #val_env_vec = VecNormalize(DummyVecEnv([lambda: self.env_val(**env_config_val)]))
        if self.config.normalize_env:
            train_env_vec = NormalizedEnv(self.env_train(**env_config_train))
            val_env_vec = NormalizedEnv(self.env_val(**env_config_val))
        else:
            train_env_vec = self.env_train(**env_config_train)
            val_env_vec = self.env_val(**env_config_val)
        return train_env_vec, val_env_vec

    def get_model(self, model_kwargs, train_data, val_data, i):
        env_train, env_val = self.build_envs(train_data, val_data, i)
        env_train.env_num = 1

        env_val.env_num = 1

        model = Arguments(env_train, env_val, self.agent)
        if self.model_name in OFF_POLICY_MODELS:
            model.if_off_policy = True
        else:
            model.if_off_policy = False

        if model_kwargs is not None:
            print(model_kwargs)
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.net_dim = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_time_gap"]
                model.episode = model_kwargs["episode"]
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def exploit_and_explore(
        self, hyperparam_names, perturb_factors=[1.2, 0.8]
    ):
        """Copy parameters from the better model and the hyperparameters
        and running averages from the corresponding optimizer."""
        #study = optuna.create_study()
        #trial = study.ask()

        sampler = {
            'PPO':sample_own_ppo_params(hyperparam_names),
            'RECURRENT_PPO': sample_own_ppo_params(hyperparam_names),
            #'SAC':sample_sac_params(hyperparam_names)
        }

        new_hparams = sampler[self.model_type.upper()]

        return new_hparams

    def _save_model_info(self, hparams, reward, start, end):
        if (hparams['learning_rate'], dict):
            hparams['learning_rate_start'] = hparams['learning_rate']['start']
            hparams['learning_rate_end'] = hparams['learning_rate']['end']
            del hparams['learning_rate']

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
            'date_training':datetime.now(),
            "runid":self.runid
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

    def summary_write(self, end_total_asset, all_assets_period) -> None:
        writter = SummaryWriter(comment=self.dataset_version + '-runid-' +str(self.runid))

        writter.add_text('Indicators', ', '.join(indicator_list))
        stocks = self.data.ticker.unique()
        writter.add_text('Stocks', ', '.join(stocks))
        period = self.config.start_date + '-' + self.config.end_date.strftime("%Y/%m/%d")
        period_trade = self.config.start_trade + '-' + self.config.end_date.strftime("%Y/%m/%d")
        writter.add_text('Period', 'Training period : ' + period + 'Trading period : ' + period_trade)

        writter.add_text('Trainer Config', ', '.join(
            [str(k + ':' + str(v) + ' | ') for k, v in self.config.__dict__.items() if '__' not in k]))

        writter.add_text('Environment Config', ', '.join(
            [str(k + ':' + str(v) + ' | ') for k, v in self.env_config.__dict__.items() if '__' not in k]))
        print('Writing hparams: ', self.config.hparams)
        try:
            writter.add_hparams(self.config.hparams,
                                {'hparam/end_total_asset': end_total_asset},
                                run_name=period_trade)
        except  Exception as e:
            print(e)
            print('Could not write the haparms', self.config.hparams, 'with end total asset of ', end_total_asset)

        for i,asset in enumerate(all_assets_period[1:]):
            writter.add_scalar('Asset over time', asset, i)

    def _increament_run_id(self) -> None:
        with open("outputs/runids.txt", "w") as f:
            self.runid += 1
            f.write(str(self.runid))

    def shannons_demon(self, pct):
        def check_diff(asset1, asset2, n):
            diff = abs(asset1 - asset2)
            if diff >= (asset2 * n):
                return True
            else:
                return False

        trade_data = data_split(
            self.data,
            start=self.unique_trade_date[0],
            end=self.unique_trade_date[-1],
        )
        stocks = self.data.ticker.dropna().unique()
        stock_dimension = len(stocks)
        trade_data.Date = pd.to_datetime(trade_data.Date)
        dolar = 10000
        print(trade_data)
        last_state = [dolar, trade_data.loc[0,'adjcp'],0]
        trade_env = self.env_trade(trade_data,
                                   trade_data,
                                   time_window=0,
                                   flag_days=[],
                                   stock_dim=stock_dimension,
                                   unique_trade_date=self.unique_trade_date,
                                   turbulence_threshold=250,
                                   initial=True,
                                   config=self.env_config,
                                   previous_state=last_state,
                                   model_name=self.model_name,
                                   iteration=1,
                                   debug=False)

        state = trade_env.reset()
        done = False
        i = 0
        df = pd.DataFrame()
        while not done:
            current_dolar = state[0]
            asset = state[2]
            price_of_asset = state[1]
            dolar_val_asset = asset * price_of_asset


            # Checks if the difference between assets are bigger than 5%
            if check_diff(current_dolar, dolar_val_asset, pct):
                if dolar_val_asset > current_dolar:
                    dolar_to_sell_from_asset = abs(current_dolar - dolar_val_asset)
                    quantity_asset_to_sell = -1* (dolar_to_sell_from_asset / price_of_asset)
                    action = np.array([quantity_asset_to_sell])
                else:
                    dolar_to_buy = abs(current_dolar - dolar_val_asset)
                    quantity_asset_to_buy = dolar_to_buy / price_of_asset
                    action = np.array([quantity_asset_to_buy])

            else:
                action = np.array([0])

            new_state, reward, terminal, _ = trade_env.step(action)

            if terminal:
                done = True

            state = new_state
            i +=1
            if i % 100 == 0:
                dc ={'date':self.unique_trade_date[i], 'assets':dolar_val_asset + current_dolar}
                print('Dolar value of assets', dolar_val_asset + current_dolar)
                df = df.append(dc, ignore_index=True)

        return df



    def run_pbt_prediction(self, total_timesteps=30000, time_frame=0, load=False,  model_to_load='', normalize: bool = False):
        start = time.time()
        print(f"======Training Agents with the population of {self.population}========")
        model_name = self.model_name + '-' + str(self.runid)
        self._increament_run_id()
        for j in range(self.population):
            previous_best_end_total_asset = 0
            self.trade_all_assets = [self.env_config.INITIAL_ACCOUNT_BALANCE]
            self.model_name = model_name + '-agent-'  + str(j)
            self.last_state = []
            check_for_progress = 0
            try:

                for i in range(
                        self.config.rebalance_window + self.config.validation_window + time_frame,
                        len(self.unique_trade_date),
                        self.config.rebalance_window,
                ):
                    print("============================================")
                    print('Agent: ', j, '/', self.population)


                    ## initial state is empty
                    if i - self.config.rebalance_window - self.config.validation_window - time_frame == 0:
                        # inital state
                        initial = True
                    else:
                        # previous state
                        initial = False

                    # turbulence_threshold= 1
                    stocks = self.data[self.data.Date == self.unique_trade_date[i]].ticker.dropna().unique()
                    stock_dimension = len(stocks)

                    train = data_split(
                        self.data,
                        start=self.config.start_date,
                        end=self.unique_trade_date[
                            i - self.config.rebalance_window - self.config.validation_window - time_frame
                            ],
                    )

                    ## validation env
                    validation = data_split(
                        self.data,
                        start=self.unique_trade_date[
                            i - self.config.rebalance_window - self.config.validation_window - time_frame
                            ],
                        end=self.unique_trade_date[i - self.config.rebalance_window],
                    )

                    model = self.get_model(self.config.hparams, train, validation, i)

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

                    hparams = self.config.hparams

                    total_reward = 0
                    model.cwd = './trained_models'
                    model.break_step = total_timesteps
                    # Add loss function tracker to train and eval function
                    reward_avg, reward_max = train_and_evaluate(model)
                    print(
                        f"======{self.model_name} Validation from: ",
                        validation['Date'].iloc[0],
                        "to ",
                        validation['Date'].iloc[-1],
                    )
                    winner_hparams = self.config.hparams

                    print("Max reward at validation for Reccurent PPO", reward_max)

                    model_ensemble = model.agent

                    ############## Training and Validation ends ##############

                    ############## Trading starts ##############
                    if hasattr(model_ensemble, 'updated_times'):
                        print('Model updated: ', model_ensemble.updated_times, ' times')
                    print(
                        "======Trading from: ",
                        self.unique_trade_date[i - self.config.rebalance_window],
                        "to ",
                        self.unique_trade_date[i],
                    )
                    # print("Used Model: ", model_ensemble)
                    trade_data = data_split(
                        self.data,
                        start=self.unique_trade_date[i - self.config.rebalance_window],
                        end=self.unique_trade_date[i],
                    )
                    trade_data.Date = pd.to_datetime(trade_data.Date)
                    trade_env = self.env_trade(trade_data,
                                trade_data,
                                time_window=time_frame,
                                flag_days=[],
                                stock_dim=stock_dimension,
                                unique_trade_date=self.unique_trade_date,
                                turbulence_threshold=250,
                                initial=initial,
                                config=self.env_config,
                                previous_state=self.last_state,
                                model_name=self.model_name,
                                iteration=i,
                                debug=False)
                    if self.config.normalize_env:
                        trade_env = NormalizedEnv(trade_env)

                    net_dimension = model.net_dim

                    all_assets_period, self.last_state = self.DRL_prediction(
                        model=model_ensemble,
                        env_trade=trade_env,
                        cwd='./trained_models',
                        net_dimension=net_dimension
                    )
                    self.trade_all_assets.extend(all_assets_period)
                    end_total_asset = all_assets_period[-1]
                    if end_total_asset > previous_best_end_total_asset:
                        previous_best_end_total_asset = end_total_asset
                        winner_hparams = self.config.hparams

                    if check_for_progress > int((len(self.unique_trade_date) //  (self.config.rebalance_window + self.config.validation_window)) / 2):

                        # If it has traded for more than half of the trading periods and still not passed buy and hold next hparams

                        bnh = self.calculate_buy_and_hold(iter=i)
                        if (bnh * 0.99) > end_total_asset:
                            print('Buy and hold bigger than performances')
                            break
                    check_for_progress += 1


                    # print("============Trading Done============")
                    ############## Trading ends ##############
            except Exception as e:
                print('Error: ', e)

                self.config.hparams = self.exploit_and_explore(hyperparam_names=self.config.hparams)

            # TO SUMMARY WRITER LIST OF THINGS TO ADD
            # MODEL VERSIONING (HParams)
            # DATASET VERSIONING (Name of stocks, Indicators)
            
            self.summary_write(end_total_asset, self.trade_all_assets)
            self.trade_all_assets = [self.env_config.INITIAL_ACCOUNT_BALANCE]
            self.config.hparams = self.exploit_and_explore(hyperparam_names=self.config.hparams)
            """self._save_model_info(
                                    winner_hparams,
                                    previous_best_end_total_asset,
                                    self.start_trading_date,
                                    self.end_date
                )"""

        print('Best hyperparameters are ', winner_hparams)
        end = time.time()

        print("Population Based Strategy took: ", (end - start) / 60, " minutes")

    def calculate_buy_and_hold(self, iter) -> int:
        """
        Calculate buy and hold value for the stock given window
        :param iter:
        :return: (int) Last portfolio value
        """
        print('Calculating buy and hold')
        start_trade_date = self.config.start_trade
        today = self.unique_trade_date[iter]

        data = data_split(self.data, start_trade_date, today)

        stock_dim = len(data.ticker.unique())
        account_balance = float(self.env_config.INITIAL_ACCOUNT_BALANCE) / stock_dim
        balances = np.array([account_balance] * stock_dim)
        first_prices = np.array(data.loc[0 , 'adjcp'].replace(0, 1).values.tolist())

        dates_prices = np.array(data.loc[int(len(data) / stock_dim) -1, 'adjcp'].replace(0, 1).values.tolist())  # todays price

        pct_changes = (dates_prices - first_prices) / first_prices
        portfolio_value = sum(balances + (balances * pct_changes))
        print('Buy and Hold value is ', portfolio_value)
        return portfolio_value

    def run_prediction(self, total_timesteps=30000, time_frame=0, load=False,  model_to_load='', normalize: bool = False):
        start = time.time()
        self.last_state = []
        self._increament_run_id()
        self.trade_all_assets = [self.env_config.INITIAL_ACCOUNT_BALANCE]

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

            # turbulence_threshold= 1
            stocks = self.data[self.data.Date == self.unique_trade_date[i]].ticker.dropna().unique()
            stock_dimension = len(stocks)

            train = data_split(
                self.data,
                start=self.config.start_date,
                end=self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window - time_frame
                    ],
            )

            ## validation env
            validation = data_split(
                self.data,
                start=self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window - time_frame
                    ],
                end=self.unique_trade_date[i - self.config.rebalance_window],
            )



            model = self.get_model(self.config.hparams, train, validation, i)

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

            seed = self.config.hparams['seed']
            hparams = self.config.hparams
            hparams['seed'] = seed

            period = validation['Date'].iloc[0] + ' ' + validation['Date'].iloc[-1]

            # model_rec_ppo = self.train_model(env_train, hparams, timesteps=timesteps, load=load)
            # study = optuna.create_study()
            # study.optimize(self.optimize_train, n_trials=100)
            total_reward = 0
            model.cwd = './trained_models'
            model.break_step = total_timesteps
            reward_avg, reward_max = train_and_evaluate(model)
            print(
                f"======{self.model_name} Validation from: ",
                validation['Date'].iloc[0],
                "to ",
                validation['Date'].iloc[-1],
            )
            hparams = self.config.hparams

            winner_hparams = hparams

            print(
                "======Recurrent PPO Validation from: ",
                self.unique_trade_date[
                    i - self.config.rebalance_window - self.config.validation_window - time_frame
                    ],
                "to ",
                self.unique_trade_date[i - self.config.rebalance_window],
            )

            print("Total reward at validation for PPO", total_reward)

            """self._save_model_info(
                winner_hparams,
                reward,
                self.unique_trade_date[i - self.config.rebalance_window - self.config.validation_window - time_frame],
                self.unique_trade_date[i - self.config.rebalance_window]
            )"""

            print('=' * 80)
            print('Best params, ', winner_hparams)
            print('=' * 80)
            self.config.hparams = winner_hparams

            # ppo_sharpe_list.append(sharpe_ppo)

            # Model Selection based on sharpe ratio
            # if (sharpe_ppo >= sharpe_a2c):

            model_ensemble = model.agent

            ############## Training and Validation ends ##############

            ############## Trading starts ##############

            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.config.rebalance_window],
                "to ",
                self.unique_trade_date[i],
            )
            # print("Used Model: ", model_ensemble)
            trade_data = data_split(
                self.data,
                start=self.unique_trade_date[i - self.config.rebalance_window],
                end=self.unique_trade_date[i],
            )
            trade_data.Date = pd.to_datetime(trade_data.Date)
            trade_env = self.env_trade(trade_data,
                        trade_data,
                        time_window=time_frame,
                        flag_days=[],
                        stock_dim=stock_dimension,
                        unique_trade_date=self.unique_trade_date,
                        turbulence_threshold=250,
                        initial=initial,
                        config=self.env_config,
                        previous_state=self.last_state,
                        model_name=self.model_name,
                        iteration=i,
                        debug=False)
            if self.config.normalize_env:
                trade_env = NormalizedEnv(trade_env)

            net_dimension = model.net_dim

            episode_assets, self.last_state = self.DRL_prediction(
                model=model_ensemble,
                env_trade=trade_env,
                cwd='./trained_models',
                net_dimension=net_dimension
            )
            self.trade_all_assets.extend(episode_assets)
            bnh = self.calculate_buy_and_hold(iter=i)

            self.summary_write(episode_assets[-1], all_assets_period=episode_assets)


            # print("============Trading Done============")
            ############## Trading ends ##############

        end = time.time()
        self.trade_all_assets = [self.env_config.INITIAL_ACCOUNT_BALANCE]
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_and_evaluate(model)

    @staticmethod
    def DRL_prediction(model, cwd, net_dimension, env_trade):
        # set trade environment

        env_trade.env_num = 1
        args = Arguments(env_train=env_trade, env_val=env_trade, agent=model)

        args.agent = model
        args.env = env_trade
        # args.agent.if_use_cri_target = True  ##Not needed for test

        # load agent
        try:
            state_dim = env_trade.state_dim
            action_dim = env_trade.action_dim

            agent = args.agent
            net_dim = net_dimension

            agent.init(net_dim, state_dim, action_dim)

            agent.save_or_load_agent(cwd=cwd, if_save=False)
            act = agent.act
            device = agent.device

        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state = env_trade.reset()


        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()

        episode_total_assets.append(env_trade.initial_total_asset)
        last_state = []
        done = False
        if agent.is_recurrent:
            hidden_state = act.init_recurent_cell_states(1, device=device)
            with _torch.no_grad():
                while not done:
                    s_tensor = _torch.as_tensor((state,), device=device).float()

                    a_tensor,_, hidden_state = act(s_tensor, hidden_state,
                                                 sequence_length=1)  # action_tanh = act.forward()

                    action = (
                        a_tensor.detach().cpu().numpy()[0]
                    )  # not need detach(), because with torch.no_grad() outside


                    state, reward, done, _ = env_trade.step(action)

                    total_asset = env_trade.total_asset

                    episode_total_assets.append(total_asset)
                    episode_return = total_asset / env_trade.initial_total_asset
                    episode_returns.append(episode_return)
                    if done:
                        last_state, _ = env_trade.render()
                        break
        else:
            with _torch.no_grad():
                while not done:
                    s_tensor = _torch.as_tensor((state,), device=device).float()

                    a_tensor = act(s_tensor)  # action_tanh = act.forward()

                    action = (
                        a_tensor.detach().cpu().numpy()[0]
                    )  # not need detach(), because with torch.no_grad() outside

                    state, reward, done, _ = env_trade.step(action)

                    total_asset = env_trade.total_asset

                    episode_total_assets.append(total_asset)
                    episode_return = total_asset / env_trade.initial_total_asset
                    episode_returns.append(episode_return)
                    if done:

                        last_state, _ = env_trade.render()
                        break
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return %", round(episode_return, 4) - 1)

        return episode_total_assets, last_state





