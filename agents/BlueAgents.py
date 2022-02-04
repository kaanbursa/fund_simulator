import torch
from agents.Base.BaseAgent import AgentDDPG, AgentPPO#, #AgentSAC, AgentTD3, AgentA2C
from agents.Base.run import Arguments, train_and_evaluate
from agents.wrappers.ObservationWrapper import NormalizedEnv

from utils.pbt import sample_own_ppo_params, sample_sac_params
import time
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from utils.helper_training import *
from data.preprocessing import *
from env.BaseEnv import EnvConfig
from utils.indicators import indicator_list, indicators_stock_stats
MODELS = {"ddpg": AgentDDPG,  "ppo": AgentPPO}
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
    clip_obs = 1
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

    def __init__(self, model_name, model_type, env_train, env_val, env_trade, data, config):
        self.env_train = env_train
        self.env_val = env_val
        self.env_trade = env_trade
        self.data = data
        self.config = TrainerConfig
        self.env_config = EnvConfig
        self.unique_trade_date = data[data.Date > self.config.start_trade].Date.unique()
        self.model_name = model_name
        self.population = 1
        self.agent = MODELS[model_type]()

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
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.net_dim = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_time_gap"]
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
            'SAC':sample_sac_params(hyperparam_names)
        }

        new_hparams = sampler[self.model_type]

        return new_hparams

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

    def run_prediction(self, model_kwargs, total_timesteps=30000, time_frame=0, load=False,  model_to_load='', normalize: bool = False):
        start = time.time()
        self.last_state = []
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



            model = self.get_model(model_kwargs, train, validation, i)

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
                if self.population > 1:  # If Population based training is being used
                    # hparams["learning_rate"] = sched_LR.value
                    try:

                        # TODO: Try different seasons for validations pick the top reward
                        total_reward = 0
                        model = self.get_model(model_kwargs, train, validation, i)
                        model.cwd = f'./trained_models/{self.model_name}-citizen{agent}'
                        model.break_step = total_timesteps
                        model.target_step = total_timesteps
                        reward_avg, reward_max = train_and_evaluate(model)
                        print(
                            f"======{self.model_name} Validation from: ",
                            validation['Date'].iloc[0],
                            "to ",
                            validation['Date'].iloc[-1],
                        )



                        if reward_avg > reward:
                            print(
                                f"Agent #{agent} has better performance for the training period with total reward: {total_reward}")
                            reward = total_reward
                            winner = model
                            winner_hparams = hparams
                        # ====================================
                        model_kwargs = self.exploit_and_explore(hyperparam_names=model_kwargs) #TODO: Change to model kwargs
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

                    print("Total reward at validation for Reccurent PPO", total_reward)

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
            # TODO: load the model
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

            end_total_asset, self.last_state = self.DRL_prediction(
                model=model_ensemble,
                env_trade=trade_env,
                cwd='./trained_models',
                net_dimension=net_dimension #TODO: find a matching way
            )
            # print("============Trading Done============")
            ############## Trading ends ##############

        end = time.time()

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

                    last_state,_ = env_trade.render()
                    break
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return %", round(episode_return, 4) - 1)

        return episode_total_assets, last_state


class Trainer:
    """
    Class for training the agent build by the team
    """
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
        self.model_type = model
        self.policy = policy
        self.model = model
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
    def get_model(self):
        model = Arguments(env, agent)
        if model_name in OFF_POLICY_MODELS:
            model.if_off_policy = True
        else:
            model.if_off_policy = False

        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.net_dim = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_time_gap"]
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, env_train, hparams, timesteps=50000, load=False, model_to_load=''):
        start = time.time()
        model_path = f"{self.config.TRAINED_MODEL_DIR}/{self.model_name}"

        end = time.time()
        model.save(model_path)
        print(
            "Training time ", self.model_name, ": ", (end - start) / 60, " minutes"
        )
        return model

