from typing import Any, Dict, Union, Callable
from datetime import datetime
import numpy as np
import optuna
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def cosine_decay_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def suggest_categorical(name, list):

    return np.random.choice(list)

def suggest_loguniform(name, low, high):
    return np.exp(np.random.uniform(low, high, 1))[0]

def suggest_uniform(name, low, high):
    return np.random.uniform(low, high, 1)[0]

def sample_ppo_params(hparams) -> Dict[str, Any]:
    """

    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    now = datetime.now()
    np.random.seed(now.second + now.microsecond)

    episode = int(suggest_uniform('episode', 1, 15))
    seed = suggest_categorical('seed', [31, 69, 42069])
    batch_size = suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    eval_time_gap = suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = round(suggest_uniform("learning_rate", 0.0001, 0.05), 5)
    target_step = int(suggest_uniform('target_step', 5000, 90000))
    net_dimension = suggest_categorical('net_dimension', [100, 128, 256, 512, 1026])
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = suggest_uniform("ent_coef", 0.000001, 0.1)
    clip_range = suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1])# 2, 5
    vf_coef = suggest_uniform("vf_coef", 0, 1)
    # Uncomment for gSDE (continuous actions)
    # log_std_init = suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = suggest_categorical('ortho_init', [False, True])
    # activation_fn = suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = suggest_categorical("activation_fn", ["tanh", "relu"])
    policy_kwargs = suggest_categorical('Policy',
        [{'net_arch': [{'pi': [256, 256], 'vf': [256, 256]}]},
                                        {'net_arch': [{'pi': [512, 256], 'vf': [ 128]}]},
    {'net_arch': [{'pi': [128, 128], 'vf': [64]}]},
                                        {'net_arch': [{'pi': [512, 256, 128], 'vf': [64, 64]}]},
                                        {'net_arch': [{'pi': [256, 128], 'vf': [256, 256]}]}]
    )



    # TODO: account when using multiple envs
    if batch_size > eval_time_gap:
        batch_size = eval_time_gap

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images

    #activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    new_hparams = {
        #"eval_time_gap": eval_time_gap,
        "batch_size": batch_size,
        "ent_coef":ent_coef,
        "gae_lambda":gae_lambda,
        "max_grad_norm":max_grad_norm,
        "vf_coef":vf_coef,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "clip_range": clip_range,
        #"net_dimension":net_dimension,
        "policy_kwargs": policy_kwargs,
        "seed":seed
    }

    return new_hparams


def sample_own_ppo_params(hparams) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    now = datetime.now()
    np.random.seed(now.second + now.microsecond)

    episode = int(suggest_uniform('episode', 1, 15))
    seed = suggest_categorical('seed', [31, 69, 42069])
    batch_size = suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    eval_time_gap = suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])

    target_step = int(suggest_uniform('target_step', 5000, 90000))
    net_dimension = suggest_categorical('net_dimension', [100, 128, 256, 512, 1026])
    lr_schedule  = suggest_categorical('lr_decay', ['stable','polynomial'])
    if lr_schedule == 'stable':
        learning_rate = round(suggest_uniform("learning_rate", 0.0001, 0.05), 5)
    else:
        learning_rate_start = round(suggest_uniform("learning_rate", 0.0011, 0.05), 5)
        learning_rate_end = round(suggest_uniform("learning_rate", 0.0001, 0.001), 5)
        assert learning_rate_start > learning_rate_end, 'Learning rate starting is bigger than ending'
        learning_rate = {
            'start':learning_rate_start,
            'end':learning_rate_end
        }

    # Uncomment to enable learning rate schedule
    # lr_schedule = suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = suggest_uniform("ent_coef", 0.000001, 0.1)
    clip_range = suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1])# 2, 5
    vf_coef = suggest_uniform("vf_coef", 0, 1)
    # Uncomment for gSDE (continuous actions)
    # log_std_init = suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = suggest_categorical('ortho_init', [False, True])
    # activation_fn = suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > eval_time_gap:
        batch_size = eval_time_gap

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images

    #activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    new_hparams = {
        "eval_time_gap": eval_time_gap,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "target_step": target_step,
        "net_dimension":net_dimension,
        "eval_time_gap":eval_time_gap,
        "episode": episode,
        "seed":seed
    }

    return new_hparams


def sample_sb_recurrent(hparams) -> Dict[str, Any]:
    """

    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    now = datetime.now()
    np.random.seed(now.second + now.microsecond)

    episode = int(suggest_uniform('episode', 1, 15))
    seed = 66#suggest_categorical('seed', [31, 69, 42069])
    batch_size = suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    activation_fn  = suggest_categorical("activation_fn ", [nn.modules.activation.Tanh,
                                                            nn.modules.activation.ReLU,
                                                            nn.modules.activation.Hardswish])
    learning_rate = cosine_decay_schedule(1)
    net_arch_list = [[dict(pi=[256, 128, 64], vf=[64])],
                [dict(pi=[256, 128, 64], vf=[256, 128])],
                [dict(pi=[256, 64], vf=[256, 128])],
                [dict(pi=[512, 128, 64], vf=[256, 128])],
                [dict(pi=[256, 128], vf=[256, 128])]]
    net_arch_int  = suggest_categorical("net_arch ", list(np.arange(0,len(net_arch_list)-1)))
    net_arch = net_arch_list[net_arch_int]

    target_step = int(suggest_uniform('target_step', 5000, 90000))
    lstm_hidden_size = suggest_categorical('net_dimension', [64, 128, 256,])
    n_lstm_layers   = suggest_categorical('n_lstm_layers ', [1,2])
    shared_lstm = suggest_categorical('shared_lstm ',[False, True])


    # Uncomment to enable learning rate schedule
    # lr_schedule = suggest_categorical('lr_schedule', ['linear', 'constant'])

    #============= hparams|
    eval_time_gap = suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = round(suggest_uniform("learning_rate", 0.0001, 0.05), 5)
    target_step = int(suggest_uniform('target_step', 5000, 90000))
    net_dimension = suggest_categorical('net_dimension', [100, 128, 256, 512, 1026])
    # Uncomment to enable learning rate schedule
    # lr_schedule = suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = suggest_uniform("ent_coef", 0.000001, 0.1)
    clip_range = suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # 2, 5
    vf_coef = suggest_uniform("vf_coef", 0, 1)



    # Independent networks usually work best
    # when not working with images

    #activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    policy_kwargs  = {
        "net_arch": net_arch,

        "activation_fn": activation_fn,
        "lstm_hidden_size":lstm_hidden_size,
        "n_lstm_layers":n_lstm_layers,
        "shared_lstm":True
    }
    new_hparams = {
        # "eval_time_gap": eval_time_gap,
        "batch_size": batch_size,
        "ent_coef": ent_coef,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "clip_range": clip_range,
        # "net_dimension":net_dimension,
        "policy_kwargs": policy_kwargs,
        "seed": seed

    }

    return new_hparams

def sample_recurrent_ppo_params(hparams) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    now = datetime.now()
    np.random.seed(now.second + now.microsecond)
    batch_size = suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    n_steps = suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])

    learning_rate = suggest_uniform("learning_rate", 0.0001, 0.05)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = suggest_uniform("ent_coef", 0.000001, 0.1)
    clip_range = suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1])# 2, 5
    vf_coef = suggest_uniform("vf_coef", 0, 1)
    net_arch = suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = suggest_categorical('ortho_init', [False, True])
    # activation_fn = suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

def sample_sac_params(hparams) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """
    now = datetime.now()
    np.random.seed(now.second + now.microsecond)
    gamma = suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = suggest_uniform("learning_rate", 0.04, 1e-5)
    batch_size = suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024])
    buffer_size = suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    learning_starts = suggest_categorical("learning_starts", [0, 1000, 10000, 20000])
    # train_freq = suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    # Polyak coeff
    tau = suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    # gradient_steps takes too much time
    # gradient_steps = suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    # ent_coef = suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    # You can comment that out when not using gSDE
    log_std_init = suggest_uniform("log_std_init", -4, 1)
    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "large": [256, 256, 256],
        # "verybig": [512, 512, 512],
    }[net_arch]

    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = suggest_uniform('target_entropy', -10, 10)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        #"train_freq": (train_freq, 'step'),
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
    }


    return hyperparams


def sample_ppo(hparam: Dict) -> Dict[str, Any]:
    """
    mutate winner hparams
    :param hparam:
    :return:
    """
    ppo_hparam_list = list(hparam.keys())
    hyper_parameter = np.random.choice(ppo_hparam_list)
    categorical = [
        "gamma",
        "cliprange",
        "n_epochs",
        "gae_lambda",
        "max_grad_norm",
        "activation_fn",
    ]
    loguniform = ["ent_coef", "vf_coef"]
    possiblity_dict = {
        "gamma": [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        "learning_rate": [1e-5, 1],
        "ent_coef": [0.00000001, 0.1],
        "cliprange": [0.1, 0.2, 0.3, 0.4],
        "gae_lambda": [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
        "max_grad_norm": [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5],
        "vf_coef": [0.5, 1],
        "activation_fn": ["tanh", "relu"],
    }
    print(hyper_parameter, " is changed")
    if hyper_parameter in categorical:

        hparam[hyper_parameter] = suggest_categorical(
            hyper_parameter, possiblity_dict[hyper_parameter]
        )
    elif hyper_parameter in loguniform:
        hparam[hyper_parameter] = round(
            suggest_loguniform(hyper_parameter, possiblity_dict[hyper_parameter]), 3
        )
    else:
        hparam[hyper_parameter] = round(
            suggest_uniform(hyper_parameter, possiblity_dict[hyper_parameter]), 3
        )
    """#batch_size = suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    #n_steps = suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    learning_rate = suggest_loguniform("lr", 1e-5, 1)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    #n_epochs = suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = suggest_uniform("vf_coef", 0, 1)
    #net_arch = suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = suggest_categorical('ortho_init', [False, True])
    # activation_fn = suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    """

    # TODO: account when using multiple envs
    # if batch_size > n_steps:
    #    batch_size = n_steps

    # if lr_schedule == "linear":
    #    learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    """net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]"""

    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return hparam
