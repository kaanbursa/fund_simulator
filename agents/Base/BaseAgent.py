import os
import torch
import numpy as np
import numpy.random as rd
from tqdm import tqdm
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from torch import nn
from agents.Base.network import QNet, QNetDuel
from agents.Base.network import Actor, ActorPPO
from agents.Base.network import Critic, CriticPPO


class AgentBase:
    def __init__(self, _net_dim=256, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        """initialize
        replace by different DRL algorithms
        explict call self.init() for multiprocessing.
        `net_dim` the dimension of networks (the width of neural networks)
        `state_dim` the dimension of state (the number of state vector)
        `action_dim` the dimension of action (the number of discrete action)
        `learning_rate` learning rate of optimizer
        `if_per_or_gae` PER (off-policy) or GAE (on-policy) for sparse reward
        `env_num` the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        `gpu_id` the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.states = None
        self.device = None
        self.traj_list = None
        self.action_dim = None
        self.if_off_policy = True

        self.env_num = 1
        self.explore_rate = 1.0
        self.explore_noise = 0.1
        self.clip_grad_norm = 4.0
        # self.amp_scale = None  # automatic mixed precision

        '''attribute'''
        self.explore_env = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """initialize the self.object in `__init__()`
        replace by different DRL algorithms
        explict call self.init() for multiprocessing.
        `net_dim` the dimension of networks (the width of neural networks)
        `state_dim` the dimension of state (the number of state vector)
        `action_dim` the dimension of action (the number of discrete action)
        `learning_rate` learning rate of optimizer
        `if_per_or_gae` PER (off-policy) or GAE (on-policy) for sparse reward
        `env_num` the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        `gpu_id` the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.action_dim = action_dim
        # self.amp_scale = torch.cuda.amp.GradScaler()
        self.traj_list = [list() for _ in range(env_num)]
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        print('Using device: ', self.device)


        self.cri = self.ClassCri(int(net_dim * 1.25), state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri

        assert isinstance(if_per_or_gae, bool)

        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_action(self, state: np.ndarray) -> np.ndarray:
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action

    def select_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Select continuous actions for exploration
        `tensor states` states.shape==(batch_size, state_dim, )
        return `tensor actions` actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """
        action = self.act(states.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu()

    def explore_one_env(self, env, target_step, reward_scale, gamma):
        """actor explores in one env, then returns the traj (env transition)
        `object env` RL training environment. env.reset() env.step()
        `int target_step` explored target_step number of step in env
        return `[traj, ...]` for off-policy ReplayBuffer, `traj = [(state, other), ...]`
        """
        state = self.states[0]
        traj = list()
        for _ in range(target_step):

            ten_state = torch.as_tensor(state, dtype=torch.float32)

            ten_action = self.select_actions(ten_state.unsqueeze(0))[0]
            action = ten_action.numpy()
            next_s, reward, done, _ = env.step(action)

            ten_other = torch.empty(2 + self.action_dim)
            ten_other[0] = reward
            ten_other[1] = done
            ten_other[2:] = ten_action
            traj.append((ten_state, ten_other))

            state = env.reset() if done else next_s

        self.states[0] = state

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [(traj_state, traj_other), ]
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step, reward_scale, gamma):
        """actor explores in VectorEnv, then returns the trajectory (env transition)
        `object env` RL training environment. env.reset() env.step()
        `int target_step` explored target_step number of step in env
        return `[traj, ...]` for off-policy ReplayBuffer, `traj = [(state, other), ...]`
        """
        ten_states = self.states

        traj = list()
        for _ in range(target_step):
            ten_actions = self.select_actions(ten_states)
            ten_next_states, ten_rewards, ten_dones = env.step(ten_actions)

            ten_others = torch.cat((ten_rewards.unsqueeze(0),
                                    ten_dones.unsqueeze(0),
                                    ten_actions))
            traj.append((ten_states, ten_others))
            ten_states = ten_next_states

        self.states = ten_states

        # traj = [(env_ten, ...), ...], env_ten = (env1_ten, env2_ten, ...)
        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [(traj_state[:, env_i, :], traj_other[:, env_i, :])
                     for env_i in range(len(self.states))]
        # traj_list = [traj_env_0, ...], traj_env_0 = (ten_state, ten_other)
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ...]

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """update the neural network by sampling batch data from ReplayBuffer
        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning
        `buffer` Experience replay buffer.
        `int batch_size` sample batch_size of data for Stochastic Gradient Descent
        `float repeat_times` the times of sample batch = int(target_step * repeat_times) in off-policy
        `float soft_update_tau` target_net = target_net * (1-tau) + current_net * tau
        `return tuple` training logging. tuple = (float, float, ...)
        """


    def optim_update(self, optimizer, objective, params):
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(params, max_norm=self.clip_grad_norm)
        optimizer.step()

    # def optim_update_amp(self, optimizer, objective):  # automatic mixed precision
    #     # self.amp_scale = torch.cuda.amp.GradScaler()
    #
    #     optimizer.zero_grad()
    #     self.amp_scale.scale(objective).backward()  # loss.backward()
    #     self.amp_scale.unscale_(optimizer)  # amp
    #
    #     # from torch.nn.utils import clip_grad_norm_
    #     # clip_grad_norm_(model.parameters(), max_norm=3.0)  # amp, clip_grad_norm_
    #     self.amp_scale.step(optimizer)  # optimizer.step()
    #     self.amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network
        `nn.Module target_net` target network update via a current network, it is more stable
        `nn.Module current_net` current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        """save or load the training files for agent from disk.
        `str cwd` current working directory, where to save training files.
        `bool if_save` True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:

                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    @staticmethod
    def convert_trajectory(traj_list, reward_scale, gamma):  # off-policy
        for ten_state, ten_other in traj_list:
            ten_other[:, 0] = ten_other[:, 0] * reward_scale  # ten_reward
            ten_other[:, 1] = (1.0 - ten_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma
        return traj_list

    def summary(self):
        raise NotImplementedError


class AgentDQN(AgentBase):
    """
        Deep Q-Network algorithm. “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al.. 2015.

        :param net_dim[int]: the dimension of networks (the width of neural networks)
        :param state_dim[int]: the dimension of state (the number of state vector)
        :param action_dim[int]: the dimension of action (the number of discrete action)
        :param learning_rate[float]: learning rate of optimizer
        :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
        :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self, _net_dim=256, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        AgentBase.__init__(self)
        self.ClassCri = None  # self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        self.if_use_dueling = True  # self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy


    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):

        self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        AgentBase.init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, states) -> np.ndarray:  # for discrete action space
        """
        Select discrete actions given an array of states.

        .. note::
            Using ϵ-greedy to uniformly random select actions for randomness.

        :param states[np.ndarray]: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = torch.randint(self.action_dim, size=(states.shape[0],))  # choosing action randomly
        else:
            action = self.act(states.to(self.device))
            a_int = action.argmax(dim=1)
        return a_int.detach().cpu()

    def explore_one_env(self, env, target_step, reward_scale, gamma) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env[object]: the DRL environment instance.
        :param target_step[int]: the total step for the interaction.
        :param reward_scale[float]: a reward scalar to clip the reward.
        :param gamma[float]: the discount factor.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj = list()
        state = self.states[0]
        for _ in range(target_step):
            ten_state = torch.as_tensor(state, dtype=torch.float32)
            ten_action = self.select_actions(ten_state.unsqueeze(0))[0]
            action = ten_action.numpy()  # isinstance(action, int)
            next_s, reward, done, _ = env.step(action)

            ten_other = torch.empty(2 + 1)
            ten_other[0] = reward
            ten_other[1] = done
            ten_other[2] = ten_action
            traj.append((ten_state, ten_other))

            state = env.reset() if done else next_s
        self.states[0] = state

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [(traj_state, traj_other), ]
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ...]

    def explore_vec_env(self, env, target_step, reward_scale, gamma) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env[object]: the DRL environment instance.
        :param target_step[int]: the total step for the interaction.
        :param reward_scale[float]: a reward scalar to clip the reward.
        :param gamma[float]: the discount factor.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        ten_states = self.states

        traj = list()
        for _ in range(target_step):
            ten_actions = self.select_actions(ten_states)
            ten_next_states, ten_rewards, ten_dones = env.step(ten_actions)

            ten_others = torch.cat((ten_rewards.unsqueeze(0),
                                    ten_dones.unsqueeze(0),
                                    ten_actions.unsqueeze(0)))
            traj.append((ten_states, ten_others))
            ten_states = ten_next_states

        self.states = ten_states

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [(traj_state[:, env_i, :], traj_other[:, env_i, :])
                     for env_i in range(len(self.states))]
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ...]

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times[float]: the re-using times of each trajectory.
        :param soft_update_tau[float]: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()
        obj_critic = q_value = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value


class AgentDDPG(AgentBase):
    """
    Bases: ``elegantrl.agent.AgentBase``

    Deep Deterministic Policy Gradient algorithm. “Continuous control with deep reinforcement learning”. T. Lillicrap et al.. 2015.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self, _net_dim=256, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        AgentBase.__init__(self)
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.3  # explore noise of action (OrnsteinUhlenbeckNoise)
        self.ou_noise = None

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing.
        """

        AgentBase.init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.explore_noise)

        if if_per_or_gae:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per_or_gae else 'mean')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per_or_gae else 'mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_action(self, state: torch.Tensor) -> torch.Tensor:

        """
                Select actions given an array of states.

                .. note::
                    Using ϵ-greedy with Ornstein–Uhlenbeck noise to add noise to actions for randomness.

                :param states[np.ndarray]: an array of states in a shape (batch_size, state_dim, ).
                :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """

        action = self.act(state.to(self.device))
        if rd.rand() < self.explore_rate:
            ou_noise = torch.as_tensor(self.ou_noise(), dtype=torch.float32, device=self.device).unsqueeze(0)
            action = (action + ou_noise).clamp(-1, 1)
        return action.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times[float]: the re-using times of each trajectory.
        :param soft_update_tau[float]: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None

        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic , state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
            action_pg = self.act(state)
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())
            self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item(), obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q

        q_value = self.cri(state, action)
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentPPO(AgentBase):
    """
    Bases: ``agents.agent.AgentBase``

    PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self, _net_dim=256, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        AgentBase.__init__(self)
        self.ClassAct = ActorPPO
        self.ClassCri = CriticPPO

        self.if_off_policy = False
        self.ratio_clip = 0.2  # could be 0.00 ~ 0.50 ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.00~0.10
        self.lambda_a_value = 1.00  # could be 0.25~8.00, the lambda of advantage value
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        AgentBase.init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)

        self.traj_list = [list() for _ in range(env_num)]
        self.env_num = env_num

        if if_per_or_gae:  # if_use_gae
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
                Select action give a state.

                :param states[np.ndarray]: an array of states in a shape (batch_size, state_dim, ).
                :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
                """
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """
            Select actions given an array of states.

            :param states[np.ndarray]: an array of states in a shape (batch_size, state_dim, ).
            :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        state = state.to(self.device)
        action, noise = self.act.get_action(state)
        return action.detach().cpu(), noise.detach().cpu()

    def explore_one_env(self, env, target_step, reward_scale, gamma):
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env[object]: the DRL environment instance.
        :param target_step[int]: the total step for the interaction.
        :param reward_scale[float]: a reward scalar to clip the reward.
        :param gamma[float]: the discount factor.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """

        state = self.states[0]

        last_done = 0
        traj = list()
        step = 0
        while step < target_step:

            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)


            ten_actions, ten_noises = self.select_actions(ten_states)

            action = ten_actions[0].numpy()

            next_s, reward, done, _ = env.step(np.tanh(action)) # Error at 82th step



            traj.append((ten_states, reward, done, ten_actions, ten_noises))
            if done:
                state = env.reset()
                last_done = step
            else:
                state = next_s
                step += 1

        self.states[0] = state

        traj_list = self.splice_trajectory([traj, ], [last_done, ])
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step, reward_scale, gamma):
        """
            Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

            :param env[object]: the DRL environment instance.
            :param target_step[int]: the total step for the interaction.
            :param reward_scale[float]: a reward scalar to clip the reward.
            :param gamma[float]: the discount factor.
            :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        ten_states = self.states

        env_num = len(self.traj_list)
        traj_list = [list() for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        for step_i in tqdm(range(target_step)):

            ten_actions, ten_noises = self.select_actions(ten_states)
            tem_next_states, ten_rewards, ten_dones = env.step(ten_actions.tanh())

            for env_i in range(env_num):
                traj_list[env_i].append((ten_states[env_i], ten_rewards[env_i], ten_dones[env_i],
                                         ten_actions[env_i], ten_noises[env_i]))
                if ten_dones[env_i]:
                    last_done_list[env_i] = step_i

            ten_states = tem_next_states

        self.states = ten_states

        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ...]

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        .. note::
            Using advantage normalization and entropy loss.

        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times[float]: the re-using times of each trajectory.
        :param soft_update_tau[float]: the soft update parameter.
        :return: a tuple of the log information.
        """

        with torch.no_grad():
            #buf_len = buffer[0].shape[0]
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]
            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (self.lambda_a_value / (buf_adv_v.std() + 1e-5))
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None
        
        assert buf_len >= batch_size, 'buf len bigger than batch size'
        update_times = int(buf_len / batch_size * repeat_times)

        for update_i in range(1, update_times + 1):

            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            """ PPO Surrogate objective of Trust Region"""

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len[int]: the length of the ``ReplayBuffer``.
        :param buf_reward[np.array]: a list of rewards for the state-action pairs.
        :param buf_mask[np.array]: a list of masks computed by the product of done signal and discount factor.
        :param buf_value[np.array]: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """

        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - (buf_mask * buf_value[:, 0])
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len[int]: the length of the ``ReplayBuffer``.
        :param buf_reward[np.array]: a list of rewards for the state-action pairs.
        :param buf_mask[np.array]: a list of masks computed by the product of done signal and discount factor.
        :param buf_value[np.array]: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """

        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device) # old policy value
        buf_adv_v = torch.empty(buf_len, dtype=torch.float32, device=self.device) # advantage value

        pre_r_sum = 0
        pre_adv_v = 0

        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_adv_v[i] = ten_reward[i] + ten_mask[i] * pre_adv_v
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv

        return buf_r_sum, buf_adv_v

    def splice_trajectory(self, traj_list, last_done_list):
        for env_i in range(self.env_num):
            last_done = last_done_list[env_i]
            traj_temp = traj_list[env_i]

            traj_list[env_i] = self.traj_list[env_i] + traj_temp[:last_done + 1]
        return traj_list

    def convert_trajectory(self, traj_list, reward_scale, gamma):
        """
        Process the trajectory list, rescale the rewards and calculate the masks.

        :param traj_list[list]: a list of trajectories.
        :param reward_scale[float]: a reward scalar to clip the reward.
        :param gamma[float]: the discount factor.
        :return: a trajectory list.
        """

        for traj in traj_list:
            temp = list(map(list, zip(*traj)))  # 2D-list transpose

            ten_state = torch.stack(temp[0])
            ten_reward = torch.as_tensor(temp[1], dtype=torch.float32) * reward_scale
            ten_mask = (1.0 - torch.as_tensor(temp[2], dtype=torch.float32)) * gamma
            ten_action = torch.stack(temp[3])
            ten_noise = torch.stack(temp[4])

            traj[:] = (ten_state, ten_reward, ten_mask, ten_action, ten_noise)

        return traj_list


class AgentRecurrentPPO(AgentBase):
    """
        Bases: ``agent.AgentBase``

        PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

        :param net_dim[int]: the dimension of networks (the width of neural networks)
        :param state_dim[int]: the dimension of state (the number of state vector)
        :param action_dim[int]: the dimension of action (the number of discrete action)
        :param learning_rate[float]: learning rate of optimizer
        :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
        :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
        """

    def __init__(self, _net_dim=256, hidden_dim=126, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        AgentBase.__init__(self)
        # Shared rec layer
        self.recurrent_layer = nn.LSTM(_state_dim, hidden_dim, batch_first=True)
        self.ClassAct = ActorPPO
        self.ClassCri = CriticPPO

        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant(param, 0)
            elif "weight" in name:
               nn.init.orthogonal_(param, np.sqrt(2))

        self.if_off_policy = False
        self.ratio_clip = 0.2  # could be 0.00 ~ 0.50 ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.00~0.10
        self.lambda_a_value = 1.00  # could be 0.25~8.00, the lambda of advantage value
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        AgentBase.init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)

        self.traj_list = [list() for _ in range(env_num)]
        self.env_num = env_num

        if if_per_or_gae:  # if_use_gae
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env
            
    def select_action(self, state: np.ndarray, hidden_state: np.ndarray) -> np.ndarray:
        """
                Select action give a state.

                :param states[np.ndarray]: an array of states in a shape (batch_size, state_dim, ).
                :param hidden_state[np.ndarray]: an array of states from previous recurrent state (batch_size, net_dim, ).
                :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
                """
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        h_tensor = torch.as_tensor(hidden_state[np.newaxis], device=self.device)

        a_tensor = self.act(s_tensor, h_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action

    def select_actions(self, state: torch.Tensor,  hidden_state: torch.Tensor) -> torch.Tensor:
        """
            Select actions given an array of states.

            :param states[np.ndarray]: an array of states in a shape (batch_size, state_dim, ).
            :param hidden_state[torch.tensor]: an array of states from previous recurrent state (batch_size, net_dim, ).
            :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        state = state.to(self.device)
        hidden_state = hidden_state.to(self.device)
        action, noise = self.act.get_action(state, hidden_state)
        return action.detach().cpu(), noise.detach().cpu()

    def explore_one_env(self, env, target_step, reward_scale, gamma):
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env[object]: the DRL environment instance.
        :param target_step[int]: the total step for the interaction.
        :param reward_scale[float]: a reward scalar to clip the reward.
        :param gamma[float]: the discount factor.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """

        state = self.states[0]

        last_done = 0
        traj = list()
        step = 0
        while step < target_step:

            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

            ten_actions, ten_noises = self.select_actions(ten_states)

            action = ten_actions[0].numpy()

            next_s, reward, done, _ = env.step(np.tanh(action))

            traj.append((ten_states, reward, done, ten_actions, ten_noises))
            if done:
                state = env.reset()
                last_done = step
            else:
                state = next_s
                step += 1

        self.states[0] = state

        traj_list = self.splice_trajectory([traj, ], [last_done, ])
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step, reward_scale, gamma):
        """
            Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

            :param env[object]: the DRL environment instance.
            :param target_step[int]: the total step for the interaction.
            :param reward_scale[float]: a reward scalar to clip the reward.
            :param gamma[float]: the discount factor.
            :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        ten_states = self.states

        env_num = len(self.traj_list)
        traj_list = [list() for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        for step_i in tqdm(range(target_step)):

            ten_actions, ten_noises = self.select_actions(ten_states)
            tem_next_states, ten_rewards, ten_dones = env.step(ten_actions.tanh())

            for env_i in range(env_num):
                traj_list[env_i].append((ten_states[env_i], ten_rewards[env_i], ten_dones[env_i],
                                         ten_actions[env_i], ten_noises[env_i]))
                if ten_dones[env_i]:
                    last_done_list[env_i] = step_i

            ten_states = tem_next_states

        self.states = ten_states

        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ...]



    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        .. note::
            Using advantage normalization and entropy loss.

        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times[float]: the re-using times of each trajectory.
        :param soft_update_tau[float]: the soft update parameter.
        :return: a tuple of the log information.
        """

        with torch.no_grad():
            # buf_len = buffer[0].shape[0]
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]
            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (self.lambda_a_value / (buf_adv_v.std() + 1e-5))
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None

        assert buf_len >= batch_size, 'buf len bigger than batch size'
        update_times = int(buf_len / batch_size * repeat_times)

        for update_i in range(1, update_times + 1):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            """ PPO Surrogate objective of Trust Region"""

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len[int]: the length of the ``ReplayBuffer``.
        :param buf_reward[np.array]: a list of rewards for the state-action pairs.
        :param buf_mask[np.array]: a list of masks computed by the product of done signal and discount factor.
        :param buf_value[np.array]: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """

        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - (buf_mask * buf_value[:, 0])
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len[int]: the length of the ``ReplayBuffer``.
        :param buf_reward[np.array]: a list of rewards for the state-action pairs.
        :param buf_mask[np.array]: a list of masks computed by the product of done signal and discount factor.
        :param buf_value[np.array]: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """

        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv_v = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0

        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_adv_v[i] = ten_reward[i] + ten_mask[i] * pre_adv_v
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv

        return buf_r_sum, buf_adv_v

    def splice_trajectory(self, traj_list, last_done_list):
        for env_i in range(self.env_num):
            last_done = last_done_list[env_i]
            traj_temp = traj_list[env_i]

            traj_list[env_i] = self.traj_list[env_i] + traj_temp[:last_done + 1]
        return traj_list

    def convert_trajectory(self, traj_list, reward_scale, gamma):
        """
        Process the trajectory list, rescale the rewards and calculate the masks.

        :param traj_list[list]: a list of trajectories.
        :param reward_scale[float]: a reward scalar to clip the reward.
        :param gamma[float]: the discount factor.
        :return: a trajectory list.
        """

        for traj in traj_list:
            temp = list(map(list, zip(*traj)))  # 2D-list transpose

            ten_state = torch.stack(temp[0])
            ten_reward = torch.as_tensor(temp[1], dtype=torch.float32) * reward_scale
            ten_mask = (1.0 - torch.as_tensor(temp[2], dtype=torch.float32)) * gamma
            ten_action = torch.stack(temp[3])
            ten_noise = torch.stack(temp[4])

            traj[:] = (ten_state, ten_reward, ten_mask, ten_action, ten_noise)

        return traj_list









