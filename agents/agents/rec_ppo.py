import os
import torch
import numpy as np
import numpy.random as rd
from tqdm import tqdm
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from agents.Base.network import ShareRecurrentPPO

class AgentRecurrentPPO:
    """

        Recurrent PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

        :param net_dim[int]: the dimension of networks (the width of neural networks)
        :param hidden_dim[int]: hidden dimension of the recurrent layer
        :param state_dim[int]: the dimension of state (the number of state vector)
        :param action_dim[int]: the dimension of action (the number of discrete action)
        :param learning_rate[float]: learning rate of optimizer
        :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
        :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
        """

    def __init__(self, config, _net_dim=256, hidden_dim=126, _state_dim=8, _action_dim=2, sequence_length=16,
                 _learning_rate=1e-4,
                 _if_per_or_gae=False, _gpu_id=0, ):

        self.network = ShareRecurrentPPO(_state_dim, _action_dim, hidden_dim, _net_dim, config)

        self.states = None
        self.device = None
        self.traj_list = None
        self.action_dim = None
        self.if_off_policy = True
        self.is_recurrent = True

        self.hidden_dim = hidden_dim

        self.env_num = 1
        self.explore_rate = 1.0
        self.explore_noise = 0.1
        self.clip_grad_norm = 4.0 # try 0.5
        # self.amp_scale = None  # automatic mixed precision

        '''attribute'''
        self.explore_env = None
        self.get_obj_critic = None
        losses = {'SmoothL1Loss':torch.nn.SmoothL1Loss}
        self.criterion = losses[config['criterion']]


        self.is_recurrent = True
        self.seq_len = sequence_length

        self.if_off_policy = False
        self.ratio_clip = 0.2  # could be 0.00 ~ 0.50 ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.00~0.10
        self.lambda_a_value = 1.00  # could be 0.25~8.00, the lambda of advantage value
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.vf_coef = 1 # Value function coefficient
        self.beta = 0.01 #VF

        # self.traj_list = [list() for _ in range(_env_num)]

        if _if_per_or_gae:  # if_use_gae
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

        self.explore_env = self.explore_one_env



    def optim_update(self, optimizer, objective, params):
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(params, max_norm=self.clip_grad_norm)
        optimizer.step()

    def select_action(self, state: np.ndarray, hidden_state: np.ndarray) -> np.ndarray:
        """
                Select action give a state.

                :param states[np.ndarray]: an array of states in a shape (batch_size, seq_len, state_dim).
                :param hidden_state[np.ndarray]: an array of states from previous recurrent state (batch_size, net_dim, ).
                :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
                """
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        h_tensor = torch.as_tensor(hidden_state[np.newaxis], device=self.device)

        a_tensor, value_tensor, hidden_state, action_noise = self.network(s_tensor, h_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action, value_tensor, hidden_state, action_noise



    def select_actions(self, state: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
            Select actions given an array of states.

            :param states[np.ndarray]: an array of states in a shape (batch_size,  seq_len, state_dim, ).
            :param hidden_state[torch.tensor]: an array of states from previous recurrent state (batch_size, net_dim, ).
            :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        state = state.to(self.device)
        hidden_state = hidden_state.to(self.device)
        action, value_tensor, hidden_state, a_noise = self.network(state, hidden_state)
        return action.detach().cpu(), value_tensor.detach().cpu(), hidden_state, a_noise

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
        #Zero initilization for hiden state staleness problem
        hidden_state = self.network.init_recurent_cell_states(self.seq_len, device=self.device)

        last_done = 0
        traj = list()
        step = 0
        while step < target_step:

            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_hidden_states = torch.as_tensor(hidden_state, dtype=torch.float32).unsqueeze(0)

            ten_actions, value_tensor, ten_hidden_state, noise = self.select_actions(ten_states, ten_hidden_states)

            action = ten_actions[0].numpy()

            next_s, reward, done, _ = env.step(np.tanh(action))

            traj.append((ten_states, ten_hidden_states, reward, done, ten_actions, noise, value_tensor))
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
        Update the neural networks by sampling batch data from ``RecurrentReplayBuffer``.

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
            buf_state, buf_hidden, buf_reward, buf_mask, buf_action, buf_noise, buf_value = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]
            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            #buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.network.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (self.lambda_a_value / (buf_adv_v.std() + 1e-5))
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None

        assert buf_len >= batch_size, 'buf len bigger than batch size'
        update_times = int(buf_len / batch_size * repeat_times)
        # train in minibatches
        # TODO: Convert samples from buffer to sequential
        samples = {
            "actions": buf_action,
            "values": buf_value,
            "log_probs": buf_logprob,
            "advantages": buf_adv_v,
            "obs": buf_state,
            'r_sum':buf_r_sum,
            "hxs":buf_hidden,
            # The loss mask is used for masking the padding while computing the loss function.
            # This is only of significance while using recurrence.
            "loss_mask": buf_mask
        }
        buffer.prepare_batch_dict(samples)
        # for i in repeat times for minibatch in batch
        for update_i in range(repeat_times):
            for mini_batch in buffer.recurrent_mini_batch_generator():

                state = mini_batch['obs']

                value = mini_batch['values']
                r_sum = mini_batch['r_sum']
                adv_v = mini_batch['advantages']
                action = mini_batch['actions']
                logprob = mini_batch['log_probs']
                masks = mini_batch['loss_mask']

                """ PPO Surrogate objective of Trust Region for shared layer network update"""
                #Check get logprobl entrop
                new_logprob, obj_entropy = self.network.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = torch.min(surrogate1, surrogate2)
                obj_surrogate = self._masked_mean(obj_surrogate, masks)

                # Value function loss

                obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

                total_loss = -(obj_surrogate - self.vf_coef * obj_critic + self.lambda_entropy * obj_entropy)

                self.optim_update(self.network, total_loss, self.network.parameters())
                #self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple

    def _masked_mean(self, tensor:torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean of the tensor but ignores the values specified by the mask
        :param loss:
        :param masks:
        :return:
        """
        return (tensor.T * mask).sum() / torch.clamp((torch.ones_like(tensor.T) * mask).float().sum(), min=1.0)

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

    def pad_sequences(self, sequence:np.ndarray, target_length:int) -> np.ndarray:
        delta_length = target_length - len(sequence)

        sequence = torch.from_numpy(np.vstack(sequence).astype(np.float))

        if delta_length <= 0:
            return sequence

        if len(sequence.shape) > 1:
            padding = torch.zeros(((delta_length,) + sequence.shape[1:]), dtype = sequence.dtype)

        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype)
        return torch.cat((sequence, padding), axis=0)

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
            ten_hidden_state = torch.stack(temp[1])
            ten_reward = torch.as_tensor(temp[2], dtype=torch.float32) * reward_scale
            ten_mask = (1.0 - torch.as_tensor(temp[3], dtype=torch.float32)) * gamma
            ten_action = torch.stack(temp[4])
            ten_noise = torch.stack(temp[5])

            traj[:] = (ten_state, ten_hidden_state, ten_reward, ten_mask, ten_action, ten_noise)

        return traj_list