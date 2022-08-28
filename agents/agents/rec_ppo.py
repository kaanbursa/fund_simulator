import os
import torch
import numpy as np
import numpy.random as rd
from tqdm import tqdm
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from agents.Base.deepagents import AgentBase
from agents.Base.network import ShareRecurrentPPO, CriticRecurrentPPO, ActorRecurrentPPO
from agents.Base.config import recurrent_config as config

class AgentRecurrentPPO(AgentBase):
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

    def __init__(self, _net_dim=256, hidden_dim=126, _state_dim=8, _action_dim=2,
                 sequence_length=16,
                 _learning_rate=1e-4,
                 _if_per_or_gae=False, _gpu_id=0, ):

        AgentBase.__init__(self)

        self.ClassAct = ActorRecurrentPPO
        self.ClassCri = CriticRecurrentPPO


        self.sequence_length = config['sequence_length']
        self.hidden_dim = hidden_dim


        self.if_off_policy = False
        self.ratio_clip = 0.2  # could be 0.00 ~ 0.50 ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.00~0.10
        self.lambda_a_value = 1.00  # could be 0.25~8.00, the lambda of advantage value
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.is_recurrent = True
        self.true_sequence_length = 0

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        AgentBase.init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)

        self.ten_hidden_state = self.act.init_recurent_cell_states(1, device=self.device)
        self.ten_hidden_state_critic = self.cri_target.init_recurent_cell_states(1, device=self.device)

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


    def optim_update(self, optimizer, objective, params):
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(params, max_norm=self.clip_grad_norm)
        optimizer.step()

    def select_action(self, state: np.ndarray, hidden_state: tuple, sequence_length: int) -> np.ndarray:
        """
                Select action give a state.

                :param states[np.ndarray]: an array of states in a shape (batch_size, seq_len, state_dim).
                :param hidden_state[np.ndarray]: an array of states from previous recurrent state (batch_size, net_dim, ).
                :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
                """
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        #h_tensor = torch.as_tensor(hidden_state[np.newaxis], device=self.device)

        a_tensor, hidden_state, action_noise = self.act(s_tensor, hidden_state, sequence_length)
        action = a_tensor.detach().cpu().numpy()
        return action, hidden_state, action_noise



    def select_actions(self, state: torch.Tensor, hidden_state: tuple, sequence_length:int) -> torch.Tensor:
        """
            Select actions given an array of states.

            :param states[np.ndarray]: an array of states in a shape (batch_size,  seq_len, state_dim, ).
            :param hidden_state[tuple]: an array of states from previous recurrent state (batch_size, seq_len, net_dim, ).
            :param sequence_length[int]: sequence length for the rnn part of the model
            :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        state = state.to(self.device)

        action, hidden_state, a_noise = self.act.get_action(state, hidden_state, sequence_length)

        return action.detach().cpu(), hidden_state, a_noise

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
        sequence_length = 1
        ten_hidden_state = self.ten_hidden_state
        ten_hidden_state_critic = self.ten_hidden_state_critic
        #first hidden state 1,1,102
        last_done = 0
        traj = list()
        step = 0
        # TODO: this verison calculates log probability when exploring the environment
        while step < target_step:

            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

            # hidden states is hidden statae + cell state
            ten_hidden_states = torch.as_tensor(ten_hidden_state[0], dtype=torch.float32) # unsqueezed on initialization
            ten_cell_states = torch.as_tensor(ten_hidden_state[1], dtype=torch.float32)

            ten_cri_hidden = torch.as_tensor(ten_hidden_state_critic[0], dtype=torch.float32)
            ten_cri_cell = torch.as_tensor(ten_hidden_state_critic[1], dtype=torch.float32)

            # sequence_length is 1 for sampling and exploring the data

            ten_actions, ten_hidden_state, noise = self.select_actions(ten_states, (ten_hidden_states, ten_cell_states), sequence_length)
            ten_values, ten_hidden_state_critic = self.cri_target(ten_states, (ten_cri_hidden, ten_cri_cell), sequence_length)
            # TODO: check if you want to add hidden state before or after new
            ten_hidden_state = (ten_hidden_states, ten_cell_states)
            # squeeze for trajectory
            ten_hidden_states = torch.as_tensor(ten_hidden_state[0],
                                                dtype=torch.float32).squeeze(0)  # unsqueezed on initialization
            ten_cell_states = torch.as_tensor(ten_hidden_state[1], dtype=torch.float32).squeeze(0)


            ten_cri_hidden = torch.as_tensor(ten_hidden_state_critic[0], dtype=torch.float32).squeeze(0)
            ten_cri_cell = torch.as_tensor(ten_hidden_state_critic[1], dtype=torch.float32).squeeze(0)
            action = ten_actions[0].numpy()

            #log_prob = self.act.get_logprob_entropy(action)

            next_s, reward, done, _ = env.step(np.tanh(action))

            traj.append((ten_states, ten_hidden_states, ten_cell_states,
                         reward, done, ten_actions, noise, ten_values, ten_cri_hidden, ten_cri_cell))

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
        #TODO: Redo for recurrent model
        raise NotImplementedError
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

            buf_state, buf_hidden, buf_cell, buf_reward, buf_mask, buf_action, buf_noise, buf_value, buf_cri_hidden, buf_cri_cell = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]
            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            # Calculate value in when sampling? TODO: check if using shared hidden state effect performance
            #buf_cri_hidden = self.cri_target.init_recurent_cell_states(1, self.device)
            #buf_cri_hidden = (buf_cri_hidden[0].unsqueeze(0), buf_cri_hidden[1].unsqueeze(0))
            #(buf_value, buf_cri_hidden) = [self.cri_target(buf_state[i:i + bs],buf_cri_hidden, sequence_length=1) for i in range(0, buf_len, bs)]
            #buf_value = torch.cat(buf_value, dim=0)

            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

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
            "log_probs": buf_logprob,
            "advantages": buf_adv_v,
            "obs": buf_state,
            'r_sum':buf_r_sum,
            "hxs":buf_hidden,
            "cxs": buf_cell,
            "cri_hidden":buf_cri_hidden,
            "cri_cell":buf_cri_cell,
            # The loss mask is used for masking the padding while computing the loss function.
            # This is only of significance while using recurrence.
            "loss_mask": buf_mask
        }
        samples_flat = self.prepare_batch(samples)
        #buffer.prepare_batch_dict(samples)
        # for i in repeat times for minibatch in batch
        #hidden_state_critic = self.cri.init_recurent_cell_states(self.num_sequences, self.device)
        # Stack sequences (target shape: (Sequence, Step, Data ...) and apply data to the samples dictionary
        # Hidden state  dimensions (number of seq,bs,hidden_state_size)
        #hidden = self.act.init_recurent_cell_states(self.sequence_length, self.device)
        # Create mini batch generator
        if batch_size > 64 : batch_size /= 2
        for i in range(repeat_times):
            #TODO: add mask
            for mini_batch in self.mini_batch_generator(samples_flat, batch_size):

                state = mini_batch['obs'].float()

                r_sum = mini_batch['r_sum']
                adv_v = mini_batch['advantages']
                action = mini_batch['actions']
                logprob = mini_batch['log_probs']
                #masks = mini_batch['loss_mask']
                hidden_state = mini_batch['hxs'].unsqueeze(0).float() # always get the first
                cell_state = mini_batch['cxs'].unsqueeze(0).float()
                cri_hidden = mini_batch['cri_hidden'].unsqueeze(0).float()
                ten_cri_cell = mini_batch['cri_cell'].unsqueeze(0).float()
                #hidden_state = torch.cat(hidden_state, dim=0) # catinate on batch
                #cell_state = torch.cat(cell_state, dim=0)
                hidden = (hidden_state, cell_state)
                critic_hidden = (cri_hidden, ten_cri_cell)

                # get hidden state of critic network
                #TODO: add mask to loss constuction
                #Check get logprobl entrop
                """ PPO Surrogate objective of Trust Region"""
                # The hidden state for new log probability is the out hidden state
                new_logprob, obj_entropy, hidden = self.act.get_logprob_entropy(state, hidden, action, self.sequence_length)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp() # a/b == log(exp(a) - exp(b))
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                #masked_actor = self._masked_mean()
                """
                UserWarning: Using a target size (torch.Size([384, 1])) 
                that is different to the input size (torch.Size([384, 30])). This will likely lead to incorrect results due to broadcasting. 
                Please ensure they have the same size.
                """
                self.optim_update(self.act_optim, obj_actor, self.act.parameters())

                value, critic_hidden = self.cri(state, critic_hidden, self.sequence_length)  # critic network predicts the reward_sum (Q value) of state

                obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

                self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
                self.soft_update(self.cri_target, self.cri,
                                 soft_update_tau) if self.cri_target is not self.cri else None

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return {'actor_loss':obj_critic.item(), 'actor_loss':obj_actor.item(), 'action_std_log':a_std_log.item() } # logging_tuple

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

    def pad_sequence(self, sequence:np.ndarray, target_length:int) -> np.ndarray:
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
        # add pad sequences here not in buffer and than use buffer.
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
        #  Convert to seqlen with pad _sequence

        for traj in traj_list:
            temp = list(map(list, zip(*traj)))  # 2D-list transpose
            # step size, 1 batch size, datasi
            ten_state = torch.stack(temp[0])
            ten_hidden_state  = torch.stack(temp[1])

            ten_cell_state = torch.stack(temp[2])
            ten_reward = torch.as_tensor(temp[3], dtype=torch.float32) * reward_scale
            ten_mask = (1.0 - torch.as_tensor(temp[4], dtype=torch.float32)) * gamma
            ten_action = torch.stack(temp[5])
            ten_noise = torch.stack(temp[6])
            ten_values = torch.stack(temp[7])
            ten_cri_hidden =torch.stack(temp[8])
            ten_cri_cell = torch.stack(temp[9])




            traj[:] = (ten_state, ten_hidden_state, ten_cell_state, ten_reward, ten_mask, ten_action, ten_noise, ten_values, ten_cri_hidden, ten_cri_cell)

        return traj_list


    def prepare_batch(self, samples):


        # Split data into sequences and apply zero-padding
        # Retrieve the indices of dones as these are the last step of a whole episode


        # Split obs, values, advantages, recurrent cell states, actions and log_probs into episodes and then into sequences
        max_sequence_length = 1
        # TODO: Change this for more generalizable modeling with done lists
        done_index = len(samples['obs'])


        for key, value in samples.items():

            sequences = []
            start_index = 0
            #
            # Split trajectory into episodes
            episode = value[start_index:done_index + 1]

            start_index = done_index + 1
            # Split episodes into sequences
            if self.sequence_length > 0:
                for start in range(0, len(episode), self.sequence_length):
                    end = start + self.sequence_length
                    sequences.append(episode[start:end])
                max_sequence_length = self.sequence_length
            else:
                # If the sequence length is not set to a proper value, sequences will be based on whole episodes
                sequences.append(episode)
                max_sequence_length = len(episode) if len(
                    episode) > max_sequence_length else max_sequence_length

            # Apply zero-padding to ensure that each sequence has the same length
            # Therfore we can train batches of sequences in parallel instead of one sequence at a time
            for i, sequence in enumerate(sequences):
                sequences[i] = self.pad_sequence(sequence, max_sequence_length)

            # Stack sequences (target shape: (Sequence, Step, Data ...) and apply data to the samples dictionary
            samples[key] = torch.stack(sequences, axis=0)


            if (key == "hxs" or key == "cxs") or (key == "cri_hidden" or key == "cri_cell"):
                # Select only the very first recurrent cell state of a sequence and add it to the samples.

                samples[key] = samples[key].double()



        # If the sequence length is based on entire episodes, it will be as long as the longest episode.
        # Hence, this information has to be stored for the mini batch generation.
        self.true_sequence_length = max_sequence_length

        # Flatten all samples and convert them to a tensor
        samples_flat = {}
        for key, value in samples.items():

            # size here is traj len / seq len
            #if not key == "hxs" and not key == "cxs":
                #convert to seq len * sample
            value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

            samples_flat[key] = value

        return samples_flat

    def mini_batch_generator(self, samples, batch_size):
        """
        Get flattened samples
        :param samples: flattened samples after prepare batch func
        :return:
        """
        num_sequences = len(samples['obs']) // self.true_sequence_length

        num_sequences_per_batch = num_sequences // batch_size

        num_sequences_per_batch = [num_sequences_per_batch] * batch_size
        remainder = num_sequences % batch_size

        for i in range(remainder):
            num_sequences_per_batch[i] += 1

        # Prepare indices,
        indices = torch.arange(0, num_sequences * self.true_sequence_length, requires_grad=False).reshape(num_sequences,
                                                                                     self.true_sequence_length)

        sequence_indices = torch.randperm(num_sequences)
        start = 0

        for n_sequences in num_sequences_per_batch:
            end = start + n_sequences
            mini_batch_indices = indices[sequence_indices[start:end]].reshape(-1)
            mini_batch = {}
            for key, value in samples.items():


                if key != 'hxs' and key != 'cxs' and key != 'cri_hidden' and key != 'cri_cell':

                    mini_batch[key] = value[mini_batch_indices].to(self.device)
                else:
                    # Collect only the recurrent cell states

                    mini_batch[key] = value[sequence_indices[start:end]].to(self.device)
            start = end

            yield mini_batch


class AgentSharedRecurrentPPO(AgentBase):
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

    def __init__(self,
                 _net_dim=256, hidden_dim=126, _state_dim=8, _action_dim=2, sequence_length=16,
                 _learning_rate=1e-4,
                 _if_per_or_gae=False, _gpu_id=0, ):

        AgentBase.__init__(self)

        self.ClassCri = None
        self.ClassAct = None

        self.sequence_length = config['sequence_length']

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
        #losses = {'SmoothL1Loss':torch.nn.SmoothL1Loss}



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


    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        self.ClassCri = ShareRecurrentPPO

        AgentBase.init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)
        # TODO: from config get if zero initialization state
        self.hidden_state_in = self.cri.init_recurent_cell_states(num_sequences=1, device=self.device)

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


        a_tensor, value_tensor, hidden_state = self.cri(s_tensor, h_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action, value_tensor, hidden_state


    def select_actions(self, state: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
            Select actions given an array of states.

            :param states[np.ndarray]: an array of states in a shape (batch_size,  seq_len, state_dim, ).
            :param hidden_state[torch.tensor]: an array of states from previous recurrent state (batch_size, net_dim, ).
            :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        state = state.to(self.device)
        self.sequence_length = 1

        #hidden_state = hidden_state.to(self.device)
        action, value_tensor, hidden_state = self.cri(state, hidden_state, self.sequence_length)
        return action.detach().cpu(), value_tensor.detach().cpu(), hidden_state

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
        hidden_state_in = self.hidden_state_in

        last_done = 0
        traj = list()
        step = 0
        while step < target_step:

            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

            ten_actions, value_tensor, ten_hidden_states_out = self.select_actions(ten_states, hidden_state_in)
            noise = self.cri.get_action_noise(ten_actions)
            action = ten_actions[0].numpy()

            #log_prob = self.cri.get_logprob_entropy(action, hidden_state_in)
            #TODO: np.tanh action for production
            next_s, reward, done, _ = env.step(np.tanh(action))
            ten_hidden_states_out = (torch.as_tensor(ten_hidden_states_out[0],
                                                dtype=torch.float32).squeeze(0),
                               torch.as_tensor(ten_hidden_states_out[1],
                                               dtype=torch.float32).squeeze(0)
                               )
            hidden_state_in = (torch.as_tensor(hidden_state_in[0],
                                                     dtype=torch.float32).squeeze(0),
                                     torch.as_tensor(hidden_state_in[1],
                                                     dtype=torch.float32).squeeze(0)
                                     )

            traj.append((ten_states,
                         hidden_state_in[0],hidden_state_in[1],
                         ten_hidden_states_out[0], ten_hidden_states_out[1]
                         , reward, done, ten_actions, noise, value_tensor))
            hidden_state_in  = (torch.as_tensor(ten_hidden_states_out[0],
                                                dtype=torch.float32).unsqueeze(0),
                               torch.as_tensor(ten_hidden_states_out[1],
                                               dtype=torch.float32).unsqueeze(0)
                               )
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
            buf_state, buf_hidden_in, buff_hidden_out, buf_reward, buf_mask, buf_action, buf_noise, buf_value = buffer#[ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]
            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            #buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]

            #buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.cri.get_old_logprob(buf_action, buf_noise, buf_hidden_in)

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
            "log_probs": buf_logprob,
            "advantages": buf_adv_v,
            "obs": buf_state,
            'r_sum': buf_r_sum,
            "hidden_in": buf_hidden_in,
            "hidden_out": buff_hidden_out,
            "values":buf_value,
            # The loss mask is used for masking the padding while computing the loss function.
            # This is only of significance while using recurrence.
            "loss_mask": buf_mask
        }

        samples_flat = self.prepare_batch(samples)
        # buffer.prepare_batch_dict(samples)
        # for i in repeat times for minibatch in batch
        # hidden_state_critic = self.cri.init_recurent_cell_states(self.num_sequences, self.device)
        # Stack sequences (target shape: (Sequence, Step, Data ...) and apply data to the samples dictionary
        # Hidden state  dimensions (number of seq,bs,hidden_state_size)
        # hidden = self.act.init_recurent_cell_states(self.sequence_length, self.device)
        # Create mini batch generator
        if batch_size > 64: batch_size /= 2
        # for i in repeat times for minibatch in batch
        for i in range(repeat_times):
            for mini_batch in self.mini_batch_generator(samples_flat, batch_size):
                state = mini_batch['obs'].float()
                value = mini_batch['values']
                r_sum = mini_batch['r_sum']
                adv_v = mini_batch['advantages']
                hidden_in = mini_batch['hidden_in']
                hidden_out = mini_batch['hidden_out']
                action = mini_batch['actions']
                logprob = mini_batch['log_probs']
                masks = mini_batch['loss_mask']

                """ PPO Surrogate objective of Trust Region for shared layer network update"""
                #Check get logprobl entrop
                new_logprob, obj_entropy = self.cri.get_logprob_entropy(state, action, hidden_out, self.sequence_length)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = torch.min(surrogate1, surrogate2)
                obj_surrogate = self._masked_mean(obj_surrogate, masks)

                # Value function loss
                obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

                total_loss = -(obj_surrogate - self.vf_coef * obj_critic + self.lambda_entropy * obj_entropy)

                self.optim_update(self.cri_optim, total_loss, self.cri.parameters())
                #self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        a_std_log = getattr(self.cri, 'a_std_log', torch.zeros(1)).mean()
        return  {'network_loss':obj_critic.item(),
                 'action_std_log':a_std_log.item() }  # logging_tuple

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

            ten_hidden_state = (torch.stack(temp[1]), torch.stack(temp[2]))

            ten_hidden_state_out = (torch.stack(temp[3]), torch.stack(temp[4]))
            ten_reward = torch.as_tensor(temp[5], dtype=torch.float32) * reward_scale
            ten_mask = (1.0 - torch.as_tensor(temp[6], dtype=torch.float32)) * gamma
            ten_action = torch.stack(temp[7])
            ten_noise = torch.stack(temp[8])
            ten_value = torch.stack(temp[9]).unsqueeze(-1)


            traj[:] = (ten_state, ten_hidden_state, ten_hidden_state_out, ten_reward, ten_mask, ten_action, ten_noise, ten_value)

        return traj_list

    def pad_sequence(self, sequence:np.ndarray, target_length:int) -> np.ndarray:
        delta_length = target_length - len(sequence)

        sequence = torch.from_numpy(np.vstack(sequence).astype(np.float))

        if delta_length <= 0:
            return sequence

        if len(sequence.shape) > 1:
            padding = torch.zeros(((delta_length,) + sequence.shape[1:]), dtype = sequence.dtype)

        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype)
        return torch.cat((sequence, padding), axis=0)


    def prepare_batch(self, samples):


        # Split data into sequences and apply zero-padding
        # Retrieve the indices of dones as these are the last step of a whole episode


        # Split obs, values, advantages, recurrent cell states, actions and log_probs into episodes and then into sequences
        max_sequence_length = 1
        # TODO: Change this for more generalizable modeling with done lists
        done_index = len(samples['obs'])

        for key, value in samples.items():
            sequences = []
            start_index = 0
            #
            # Split trajectory into episodes
            episode = value[start_index:done_index + 1]

            start_index = done_index + 1
            # Split episodes into sequences
            if ((key is not "hidden_in") and (key is not 'hidden_out')):

                if self.sequence_length > 0:
                    for start in range(0, len(episode), self.sequence_length):
                        end = start + self.sequence_length
                        sequences.append(episode[start:end])
                    max_sequence_length = self.sequence_length
                else:
                    # If the sequence length is not set to a proper value, sequences will be based on whole episodes
                    sequences.append(episode)
                    max_sequence_length = len(episode) if len(
                        episode) > max_sequence_length else max_sequence_length

                # Apply zero-padding to ensure that each sequence has the same length
                # Therfore we can train batches of sequences in parallel instead of one sequence at a time
                for i, sequence in enumerate(sequences):

                    sequences[i] = self.pad_sequence(sequence, max_sequence_length)

                # Stack sequences (target shape: (Sequence, Step, Data ...) and apply data to the samples dictionary
                samples[key] = torch.stack(sequences, axis=0)


            if (key == "hidden_in" or key == "hidden_out") :
                # Select only the very first recurrent cell state of a sequence and add it to the samples.

                samples[key] = (samples[key][0].double(),samples[key][1].double())


        # If the sequence length is based on entire episodes, it will be as long as the longest episode.
        # Hence, this information has to be stored for the mini batch generation.
        self.true_sequence_length = max_sequence_length

        # Flatten all samples and convert them to a tensor
        samples_flat = {}
        for key, value in samples.items():
            # size here is traj len / seq len

            if not key == "hidden_in" and not key == "hidden_out":
                #convert to seq len * sample
                value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

            else:

                value = (
                    value[0].reshape(value[0].shape[0] * value[0].shape[1], * value[0].shape[2:]),
                    value[1].reshape(value[1].shape[0] * value[1].shape[1], *value[1].shape[2:])
                    )


            samples_flat[key] = value

        return samples_flat

    def mini_batch_generator(self, samples, batch_size):
        """
        Get flattened samples
        :param samples: flattened samples after prepare batch func
        :return:
        """
        num_sequences = len(samples['obs']) // self.true_sequence_length

        num_sequences_per_batch = num_sequences // batch_size

        num_sequences_per_batch = [num_sequences_per_batch] * batch_size
        remainder = num_sequences % batch_size

        for i in range(remainder):
            num_sequences_per_batch[i] += 1

        # Prepare indices,
        indices = torch.arange(0,
                               num_sequences * self.true_sequence_length,
                               requires_grad=False)\
            .reshape(num_sequences, self.true_sequence_length)

        sequence_indices = torch.randperm(num_sequences)
        start = 0

        for n_sequences in num_sequences_per_batch:
            end = start + n_sequences
            mini_batch_indices = indices[sequence_indices[start:end]].reshape(-1)
            mini_batch = {}
            for key, value in samples.items():
                if key != 'hidden_in' and key != 'hidden_out':
                    mini_batch[key] = value[mini_batch_indices].to(self.device)
                else:
                    # Collect only the recurrent cell states
                    mini_batch[key] = (
                        value[0][mini_batch_indices].unsqueeze(0).to(self.device).float(),
                        value[1][mini_batch_indices].unsqueeze(0).to(self.device).float(),
                                       )

                    #mini_batch[key] = value
            start = end

            yield mini_batch