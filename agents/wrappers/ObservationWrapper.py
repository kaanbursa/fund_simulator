import gym
import numpy as np

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


class NormalizedEnv(gym.core.Wrapper):
    def __init__(self, env, ob=True, ret=True, clipob=1., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        infos['real_reward'] = rews
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret = self.ret * (1-float(dones))
        return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(())

        obs = self.env.reset()

        return self._obfilt(obs)

class ObservationWrapper(gym.ObservationWrapper):
    def  __init__(self, env, clip_obs: float = 10.0, clip_reward: float = 10.0, gamma: float = 0.99, epsilon: float = 1e-8):
        super().__init__(env)


        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma =  gamma
        self.epsilon = epsilon
        self.old_obs = np.array([])
        self.old_reward = np.array([])


    def _sanity_checks(self):
        """
        Check for types
        :return:
        """
        if isinstance(self.observation_space, gym.spaces.Dict):
            # By default, we normalize all keys
            if self.norm_obs_keys is None:
                self.norm_obs_keys = list(self.observation_space.spaces.keys())
            # Check that all keys are of type Box
            for obs_key in self.norm_obs_keys:
                if not isinstance(self.observation_space.spaces[obs_key], gym.spaces.Box):
                    raise ValueError(
                        f"This wrapper only supports `gym.spaces.Box` observation spaces but {obs_key} "
                        f"is of type {self.observation_space.spaces[obs_key]}. "
                        "You should probably explicitely pass the observation keys "
                        " that should be normalized via the `norm_obs_keys` parameter."
                    )

        elif isinstance(self.observation_space, gym.spaces.Box):
            if self.norm_obs_keys is not None:
                raise ValueError("`norm_obs_keys` param is applicable only with `gym.spaces.Dict` observation spaces")

        else:
            raise ValueError(
                "VecNormalize only supports `gym.spaces.Box` and `gym.spaces.Dict` observation spaces, "
                f"not {self.observation_space}"
            )

    def observation(self, obs):
        # Modify here

        return obs
