from gym import utils
from gym.spaces import Box
from gym.envs.mujoco import MujocoEnv
import numpy as np
import os


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class A1Env(MujocoEnv, utils.EzPickle):
    """
    ### Action Space
    The action space is a `Box(-1, 1, (12,), float32)`. The A1 robot consists of 12 actuators:   
    | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | 
    | --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | 
    | 0   | Servo motor position                    | -0.802851   | 0.802851    | FR_abduction                     | 
    | 1   | Servo motor position                    | -1.0472     | 4.18879     | FR_hip                           | 
    | 2   | Servo motor position                    | -2.69653    | -0.916298   | FR_calf                          | 
    | 3   | Servo motor position                    | -0.802851   | 0.802851    | FL_abduction                     | 
    | 4   | Servo motor position                    | -1.0472     | 4.18879     | FL_hip                           | 
    | 5   | Servo motor position                    | -2.69653    | -0.916298   | FL_calf                          | 
    | 6   | Servo motor position                    | -0.802851   | 0.802851    | RR_abduction                     |
    | 7   | Servo motor position                    | -1.0472     | 4.18879     | RR_hip                           | 
    | 8   | Servo motor position                    | -2.69653    | -0.916298   | RR_calf                          |
    | 9   | Servo motor position                    | -0.802851   | 0.802851    | RL_abduction                     | 
    | 10  | Servo motor position                    | -1.0472     | 4.18879     | RL_hip                           | 
    | 11  | Servo motor position                    | -2.69653    | -0.916298   | RL_calf                          |

    ### Observation Space
    (-inf, inf, (37,), float32). 37 observations are used:
     - qpos: positional state of each joint
     - qvel: velocity of each joint
    
    ### Rewards
    The reward consists of two parts:
     - *forward_reward*: A reward of moving forward which is measured
        as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*.
        *dt* is the time between actions and is dependent on the frame_skip parameter (fixed to 5), 
        where the frametime is 0.01 - making the default *dt = 5 * 0.01 = 0.05*. This reward would 
        be positive if the robot runs forward (right).
     - *ctrl_cost*: A cost for penalising the robot if it takes actions that are too large. It is 
        measured as *`ctrl_cost_weight` * sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is 
        a parameter set for thecontrol. The total reward returned is ***reward*** 
        *=* *forward_reward - ctrl_cost* and `info` will also contain the individual reward terms
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 20}


    def __init__(
        self,
        frame_skip=0,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.03,
        reset_noise_scale=0.0,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        exclude_current_positions_from_observation=False,
        disable_abduction=True,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )

        self.frame_skip = frame_skip
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)
        self._disable_abduction = disable_abduction
        self.prev_action = np.zeros(12)
        self.abduction_range = [-0.802851, 0.802851]
        self.hip_range = [-1.0472, 4.18879]
        self.knee_range = [-2.69653, -0.916298]

        observation_space = Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float64)
        xmlPath = os.path.dirname(os.path.realpath(__file__)) + '/assets/scene.xml'
        MujocoEnv.__init__(self, xmlPath, 25, observation_space=observation_space, **kwargs)


    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )


    def control_cost(self, action):
        """ Penalize the large action value """
        action = self.prev_action - action   
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
    

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy


    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated


    def step(self, action):
        
        action_cp = action.copy()

        # Covert the activation output range[-1, 1] to range for each actuator
        for i, a in enumerate(action):
            if i%3 == 0:
                if self._disable_abduction:
                    action[i] = 0
                else:
                    action[i] = np.interp(a, [-1, 1], self.abduction_range) 
            if i%3 == 1:
                action[i] = np.interp(a, [-1, 1], self.hip_range) 
            if i%3 ==2:
                action[i] = np.interp(a, [-1, 1], self.knee_range) 

        # Get reward for distance travelled
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt 
        forward_reward = self._forward_reward_weight * x_velocity
        
        # Deduct control cost from the reward
        ctrl_cost = self.control_cost(action_cp)
        self.prev_action = np.array(action_cp)
        reward = forward_reward - ctrl_cost

        observation = self._get_obs()
        terminated = self.terminated
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "action": action
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info


    def _get_obs(self):
        """ return observations (qpos + qvel) """
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]
        
        observation = np.concatenate((position, velocity)).ravel()
        return observation


    def reset_model(self):
        """ resetting the environment """
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation


    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)