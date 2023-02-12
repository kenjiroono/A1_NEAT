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
    ### Description

    
    ### Action Space
    The action space is a `Box(-1, 1, (12,), float32)`. An action represents the torques applied between *links*.
    | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Servo motor position                    | -1          | 1           | FR_hip                           | hinge | torque (N m) |
    | 1   | Servo motor position                    | -1          | 1           | FR_thigh                         | hinge | torque (N m) |
    | 2   | Servo motor position                    | -1          | 1           | FR_calf                          | hinge | torque (N m) |
    | 3   | Servo motor position                    | -1          | 1           | FL_hip                           | hinge | torque (N m) |
    | 4   | Servo motor position                    | -1          | 1           | FL_thigh                         | hinge | torque (N m) |
    | 5   | Servo motor position                    | -1          | 1           | FL_calf                          | hinge | torque (N m) |
    | 6   | Servo motor position                    | -1          | 1           | RR_hip                           | hinge | torque (N m) |
    | 7   | Servo motor position                    | -1          | 1           | RR_thigh                         | hinge | torque (N m) |
    | 8   | Servo motor position                    | -1          | 1           | RR_calf                          | hinge | torque (N m) |
    | 9   | Servo motor position                    | -1          | 1           | RL_hip                           | hinge | torque (N m) |
    | 10  | Servo motor position                    | -1          | 1           | RL_thigh                         | hinge | torque (N m) |
    | 11  | Servo motor position                    | -1          | 1           | RL_calf                          | hinge | torque (N m) |

    
    ### Observation Space
    Observations consist of positional values of different body parts of the
    cheetah, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.
    By default, observations do not include the x-coordinate of the cheetah's center of mass. It may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will have 18 dimensions where the first dimension
    represents the x-coordinate of the cheetah's center of mass.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
    will be returned in `info` with key `"x_position"`.
    However, by default, the observation is a `ndarray` with shape `(17,)` where the elements correspond to the following:
    | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | z-coordinate of the front tip        | -Inf | Inf | rootz                            | slide | position (m)             |
    | 1   | angle of the front tip               | -Inf | Inf | rooty                            | hinge | angle (rad)              |
    | 2   | angle of the second rotor            | -Inf | Inf | bthigh                           | hinge | angle (rad)              |
    | 3   | angle of the second rotor            | -Inf | Inf | bshin                            | hinge | angle (rad)              |
    | 4   | velocity of the tip along the x-axis | -Inf | Inf | bfoot                            | hinge | angle (rad)              |
    | 5   | velocity of the tip along the y-axis | -Inf | Inf | fthigh                           | hinge | angle (rad)              |
    | 6   | angular velocity of front tip        | -Inf | Inf | fshin                            | hinge | angle (rad)              |
    | 7   | angular velocity of second rotor     | -Inf | Inf | ffoot                            | hinge | angle (rad)              |
    | 8   | x-coordinate of the front tip        | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 9   | y-coordinate of the front tip        | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 10  | angle of the front tip               | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 11  | angle of the second rotor            | -Inf | Inf | bthigh                           | hinge | angular velocity (rad/s) |
    | 12  | angle of the second rotor            | -Inf | Inf | bshin                            | hinge | angular velocity (rad/s) |
    | 13  | velocity of the tip along the x-axis | -Inf | Inf | bfoot                            | hinge | angular velocity (rad/s) |
    | 14  | velocity of the tip along the y-axis | -Inf | Inf | fthigh                           | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of front tip        | -Inf | Inf | fshin                            | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of second rotor     | -Inf | Inf | ffoot                            | hinge | angular velocity (rad/s) |
    
    ### Rewards
    The reward consists of two parts:
    - *forward_reward*: A reward of moving forward which is measured
    as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*. *dt* is
    the time between actions and is dependent on the frame_skip parameter
    (fixed to 5), where the frametime is 0.01 - making the
    default *dt = 5 * 0.01 = 0.05*. This reward would be positive if the cheetah
    runs forward (right).
    - *ctrl_cost*: A cost for penalising the cheetah if it takes
    actions that are too large. It is measured as *`ctrl_cost_weight` *
    sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is a parameter set for the
    control and has a default value of 0.1
    The total reward returned is ***reward*** *=* *forward_reward - ctrl_cost* and `info` will also contain the individual reward terms

    ```
    | Parameter                                    | Type      | Default              | Description                                                                                                                                                       |
    | -------------------------------------------- | --------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"scene.xml"`        | Path to a Mujoco a1.xml                                                                                                                                            |
    | `forward_reward_weight`                      | **float** | `1.0`                | Weight for _forward_reward_ term (see section on reward)                                                                                                          |
    | `ctrl_cost_weight`                           | **float** | `0.1`                | Weight for _ctrl_cost_ weight (see section on reward)                                                                                                             |
    | `reset_noise_scale`                          | **float** | `0.1`                | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                    |
    | `exclude_current_positions_from_observation` | **bool**  | `True`               | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |    
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 20}


    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.2,
        reset_noise_scale=0.1,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        exclude_current_positions_from_observation=False,
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

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float64)
    
        xmlPath = os.path.dirname(os.path.realpath(__file__)) + '/assets/scene.xml'
        MujocoEnv.__init__(self, xmlPath, 25, observation_space=observation_space, **kwargs)


    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )


    def control_cost(self, action):
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
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt 
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)
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
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        print

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation


    def reset_model(self):
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