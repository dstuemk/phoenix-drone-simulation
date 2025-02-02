import numpy as np
from phoenix_drone_simulation.envs.base import DroneBaseEnv
from phoenix_drone_simulation.envs.utils import deg2rad


class DroneCircleBaseEnv(DroneBaseEnv):
    def __init__(
            self,
            physics,
            control_mode: str,
            drone_model: str,
            observation_noise=1,  # must be positive in order to add noise
            domain_randomization: float = 0.10,  # use 10% DR as default
            target_pos: np.ndarray = np.array([0, 0, 1.0], dtype=np.float32),
            sim_freq=200,  # in Hz
            aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
            observation_frequency=100,  # in Hz
            penalty_action: float = 1e-4,
            penalty_angle: float = 0,
            penalty_spin: float = 1e-3,  # Note: 1e-4 resulted in fast yaw spin
            penalty_terminal: float = 100,
            penalty_velocity: float = 1e-4,
            **kwargs
    ):
        # === Circle task specific attributes
        # must be defined before calling super class constructor:
        self.target_pos = target_pos  # used in _computePotential()
        self.ARP = 1e-3  # action rate penalty
        self.penalty_action = penalty_action
        self.penalty_angle = penalty_angle
        self.penalty_spin = penalty_spin
        self.penalty_terminal = penalty_terminal
        self.penalty_velocity = penalty_velocity

        self.done_dist_threshold = 0.25

        # === Costs: The following constants are used for cost calculation:
        self.vel_limit = 0.25  # [m/s]
        self.roll_pitch_limit = deg2rad(10)  # [rad]
        self.rpy_dot_limit = deg2rad(200)  # [rad/s]
        self.x_lim = 0.10
        self.y_lim = 0.10
        self.z_lim = 1.20

        # === Reference trajectory
        #self.episode_length = 500
        self.circle_time = 3  # [s]
        circle_radius = 0.25  # [m]
        self.num_ref_points = N = self.circle_time * observation_frequency  # [1]
        ts = 2 * np.pi * np.arange(self.num_ref_points) / self.num_ref_points

        self.ref_offset = 0  # set by task_specific_reset()-method
        self.ref = np.zeros((self.num_ref_points, 3))
        self.ref[:, 2] = 1.  # z-position
        self.ref[:, 1] = circle_radius * np.sin(ts)  # y-position
        self.ref[:, 0] = circle_radius * (1 - np.cos(ts))  # x-position

        # task specific parameters - init drone state
        init_xyz = np.array([0, 0, 1], dtype=np.float32)
        init_rpy = np.zeros(3)
        init_xyz_dot = np.zeros(3)
        init_rpy_dot = np.zeros(3)

        super(DroneCircleBaseEnv, self).__init__(
            **{
                **dict(
                    control_mode=control_mode,
                    drone_model=drone_model,
                    physics=physics,
                    observation_noise=observation_noise,
                    domain_randomization=domain_randomization,
                    sim_freq=sim_freq,
                    aggregate_phy_steps=aggregate_phy_steps,
                    observation_frequency=observation_frequency,
                ),
                **kwargs,
                **dict(
                    init_xyz=init_xyz,
                    init_rpy=init_rpy,
                    init_xyz_dot=init_xyz_dot,
                    init_rpy_dot=init_rpy_dot,
                )
            },
            #control_mode=control_mode,
            #drone_model=drone_model,
            #init_xyz=init_xyz,
            #init_rpy=init_rpy,
            #init_xyz_dot=init_xyz_dot,
            #init_rpy_dot=init_rpy_dot,
            #physics=physics,
            #observation_noise=observation_noise,
            #domain_randomization=domain_randomization,
            #sim_freq=sim_freq,
            #aggregate_phy_steps=aggregate_phy_steps,
            #observation_frequency=observation_frequency,
            #**kwargs
        )

    def _setup_task_specifics(self):
        """Initialize task specifics. Called by _setup_simulation()."""
        target_visual = self.bc.createVisualShape(
            self.bc.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0.95, 0.1, 0.05, 0.4],
        )
        # Spawn visual without collision shape
        self.target_body_id = self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_pos
        )

        # === Draw reference circle
        for k in range(0, self.num_ref_points, 10):
            self.bc.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=self.bc.createVisualShape(
                    self.bc.GEOM_SPHERE,
                    radius=0.003,
                    rgbaColor=[0.95, 0.15, 0.10, 0.7],
                ),
                basePosition=self.ref[k]
            )

        # === Set camera position
        self.bc.resetDebugVisualizerCamera(
            cameraTargetPosition=(0.0, 0.0, 0.0),
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-70
        )

    def compute_done(self) -> bool:
        """Compute end of episode if dist(drone - ref) > d."""
        dist = np.linalg.norm(self.drone.xyz - self.target_pos)
        done = True if dist > self.done_dist_threshold else False
        return done

    def compute_info(self) -> dict:
        # state = self.drone.get_state()
        c = 0.
        info = {'cost': c, **super().compute_info()}
        return info

    def compute_observation(self) -> np.ndarray:
        r"""Returns the current observation of the environment."""
        t = (self.iteration // self.aggregate_phy_steps + self.ref_offset) % self.num_ref_points
        self.target_pos = self.ref[t]
        # update target visual:
        self.bc.resetBasePositionAndOrientation(
            self.target_body_id,
            posObj=self.target_pos,
            ornObj=(0, 0, 0, 1)
        )
        return super().compute_observation(target_pos=self.target_pos)

    def compute_potential(self) -> float:
        """Euclidean distance from current ron position to target position."""
        return np.linalg.norm(self.drone.xyz - self.target_pos)

    def compute_reward(self, action) -> float:
        # Determine penalties
        # spin_penalty = 1e-4 * np.linalg.norm(self.drone.rpy_dot)**2
        act_diff = action - self.last_action
        normed_clipped_a = 0.5 * (np.clip(action, -1, 1) + 1)

        penalty_action = self.penalty_action * np.linalg.norm(normed_clipped_a)
        penalty_action_rate = self.ARP * np.linalg.norm(act_diff)
        penalty_rpy = self.penalty_angle * np.linalg.norm(self.drone.rpy)
        penalty_spin = self.penalty_spin * np.linalg.norm(self.drone.rpy_dot)
        penalty_terminal = self.penalty_terminal if self.compute_done() else 0.
        penalty_velocity = self.penalty_velocity * np.linalg.norm(
            self.drone.xyz_dot)

        penalties = np.sum([penalty_rpy, penalty_action_rate, penalty_spin,
                            penalty_velocity, penalty_action, penalty_terminal])
        # L2 norm
        dist = np.linalg.norm(self.drone.xyz - self.target_pos)
        reward = -dist - penalties
        return reward

    def task_specific_reset(self):

        # Note:
        #  - use copy since += operations are call by reference
        #  - init_xyz, init_xyz_dot, ... are set the SimOpt classes
        pos = self.init_xyz.copy()
        xyz_dot = self.init_xyz_dot.copy()
        rpy_dot = self.init_rpy_dot.copy()
        quat = self.init_quaternion.copy()

        if self.enable_reset_distribution:
            # compute index where to position drone on reference circle:
            self.ref_offset = np.random.randint(0, self.num_ref_points)
            self.target_pos = self.ref[self.ref_offset]
            pos = self.target_pos.copy()

            pos_lim = 0.05  # should be at least 0.15 for hover task since position PID shoots up to (0,0,1.15)
            pos += np.random.uniform(-pos_lim, pos_lim, size=3)

            init_angle = deg2rad(20)  # default: 20°
            rpy = np.random.uniform(-init_angle, init_angle, size=3)
            yaw_init = 0.1 * np.pi  # use 2*pi as init
            rpy[2] = np.random.uniform(-yaw_init, yaw_init)
            quat = self.bc.getQuaternionFromEuler(rpy)

            # set random initial velocities
            vel_lim = 0.1  # default: 0.05
            xyz_dot = xyz_dot + np.random.uniform(-vel_lim, vel_lim, size=3)

            rpy_dot_lim = deg2rad(50)  # default: 50°/s
            rpy_dot[:2] = np.random.uniform(-rpy_dot_lim, rpy_dot_lim, size=2)
            rpy_dot[2] = np.random.uniform(-deg2rad(20), deg2rad(20))

            # init low pass filter(s) with new values:
            self.gyro_lpf.set(x=rpy_dot)

            # set drone internals
            self.drone.x = np.random.normal(self.drone.HOVER_X, scale=0.02,
                                            size=(4,))
            self.drone.y = self.drone.K * self.drone.x
            self.drone.action_buffer = np.clip(
                np.random.normal(self.drone.HOVER_ACTION, 0.02,
                                 size=self.drone.action_buffer.shape), -1, 1)
            self.drone.last_action = self.drone.action_buffer[-1, :]


        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            posObj=pos,
            ornObj=quat
        )

        R = np.array(self.bc.getMatrixFromQuaternion(quat)).reshape(3, 3)
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            linearVelocity=xyz_dot,
            # PyBullet assumes world frame, so local frame -> world frame
            angularVelocity=R.T @ rpy_dot
        )

        self.bc.resetBasePositionAndOrientation(
            self.target_body_id,
            posObj=self.target_pos,
            ornObj=(0, 0, 0, 1)
        )


""" ==================
        PWM control
    ==================
"""


class DroneCircleSimpleEnv(DroneCircleBaseEnv):
    def __init__(self,
                 aggregate_phy_steps: int = 1,
                 control_mode='PWM',
                 **kwargs):
        super(DroneCircleSimpleEnv, self).__init__(
            control_mode=control_mode,
            drone_model='cf21x_sys_eq',
            physics='SimplePhysics',
            # use 100 Hz since no motor dynamics and PID is used
            sim_freq=100,
            aggregate_phy_steps=aggregate_phy_steps,
            **kwargs
        )


class DroneCircleBulletEnv(DroneCircleBaseEnv):
    def __init__(self,
                 aggregate_phy_steps: int = 2,  # sub-steps used to calculate motor dynamics
                 control_mode: str = 'PWM',
                 **kwargs):
        super(DroneCircleBulletEnv, self).__init__(
            **{
                'aggregate_phy_steps': aggregate_phy_steps,
                'control_mode': control_mode,
                'drone_model': 'cf21x_bullet',
                'physics': 'PyBulletPhysics',
                'observation_frequency': 100,
                'sim_freq': 200,
                **kwargs,
            },
            #aggregate_phy_steps=aggregate_phy_steps,
            #control_mode=control_mode,
            #drone_model='cf21x_bullet',
            #physics='PyBulletPhysics',
            #observation_frequency=100,  # use 100Hz PWM control loop
            #sim_freq=200,  # but step physics with 200Hz
            #**kwargs
        )
