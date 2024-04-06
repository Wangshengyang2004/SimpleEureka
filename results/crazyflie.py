import torch
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.crazyflie import Crazyflie
from omniisaacgymenvs.robots.articulations.views.crazyflie_view import CrazyflieView
import torch.distributions as D
from functorch import vmap
import numpy as np

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)

class CrazyflieTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self._num_observations = 18
        self._num_actions = 4
        RLTask.__init__(self, name=name, env=env)
        self._ball_position = torch.tensor([0, 0, 2.0], device=self.device)
        
        self._crazyflie_position = torch.tensor([0, 0, 2.0], device=self.device)
        self.traj_c_dist = D.Uniform(torch.tensor(-0.6, device=self.device), torch.tensor(0.6, device=self.device))
        self.traj_scale_dist = D.Uniform(torch.tensor([1.8, 1.8, 1.], device=self.device), torch.tensor([3.2, 3.2, 1.5], device=self.device))
        self.traj_w_dist = D.Uniform(torch.tensor(0.8, device=self.device), torch.tensor(1.1, device=self.device))
        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_t0 = torch.pi / 2
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_w = torch.ones(self.num_envs, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, 4, 3, device=self.device)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self.dt = self._task_cfg["sim"]["dt"]
        self.arm_length = 0.05
        self.motor_damp_time_up = 0.15
        self.motor_damp_time_down = 0.15
        self.motor_tau_up = 4 * self.dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4 * self.dt / (self.motor_damp_time_down + EPS)
        self.mass = 0.028
        self.thrust_to_weight = 1.9
        self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
        self.motor_assymetry = self.motor_assymetry * 4.0 / np.sum(self.motor_assymetry)
        self.grav_z = -1.0 * self._task_cfg["sim"]["gravity"][2]

    def set_up_scene(self, scene) -> None:
        self.get_crazyflie()
        self.get_target()
        RLTask.set_up_scene(self, scene)
        self._copters = CrazyflieView(prim_paths_expr="/World/envs/.*/Crazyflie", name="crazyflie_view")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="ball_view")
        scene.add(self._copters)
        scene.add(self._balls)
        for i in range(4):
            scene.add(self._copters.physics_rotors[i])

    def initialize_views(self, scene) -> None:
        super().initialize_views(scene)
        if scene.object_exists("crazyflie_view"):
            scene.remove_object("crazyflie_view", registry_only=True)
        if scene.object_exists("ball_view"):
            scene.remove_object("ball_view", registry_only=True)
        for i in range(1, 5):
            scene.remove_object(f"m{i}_prop_view", registry_only=True)
        self._copters = CrazyflieView(prim_paths_expr="/World/envs/.*/Crazyflie", name="crazyflie_view")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="ball_view")
        scene.add(self._copters)
        scene.add(self._balls)
        for i in range(4):
            scene.add(self._copters.physics_rotors[i])

    def get_crazyflie(self) -> None:
        copter = Crazyflie(
            prim_path=self.default_zero_env_path + "/Crazyflie",
            name="crazyflie",
            translation=self._crazyflie_position
        )
        self._sim_config.apply_articulation_settings(
            "crazyflie", get_prim_at_path(copter.prim_path), self._sim_config.parse_actor_config("crazyflie")
        )

    def get_target(self) -> None:
        radius = 0.01
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball",
            translation=self._ball_position,
            name="target_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings(
            "ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball")
        )
        ball.set_collision_enabled(False)

    def get_observations(self) -> dict:
        self.root_pos, self.root_rot = self._copters.get_world_poses(clone=False)
        self.root_velocities = self._copters.get_velocities(clone=False)

        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot

        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)

        root_linvels = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]

        self.obs_buf[..., 0:3] = self.target_positions - root_positions

        self.obs_buf[..., 3:6] = rot_x
        self.obs_buf[..., 6:9] = rot_y
        self.obs_buf[..., 9:12] = rot_z

        self.obs_buf[..., 12:15] = root_linvels
        self.obs_buf[..., 15:18] = root_angvels

        observations = {self._copters.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)

        actions = actions.clone().to(self._device)
        self.actions = actions

        thrust_cmds = torch.clamp(actions, min=-1.0, max=1.0)
        thrust_cmds = (thrust_cmds + 1.0) / 2.0

        motor_tau = self.motor_tau_up * torch.ones((self._num_envs, 4), dtype=torch.float32, device=self._device)
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down
        motor_tau[motor_tau > 1.0] = 1.0

        thrust_rot = thrust_cmds**0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp**2

        thrust_noise = 0.01 * torch.randn(4, dtype=torch.float32, device=self._device)
        thrust_noise = thrust_cmds * thrust_noise
        self.thrust_cmds_damp = torch.clamp(self.thrust_cmds_damp + thrust_noise, min=0.0, max=1.0)

        thrusts = self.thrust_max * self.thrust_cmds_damp

        root_quats = self.root_rot
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        rot_matrix = torch.cat((rot_x, rot_y, rot_z), 1).reshape(-1, 3, 3)

        force_x = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        force_y = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        force_xy = torch.cat((force_x, force_y), 1).reshape(-1, 4, 2)
        thrusts = thrusts.reshape(-1, 4, 1)
        thrusts = torch.cat((force_xy, thrusts), 2)

        thrusts_0 = thrusts[:, 0]
        thrusts_0 = thrusts_0[:, :, None]

        thrusts_1 = thrusts[:, 1]
        thrusts_1 = thrusts_1[:, :, None]

        thrusts_2 = thrusts[:, 2]
        thrusts_2 = thrusts_2[:, :, None]

        thrusts_3 = thrusts[:, 3]
        thrusts_3 = thrusts_3[:, :, None]

        mod_thrusts_0 = torch.matmul(rot_matrix, thrusts_0)
        mod_thrusts_1 = torch.matmul(rot_matrix, thrusts_1)
        mod_thrusts_2 = torch.matmul(rot_matrix, thrusts_2)
        mod_thrusts_3 = torch.matmul(rot_matrix, thrusts_3)

        self.thrusts[:, 0] = torch.squeeze(mod_thrusts_0)
        self.thrusts[:, 1] = torch.squeeze(mod_thrusts_1)
        self.thrusts[:, 2] = torch.squeeze(mod_thrusts_2)
        self.thrusts[:, 3] = torch.squeeze(mod_thrusts_3)

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0

        # spin spinning rotors
        prop_rot = self.thrust_cmds_damp * self.prop_max_rot
        self.dof_vel[:, 0] = prop_rot[:, 0]
        self.dof_vel[:, 1] = -1.0 * prop_rot[:, 1]
        self.dof_vel[:, 2] = prop_rot[:, 2]
        self.dof_vel[:, 3] = -1.0 * prop_rot[:, 3]

        self._copters.set_joint_velocities(self.dof_vel)

        # apply actions
        for i in range(4):
            self._copters.physics_rotors[i].apply_forces(self.thrusts[:, i], indices=self.all_indices)

    def post_reset(self) -> None:
        thrust_max = self.grav_z * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0
        self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.thrust_cmds_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_rot_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_max = torch.tensor(thrust_max, device=self._device, dtype=torch.float32)

        self.motor_linearity = 1.0
        self.prop_max_rot = 433.3

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 1
        self.actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        # Extra info
        self.extras = {}

        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {
            "rew_pos": torch_zeros(),
            "rew_orient": torch_zeros(),
            "rew_effort": torch_zeros(),
            "rew_spin": torch_zeros(),
            "raw_dist": torch_zeros(),
            "raw_orient": torch_zeros(),
            "raw_effort": torch_zeros(),
            "raw_spin": torch_zeros(),
        }

        self.root_pos, self.root_rot = self._copters.get_world_poses()
        self.root_velocities = self._copters.get_velocities()
        self.dof_pos = self._copters.get_joint_positions()
        self.dof_vel = self._copters.get_joint_velocities()

        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        # control parameters
        self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.thrust_cmds_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_rot_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)

        self.set_targets(self.all_indices)

    def set_targets(self, env_ids) -> None:
        num_sets = len(env_ids)
        envs_long = env_ids.long()
        # set target position randomly with x, y in (0, 0) and z in (2)
        self.target_positions[envs_long, 0:2] = torch.zeros((num_sets, 2), device=self._device)
        self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device) * 2.0

        # shift the target up so it visually aligns better
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        ball_pos[:, 2] += 0.0
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

        self.target_pos = self._compute_traj(4, env_ids, step_size=5)
        pos_expanded = self.root_pos.unsqueeze(1)
        self.rpos = self.target_pos - pos_expanded

    def reset_idx(self, env_ids) -> None:
        num_resets = len(env_ids)
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w

        self.dof_pos[env_ids, :] = torch_rand_float(-0.0, 0.0, (num_resets, self._copters.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0

        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        self._copters.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._copters.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)

        self._copters.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._copters.set_velocities(root_velocities[env_ids], indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.thrust_cmds_damp[env_ids] = 0
        self.thrust_rot_damp[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids]) / self._max_episode_length
            self.episode_sums[key][env_ids] = 0.0

    def lemniscate(self, t, c):
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        sin2p1 = torch.square(sin_t) + 1

        x = torch.stack([
            cos_t, sin_t * cos_t, c * sin_t
        ], dim=-1) / sin2p1.unsqueeze(-1)

        return x
    
    def scale_time(self, t, a: float=1.0):
        return t / (1 + 1/(a*torch.abs(t)))
    
    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + self.scale_time(self.traj_w[env_ids].unsqueeze(1) * t * self.dt)
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)
        
        target_pos = vmap(self.lemniscate)(t, self.traj_c[env_ids])
        target_pos = vmap(quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)

        crazyflie_pos_expanded = self._crazyflie_position.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, 3]

        dynamic_target_pos = target_pos + crazyflie_pos_expanded  # Resulting shape will be [4096, 4, 3]

        print("return 1: ",dynamic_target_pos.shape)

        return dynamic_target_pos
    
    def calculate_metrics(self) -> None:
        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        root_angvels = self.root_velocities[:, 3:]

        reward_distance_scale=1.2

        target_dist = torch.norm(self.rpos[:, [0]], dim=-1)
        pos_reward=torch.exp(-reward_distance_scale * target_dist)

        self.target_dist = target_dist
        self.root_positions = root_positions

        # orient reward
        ups = quat_axis(root_quats, 2)
        self.orient_z = ups[..., 2]
        up_reward = torch.clamp(ups[..., 2], min=0.0, max=1.0)

        # effort reward
        effort = torch.square(self.actions).sum(-1)
        effort_reward = 0.05 * torch.exp(-0.5 * effort)

        # spin reward
        spin = torch.square(root_angvels).sum(-1)
        spin_reward = 0.01 * torch.exp(-1.0 * spin)

        # combined reward
        self.rew_buf[:] = pos_reward + pos_reward * (up_reward + spin_reward) - effort_reward

        # log episode reward sums
        self.episode_sums["rew_pos"] += pos_reward
        self.episode_sums["rew_orient"] += up_reward
        self.episode_sums["rew_effort"] += effort_reward
        self.episode_sums["rew_spin"] += spin_reward

        # log raw info
        self.episode_sums["raw_dist"] += target_dist
        self.episode_sums["raw_orient"] += ups[..., 2]
        self.episode_sums["raw_effort"] += effort
        self.episode_sums["raw_spin"] += spin

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > 0.5, ones, die)

        # z >= 0.5 & z <= 5.0 & up > 0
        die = torch.where(self.root_positions[..., 2] < 1.5, ones, die)
        die = torch.where(self.root_positions[..., 2] > 6.0, ones, die)
        die = torch.where(self.orient_z < 0.0, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)#END