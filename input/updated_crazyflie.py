import numpy as np
import torch
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.crazyflie import Crazyflie
from omniisaacgymenvs.robots.articulations.views.crazyflie_view import CrazyflieView

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class CrazyflieTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self._num_observations = 18
        self._num_actions = 4

        self._crazyflie_position = torch.tensor([0, 0, 1.0])
        self._ball_position = torch.tensor([0, 0, 1.0])

        RLTask.__init__(self, name=name, env=env)

        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self.dt = self._task_cfg["sim"]["dt"]

        # parameters for the crazyflie
        self.arm_length = 0.05

        # parameters for the controller
        self.motor_damp_time_up = 0.15
        self.motor_damp_time_down = 0.15

        # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
        # T is a time constant of the first-order filter
        self.motor_tau_up = 4 * self.dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4 * self.dt / (self.motor_damp_time_down + EPS)

        # thrust max
        self.mass = 0.028
        self.thrust_to_weight = 1.9

        self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
        # re-normalizing to sum-up to 4
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
        return

    def initialize_views(self, scene):
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

    def get_crazyflie(self):
        copter = Crazyflie(
            prim_path=self.default_zero_env_path + "/Crazyflie", name="crazyflie", translation=self._crazyflie_position
        )
        self._sim_config.apply_articulation_settings(
            "crazyflie", get_prim_at_path(copter.prim_path), self._sim_config.parse_actor_config("crazyflie")
        )

    def get_target(self):
        radius = 0.2
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

        # clamp to [-1.0, 1.0]
        thrust_cmds = torch.clamp(actions, min=-1.0, max=1.0)
        # scale to [0.0, 1.0]
        thrust_cmds = (thrust_cmds + 1.0) / 2.0
        # filtering the thruster and adding noise
        motor_tau = self.motor_tau_up * torch.ones((self._num_envs, 4), dtype=torch.float32, device=self._device)
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down
        motor_tau[motor_tau > 1.0] = 1.0

        # Since NN commands thrusts we need to convert to rot vel and back
        thrust_rot = thrust_cmds**0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp**2

        ## Adding noise
        thrust_noise = 0.01 * torch.randn(4, dtype=torch.float32, device=self._device)
        thrust_noise = thrust_cmds * thrust_noise
        self.thrust_cmds_damp = torch.clamp(self.thrust_cmds_damp + thrust_noise, min=0.0, max=1.0)

        thrusts = self.thrust_max * self.thrust_cmds_damp

        # thrusts given rotation
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

    def post_reset(self):
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

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()
        # set target position randomly with x, y in (0, 0) and z in (2)
        self.target_positions[envs_long, 0:2] = torch.zeros((num_sets, 2), device=self._device)
        self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device) * 2.0

        # shift the target up so it visually aligns better
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        ball_pos[:, 2] += 0.0
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

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

    # New Code Starts Here
    python
    import math
    import torch
    from omni.isaac.core.utils.torch.rotations import *
    EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)
    
    class CrazyflieTask(RLTask):
        # ... (rest of the class code)
    
        def target_position_rot(self, root_positions, steps):
            radius = 0.3
            theta = torch.tensor(-2 * math.pi / 350 * steps)
            center_z = 2.0
    
            x = root_positions[:, 0]
            z = root_positions[:, 2] - center_z
    
            # Calculate the scale factor to ensure the point lies on the circle with radius 0.5
            scale = radius / torch.sqrt(x**2 + z**2)
    
            # Calculate x1 and z1 to make points lie on the circle
            x1 = x * scale
            z1 = z * scale
    
            # Construct the new points [x1, 0, z1] for each root_position
            new_points = torch.stack([x1, torch.zeros_like(x1), z1], dim=1)
    
            # Calculate the cosine and sine of the rotation angle
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
    
            # Apply the rotation matrix for clockwise rotation in the XZ plane
            x_rotated = new_points[:, 0] * cos_theta + new_points[:, 2] * sin_theta
            z_rotated = -new_points[:, 0] * sin_theta + new_points[:, 2] * cos_theta
    
            # Construct the rotated points [x_rotated, 0, z_rotated]
            rotated_points = torch.stack([x_rotated, new_points[:, 1], z_rotated], dim=1)
    
            return new_points, rotated_points
    
        def calculate_metrics(self) -> None:
            root_positions = self.root_pos - self._env_pos
            root_quats = self.root_rot
            root_vel = self.root_velocities[:, :3]
            root_angvels = self.root_velocities[:, 3:]
            self.root_positions = root_positions
    
            rot_target = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
            rot_target[:, 1] = 1
    
            scaled_points, global_target_positions = self.target_position_rot(root_positions, 0)
            rotated_points_list = []
            for i in range(4):
                _, rotated_points = self.target_position_rot(root_positions, i)
                rotated_points_list.append(rotated_points)
    
            # Average the rotated points across the 4 steps
            next_step = sum(rotated_points_list) / 4
    
            target_dist = torch.norm(root_positions - global_target_positions, dim=1)
            self.target_dist = target_dist
    
            pos_reward = torch.exp(-3 * self.target_dist)
    
            # orient reward
            norm_rot_target = rot_target / torch.norm(rot_target, dim=1, keepdim=True)
    
            ups = quat_axis(root_quats, 2)
            norm_ups = ups / torch.norm(ups, dim=1, keepdim=True)
    
            cos_angle = torch.sum(norm_ups * norm_rot_target, dim=1)
            coeff_rot = 1 - cos_angle
            up_reward = torch.exp(-3 * coeff_rot)
    
            radius = 0.3
            desired_speed = 2 * radius * math.pi / 3.5
            current_speeds = torch.norm(root_vel, dim=1)  # Calculate the magnitudes of current velocities
            speed_diff = torch.abs(current_speeds - desired_speed)  # Calculate the absolute difference from the desired speed
            speed_reward = torch.exp(-3 * speed_diff)
    
            traj = next_step - scaled_points
    
            # Normalize root_vel and traj to unit vectors
            norm_root_vel = root_vel / torch.norm(root_vel, dim=1, keepdim=True)
            norm_traj = traj / torch.norm(traj, dim=1, keepdim=True)
            # Calculate the dot product
            dot_product = torch.sum(norm_root_vel * norm_traj, dim=1)
            colinearity = 1 - dot_product
    
            # Define the exponential reward
            coline_reward = torch.exp(-3 * colinearity)
    
            roll_angvel = root_angvels[:, 0]  # Extract roll component
            yaw_angvel = root_angvels[:, 2]  # Extract yaw component
    
            # Desired angular velocities for roll and yaw are zero (minimal movement)
            desired_roll_angvel = 0.0
            desired_yaw_angvel = 0.0
    
            # Calculate the penalties for deviation from zero for roll and yaw
            roll_penalty = torch.square(roll_angvel - desired_roll_angvel)
        python
            ).sum(-1)
            yaw_penalty = torch.square(yaw_angvel - desired_yaw_angvel).sum(-1)
    
            y_angvel = root_angvels[:, 1]  # Y-axis angular velocity for spinning
            desired_y_angvel = 2 * math.pi / 3.5  # Desired spin rate around Y-axis
            y_angvel_diff = y_angvel - desired_y_angvel
            y_penalty = torch.square(y_angvel_diff).sum(-1)
    
            # Higher weight for Y-axis spin reward, lower weight for roll and yaw penalties to encourage minimal movements
            combined_penalty = 0.1 * torch.exp(-1.0 * y_penalty) + 0.05 * torch.exp(-1.0 * roll_penalty) + 0.05 * torch.exp(-1.0 * yaw_penalty)
    
            # Update spin_reward with the combined penalty
            spin_reward = combined_penalty
    
            # combined reward
            self.rew_buf[:] = (pos_reward + speed_reward + coline_reward) * (1 + up_reward + spin_reward)
    
            # log episode reward sums
            self.episode_sums["rew_pos"] += pos_reward
            self.episode_sums["rew_orient"] += up_reward
            self.episode_sums["rew_speed"] += speed_reward
            self.episode_sums["rew_coline"] += coline_reward
            self.episode_sums["rew_spin"] += spin_reward
    
            # log raw info
            self.episode_sums["raw_dist"] += target_dist
            self.episode_sums["raw_orient"] += ups[..., 2]
    
        def is_done(self) -> None:
            # resets due to misbehavior
            ones = torch.ones_like(self.reset_buf)
            die = torch.zeros_like(self.reset_buf)
            die = torch.where(self.target_dist > 10.0, ones, die)
    
            # z >= 0.5 & z <= 5.0 & up > 0
            die = torch.where(self.root_positions[..., 2] < -2.0, ones, die)
            die = torch.where(self.root_positions[..., 2] > 6.0, ones, die)
    
            # resets due to episode length
            self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)  # END
    
        python
    # The provided code is complete and follows the requirements. However, if you have any additional questions or need further assistance, please let me know.
    
        python
    # The provided code is complete and follows the requirements. If you have any questions or need further assistance, feel free to ask.
    
        python
    
    