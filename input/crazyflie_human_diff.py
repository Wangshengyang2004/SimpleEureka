# Features: Use quaternion to control the orientation of the drone, (0,0,0,1) means a 360 degree flip
# Use Curiosity reward to encourage the agent to perform more flips   
def calculate_metrics(self) -> None:
       root_positions = self.root_pos - self._env_pos
       root_angvels = self.root_velocities[:, 3:]
       root_quats = self.root_rot

       target_dist = torch.sqrt(torch.square(self.target_positions - root_positions).sum(-1))
       self.target_dist = target_dist
       self.root_positions = root_positions

       position_temp: float = 0.1  # Lower temperature to widen the effective range
       flip_temp: float = 0.05     # Lower temperature to reduce sensitivity to exact angular velocity
       quat_temp: float= 0.1
       spin_temp: float=0.1

       # pos reward
       position_error = torch.norm(self.target_positions - root_positions, dim=-1)
       position_reward = torch.exp(-position_temp * position_error ** 2)

       #phase 1
       target_angvel_y = 5.0  # Reduced target angular velocity threshold to make it achievable
       angvel_error = torch.abs(root_angvels[..., 1] - target_angvel_y)
       flip_reward = torch.exp(-flip_temp * angvel_error ** 2)

       target_quat=self.target_quats
       quat_variation = torch.norm(root_quats - target_quat, dim=-1)
       quaternion_reward = torch.exp(-quat_temp*quat_variation**2)

       #phase 2

       spin = torch.abs(root_angvels).sum(-1)
       spin_reward = torch.exp(-spin_temp * spin)

       ups = quat_axis(root_quats, 2)
       self.orient_z = ups[..., 2]
       up_reward = torch.clamp(ups[..., 2], min=0.0, max=1.0)

       successful_flip_threshold = 0.1  # Stricter threshold
       flip_completion = (torch.abs(root_quats[:, 2] - 1.0) < successful_flip_threshold).float()
       successful_flip_reward = flip_completion  # High reward for successful flip

       num_flips = self.episode_sums["successful_flip_reward"]

       # Create boolean mask for environments with fewer than 2 successful flips
       flip_mask = num_flips < 6

       # Set curiosity reward if 2 or more successful flips have been achieved
       curiosity_reward = torch.where(num_flips >= 2, torch.tensor(0.5, device=self._device), torch.tensor(0.0, device=self._device))

       # Apply the mask to rewards
       flip_reward = torch.where(flip_mask, flip_reward, spin_reward)
       quaternion_reward = torch.where(flip_mask, quaternion_reward, up_reward)
       successful_flip_reward = torch.where(flip_mask, successful_flip_reward, torch.tensor(1.0, device=self._device))

       # combined reward
       self.rew_buf[:] = 0.2*flip_reward+0.2*position_reward+0.2*quaternion_reward+0.2*successful_flip_reward+0.2*curiosity_reward

       # log episode reward sums
       self.episode_sums["position_reward"] += position_reward
       self.episode_sums["flip_reward"] += flip_reward
       self.episode_sums["quaternion_reward"] += quaternion_reward
       self.episode_sums["successful_flip_reward"] += successful_flip_reward
       self.episode_sums["curiosity_reward"] += curiosity_reward

   def is_done(self) -> None:
       # resets due to misbehavior
       ones = torch.ones_like(self.reset_buf)
       die = torch.zeros_like(self.reset_buf)
       die = torch.where(self.target_dist > 5.0, ones, die)

       # z >= 0.5 & z <= 5.0 & up > 0
       die = torch.where(self.root_positions[..., 2] < 0.5, ones, die)
       die = torch.where(self.root_positions[..., 2] > 5.0, ones, die)

       # resets due to episode length
       self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)