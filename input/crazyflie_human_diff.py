    # Written by human, the parts of modification different from the original one, 
    # including new buffer contents, new task goal, and new reward computation.
    def target_position_rot(self, root_positions, steps):
        radius = 0.3
        theta = torch.tensor(-2 * math.pi / 350 * steps)
        center_z = 2.0

        x = root_positions[:, 0]
        z = root_positions[:, 2]-center_z
        
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
        #root_positions[:, 0] -= 1
        root_quats = self.root_rot
        root_vel = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]
        self.root_positions = root_positions
        #self.root_positions[:, 0] += 1

        rot_target = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        rot_target[:, 1] = 1

        scaled_points, global_target_positions = self.target_position_rot(root_positions,0)
        rotated_points_list = []
        for i in range(4):
            _, rotated_points = self.target_position_rot(root_positions, i)
            rotated_points_list.append(rotated_points)

        # Average the rotated points across the 4 steps
        next_step = sum(rotated_points_list) / 4

        target_dist = torch.norm(root_positions - global_target_positions, dim=1)
        self.target_dist=target_dist

        pos_reward = torch.exp(-3*self.target_dist)

        # orient reward
        norm_rot_target = rot_target / torch.norm(rot_target, dim=1, keepdim=True)

        ups = quat_axis(root_quats, 2)
        norm_ups = ups / torch.norm(ups, dim=1, keepdim=True)

        cos_angle = torch.sum(norm_ups * norm_rot_target, dim=1)
        coeff_rot=1-cos_angle
        up_reward = torch.exp(-3*coeff_rot)

        radius=0.3
        desired_speed = 2*radius*math.pi / 3.5
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
        roll_penalty = torch.square(roll_angvel - desired_roll_angvel).sum(-1)
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
        self.rew_buf[:] = (pos_reward +speed_reward+coline_reward)(1+up_reward+spin_reward) 
        
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
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)