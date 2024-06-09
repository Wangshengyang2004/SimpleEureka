import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import os
from loguru import logger
from utils.exceptions import BLANK_TENSORBOARD_LOG_ERROR

class tensorboard_parser:
    def __init__(self, log_path, save=False, plot=False, dir_path='./plot', name='all_metrics'):
        self.log_path = log_path
        self.ea = event_accumulator.EventAccumulator(log_path)
        self.ea.Reload()
        self.scalar_keys = self.ea.scalars.Keys()
        self.stats_df = pd.DataFrame(columns=['Metric', 'Mean', 'Max', 'Min', 'Std Dev', 'Variance', 'Sample'])
        self.save = save
        self.plot = plot
        self.dir_path = dir_path
        self.name = name
        if self.save and not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    # Function to compute descriptive statistics
    def compute_statistics(self, values):
        mean = np.mean(values)
        max_val = np.max(values)
        min_val = np.min(values)
        std_dev = np.std(values)
        variance = np.var(values)
        return mean, max_val, min_val, std_dev, variance

    # @logger.catch
    def parse_and_plot(self):
        if not self.scalar_keys:
            raise BLANK_TENSORBOARD_LOG_ERROR
        num_metrics = len(self.scalar_keys)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(15, 5 * num_metrics))
        if num_metrics == 1:
            axes = [axes]  # Ensure axes is a list even with one plot
        
        for idx, key in enumerate(self.scalar_keys):
            events = self.ea.Scalars(key)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            mean, max_val, min_val, std_dev, variance = self.compute_statistics(values)
            
            new_row = pd.DataFrame({
                'Metric': [key],
                'Mean': [mean],
                'Max': [max_val],
                'Min': [min_val],
                'Std Dev': [std_dev],
                'Variance': [variance],
                'Sample': [None]  # Placeholder for the 'Sample' column
            })
            self.stats_df = pd.concat([self.stats_df, new_row], ignore_index=True)
            
            axes[idx].plot(steps, values)
            axes[idx].set_title(f'{key}\nMean: {mean:.4f}, Max: {max_val:.4f}, Min: {min_val:.4f}, Std Dev: {std_dev:.4f}, Variance: {variance:.4f}')
            axes[idx].set_xlabel('Steps')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True)
        
        plt.tight_layout()
        if self.save:
            plt.savefig(os.path.join(self.dir_path, f'{self.name}.png'), dpi=300)
            logger.info(f"Saved the plot to {os.path.join(self.dir_path, f'{self.name}.png')}")
        if self.plot:
            plt.show()
        print(self.stats_df)
        self.stats_df.to_csv(os.path.join(self.dir_path, f'{self.name}.csv'), index=False)

    # @logger.catch
    def parse(self, field=["Episode/raw_dist", "Episode/raw_effort", "Episode/raw_orient", "Episode/raw_spin", "Episode/rew_effort", "Episode/rew_orient", "Episode/rew_pos","Episode/rew_spin","rewards/step"]) -> pd.DataFrame:
        if not self.scalar_keys:
            raise BLANK_TENSORBOARD_LOG_ERROR
        for key in self.scalar_keys:
            if key not in field:
                continue
            events = self.ea.Scalars(key)
            values = [e.value for e in events]
            mean, max_val, min_val, std_dev, variance = self.compute_statistics(values)
            sample_condition = np.arange(len(values)) % 50 == 0
            sample = np.array(values)[sample_condition]
            
            new_row = pd.DataFrame({
                'Metric': [key],
                'Mean': [mean],
                'Max': [max_val],
                'Min': [min_val],
                'Std Dev': [std_dev],
                'Variance': [variance],
                'Sample': [sample]
            })
            self.stats_df = pd.concat([self.stats_df, new_row], ignore_index=True)
        return self.stats_df
    
    @staticmethod
    def parse_tensorboard(log_path, *args, **kwargs):
        tb_parser = tensorboard_parser(log_path)
        return tb_parser.parse(*args, **kwargs)
