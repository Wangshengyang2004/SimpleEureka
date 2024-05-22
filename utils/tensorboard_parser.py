import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import os
from loguru import logger
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
        if self.save:
            if os.path.exists(self.dir_path):
                pass
            else:
                os.makedirs(self.dir_path)

    # Function to compute descriptive statistics
    def compute_statistics(self, values):
        mean = np.mean(values)
        max_val = np.max(values)
        min_val = np.min(values)
        std_dev = np.std(values)
        variance = np.var(values)
        return mean, max_val, min_val, std_dev, variance

    def parse_and_plot(self):
        num_metrics = len(self.scalar_keys)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(15, 5 * num_metrics))
        if num_metrics == 1:
            axes = [axes]  # Ensure axes is a list even with one plot
        
        for idx, key in enumerate(self.scalar_keys):
            # Retrieve scalar events for the key
            events = self.ea.Scalars(key)
            
            # Extract steps and values
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            # Compute statistics
            mean, max_val, min_val, std_dev, variance = self.compute_statistics(values)
            
            # Append statistics to the DataFrame
            self.stats_df = self.stats_df._append({
                'Metric': key,
                'Mean': mean,
                'Max': max_val,
                'Min': min_val,
                'Std Dev': std_dev,
                'Variance': variance,
                'Sample': None  # Placeholder for the 'Sample' column
            }, ignore_index=True)
            
            # Plot the scalar metric
            axes[idx].plot(steps, values)
            axes[idx].set_title(f'{key}\nMean: {mean:.4f}, Max: {max_val:.4f}, Min: {min_val:.4f}, Std Dev: {std_dev:.4f}, Variance: {variance:.4f}')
            axes[idx].set_xlabel('Steps')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True)
        
        plt.tight_layout()
        
        if self.save:
            plt.savefig(os.path.join(self.dir_path, f'{self.name}.png'), dpi=300)
            logger.info(f"Saved the plot to {os.path.join(self.dir_path, f'{self.name}.png')}")
        
        # Show the plot
        if self.plot:
            plt.show()

        # Display the statistics DataFrame
        print(self.stats_df)
        self.stats_df.to_csv(os.path.join(self.dir_path, f'{self.name}.csv'), index=False)

    def parse(self, field=["Episode/raw_dist", "Episode/raw_effort", "Episode/raw_orient", "Episode/raw_spin", "Episode/rew_effort", "Episode/rew_orient", "Episode/rew_pos","Episode/rew_spin","rewards/step"]) -> pd.DataFrame:
        for key in self.scalar_keys:
            if key not in field:
                continue
            # Retrieve scalar events for the key
            events = self.ea.Scalars(key)
            
            # Extract steps and values
            values = [e.value for e in events]
            
            # Compute statistics
            mean, max_val, min_val, std_dev, variance = self.compute_statistics(values)
            
            # Sample every 50th value
            sample_condition = np.arange(len(values)) % 50 == 0
            sample = np.array(values)[sample_condition]
            
            # Append statistics to the DataFrame
            self.stats_df = self.stats_df._append({
                'Metric': key,
                'Mean': mean,
                'Max': max_val,
                'Min': min_val,
                'Std Dev': std_dev,
                'Variance': variance,
                'Sample': sample
            }, ignore_index=True)

        return self.stats_df
    
    @staticmethod
    def parse_tensorboard(log_path, *args, **kwargs):
        tb_parser = tensorboard_parser(log_path)
        return tb_parser.parse(*args, **kwargs)
    
if __name__ == '__main__':
    # Create an instance of the tensorboard_parser class
    path = r"H:\Omniverse\Library\isaac_sim-2023.1.1\SimpleEureka\results\2024-05-21_23-51-53\iter0\1\summaries\events.out.tfevents.1716307000.DESKTOP-P1IIRLN"
    tb_parser = tensorboard_parser(path, save=True, plot=True)
    # Parse the log files and plot the scalar metrics
    tb_parser.parse_and_plot()
    # Parse the log files and return the DataFrame
    df = tb_parser.parse()
    print(df)
