import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import os

class tensorboard_parser:
    def __init__(self, log_path):
        self.log_path = log_path
        self.ea = event_accumulator.EventAccumulator(log_path)
        self.ea.Reload()
        self.scalar_keys = self.ea.scalars.Keys()
        self.stats_df = pd.DataFrame(columns=['Metric', 'Mean', 'Max', 'Min', 'Std Dev', 'Variance'])
        # Remove the plot directory if it exists
        if os.path.exists('./plot'):
            import shutil
            shutil.rmtree('./plot')
        os.makedirs('./plot')

    # Function to compute descriptive statistics
    def compute_statistics(self, values):
        mean = np.mean(values)
        max_val = np.max(values)
        min_val = np.min(values)
        std_dev = np.std(values)
        variance = np.var(values)
        return mean, max_val, min_val, std_dev, variance

    def parse_and_plot(self):
        for key in self.scalar_keys:
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
                'Variance': variance
            }, ignore_index=True)
            
            # Create a plot for the scalar metric
            plt.figure()
            plt.plot(steps, values)
            plt.title(f'{key}\nMean: {mean:.4f}, Max: {max_val:.4f}, Min: {min_val:.4f}, Std Dev: {std_dev:.4f}, Variance: {variance:.4f}')
            plt.xlabel('Steps')
            plt.ylabel('Value')
            plt.grid(True)
            # Make sure the plot directory exists
            os.makedirs(f"./plot/{key.split('/')[0]}", exist_ok=True)
            plt.savefig(f'./plot/{key}.png',dpi=300)
            # Show the plot
            plt.show()

        # Display the statistics DataFrame
        print(self.stats_df)
        self.stats_df.to_csv('./stats.csv', index=False)

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
            sample_condition = np.mod(np.array(values).__index__(), 50) == 0
            sample = np.array(values)[sample_condition]
            # Append statistics to the DataFrame
            self.stats_df = self.stats_df.append({
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
    tb_parser = tensorboard_parser('runs/ppo_1')
    # Parse the log files and plot the scalar metrics
    tb_parser.parse_and_plot()
    # Parse the log files and return the DataFrame
    df = tb_parser.parse()
    print(df)