import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import os
from loguru import logger
from utils.exceptions import BLANK_TENSORBOARD_LOG_ERROR
class tensorboard_parser:
    def __init__(self, log_path, save=False, plot=False, dir_path='./plot', name='all_metrics', precision=2):
        self.log_path = log_path
        self.ea = event_accumulator.EventAccumulator(log_path)
        self.ea.Reload()
        self.scalar_keys = self.ea.scalars.Keys()
        self.stats_df = pd.DataFrame(columns=['Metric', 'Mean', 'Max', 'Min', 'Std Dev', 'Variance', 'Sample'])
        self.save = save
        self.plot = plot
        self.dir_path = dir_path
        self.name = name
        self.precision = precision
        if self.save and not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    def compute_statistics(self, values):
        mean = round(np.mean(values), self.precision)
        max_val = round(np.max(values), self.precision)
        min_val = round(np.min(values), self.precision)
        std_dev = round(np.std(values, ddof=1), self.precision)
        variance = round(np.var(values, ddof=1), self.precision)
        return mean, max_val, min_val, std_dev, variance

    def format_samples(self, values):
        return [round(v, self.precision) for v in values]

    def parse_and_plot(self, dpi=300):
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
            sample_condition = np.arange(len(values)) % 50 == 0
            sample = self.format_samples(np.array(values)[sample_condition])
            new_row = pd.DataFrame({
                'Metric': [key],
                'Mean': [mean],
                'Max': [max_val],
                'Min': [min_val],
                'Std Dev': [std_dev],
                'Variance': [variance],
                'Sample': [sample]  # Formatted 'Sample' column
            })
            self.stats_df = pd.concat([self.stats_df, new_row], ignore_index=True)
            axes[idx].plot(steps, values)
            axes[idx].set_title(f'{key} - Mean: {mean}, Max: {max_val}, Min: {min_val}, Std Dev: {std_dev}, Variance: {variance}')
            axes[idx].set_xlabel('Steps')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True)
        
        plt.tight_layout()
        if self.save:
            plt.savefig(os.path.join(self.dir_path, f'{self.name}.png'), dpi=dpi)
            self.stats_df.to_pickle(os.path.join(self.dir_path, f'{self.name}.pkl'))
            logger.info(f"Saved the plot to {os.path.join(self.dir_path, f'{self.name}.png')}")
            logger.info(f"Saved the statistics to {os.path.join(self.dir_path, f'{self.name}.pkl')}")
        if self.plot:
            plt.show()
        
        

    def parse(self) -> pd.DataFrame:
        if not self.scalar_keys:
            raise BLANK_TENSORBOARD_LOG_ERROR
        for key in self.scalar_keys:
            events = self.ea.Scalars(key)
            values = [e.value for e in events]
            mean, max_val, min_val, std_dev, variance = self.compute_statistics(values)
            sample_condition = np.arange(len(values)) % 50 == 0
            sample = self.format_samples(np.array(values)[sample_condition])

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
        logger.info(self.stats_df)
        return self.stats_df


if __name__ == '__main__':
    log_path = '/home/simonwsy/SimpleEureka/results/2024-06-15_09-19-40/iter3/0/summaries/events.out.tfevents.1718415621.simonwsy-Precision-5820-Tower'
    tb_parser = tensorboard_parser(log_path, save=False, plot=True)
    tb_parser.parse_and_plot()
    print(tb_parser.parse())