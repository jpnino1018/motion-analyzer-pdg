import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional
from src.preprocessing.signal_processing import AccelerometerData

class MovementVisualizer:
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]

    def plot_movement_data(self, 
                         acc_data: AccelerometerData, 
                         peaks: np.ndarray,
                         title: str,
                         save_path: Optional[str] = None):
        """Plot movement data with detected peaks"""
        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])
        
        # Plot magnitude with peaks
        magnitude = acc_data.magnitude
        time_seconds = (acc_data.timestamps - acc_data.timestamps[0]) / 1000  # Convert to seconds
        
        ax1.plot(time_seconds, magnitude, label='Magnitude')
        if len(peaks) > 0:
            ax1.plot(time_seconds[peaks], magnitude[peaks], 'ro', label='Peaks')
        
        ax1.set_title(f'Movement Analysis - {title}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration Magnitude (m/s²)')
        ax1.legend()
        
        # Plot individual axes
        ax2.plot(time_seconds, acc_data.x, 'r-', label='X', alpha=0.7)
        ax2.plot(time_seconds, acc_data.y, 'g-', label='Y', alpha=0.7)
        ax2.plot(time_seconds, acc_data.z, 'b-', label='Z', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Acceleration (m/s²)')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_movement_comparison(self,
                               left_data: AccelerometerData,
                               right_data: AccelerometerData,
                               left_peaks: np.ndarray,
                               right_peaks: np.ndarray,
                               title: str,
                               save_path: Optional[str] = None):
        """Plot left and right movement data side by side"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Left side plots
        time_left = (left_data.timestamps - left_data.timestamps[0]) / 1000
        mag_left = left_data.magnitude
        
        ax1.plot(time_left, mag_left, 'b-', label='Magnitude')
        if len(left_peaks) > 0:
            ax1.plot(time_left[left_peaks], mag_left[left_peaks], 'ro', label='Peaks')
        ax1.set_title('Left Side')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Magnitude (m/s²)')
        ax1.legend()
        
        ax3.plot(time_left, left_data.x, 'r-', label='X', alpha=0.7)
        ax3.plot(time_left, left_data.y, 'g-', label='Y', alpha=0.7)
        ax3.plot(time_left, left_data.z, 'b-', label='Z', alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.legend()
        
        # Right side plots
        time_right = (right_data.timestamps - right_data.timestamps[0]) / 1000
        mag_right = right_data.magnitude
        
        ax2.plot(time_right, mag_right, 'b-', label='Magnitude')
        if len(right_peaks) > 0:
            ax2.plot(time_right[right_peaks], mag_right[right_peaks], 'ro', label='Peaks')
        ax2.set_title('Right Side')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Magnitude (m/s²)')
        ax2.legend()
        
        ax4.plot(time_right, right_data.x, 'r-', label='X', alpha=0.7)
        ax4.plot(time_right, right_data.y, 'g-', label='Y', alpha=0.7)
        ax4.plot(time_right, right_data.z, 'b-', label='Z', alpha=0.7)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Acceleration (m/s²)')
        ax4.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_metrics_comparison(self, 
                              metrics_list: List[dict], 
                              groups: List[str],
                              save_path: Optional[str] = None):
        """Plot comparison of movement metrics between groups"""
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df['group'] = groups
        
        metrics_to_plot = ['mag_prom', 'tiempo_prom_rep', 'fatiga', 'enlentecimiento', 'titubeos']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics_to_plot):
            sns.boxplot(data=metrics_df, x='group', y=metric, ax=axes[i])
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_xlabel('')
        
        if len(axes) > len(metrics_to_plot):
            fig.delaxes(axes[-1])
        
        plt.suptitle('Movement Metrics Comparison')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()