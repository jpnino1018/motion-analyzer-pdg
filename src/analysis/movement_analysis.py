import numpy as np
from scipy.stats import linregress
from dataclasses import dataclass
from typing import List

@dataclass
class MovementMetrics:
    n_reps: int
    magnitude_mean: float
    magnitude_max: float
    rep_time_mean: float
    rep_time_std: float
    fatigue_index: float
    slowdown_rate: float
    vertical_amplitude: float
    hesitations: int

class MovementAnalyzer:
    def __init__(self, std_threshold: float = 1.5):
        self.std_threshold = std_threshold

    def calculate_fatigue(self, peak_magnitudes: np.ndarray) -> float:
        """Calculate fatigue index using first and second half comparison"""
        if len(peak_magnitudes) < 4:
            return 0.0
            
        mid_point = len(peak_magnitudes) // 2
        first_half = peak_magnitudes[:mid_point].mean()
        second_half = peak_magnitudes[mid_point:].mean()
        
        return (first_half - second_half) / first_half if first_half != 0 else 0.0

    def calculate_slowdown(self, intervals: np.ndarray) -> float:
        """Calculate movement slowdown rate"""
        if len(intervals) < 3:
            return 0.0
            
        x = np.arange(len(intervals))
        slope, _, _, _, _ = linregress(x, intervals)
        return slope

    def count_hesitations(self, intervals: np.ndarray) -> int:
        """Count movements that took significantly longer than average"""
        if len(intervals) < 3:
            return 0
            
        threshold = intervals.mean() + self.std_threshold * intervals.std()
        return np.sum(intervals > threshold)

    def analyze_movement(self, 
                        magnitudes: np.ndarray,
                        peaks: np.ndarray,
                        intervals: np.ndarray,
                        vertical_amplitude: float) -> MovementMetrics:
        """Comprehensive movement analysis"""
        if len(peaks) == 0:
            return MovementMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        peak_magnitudes = magnitudes[peaks]
        
        return MovementMetrics(
            n_reps=len(peaks),
            magnitude_mean=peak_magnitudes.mean(),
            magnitude_max=peak_magnitudes.max(),
            rep_time_mean=intervals.mean() if len(intervals) > 0 else 0.0,
            rep_time_std=intervals.std() if len(intervals) > 0 else 0.0,
            fatigue_index=self.calculate_fatigue(peak_magnitudes),
            slowdown_rate=self.calculate_slowdown(intervals),
            vertical_amplitude=vertical_amplitude,
            hesitations=self.count_hesitations(intervals)
        )