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
    vertical_amplitude_mean: float  # Mean vertical displacement (cm) via double integration
    vertical_amplitude_decay: float  # Peak magnitude decay rate (m/s² per rep, negative = reduction)
    vertical_amplitude_ratio: float  # Peak magnitude first/second half ratio (> 1.0 = reduction)
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

    def calculate_amplitude_decay(self, peak_values: np.ndarray) -> float:
        """Calculate amplitude decay rate using linear regression on peak magnitudes.
        
        Args:
            peak_values: Array of peak magnitudes (m/s²) in temporal order
            
        Returns:
            Decay rate in m/s² per repetition (negative = progressive reduction)
        """
        if len(peak_values) < 3:
            return 0.0
        
        x = np.arange(len(peak_values))
        slope, _, _, _, _ = linregress(x, peak_values)
        return slope  # m/s² per repetition

    def calculate_amplitude_ratio(self, peak_values: np.ndarray) -> float:
        """Calculate first-half vs second-half peak magnitude ratio.
        
        Args:
            peak_values: Array of peak magnitudes (m/s²) in temporal order
            
        Returns:
            Ratio of first_half_mean / second_half_mean 
            (> 1.0 = reduction in second half, < 1.0 = increase/warm-up)
        """
        if len(peak_values) < 4:
            return 1.0
        
        mid = len(peak_values) // 2
        first_half = np.mean(peak_values[:mid])
        second_half = np.mean(peak_values[mid:])
        
        if second_half == 0:
            return 1.0
        
        return first_half / second_half

    def calculate_vertical_displacement(self, z_accel: np.ndarray, timestamps: np.ndarray, peaks: np.ndarray):
        """Calculate vertical displacement per repetition by double integration.
        
        Args:
            z_accel: Z-axis acceleration data (m/s²)
            timestamps: Timestamps in milliseconds
            peaks: Indices of detected peaks (repetitions)
            
        Returns:
            Tuple of (mean_amplitude_cm, per_rep_amplitudes_cm)
        """
        if len(z_accel) < 2 or len(peaks) < 2:
            return 0.0, []
        
        # Convert timestamps to seconds and calculate dt
        time_seconds = timestamps / 1000.0
        dt = np.diff(time_seconds)
        
        if len(dt) == 0 or np.any(dt <= 0):
            return 0.0, []
        
        # Remove gravity offset
        z_accel_centered = z_accel - np.mean(z_accel)
        
        # First integration: acceleration -> velocity
        velocity = np.zeros_like(z_accel)
        for i in range(1, len(z_accel)):
            velocity[i] = velocity[i-1] + (z_accel_centered[i-1] + z_accel_centered[i]) / 2 * dt[i-1]
        
        # Remove velocity drift
        velocity_centered = velocity - np.mean(velocity)
        
        # Second integration: velocity -> displacement
        displacement = np.zeros_like(velocity)
        for i in range(1, len(velocity)):
            displacement[i] = displacement[i-1] + (velocity_centered[i-1] + velocity_centered[i]) / 2 * dt[i-1]
        
        # Calculate amplitude for each repetition
        rep_amplitudes = []
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            
            rep_displacement = displacement[start_idx:end_idx]
            if len(rep_displacement) > 0:
                rep_amplitude = np.max(rep_displacement) - np.min(rep_displacement)
                rep_amplitudes.append(rep_amplitude * 100)  # convert to cm
        
        if not rep_amplitudes:
            return 0.0, []
        
        mean_amplitude_cm = np.mean(rep_amplitudes)
        return mean_amplitude_cm, rep_amplitudes

    def analyze_movement(self, 
                        magnitudes: np.ndarray,
                        peaks: np.ndarray,
                        intervals: np.ndarray,
                        z_accel: np.ndarray,
                        timestamps: np.ndarray) -> MovementMetrics:
        """Comprehensive movement analysis"""
        if len(peaks) == 0:
            return MovementMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0)

        peak_magnitudes = magnitudes[peaks]
        mean_amplitude, per_rep_amplitudes = self.calculate_vertical_displacement(z_accel, timestamps, peaks)
        
        # Use peak magnitudes for progression metrics (clearer signal than displacement)
        amplitude_decay = self.calculate_amplitude_decay(peak_magnitudes)
        amplitude_ratio = self.calculate_amplitude_ratio(peak_magnitudes)
        
        return MovementMetrics(
            n_reps=len(peaks),
            magnitude_mean=peak_magnitudes.mean(),
            magnitude_max=peak_magnitudes.max(),
            rep_time_mean=intervals.mean() if len(intervals) > 0 else 0.0,
            rep_time_std=intervals.std() if len(intervals) > 0 else 0.0,
            fatigue_index=self.calculate_fatigue(peak_magnitudes),
            slowdown_rate=self.calculate_slowdown(intervals),
            vertical_amplitude_mean=mean_amplitude,
            vertical_amplitude_decay=amplitude_decay,
            vertical_amplitude_ratio=amplitude_ratio,
            hesitations=self.count_hesitations(intervals)
        )