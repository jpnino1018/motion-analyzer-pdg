import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class AccelerometerData:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    timestamps: np.ndarray

    @property
    def magnitude(self) -> np.ndarray:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

class SignalProcessor:
    def __init__(self, height_threshold: float = 0.2, min_distance: int = 10):
        self.height_threshold = height_threshold
        self.min_distance = min_distance

    def detect_peaks(self,
                     signal: np.ndarray,
                     n_reps: Optional[int] = None,
                     height: Optional[float] = None,
                     distance: Optional[int] = None,
                     prominence: Optional[float] = None,
                     width: Optional[float] = None) -> np.ndarray:
        """Enhanced peak detection with optional prominence and peak-count limiting.

        Args:
            signal: 1D numpy array with the signal to analyze
            n_reps: If provided, limit to the top n peaks by prominence/height
            height: Minimal height for peaks (overrides instance threshold)
            distance: Minimal distance between peaks (overrides instance value)
            prominence: Minimal prominence for peaks (preferred for ignoring small bumps)
            width: Minimal width for peaks

        Returns:
            numpy array with indices of detected peaks
        """
        # Use provided params or fall back to instance defaults
        height = self.height_threshold if height is None else height
        distance = self.min_distance if distance is None else distance

        # Call scipy find_peaks with requested filters
        peaks, props = find_peaks(signal,
                                  height=height,
                                  distance=distance,
                                  prominence=prominence,
                                  width=width)

        if n_reps and len(peaks) > n_reps:
            # Prefer peaks with larger prominence first, then height
            prominences = props.get('prominences')
            heights = props.get('peak_heights')
            # Build sorting key: (prominence, height)
            if prominences is not None:
                order = np.argsort(prominences)
            elif heights is not None:
                order = np.argsort(heights)
            else:
                order = np.argsort(signal[peaks])

            top_idx = order[-n_reps:]
            peaks = peaks[top_idx]
            peaks.sort()

        return peaks

    def calculate_intervals(self, timestamps: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """Calculate time intervals between consecutive peaks"""
        return np.diff(timestamps[peaks])

    def calculate_vertical_range(self, acc_data: AccelerometerData) -> float:
        """Calculate maximum range across all axes"""
        ranges = [
            np.ptp(acc_data.x),
            np.ptp(acc_data.y),
            np.ptp(acc_data.z)
        ]
        return max(ranges)