import numpy as np
from typing import Dict, List, Any
from src.preprocessing.signal_processing import AccelerometerData, SignalProcessor
from src.analysis.movement_analysis import MovementAnalyzer, MovementMetrics
from src.preprocessing.cleaners import recortar_inactividad

class MovementProcessor:
    def __init__(self):
        self.signal_processor = SignalProcessor()
        self.movement_analyzer = MovementAnalyzer()

    def process_movement_data(self, data: List[Dict[str, Any]], n_reps: int = 10, 
                             trim_inactive: bool = True, inactive_threshold: float = 0.5, 
                             min_active_length: int = 50) -> MovementMetrics:
        """Process movement data with optional automatic trimming of inactive initial periods.
        
        Args:
            data: List of sensor readings with timestamp, accelerometer, gyroscope
            n_reps: Expected number of repetitions to detect
            trim_inactive: If True, automatically remove inactive initial period
            inactive_threshold: Magnitude threshold (m/s²) to consider movement active
            min_active_length: Minimum consecutive samples above threshold to confirm activity start
            
        Returns:
            MovementMetrics with extracted features
        """
        # Store original length for debugging
        original_length = len(data)
        
        # Trim inactive initial period if requested
        if trim_inactive and len(data) > 0:
            data = recortar_inactividad(data, umbral=inactive_threshold, min_len=min_active_length)
            trimmed_count = original_length - len(data)
            if trimmed_count > 0:
                # Store trimming info for debugging (could be exposed via return tuple if needed)
                self._last_trim_info = {"original": original_length, "trimmed": trimmed_count, "remaining": len(data)}
        
        # Convert raw data to numpy arrays
        timestamps = np.array([d['timestamp'] for d in data])
        acc_data = self._extract_accelerometer_data(data, timestamps)
        
        if len(timestamps) == 0:
            return self._get_empty_metrics()

        # Process signals
        magnitudes = acc_data.magnitude

        # Compute dynamic prominence threshold so small fluctuations are ignored
        mag_min = magnitudes.min() if magnitudes.size else 0.0
        mag_max = magnitudes.max() if magnitudes.size else 0.0
        mag_range = mag_max - mag_min
        # prominence tuned to capture clear movement peaks; floor to small value
        dynamic_prominence = max(0.2, 0.25 * mag_range)

        peaks = self.signal_processor.detect_peaks(
            magnitudes,
            n_reps=n_reps,
            prominence=dynamic_prominence,
            distance=10
        )
        
        if len(peaks) == 0:
            return self._get_empty_metrics()

        intervals = self.signal_processor.calculate_intervals(timestamps, peaks)

        # Analyze movement patterns
        return self.movement_analyzer.analyze_movement(
            magnitudes=magnitudes,
            peaks=peaks,
            intervals=intervals,
            z_accel=acc_data.z,
            timestamps=timestamps
        )

    def _extract_accelerometer_data(self, data: List[Dict[str, Any]], timestamps: np.ndarray) -> AccelerometerData:
        x_values = []
        y_values = []
        z_values = []
        valid_timestamps = []

        for i, d in enumerate(data):
            acc = d.get('accelerometer', {})
            if all(k in acc for k in ('x', 'y', 'z')):
                x_values.append(acc['x'])
                y_values.append(acc['y'])
                z_values.append(acc['z'])
                valid_timestamps.append(timestamps[i])

        return AccelerometerData(
            x=np.array(x_values),
            y=np.array(y_values),
            z=np.array(z_values),
            timestamps=np.array(valid_timestamps)
        )

    def _get_empty_metrics(self) -> MovementMetrics:
        return MovementMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0)