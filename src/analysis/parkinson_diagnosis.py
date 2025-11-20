"""
Parkinson's Disease Severity Diagnosis System
Based on UPDRS (Unified Parkinson's Disease Rating Scale) motor assessment
Grades 0-4 based on movement analysis metrics
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class DiagnosisResult:
    """Container for diagnosis results"""
    severity_score: int  # 0-4 scale
    severity_label: str
    confidence: float  # 0-1
    contributing_factors: Dict[str, str]
    clinical_notes: str
    
    
class ParkinsonDiagnosisSystem:
    """
    Expert system for Parkinson's severity assessment (0-4 scale)
    
    Severity Scale:
    - 0: Normal - No signs of motor impairment
    - 1: Mild - Slight amplitude reduction or rhythm irregularity, unilateral
    - 2: Moderate - Clear progressive amplitude decay, bilateral involvement possible
    - 3: Marked - Severe bradykinesia with significant freezing/hesitations
    - 4: Severe - Barely able to perform movement, extreme slowness
    """
    
    def __init__(self):
        """Initialize diagnostic thresholds based on clinical research"""
        
        # Amplitude Decay Rate thresholds (m/s²/rep)
        # Based on linear regression of peak magnitudes
        self.decay_thresholds = {
            'normal': -0.03,      # Minimal to no decay
            'mild': -0.08,        # Slight progressive reduction
            'moderate': -0.15,    # Clear bradykinesia pattern
            'marked': -0.25,      # Severe progressive reduction
            'severe': -0.40       # Extreme decay
        }
        
        # Amplitude Ratio thresholds (first_half / second_half)
        # Ratio > 1.0 indicates reduction in second half
        self.ratio_thresholds = {
            'normal': 1.05,       # Minimal difference
            'mild': 1.15,         # Slight reduction
            'moderate': 1.30,     # Clear reduction
            'marked': 1.50,       # Severe reduction
            'severe': 2.00        # Extreme reduction
        }
        
        # Mean magnitude thresholds (m/s²)
        # Lower values indicate reduced movement amplitude
        self.magnitude_thresholds = {
            'normal': 3.0,        # Strong movement
            'mild': 2.2,          # Slightly reduced
            'moderate': 1.5,      # Clearly reduced
            'marked': 1.0,        # Severely reduced
            'severe': 0.6         # Barely moving
        }
        
        # Rhythm variability (STD of inter-rep intervals, ms)
        # Higher values indicate more irregular rhythm
        self.rhythm_std_thresholds = {
            'normal': 150,        # Consistent rhythm
            'mild': 250,          # Slight irregularity
            'moderate': 400,      # Clear irregularity
            'marked': 600,        # Severe irregularity
            'severe': 800         # Extreme irregularity
        }
        
        # Mean repetition time (ms)
        # Higher values indicate slower movement
        self.rep_time_thresholds = {
            'normal': 600,        # Normal speed
            'mild': 800,          # Slightly slow
            'moderate': 1100,     # Clearly slow
            'marked': 1500,       # Severely slow
            'severe': 2000        # Extremely slow
        }
        
        # Hesitation count (normalized per 10 reps)
        self.hesitation_thresholds = {
            'normal': 0.5,        # Rare
            'mild': 1.5,          # Occasional
            'moderate': 3.0,      # Frequent
            'marked': 5.0,        # Very frequent
            'severe': 7.0         # Almost constant
        }
        
        # Feature weights for final scoring (sum to 1.0)
        self.feature_weights = {
            'decay_rate': 0.30,      # Most important - captures bradykinesia
            'amplitude_ratio': 0.25,  # Important - validates decay pattern
            'magnitude': 0.15,        # Moderate - overall amplitude
            'rhythm_std': 0.15,       # Moderate - rhythm consistency
            'rep_time': 0.10,         # Less critical - speed
            'hesitations': 0.05       # Least critical - freezing episodes
        }
    
    def _score_feature(self, value: float, thresholds: dict, inverse: bool = False) -> Tuple[float, str]:
        """
        Score a single feature on 0-4 scale
        
        Args:
            value: Feature value to score
            thresholds: Dictionary with severity level thresholds
            inverse: If True, higher values = better (e.g., magnitude)
        
        Returns:
            (score, severity_label)
        """
        if inverse:
            # For features where higher is better (magnitude)
            if value >= thresholds['normal']:
                return 0.0, 'normal'
            elif value >= thresholds['mild']:
                # Interpolate between 0 and 1
                range_span = thresholds['normal'] - thresholds['mild']
                score = 1.0 - ((value - thresholds['mild']) / range_span)
                return score, 'mild'
            elif value >= thresholds['moderate']:
                range_span = thresholds['mild'] - thresholds['moderate']
                score = 1.0 + (1.0 - ((value - thresholds['moderate']) / range_span))
                return min(score, 2.0), 'moderate'
            elif value >= thresholds['marked']:
                range_span = thresholds['moderate'] - thresholds['marked']
                score = 2.0 + (1.0 - ((value - thresholds['marked']) / range_span))
                return min(score, 3.0), 'marked'
            else:
                # Below marked threshold = severe
                if value >= thresholds['severe']:
                    range_span = thresholds['marked'] - thresholds['severe']
                    score = 3.0 + (1.0 - ((value - thresholds['severe']) / range_span))
                    return min(score, 4.0), 'severe'
                else:
                    return 4.0, 'severe'
        else:
            # For features where lower is better (decay, ratio, std, time, hesitations)
            if value <= thresholds['normal']:
                return 0.0, 'normal'
            elif value <= thresholds['mild']:
                range_span = thresholds['mild'] - thresholds['normal']
                score = (value - thresholds['normal']) / range_span
                return score, 'mild'
            elif value <= thresholds['moderate']:
                range_span = thresholds['moderate'] - thresholds['mild']
                score = 1.0 + ((value - thresholds['mild']) / range_span)
                return min(score, 2.0), 'moderate'
            elif value <= thresholds['marked']:
                range_span = thresholds['marked'] - thresholds['moderate']
                score = 2.0 + ((value - thresholds['moderate']) / range_span)
                return min(score, 3.0), 'marked'
            else:
                # Above marked threshold = severe
                if value <= thresholds['severe']:
                    range_span = thresholds['severe'] - thresholds['marked']
                    score = 3.0 + ((value - thresholds['marked']) / range_span)
                    return min(score, 4.0), 'severe'
                else:
                    return 4.0, 'severe'
    
    def diagnose(self, metrics: Dict) -> DiagnosisResult:
        """
        Perform comprehensive diagnosis based on movement metrics
        
        Args:
            metrics: Dictionary with movement analysis results (from MovementProcessor)
        
        Returns:
            DiagnosisResult with severity score (0-4) and clinical interpretation
        """
        # Extract key metrics (use active side)
        decay_rate = abs(metrics.get('active_vertical_amplitude_decay', 0))  # Use absolute value
        amplitude_ratio = metrics.get('active_vertical_amplitude_ratio', 1.0)
        magnitude = metrics.get('active_magnitude_mean', 0)
        rhythm_std = metrics.get('active_rep_time_std', 0)
        rep_time = metrics.get('active_rep_time_mean', 0)
        hesitations = metrics.get('active_hesitations', 0)
        num_peaks = metrics.get('active_peaks_count', 10)
        
        # Normalize hesitations per 10 repetitions
        hesitations_normalized = (hesitations / max(num_peaks, 1)) * 10
        
        # Score each feature
        decay_score, decay_label = self._score_feature(decay_rate, self.decay_thresholds, inverse=False)
        ratio_score, ratio_label = self._score_feature(amplitude_ratio, self.ratio_thresholds, inverse=False)
        magnitude_score, magnitude_label = self._score_feature(magnitude, self.magnitude_thresholds, inverse=True)
        rhythm_score, rhythm_label = self._score_feature(rhythm_std, self.rhythm_std_thresholds, inverse=False)
        time_score, time_label = self._score_feature(rep_time, self.rep_time_thresholds, inverse=False)
        hesitation_score, hesitation_label = self._score_feature(hesitations_normalized, self.hesitation_thresholds, inverse=False)
        
        # Calculate weighted severity score
        weighted_score = (
            decay_score * self.feature_weights['decay_rate'] +
            ratio_score * self.feature_weights['amplitude_ratio'] +
            magnitude_score * self.feature_weights['magnitude'] +
            rhythm_score * self.feature_weights['rhythm_std'] +
            time_score * self.feature_weights['rep_time'] +
            hesitation_score * self.feature_weights['hesitations']
        )
        
        # Round to nearest integer (0-4)
        severity_score = int(round(weighted_score))
        severity_score = max(0, min(4, severity_score))  # Clamp to 0-4
        
        # Map score to label
        severity_labels = {
            0: "Normal - Sin signos de deterioro motor",
            1: "Leve - Reducción leve de amplitud o ritmo irregular",
            2: "Moderado - Clara bradicinesia con reducción progresiva",
            3: "Marcado - Bradicinesia severa con congelamiento frecuente",
            4: "Severo - Movimiento extremadamente limitado"
        }
        
        severity_label = severity_labels[severity_score]
        
        # Calculate confidence based on consistency of feature scores
        score_variance = np.var([decay_score, ratio_score, magnitude_score, 
                                 rhythm_score, time_score, hesitation_score])
        # Lower variance = higher confidence
        confidence = max(0.5, min(1.0, 1.0 - (score_variance / 4.0)))
        
        # Document contributing factors
        contributing_factors = {
            'decay_rate': f"{decay_label} ({decay_rate:.3f} m/s²/rep)",
            'amplitude_ratio': f"{ratio_label} ({amplitude_ratio:.2f})",
            'magnitude': f"{magnitude_label} ({magnitude:.2f} m/s²)",
            'rhythm_variability': f"{rhythm_label} ({rhythm_std:.0f} ms)",
            'repetition_time': f"{time_label} ({rep_time:.0f} ms)",
            'hesitations': f"{hesitation_label} ({hesitations}/{num_peaks} reps)"
        }
        
        # Generate clinical notes
        clinical_notes = self._generate_clinical_notes(
            severity_score, decay_label, ratio_label, magnitude_label,
            rhythm_label, time_label, hesitation_label
        )
        
        return DiagnosisResult(
            severity_score=severity_score,
            severity_label=severity_label,
            confidence=confidence,
            contributing_factors=contributing_factors,
            clinical_notes=clinical_notes
        )
    
    def _generate_clinical_notes(self, score: int, decay: str, ratio: str, 
                                  magnitude: str, rhythm: str, time: str, 
                                  hesitation: str) -> str:
        """Generate detailed clinical interpretation"""
        
        notes = []
        
        if score == 0:
            notes.append("✓ Movimiento dentro de parámetros normales")
            notes.append("✓ No se observan signos de bradicinesia")
            notes.append("✓ Ritmo consistente y amplitud estable")
        
        elif score == 1:
            notes.append("⚠️ Signos tempranos de deterioro motor")
            if decay != 'normal':
                notes.append(f"• Reducción leve de amplitud: {decay}")
            if ratio != 'normal':
                notes.append(f"• Fatiga progresiva detectada: {ratio}")
            if rhythm != 'normal':
                notes.append(f"• Ligera irregularidad del ritmo: {rhythm}")
            notes.append("→ Recomendación: Monitoreo periódico")
        
        elif score == 2:
            notes.append("⚠️ Bradicinesia moderada detectada")
            if decay in ['moderate', 'marked', 'severe']:
                notes.append(f"• Reducción progresiva clara: {decay}")
            if ratio in ['moderate', 'marked', 'severe']:
                notes.append(f"• Fatiga significativa en segunda mitad: {ratio}")
            if magnitude in ['moderate', 'marked', 'severe']:
                notes.append(f"• Amplitud reducida: {magnitude}")
            notes.append("→ Recomendación: Evaluación neurológica completa")
        
        elif score == 3:
            notes.append("🔴 Bradicinesia marcada con compromiso funcional")
            notes.append(f"• Reducción severa de amplitud: {decay}")
            notes.append(f"• Fatiga extrema: {ratio}")
            if hesitation in ['marked', 'severe']:
                notes.append(f"• Congelamiento frecuente: {hesitation}")
            notes.append("→ Recomendación: Intervención terapéutica urgente")
        
        else:  # score == 4
            notes.append("🔴 Compromiso motor severo")
            notes.append("• Capacidad de movimiento extremadamente limitada")
            notes.append("• Bradykinesia extrema con posible congelamiento")
            notes.append("→ Recomendación: Ajuste inmediato de tratamiento")
        
        return "\n".join(notes)
