#!/usr/bin/env python3
"""
Motor Anomaly Detection - Data Preprocessing Module
This module handles preprocessing of collected vibration data for AI model training.
"""

import numpy as np
import pandas as pd
import h5py
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    """Container for extracted features"""
    time_domain: Dict[str, float]
    frequency_domain: Dict[str, float]
    statistical: Dict[str, float]
    sample_id: int
    fault_type: int

class VibrationPreprocessor:
    """
    Class for preprocessing vibration data and extracting features
    """
    
    def __init__(self, sample_rate: int = 1000):
        """
        Initialize preprocessor
        
        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data_from_hdf5(self, filename: str) -> List[Dict]:
        """
        Load data from HDF5 file
        
        Args:
            filename: Path to HDF5 file
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        with h5py.File(filename, 'r') as f:
            for sample_key in f.keys():
                sample_group = f[sample_key]
                
                sample_data = {
                    'sample_id': int(sample_key.split('_')[1]),
                    'fault_type': sample_group.attrs['fault_type'],
                    'sample_rate': sample_group.attrs['sample_rate'],
                    'duration_ms': sample_group.attrs['duration_ms'],
                    'timestamp': sample_group.attrs['timestamp'],
                    'x_data': np.array(sample_group['x_data']),
                    'y_data': np.array(sample_group['y_data']),
                    'z_data': np.array(sample_group['z_data'])
                }
                samples.append(sample_data)
        
        logger.info(f"Loaded {len(samples)} samples from {filename}")
        return samples
    
    def apply_filters(self, data: np.ndarray, 
                     filter_type: str = 'bandpass',
                     low_freq: float = 1.0,
                     high_freq: float = 100.0) -> np.ndarray:
        """
        Apply digital filters to vibration data
        
        Args:
            data: Input vibration data
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz
            
        Returns:
            Filtered data
        """
        nyquist = self.sample_rate / 2
        
        if filter_type == 'bandpass':
            low = low_freq / nyquist
            high = high_freq / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
        elif filter_type == 'lowpass':
            high = high_freq / nyquist
            b, a = signal.butter(4, high, btype='low')
        elif filter_type == 'highpass':
            low = low_freq / nyquist
            b, a = signal.butter(4, low, btype='high')
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        return signal.filtfilt(b, a, data)
    
    def extract_time_domain_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract time domain features from vibration signal
        
        Args:
            data: Vibration signal data
            
        Returns:
            Dictionary of time domain features
        """
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(data)
        features['std'] = np.std(data)
        features['var'] = np.var(data)
        features['rms'] = np.sqrt(np.mean(data**2))
        
        # Peak and shape features
        features['peak'] = np.max(np.abs(data))
        features['peak_to_peak'] = np.max(data) - np.min(data)
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0
        features['form_factor'] = features['rms'] / np.mean(np.abs(data)) if np.mean(np.abs(data)) > 0 else 0
        
        # Higher order moments
        features['skewness'] = self.calculate_skewness(data)
        features['kurtosis'] = self.calculate_kurtosis(data)
        
        # Energy features
        features['energy'] = np.sum(data**2)
        features['power'] = features['energy'] / len(data)
        
        return features
    
    def extract_frequency_domain_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features from vibration signal
        
        Args:
            data: Vibration signal data
            
        Returns:
            Dictionary of frequency domain features
        """
        features = {}
        
        # Compute FFT
        fft_data = fft(data)
        freqs = fftfreq(len(data), 1/self.sample_rate)
        
        # Use only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft_data[:len(fft_data)//2])
        power_spectrum = magnitude**2
        
        # Spectral features
        features['spectral_centroid'] = np.sum(positive_freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        features['spectral_rolloff'] = self.calculate_spectral_rolloff(positive_freqs, magnitude)
        features['spectral_flux'] = np.sum(np.diff(magnitude)**2)
        
        # Frequency band energy ratios
        features['low_freq_energy'] = self.calculate_band_energy(positive_freqs, power_spectrum, 0, 10)
        features['mid_freq_energy'] = self.calculate_band_energy(positive_freqs, power_spectrum, 10, 100)
        features['high_freq_energy'] = self.calculate_band_energy(positive_freqs, power_spectrum, 100, 500)
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(magnitude)
        features['dominant_frequency'] = positive_freqs[dominant_freq_idx]
        features['dominant_magnitude'] = magnitude[dominant_freq_idx]
        
        return features
    
    def extract_statistical_features(self, x_data: np.ndarray, 
                                   y_data: np.ndarray, 
                                   z_data: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from 3-axis data
        
        Args:
            x_data, y_data, z_data: 3-axis vibration data
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        # Magnitude vector
        magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        
        # Cross-correlation between axes
        features['xy_correlation'] = np.corrcoef(x_data, y_data)[0, 1]
        features['xz_correlation'] = np.corrcoef(x_data, z_data)[0, 1]
        features['yz_correlation'] = np.corrcoef(y_data, z_data)[0, 1]
        
        # Magnitude statistics
        features['magnitude_mean'] = np.mean(magnitude)
        features['magnitude_std'] = np.std(magnitude)
        features['magnitude_max'] = np.max(magnitude)
        features['magnitude_min'] = np.min(magnitude)
        
        # Axis ratios
        x_energy = np.sum(x_data**2)
        y_energy = np.sum(y_data**2)
        z_energy = np.sum(z_data**2)
        total_energy = x_energy + y_energy + z_energy
        
        if total_energy > 0:
            features['x_energy_ratio'] = x_energy / total_energy
            features['y_energy_ratio'] = y_energy / total_energy
            features['z_energy_ratio'] = z_energy / total_energy
        else:
            features['x_energy_ratio'] = 0
            features['y_energy_ratio'] = 0
            features['z_energy_ratio'] = 0
        
        return features
    
    def calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def calculate_spectral_rolloff(self, freqs: np.ndarray, magnitude: np.ndarray, 
                                 rolloff_percent: float = 0.85) -> float:
        """Calculate spectral rolloff frequency"""
        total_energy = np.sum(magnitude)
        cumulative_energy = np.cumsum(magnitude)
        rolloff_idx = np.where(cumulative_energy >= rolloff_percent * total_energy)[0]
        return freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    def calculate_band_energy(self, freqs: np.ndarray, power_spectrum: np.ndarray,
                            low_freq: float, high_freq: float) -> float:
        """Calculate energy in a specific frequency band"""
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.sum(power_spectrum[mask])
    
    def extract_all_features(self, sample: Dict) -> FeatureSet:
        """
        Extract all features from a sample
        
        Args:
            sample: Sample dictionary with x_data, y_data, z_data
            
        Returns:
            FeatureSet object with all extracted features
        """
        x_data = sample['x_data']
        y_data = sample['y_data']
        z_data = sample['z_data']
        
        # Apply filtering
        x_filtered = self.apply_filters(x_data)
        y_filtered = self.apply_filters(y_data)
        z_filtered = self.apply_filters(z_data)
        
        # Extract features for each axis
        time_features = {}
        freq_features = {}
        
        for axis, data in [('x', x_filtered), ('y', y_filtered), ('z', z_filtered)]:
            # Time domain features
            axis_time_features = self.extract_time_domain_features(data)
            for key, value in axis_time_features.items():
                time_features[f'{axis}_{key}'] = value
            
            # Frequency domain features
            axis_freq_features = self.extract_frequency_domain_features(data)
            for key, value in axis_freq_features.items():
                freq_features[f'{axis}_{key}'] = value
        
        # Statistical features (cross-axis)
        statistical_features = self.extract_statistical_features(x_filtered, y_filtered, z_filtered)
        
        return FeatureSet(
            time_domain=time_features,
            frequency_domain=freq_features,
            statistical=statistical_features,
            sample_id=sample['sample_id'],
            fault_type=sample['fault_type']
        )
    
    def process_dataset(self, samples: List[Dict]) -> pd.DataFrame:
        """
        Process entire dataset and extract features
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Processing {len(samples)} samples...")
        
        feature_list = []
        
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                logger.info(f"Processing sample {i+1}/{len(samples)}")
            
            try:
                features = self.extract_all_features(sample)
                
                # Combine all features into a single dictionary
                feature_dict = {
                    'sample_id': features.sample_id,
                    'fault_type': features.fault_type
                }
                feature_dict.update(features.time_domain)
                feature_dict.update(features.frequency_domain)
                feature_dict.update(features.statistical)
                
                feature_list.append(feature_dict)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample['sample_id']}: {str(e)}")
        
        df = pd.DataFrame(feature_list)
        logger.info(f"Extracted {len(df.columns)-2} features from {len(df)} samples")
        
        return df
    
    def create_sliding_windows(self, data: np.ndarray, window_size: int, 
                              overlap: float = 0.5) -> List[np.ndarray]:
        """
        Create sliding windows from time series data
        
        Args:
            data: Input time series data
            window_size: Size of each window
            overlap: Overlap between windows (0.0 to 1.0)
            
        Returns:
            List of windowed data arrays
        """
        step_size = int(window_size * (1 - overlap))
        windows = []
        
        for i in range(0, len(data) - window_size + 1, step_size):
            windows.append(data[i:i + window_size])
        
        return windows
    
    def visualize_features(self, df: pd.DataFrame, output_dir: str = "plots"):
        """
        Create visualizations of extracted features
        
        Args:
            df: DataFrame with extracted features
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Feature correlation heatmap
        plt.figure(figsize=(20, 16))
        feature_cols = [col for col in df.columns if col not in ['sample_id', 'fault_type']]
        correlation_matrix = df[feature_cols].corr()
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature distributions by fault type
        fault_names = {0: 'Normal', 1: 'Imbalance', 2: 'Bearing', 3: 'Misalignment'}
        df['fault_name'] = df['fault_type'].map(fault_names)
        
        # Select top features for visualization
        important_features = ['x_rms', 'y_rms', 'z_rms', 'x_peak', 'y_peak', 'z_peak',
                            'x_spectral_centroid', 'y_spectral_centroid', 'z_spectral_centroid',
                            'magnitude_mean', 'magnitude_std']
        
        available_features = [f for f in important_features if f in df.columns]
        
        if available_features:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, feature in enumerate(available_features[:6]):
                sns.boxplot(data=df, x='fault_name', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} by Fault Type')
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Example usage of the preprocessor"""
    
    # Configuration
    INPUT_FILE = "collected_data/motor_data_20240101_120000.h5"  # Update with your file
    OUTPUT_DIR = "processed_data"
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = VibrationPreprocessor(sample_rate=1000)
    
    try:
        # Load data
        samples = preprocessor.load_data_from_hdf5(INPUT_FILE)
        
        # Process dataset
        features_df = preprocessor.process_dataset(samples)
        
        # Save processed features
        output_file = f"{OUTPUT_DIR}/processed_features.csv"
        features_df.to_csv(output_file, index=False)
        logger.info(f"Processed features saved to {output_file}")
        
        # Create visualizations
        preprocessor.visualize_features(features_df, f"{OUTPUT_DIR}/plots")
        
        # Print summary statistics
        print("\n=== Dataset Summary ===")
        print(f"Total samples: {len(features_df)}")
        print(f"Total features: {len(features_df.columns) - 2}")
        print("\nSamples per fault type:")
        fault_names = {0: 'Normal', 1: 'Imbalance', 2: 'Bearing', 3: 'Misalignment'}
        for fault_type, count in features_df['fault_type'].value_counts().sort_index().items():
            print(f"  {fault_names.get(fault_type, f'Unknown ({fault_type})')}: {count}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")


if __name__ == "__main__":
    main()
