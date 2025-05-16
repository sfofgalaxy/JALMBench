import numpy as np
import matplotlib.pyplot as plt

class AudioMetrics:
    """Calculate audio quality metrics"""
    @staticmethod
    def calculate_snr(original, processed):
        """Calculate signal-to-noise ratio"""
        noise = original - processed
        signal_power = np.sum(original ** 2)
        noise_power = np.sum(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def calculate_mse(original, processed):
        """Calculate mean squared error"""
        return np.mean((original - processed) ** 2)
    
    @staticmethod
    def plot_waveforms(original, processed, title, filename):
        """Plot original and processed waveforms"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(original)
        plt.title('Original')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        plt.subplot(2, 1, 2)
        plt.plot(processed)
        plt.title(f'Processed: {title}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close() 