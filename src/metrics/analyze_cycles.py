"""
Comprehensive Cycle Analysis Tool for Time Series Data

This module provides advanced cycle analysis capabilities for time series data extracted
from RL training logs. It implements multiple periodicity detection
methods to identify temporal patterns and cyclic behaviors in RL environments.

Analysis Methods:
- Autocorrelation: Identifies self-similarity patterns across time lags
- FFT Analysis: Frequency domain analysis for dominant periodicities
- Periodogram: Welch's method for power spectral density estimation
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from pathlib import Path
from scipy.fft import fft, fftfreq

from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

def get_dominant_period(df_cycles):
    """
    Find the period length with the highest total confidence across all detections.
    
    Args:
        df_cycles: DataFrame containing cycle analysis results
        
    Returns:
        tuple: (period, best_row_for_that_period)
    """
    period_conf_sums = df_cycles.groupby('detected_period')['confidence'].sum()
    best_period = period_conf_sums.idxmax()
    period_entries = df_cycles[df_cycles['detected_period'] == best_period]
    best_overall = period_entries.loc[period_entries['confidence'].idxmax()]
    return best_period, best_overall

def analyze_cycles_from_csv(csv_files, output_path=None, max_cycle_length=200, is_cyclenet=False):
    """
    Perform comprehensive cycle analysis on CSV files containing time series data.
    
    Args:
        csv_files: List of CSV file paths or a single path
        output_path: Base path for output files (optional, derived from input if not given)
        max_cycle_length: Maximum cycle length to search for
        is_cyclenet: If true, excludes the last dimension (cycle index) from observations

    Returns:
        tuple: (cycle_analysis_csv_path, cycle_plots_path, summary_json_path)
    """
    
    # Normalize input to list
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    
    # Determine output path and ensure directory exists
    if output_path is None:
        if csv_files:
            first_file = csv_files[0]
            base_path = Path(first_file).parent / Path(first_file).stem
            output_path = str(base_path)
        else:
            output_path = "cycle_analysis"
    
    # Check if output_path should be treated as a directory or file prefix
    output_path = Path(output_path)
    
    # If the output_path ends with a directory name (like 'cycle_data'), 
    # treat it as a directory and create it
    if output_path.name and not output_path.suffix:
        # This looks like a directory path
        output_path.mkdir(parents=True, exist_ok=True)
        # Use the directory path with a base filename
        output_path_str = str(output_path / "cycle_data")
    else:
        # This looks like a file prefix path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path_str = str(output_path)
    
    cycle_results = []
    all_timeseries = {}
    
    # Process all CSV files
    for i, csv_path in enumerate(csv_files):
        if not os.path.exists(csv_path):
            logger.warning(f"Warning: File {csv_path} not found, skipping...")
            continue

        logger.cycle(f"Loading data from: {csv_path}.")
        df = pd.read_csv(csv_path)
        
        # Extract all mean_ch columns
        mean_channels = [col for col in df.columns if col.startswith('mean_ch')]
        # For CycleNet runs, exclude the last channel (cycle index) from observations
        if is_cyclenet and 'observations' in Path(csv_path).stem:
            if len(mean_channels) > 0:
                # Remove the last mean_ch column (highest channel number)
                channel_numbers = []
                for col in mean_channels:
                    try:
                        ch_num = int(col.split('ch')[1])
                        channel_numbers.append(ch_num)
                    except (IndexError, ValueError):
                        continue
                
                if channel_numbers:
                    max_channel = max(channel_numbers)
                    excluded_channel = f'mean_ch{max_channel}'
                    if excluded_channel in mean_channels:
                        mean_channels.remove(excluded_channel)
                        logger.cycle(f"Excluded cycle index channel {excluded_channel} from CycleNet observations analysis.")
        logger.cycle(f"Found {len(mean_channels)} channels in {Path(csv_path).name}.")

        # Use the filename as prefix
        file_stem = Path(csv_path).stem
        data_type = file_stem
        
        for col in mean_channels:
            channel_idx = col.split('ch')[1]
            channel_name = f"{data_type}_ch{channel_idx}"
            
            # Extract time series (ignore NaN values)
            timeseries = df[col].dropna().values
            
            if len(timeseries) < max_cycle_length:
                logger.cycle(f"Skipping {channel_name}: too short ({len(timeseries)} < {max_cycle_length}).")
                continue

            all_timeseries[channel_name] = timeseries
            
            # Perform periodicity analysis
            results = _analyze_periodicity(timeseries, max_cycle_length, channel_name)
            cycle_results.extend(results)
    
    if not cycle_results:
        logger.warning("No cycle analysis results found. This may indicate insufficient data.")
        return None, None, None

    # CSV with results
    cycle_csv_path = f"{output_path_str}_cycle_analysis.csv"
    df_cycles = pd.DataFrame(cycle_results)
    df_cycles.to_csv(cycle_csv_path, index=False)
    
    # Create comprehensive plots
    cycle_plots_path = _create_cycle_plots(
        df_cycles,
        all_timeseries,
        f"{output_path_str}_cycle_analysis.svg"
    )
    
    # Summary statistics
    summary_info = _get_cycle_summary_info(df_cycles)

    # Save summary as JSON
    summary_json_path = f"{output_path_str}_cycle_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_info, f, indent=2, ensure_ascii=False)
    
    # Save results
    logger.cycle(f"Saved results to: {output_path_str}.")

    return cycle_csv_path, cycle_plots_path, summary_json_path

def _get_cycle_summary_info(df_cycles):
    """
    Create a comprehensive summary dictionary from the cycle analysis DataFrame.
    
    Args:
        df_cycles: DataFrame containing cycle analysis results
        
    Returns:
        dict: Summary statistics and information
    """
    if len(df_cycles) == 0:
        return {}
    
    # Find dominant cycle (across all methods)
    def get_dominant_period(df):
        period_conf_sums = df.groupby('detected_period')['confidence'].sum()
        best_period = period_conf_sums.idxmax()
        period_entries = df[df['detected_period'] == best_period]
        best_overall = period_entries.loc[period_entries['confidence'].idxmax()]
        return float(best_period), best_overall
    
    best_period, best_row = get_dominant_period(df_cycles)
    summary = {
        "dominant_period": best_period,
        "dominant_channel": best_row['channel'],
        "dominant_method": best_row['method'],
        "dominant_confidence": float(best_row['confidence']),
        "dominant_mean_value": float(best_row['mean_value']),
        "dominant_std_value": float(best_row['std_value']),
        "all_methods": {},
        "period_conf_sums": df_cycles.groupby('detected_period')['confidence'].sum().to_dict(),
        "period_counts": df_cycles['detected_period'].value_counts().to_dict(),
    }
    
    # For each method, get the best result
    for method in ['autocorr', 'fft', 'periodogram']:
        method_data = df_cycles[df_cycles['method'] == method]
        if len(method_data) > 0:
            best = method_data.loc[method_data['confidence'].idxmax()]
            summary['all_methods'][method] = {
                "channel": best['channel'],
                "period": float(best['detected_period']),
                "confidence": float(best['confidence'])
            }
    return summary


def _analyze_periodicity(data, max_cycle_length, channel_name):
    """
    Perform comprehensive periodicity analysis using multiple methods.
    
    Args:
        data: 1D numpy array of the time series
        max_cycle_length: Maximum period length to search for
        channel_name: Name of the channel for results
    
    Returns:
        list: List of result dictionaries from different analysis methods
    """
    results = []
    data = np.array(data).flatten()
    n = len(data)
    
    if n < max_cycle_length:
        return results
    
    # Preprocessing: Linear detrending
    x = np.arange(n)
    coeffs = np.polyfit(x, data, 1)
    detrended = data - np.polyval(coeffs, x)
    
    # Normalization
    if np.std(detrended) > 0:
        normalized = (detrended - np.mean(detrended)) / np.std(detrended)
    else:
        normalized = detrended
    
    # Apply different analysis methods
    autocorr_result = _autocorrelation_analysis(normalized, max_cycle_length, channel_name)
    if autocorr_result:
        results.append(autocorr_result)
    
    fft_result = _fft_analysis(normalized, max_cycle_length, channel_name)
    if fft_result:
        results.append(fft_result)
    
    periodogram_result = _periodogram_analysis(normalized, max_cycle_length, channel_name)
    if periodogram_result:
        results.append(periodogram_result)
    
    return results


def _autocorrelation_analysis(data, max_cycle_length, channel_name):
    """
    Perform autocorrelation analysis for period detection.
    
    Args:
        data: Normalized time series data
        max_cycle_length: Maximum lag to consider
        channel_name: Name of the channel
        
    Returns:
        dict: Analysis results or None if no peaks found
    """
    n = len(data)
    max_lag = min(max_cycle_length, n // 2)
    
    # Compute autocorrelation
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    
    if autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]  # Normalize
    
    # Search for local maxima (excluding lag=0)
    if len(autocorr) <= max_lag:
        return None
    
    autocorr_subset = autocorr[1:max_lag+1]
    peaks, _ = signal.find_peaks(autocorr_subset, height=0.1, distance=5)
    
    if len(peaks) == 0:
        return None
    
    # Find strongest peak
    peak_heights = autocorr_subset[peaks]
    strongest_peak_idx = np.argmax(peak_heights)
    detected_period = peaks[strongest_peak_idx] + 1  # +1 due to index offset
    confidence = peak_heights[strongest_peak_idx]
    
    return {
        'channel': channel_name,
        'method': 'autocorr',
        'detected_period': detected_period,
        'confidence': confidence,
        'secondary_peaks': len(peaks),
        'mean_value': np.mean(data),
        'std_value': np.std(data)
    }


def _fft_analysis(data, max_cycle_length, channel_name):
    """
    Perform FFT-based frequency analysis for period detection.
    
    Args:
        data: Normalized time series data
        max_cycle_length: Maximum period length to consider
        channel_name: Name of the channel
        
    Returns:
        dict: Analysis results or None if no valid frequencies found
    """
    n = len(data)
    
    # Compute FFT
    fft_data = fft(data)
    freqs = fftfreq(n, d=1.0)
    
    # Power spectrum (only positive frequencies)
    power = np.abs(fft_data[:n//2]) ** 2
    freqs_pos = freqs[:n//2]
    
    # Convert frequencies to periods (avoid division by zero)
    valid_freq_mask = (freqs_pos > 1.0/max_cycle_length) & (freqs_pos < 0.5)  # Nyquist
    
    if not np.any(valid_freq_mask):
        return None
    
    valid_power = power[valid_freq_mask]
    valid_freqs = freqs_pos[valid_freq_mask]
    
    # Find strongest frequency
    max_power_idx = np.argmax(valid_power)
    dominant_freq = valid_freqs[max_power_idx]
    detected_period = 1.0 / dominant_freq
    
    # Confidence based on power ratio
    total_power = np.sum(valid_power)
    confidence = valid_power[max_power_idx] / total_power if total_power > 0 else 0
    
    return {
        'channel': channel_name,
        'method': 'fft',
        'detected_period': detected_period,
        'confidence': confidence,
        'dominant_frequency': dominant_freq,
        'mean_value': np.mean(data),
        'std_value': np.std(data)
    }


def _periodogram_analysis(data, max_cycle_length, channel_name):
    """
    Perform periodogram analysis using Welch's method for period detection.
    
    Args:
        data: Normalized time series data
        max_cycle_length: Maximum period length to consider
        channel_name: Name of the channel
        
    Returns:
        dict: Analysis results or None if insufficient data or no valid frequencies
    """
    n = len(data)
    
    if n < 64:  # Minimum length for Welch
        return None
    
    # Compute periodogram with Welch's method
    nperseg = min(n // 4, 256)  # Segment length
    freqs, psd = signal.welch(data, nperseg=nperseg, scaling='density')
    
    # Filter valid frequencies
    valid_freq_mask = (freqs > 1.0/max_cycle_length) & (freqs < 0.5)
    
    if not np.any(valid_freq_mask):
        return None
    
    valid_psd = psd[valid_freq_mask]
    valid_freqs = freqs[valid_freq_mask]
    
    # Find peak in PSD
    max_psd_idx = np.argmax(valid_psd)
    dominant_freq = valid_freqs[max_psd_idx]
    detected_period = 1.0 / dominant_freq
    
    # Confidence based on PSD ratio
    total_psd = np.sum(valid_psd)
    confidence = valid_psd[max_psd_idx] / total_psd if total_psd > 0 else 0
    
    return {
        'channel': channel_name,
        'method': 'periodogram',
        'detected_period': detected_period,
        'confidence': confidence,
        'dominant_frequency': dominant_freq,
        'mean_value': np.mean(data),
        'std_value': np.std(data)
    }


def _create_cycle_plots(df_cycles, all_timeseries, plot_path):
    """
    Create comprehensive visualization plots for the cycle analysis results.
    
    Args:
        df_cycles: DataFrame containing cycle analysis results
        all_timeseries: Dictionary of time series data by channel
        plot_path: Path to save the plot file
        
    Returns:
        str: Path to the saved plot file
    """
    logger.cycle(f"Creating plots for {len(df_cycles)} cycles.")

    if len(df_cycles) == 0:
        logger.warning("No cycle data to plot.")
        return plot_path
    
    period, best_overall = get_dominant_period(df_cycles)
    # Use int for range if possible
    try:
        period_int = int(round(period))
    except Exception:
        period_int = period
    channel_name = best_overall['channel']

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Bubble Plot: Channel vs Detected Period (Bubble = Confidence, Color = Method)
    ax1 = plt.subplot(3, 2, 1)
    method_label_map = {'autocorr': 'Autocorrelation', 'fft': 'FFT', 'periodogram': 'Periodogram'}
    if len(df_cycles) > 0:
        def channel_sort_key(ch):
            m = re.search(r'_ch(\d+)$', ch)
            return int(m.group(1)) if m else 9999
        df_cycles_sorted = df_cycles.copy()
        df_cycles_sorted['channel_num'] = df_cycles_sorted['channel'].apply(channel_sort_key)
        method_colors = {m: c for m, c in zip(df_cycles_sorted['method'].unique(), ['red', 'blue', 'green', 'orange', 'purple', 'cyan'])}
        for method in df_cycles_sorted['method'].unique():
            sub = df_cycles_sorted[df_cycles_sorted['method'] == method]
            label = method_label_map.get(method, method)
            ax1.scatter(
                sub['channel_num'],
                sub['detected_period'],
                s=120 * sub['confidence'],
                alpha=0.6,
                color=method_colors[method],
                label=label,
                edgecolors='k',
                linewidths=0.5
            )
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Detected Period')
        ax1.set_title('Period Length per Channel with Confidence')
        ax1.legend(title='Method')
        ax1.grid(True, alpha=0.2)

    # 2. Confidence heatmap by channel and method
    ax2 = plt.subplot(3, 2, 2)
    pivot_data = df_cycles.copy()
    # Rename columns for display
    method_label_map = {'autocorr': 'Autocorrelation', 'fft': 'FFT', 'periodogram': 'Periodogram'}
    pivot_data['method_disp'] = pivot_data['method'].map(method_label_map).fillna(pivot_data['method'])
    pivot_table = pivot_data.pivot_table(
        values='confidence', 
        index='channel', 
        columns='method_disp', 
        aggfunc='max'
    ).fillna(0)

    if len(pivot_table) > 0:
        # Sort channels by number
        def channel_sort_key(ch):
            m = re.search(r'_ch(\d+)$', ch)
            return int(m.group(1)) if m else 9999

        sorted_channels = sorted(pivot_table.index, key=channel_sort_key)
        pivot_table = pivot_table.loc[sorted_channels]

        # Extract only channel numbers for y-axis
        channel_numbers = []
        for ch in pivot_table.index:
            m = re.search(r'_ch(\d+)$', ch)
            channel_numbers.append(int(m.group(1)) if m else ch)

        im = ax2.imshow(pivot_table.values, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(pivot_table.columns)))
        ax2.set_xticklabels(pivot_table.columns)
        ax2.set_yticks(range(len(pivot_table.index)))
        ax2.set_yticklabels(channel_numbers, fontsize=8)
        ax2.set_ylabel('Channel')
        ax2.set_title('Confidence Heatmap (Channel vs Method)')
        plt.colorbar(im, ax=ax2)
    
    # 3. Time series overview: all channels with global best period length
    ax3 = plt.subplot(3, 2, 3)
    colors = plt.get_cmap('tab10').colors
    if len(df_cycles) > 0:
        def channel_sort_key(ch):
            m = re.search(r'_ch(\d+)$', ch)
            return int(m.group(1)) if m else 9999
        sorted_channels = sorted(all_timeseries.keys(), key=channel_sort_key)
        for idx, channel in enumerate(sorted_channels):
            data = all_timeseries[channel]
            sample_length = min(1000, len(data))
            x_vals = np.arange(sample_length)
            y_vals = data[:sample_length]
            color = colors[idx % len(colors)]
            m = re.search(r'_ch(\d+)$', channel)
            ch_num = m.group(1) if m else str(idx)
            ax3.plot(x_vals, y_vals, color=color, alpha=0.7, linewidth=1, label=f'Channel {ch_num}')
        for i in range(0, sample_length, period_int):
            ax3.axvline(i, color='black', alpha=0.5, linestyle='--', linewidth=1.2)
        ax3.set_title(f'All Channels: Time Series (Global Best Period = {period:.2f})')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Value')
        ax3.set_xlim(0, sample_length)
        ax3.legend(fontsize=8, ncol=2)

    # 4. Time series of the global best channel (highest confidence in the entire dataset)
    ax4 = plt.subplot(3, 2, 4)
    if len(df_cycles) > 0:
        if channel_name in all_timeseries:
            data = all_timeseries[channel_name]
            sample_length = min(1000, len(data))
            x_vals = np.arange(sample_length)
            y_vals = data[:sample_length]
            ax4.plot(x_vals, y_vals, 'b-', alpha=0.7, linewidth=1)
            for i in range(0, sample_length, period_int):
                ax4.axvline(i, color='black', alpha=0.5, linestyle='--', linewidth=1.2)
            m = re.search(r'_ch(\d+)$', channel_name)
            ch_num = m.group(1) if m else channel_name
            ax4.set_title(f'Best Channel: {ch_num} (Period: {period:.2f})')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Value')
            ax4.set_xlim(0, sample_length)
    
    plt.tight_layout()
    
    # Save as SVG
    plt.savefig(plot_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.cycle(f"Saved cycle analysis plot: {plot_path}.")

    return plot_path

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive cycle analysis for time series data from CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('csv_files', nargs='+', help='Path(s) to CSV file(s) with mean_ch columns')
    parser.add_argument('--output', help='Output base path (default: derived from input)')
    parser.add_argument('--max-cycle-length', type=int, default=200, 
                       help='Maximum cycle length to search for (default: 200)')
    parser.add_argument('--is-cyclenet', action='store_true',
                       help='Exclude last dimension (cycle index) from observations for CycleNet runs')
    
    args = parser.parse_args()

    logger.cycle("Starting cycle analysis...")

    # Check file existence
    valid_files = []
    for csv_file in args.csv_files:
        if os.path.exists(csv_file):
            valid_files.append(csv_file)
        else:
            logger.warning(f"CSV file not found: {csv_file}.")

    if not valid_files:
        parser.error("No valid CSV files found")

    # Run cycle analysis
    try:
        cycle_csv, cycle_plot, _ = analyze_cycles_from_csv(
            csv_files=valid_files,
            output_path=args.output,
            max_cycle_length=args.max_cycle_length,
            is_cyclenet=args.is_cyclenet
        )
        
        if cycle_csv and cycle_plot:
            logger.cycle(f"Cycle analysis completed successfully!")
        else:
            logger.cycle(f"Cycle analysis completed successfully (but there may be missing or incomplete results due to insufficient data)!")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
