"""
Stock Pattern Recognition System
Processes stock price data into "strokes" (sequences between critical points)
Similar to handwriting/ECG pattern recognition applications

Critical Points in Stock Data:
- Local peaks (resistance levels)
- Local troughs (support levels)
- Trend reversals
- Volume spikes
- Breakout points
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime


class StockPatternRecognizer:
    """
    Processes stock data and extracts stroke sequences between critical points.
    
    A "stroke" represents a continuous price movement between two critical points,
    analogous to pen strokes in handwriting or QRS complexes in ECG data.
    """
    
    def __init__(self, prominence_threshold: float = 0.02, 
                 volume_threshold: float = 1.5,
                 smoothing_window: int = 5):
        """
        Initialize the pattern recognizer.
        
        Parameters:
        - prominence_threshold: Minimum prominence for peak/trough detection (as % of price)
        - volume_threshold: Multiplier for volume spike detection (vs rolling average)
        - smoothing_window: Window size for price smoothing (must be odd)
        """
        self.prominence_threshold = prominence_threshold
        self.volume_threshold = volume_threshold
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
    
    def identify_critical_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify critical points in stock price data.
        
        Critical points include:
        1. Local maxima (peaks/resistance)
        2. Local minima (troughs/support)
        3. Trend reversal points
        4. Volume anomalies
        
        Parameters:
        - df: DataFrame with columns ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
        - DataFrame with additional columns marking critical points
        """
        if df.empty or 'close' not in df.columns:
            return df
        
        df = df.copy()
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Extract price series
        prices = df['close'].values
        
        # Smooth the price data to reduce noise
        if len(prices) >= self.smoothing_window:
            smoothed_prices = savgol_filter(prices, self.smoothing_window, 3)
        else:
            smoothed_prices = prices
        
        df['smoothed_close'] = smoothed_prices
        
        # Calculate dynamic prominence threshold based on price range
        price_range = np.max(prices) - np.min(prices)
        prominence = price_range * self.prominence_threshold
        
        # Find peaks (local maxima - resistance levels)
        peaks, peak_properties = find_peaks(
            smoothed_prices, 
            prominence=prominence,
            distance=3  # Minimum 3 data points between peaks
        )
        
        # Find troughs (local minima - support levels)
        troughs, trough_properties = find_peaks(
            -smoothed_prices,
            prominence=prominence,
            distance=3
        )
        
        # Initialize critical point markers
        df['is_critical'] = False
        df['critical_type'] = None
        df['critical_subtype'] = None
        
        # Mark peaks
        df.loc[peaks, 'is_critical'] = True
        df.loc[peaks, 'critical_type'] = 'extremum'
        df.loc[peaks, 'critical_subtype'] = 'peak'
        
        # Mark troughs
        df.loc[troughs, 'is_critical'] = True
        df.loc[troughs, 'critical_type'] = 'extremum'
        df.loc[troughs, 'critical_subtype'] = 'trough'
        
        # Detect trend reversals using moving average crossovers
        if len(df) >= 20:
            df['ma_short'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['ma_long'] = df['close'].rolling(window=20, min_periods=1).mean()
            
            # Bullish crossover (golden cross)
            bullish_cross = (df['ma_short'] > df['ma_long']) & (df['ma_short'].shift(1) <= df['ma_long'].shift(1))
            df.loc[bullish_cross & ~df['is_critical'], 'is_critical'] = True
            df.loc[bullish_cross & ~df['is_critical'], 'critical_type'] = 'trend_reversal'
            df.loc[bullish_cross & ~df['is_critical'], 'critical_subtype'] = 'bullish_cross'
            
            # Bearish crossover (death cross)
            bearish_cross = (df['ma_short'] < df['ma_long']) & (df['ma_short'].shift(1) >= df['ma_long'].shift(1))
            df.loc[bearish_cross & ~df['is_critical'], 'is_critical'] = True
            df.loc[bearish_cross & ~df['is_critical'], 'critical_type'] = 'trend_reversal'
            df.loc[bearish_cross & ~df['is_critical'], 'critical_subtype'] = 'bearish_cross'
        
        # Detect volume spikes
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
            volume_spikes = df['volume'] > (df['volume_ma'] * self.volume_threshold)
            
            df.loc[volume_spikes & ~df['is_critical'], 'is_critical'] = True
            df.loc[volume_spikes & ~df['is_critical'], 'critical_type'] = 'volume_anomaly'
            df.loc[volume_spikes & ~df['is_critical'], 'critical_subtype'] = 'spike'
        
        return df
    
    def strokes(self, df: pd.DataFrame, include_time: bool = True) -> List[np.ndarray]:
        """
        Extract stroke sequences from stock data.
        
        A stroke is a continuous price movement between two critical points,
        represented as a matrix of (price, time) or (price, index) pairs.
        
        This is analogous to:
        - Handwriting: pen strokes between critical points (corners, endpoints)
        - ECG: segments between R-peaks or other fiducial points
        
        Parameters:
        - df: DataFrame with stock data
        - include_time: If True, include timestamps; if False, use sequential indices
        
        Returns:
        - List of stroke matrices, where each stroke is an Nx2 or Nx3 array:
          - Column 0: Normalized price (0-1 range within stroke)
          - Column 1: Time (seconds from start) or sequential index
          - Column 2 (optional): Volume (normalized)
        """
        # Identify critical points
        df_marked = self.identify_critical_points(df)
        
        # Get indices of critical points
        critical_indices = df_marked[df_marked['is_critical']].index.tolist()
        
        if len(critical_indices) < 2:
            # Not enough critical points to form strokes
            return []
        
        # Ensure first and last points are included
        if 0 not in critical_indices:
            critical_indices = [0] + critical_indices
        if len(df_marked) - 1 not in critical_indices:
            critical_indices.append(len(df_marked) - 1)
        
        critical_indices = sorted(critical_indices)
        
        # Extract strokes between consecutive critical points
        strokes_list = []
        
        for i in range(len(critical_indices) - 1):
            start_idx = critical_indices[i]
            end_idx = critical_indices[i + 1]
            
            # Extract data for this stroke
            stroke_data = df_marked.iloc[start_idx:end_idx + 1].copy()
            
            if len(stroke_data) < 2:
                continue
            
            # Build stroke matrix
            prices = stroke_data['close'].values
            
            # Normalize prices to 0-1 range within this stroke
            price_min = prices.min()
            price_max = prices.max()
            if price_max - price_min > 0:
                normalized_prices = (prices - price_min) / (price_max - price_min)
            else:
                normalized_prices = np.zeros_like(prices)
            
            if include_time:
                # Use actual timestamps
                times = pd.to_datetime(stroke_data['datetime']).values
                time_seconds = (times - times[0]) / np.timedelta64(1, 's')
                stroke_matrix = np.column_stack([normalized_prices, time_seconds])
            else:
                # Use sequential indices
                indices = np.arange(len(stroke_data))
                stroke_matrix = np.column_stack([normalized_prices, indices])
            
            # Optionally add volume as third dimension
            if 'volume' in stroke_data.columns:
                volumes = stroke_data['volume'].values
                vol_min = volumes.min()
                vol_max = volumes.max()
                if vol_max - vol_min > 0:
                    normalized_volumes = (volumes - vol_min) / (vol_max - vol_min)
                else:
                    normalized_volumes = np.zeros_like(volumes)
                stroke_matrix = np.column_stack([stroke_matrix, normalized_volumes])
            
            strokes_list.append(stroke_matrix)
        
        return strokes_list
    
    def get_stroke_features(self, stroke: np.ndarray) -> Dict[str, float]:
        """
        Extract features from a single stroke for pattern recognition.
        
        Features include:
        - Duration (time span)
        - Amplitude (price change)
        - Direction (up/down/flat)
        - Volatility (price variance)
        - Volume change (if available)
        
        Parameters:
        - stroke: Nx2 or Nx3 array from strokes() method
        
        Returns:
        - Dictionary of feature values
        """
        if len(stroke) < 2:
            return {}
        
        features = {}
        
        # Extract columns
        prices = stroke[:, 0]  # Normalized prices
        times = stroke[:, 1]   # Time or index
        
        # Duration
        features['duration'] = times[-1] - times[0]
        
        # Price change (amplitude)
        features['amplitude'] = prices[-1] - prices[0]
        features['abs_amplitude'] = abs(features['amplitude'])
        
        # Direction
        if features['amplitude'] > 0.1:
            features['direction'] = 'up'
        elif features['amplitude'] < -0.1:
            features['direction'] = 'down'
        else:
            features['direction'] = 'flat'
        
        # Volatility (standard deviation of prices)
        features['volatility'] = np.std(prices)
        
        # Price range within stroke
        features['price_range'] = np.max(prices) - np.min(prices)
        
        # Curvature (second derivative approximation)
        if len(prices) >= 3:
            second_diff = np.diff(prices, n=2)
            features['curvature'] = np.mean(np.abs(second_diff))
        else:
            features['curvature'] = 0.0
        
        # Volume features (if available)
        if stroke.shape[1] >= 3:
            volumes = stroke[:, 2]
            features['volume_change'] = volumes[-1] - volumes[0]
            features['avg_volume'] = np.mean(volumes)
            features['volume_volatility'] = np.std(volumes)
        
        return features
    
    def classify_pattern(self, strokes_list: List[np.ndarray], 
                        window_size: int = 3) -> List[Dict]:
        """
        Classify patterns based on sequences of strokes.
        
        Common patterns:
        - Head and Shoulders (peak-trough-peak-trough-peak)
        - Double Top/Bottom (peak-trough-peak or trough-peak-trough)
        - Triangle (converging peaks and troughs)
        - Flag/Pennant (consolidation after sharp move)
        
        Parameters:
        - strokes_list: List of stroke matrices from strokes()
        - window_size: Number of consecutive strokes to analyze
        
        Returns:
        - List of dictionaries containing pattern classifications
        """
        patterns = []
        
        if len(strokes_list) < window_size:
            return patterns
        
        # Analyze windows of consecutive strokes
        for i in range(len(strokes_list) - window_size + 1):
            window = strokes_list[i:i + window_size]
            
            # Extract features for each stroke in window
            features_list = [self.get_stroke_features(stroke) for stroke in window]
            
            # Pattern detection logic
            directions = [f.get('direction', 'flat') for f in features_list]
            amplitudes = [f.get('amplitude', 0) for f in features_list]
            
            pattern_dict = {
                'start_index': i,
                'end_index': i + window_size - 1,
                'pattern_type': None,
                'confidence': 0.0
            }
            
            # Simple pattern detection examples
            if window_size == 3:
                # Double Top: up, down, up with similar heights
                if (directions == ['up', 'down', 'up'] and 
                    abs(amplitudes[0] - amplitudes[2]) < 0.2):
                    pattern_dict['pattern_type'] = 'double_top'
                    pattern_dict['confidence'] = 0.8
                
                # Double Bottom: down, up, down with similar depths
                elif (directions == ['down', 'up', 'down'] and 
                      abs(amplitudes[0] - amplitudes[2]) < 0.2):
                    pattern_dict['pattern_type'] = 'double_bottom'
                    pattern_dict['confidence'] = 0.8
                
                # Ascending Triangle: up, flat, up
                elif directions == ['up', 'flat', 'up']:
                    pattern_dict['pattern_type'] = 'ascending_triangle'
                    pattern_dict['confidence'] = 0.7
                
                # Descending Triangle: down, flat, down
                elif directions == ['down', 'flat', 'down']:
                    pattern_dict['pattern_type'] = 'descending_triangle'
                    pattern_dict['confidence'] = 0.7
            
            if pattern_dict['pattern_type']:
                patterns.append(pattern_dict)
        
        return patterns
    
    def visualize_strokes(self, df: pd.DataFrame, strokes_list: List[np.ndarray],
                         symbol: str = "Stock", save_path: Optional[str] = None):
        """
        Visualize the original price data with strokes and critical points marked.
        
        Parameters:
        - df: Original DataFrame with stock data
        - strokes_list: List of stroke matrices from strokes()
        - symbol: Stock symbol for title
        - save_path: Optional path to save the figure
        """
        df_marked = self.identify_critical_points(df)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Top plot: Price with critical points
        ax1.plot(df_marked['datetime'], df_marked['close'], 
                label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
        ax1.plot(df_marked['datetime'], df_marked['smoothed_close'],
                label='Smoothed Price', color='orange', linewidth=1, alpha=0.8)
        
        # Mark critical points
        critical_points = df_marked[df_marked['is_critical']]
        
        peaks = critical_points[critical_points['critical_subtype'] == 'peak']
        troughs = critical_points[critical_points['critical_subtype'] == 'trough']
        
        ax1.scatter(peaks['datetime'], peaks['close'], 
                   color='red', marker='v', s=100, label='Peaks', zorder=5)
        ax1.scatter(troughs['datetime'], troughs['close'],
                   color='green', marker='^', s=100, label='Troughs', zorder=5)
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'{symbol} - Price with Critical Points', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Strokes visualization
        colors = plt.cm.rainbow(np.linspace(0, 1, len(strokes_list)))
        
        for idx, stroke in enumerate(strokes_list):
            # Denormalize for visualization purposes
            # Plot as segments
            ax2.plot(range(len(stroke)), stroke[:, 0], 
                    color=colors[idx], linewidth=2, alpha=0.7,
                    label=f'Stroke {idx+1}' if idx < 10 else '')
        
        ax2.set_xlabel('Normalized Time Index', fontsize=12)
        ax2.set_ylabel('Normalized Price', fontsize=12)
        ax2.set_title(f'{symbol} - Extracted Strokes ({len(strokes_list)} total)', 
                     fontsize=14, fontweight='bold')
        if len(strokes_list) <= 10:
            ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()


def main():
    """
    Example usage of the Stock Pattern Recognition system.
    Demonstrates how to process stock data into strokes and extract patterns.
    """
    # Import the collector
    from stock_news_collector import StockDataCollector
    
    print("="*70)
    print("STOCK PATTERN RECOGNITION SYSTEM")
    print("="*70)
    print("\nThis system processes stock data similarly to handwriting/ECG analysis:")
    print("1. Identifies critical points (peaks, troughs, reversals)")
    print("2. Segments data into 'strokes' between critical points")
    print("3. Extracts features and classifies patterns")
    print("="*70)
    
    # Configuration
    TWELVE_DATA_API_KEY = "ac3aa4f4061b4c4f91a9e3636dbde84a"
    symbol = "AAPL"
    
    print(f"\n📊 Collecting data for {symbol}...")
    
    # Collect stock data
    collector = StockDataCollector(TWELVE_DATA_API_KEY)
    df = collector.get_time_series(symbol=symbol, interval="1day", outputsize=60)
    
    if df.empty:
        print("❌ Failed to collect stock data")
        return
    
    print(f"✓ Retrieved {len(df)} data points")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Initialize pattern recognizer
    print("\n🔍 Initializing pattern recognizer...")
    recognizer = StockPatternRecognizer(
        prominence_threshold=0.02,  # 2% price change for critical points
        volume_threshold=1.5,       # 50% above average for volume spikes
        smoothing_window=5          # 5-day smoothing
    )
    
    # Identify critical points
    print("\n📍 Identifying critical points...")
    df_marked = recognizer.identify_critical_points(df)
    
    critical_points = df_marked[df_marked['is_critical']]
    print(f"✓ Found {len(critical_points)} critical points:")
    print(f"  - Peaks: {len(critical_points[critical_points['critical_subtype'] == 'peak'])}")
    print(f"  - Troughs: {len(critical_points[critical_points['critical_subtype'] == 'trough'])}")
    print(f"  - Other: {len(critical_points[~critical_points['critical_subtype'].isin(['peak', 'trough'])])}")
    
    # Extract strokes
    print("\n✂️  Extracting stroke sequences...")
    strokes_list = recognizer.strokes(df, include_time=True)
    print(f"✓ Extracted {len(strokes_list)} strokes")
    
    # Analyze stroke features
    print("\n📊 Analyzing stroke features...")
    for i, stroke in enumerate(strokes_list[:5]):  # Show first 5
        features = recognizer.get_stroke_features(stroke)
        print(f"\n  Stroke {i+1}:")
        print(f"    - Duration: {features['duration']:.2f} time units")
        print(f"    - Direction: {features['direction']}")
        print(f"    - Amplitude: {features['amplitude']:.4f}")
        print(f"    - Volatility: {features['volatility']:.4f}")
    
    if len(strokes_list) > 5:
        print(f"\n  ... and {len(strokes_list) - 5} more strokes")
    
    # Classify patterns
    print("\n🎯 Classifying patterns...")
    patterns = recognizer.classify_pattern(strokes_list, window_size=3)
    
    if patterns:
        print(f"✓ Found {len(patterns)} pattern(s):")
        for pattern in patterns:
            print(f"\n  Pattern: {pattern['pattern_type']}")
            print(f"    - Strokes: {pattern['start_index']} to {pattern['end_index']}")
            print(f"    - Confidence: {pattern['confidence']:.2%}")
    else:
        print("  No patterns detected in this dataset")
    
    # Save stroke data
    print("\n💾 Saving stroke data...")
    stroke_data = []
    for i, stroke in enumerate(strokes_list):
        features = recognizer.get_stroke_features(stroke)
        features['stroke_id'] = i
        features['stroke_length'] = len(stroke)
        stroke_data.append(features)
    
    stroke_df = pd.DataFrame(stroke_data)
    stroke_df.to_csv(f"{symbol}_strokes.csv", index=False)
    print(f"✓ Saved {symbol}_strokes.csv")
    
    # Visualize
    print("\n📈 Generating visualization...")
    recognizer.visualize_strokes(df, strokes_list, symbol=symbol, 
                                 save_path=f"{symbol}_pattern_analysis.png")
    
    print("\n" + "="*70)
    print("✅ PATTERN RECOGNITION COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - {symbol}_strokes.csv (stroke features)")
    print(f"  - {symbol}_pattern_analysis.png (visualization)")


if __name__ == "__main__":
    main()
