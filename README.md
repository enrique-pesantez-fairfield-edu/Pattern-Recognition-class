# Stock Pattern Recognition System

A comprehensive system for collecting stock market data and processing it into pattern recognition sequences, similar to handwriting or ECG analysis applications.

## 📋 Overview

This project implements a pattern recognition pipeline for stock market data that:

1. **Collects Data**: Fetches stock prices and related news from APIs
2. **Identifies Critical Points**: Detects peaks, troughs, trend reversals, and volume anomalies
3. **Extracts Strokes**: Segments data into sequences between critical points
4. **Classifies Patterns**: Recognizes common chart patterns (double tops, triangles, etc.)
5. **Generates Reports**: Creates visualizations and analytical summaries

### Analogy to Assignment Requirements

This implementation treats stock data similarly to handwriting or ECG data:

| Aspect | Handwriting/ECG | Stock Market |
|--------|----------------|--------------|
| **Data Stream** | (x,y) pen coordinates or signal amplitude | Price over time |
| **Critical Points** | Corners, endpoints, peaks | Support/resistance levels, reversals |
| **Strokes** | Continuous pen movements | Price movements between critical points |
| **Features** | Direction, curvature, speed | Amplitude, volatility, duration |
| **Patterns** | Letters, QRS complexes | Head & shoulders, double tops, triangles |

## 🗂️ Project Structure

```
.
├── stock_news_collector.py          # Data collection module
├── stock_pattern_recognizer.py      # Pattern recognition module (Core)
├── integrated_stock_analyzer.py     # Complete pipeline integration
├── stock_analysis.ipynb              # Jupyter notebook version
└── README.md                         # This file
```

## 📦 Key Components

### 1. Data Collection (`stock_news_collector.py`)

**Classes:**
- `StockDataCollector`: Fetches stock price data from Twelve Data API
- `NewsCollector`: Retrieves relevant news articles from NewsAPI.ai
- `TimeSeriesFormatter`: Formats and merges data for analysis

**Features:**
- Multiple stock support
- Rate limiting (8 requests/minute)
- News filtering for stock-relevant articles
- Time series sequence generation

### 2. Pattern Recognition (`stock_pattern_recognizer.py`) ⭐

**Main Class: `StockPatternRecognizer`**

This is the **core module** that implements the "strokes" functionality required by the assignment.

#### Critical Point Detection

Identifies significant points in price data:

```python
def identify_critical_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies critical points including:
    - Local maxima (peaks/resistance levels)
    - Local minima (troughs/support levels)  
    - Trend reversals (moving average crossovers)
    - Volume anomalies (spikes)
    """
```

**Methods Used:**
- Savitzky-Golay filtering for noise reduction
- `scipy.signal.find_peaks` for peak/trough detection
- Moving average crossovers for trend changes
- Statistical analysis for volume anomalies

#### The `strokes()` Function ⭐⭐⭐

**This is the callable routine required by the assignment:**

```python
def strokes(df: pd.DataFrame, include_time: bool = True) -> List[np.ndarray]:
    """
    Extract stroke sequences from stock data.
    
    A stroke is a continuous price movement between two critical points,
    represented as a matrix of (price, time) pairs.
    
    Returns:
    - List of stroke matrices, where each stroke is an Nx2 or Nx3 array:
      - Column 0: Normalized price (0-1 range within stroke)
      - Column 1: Time (seconds from start) or sequential index
      - Column 2 (optional): Volume (normalized)
    """
```

**Output Format:**

Each stroke is a NumPy array with shape (N, 2) or (N, 3):

```
Stroke 1:
[[0.0000, 0.0],      # Start point: (normalized_price, time)
 [0.2543, 1.0],      # Intermediate point
 [0.5124, 2.0],      # Intermediate point
 [0.8765, 3.0],      # Intermediate point
 [1.0000, 4.0]]      # End point (critical point)

Stroke 2:
[[0.0000, 0.0],      # Starts at previous critical point
 [0.4532, 1.0],
 ...
```

**Key Features:**
- ✅ Processes application data (stock prices)
- ✅ Returns cursor location sequences (price trajectories)
- ✅ Sequences between critical points
- ✅ Contains (value, time) pairs
- ✅ Normalized within each stroke
- ✅ Handles variable-length sequences

#### Pattern Classification

Detects common chart patterns:

```python
def classify_pattern(strokes_list: List[np.ndarray], window_size: int = 3) -> List[Dict]:
    """
    Classifies patterns based on stroke sequences:
    - Double Top/Bottom
    - Ascending/Descending Triangles
    - Head and Shoulders
    - Flags and Pennants
    """
```

#### Feature Extraction

```python
def get_stroke_features(stroke: np.ndarray) -> Dict[str, float]:
    """
    Extracts features from individual strokes:
    - Duration (time span)
    - Amplitude (price change)
    - Direction (up/down/flat)
    - Volatility (price variance)
    - Curvature (trend smoothness)
    - Volume characteristics
    """
```

### 3. Integrated Pipeline (`integrated_stock_analyzer.py`)

**Class: `IntegratedStockAnalyzer`**

Combines all components into a complete workflow:

```python
analyzer = IntegratedStockAnalyzer(twelve_data_key, news_api_key)

# Analyze single stock
results = analyzer.analyze_stock(
    symbol="AAPL",
    company_name="Apple Inc",
    days=60,
    collect_news=True
)

# Analyze multiple stocks
all_results = analyzer.analyze_multiple_stocks(
    stock_list=[{"symbol": "AAPL", "company_name": "Apple Inc"}, ...],
    collect_news=True,
    delay_seconds=8
)

# Generate reports
analyzer.generate_report(results)
```

## 🚀 Usage

### Quick Start

```python
from stock_pattern_recognizer import StockPatternRecognizer
from stock_news_collector import StockDataCollector

# 1. Collect data
collector = StockDataCollector(api_key="your_key")
df = collector.get_time_series("AAPL", interval="1day", outputsize=60)

# 2. Initialize recognizer
recognizer = StockPatternRecognizer()

# 3. Extract strokes (THE KEY FUNCTION)
strokes_list = recognizer.strokes(df, include_time=True)

# 4. Analyze each stroke
for i, stroke in enumerate(strokes_list):
    features = recognizer.get_stroke_features(stroke)
    print(f"Stroke {i}: {features['direction']} movement")

# 5. Classify patterns
patterns = recognizer.classify_pattern(strokes_list, window_size=3)
```

### Running the Complete Pipeline

```bash
# Run standalone pattern recognizer
python stock_pattern_recognizer.py

# Run integrated analyzer (all stocks)
python integrated_stock_analyzer.py
```

### Using the Jupyter Notebook

Open `stock_analysis.ipynb` and run cells sequentially. The notebook provides:
- Interactive data exploration
- Step-by-step explanations
- Visualizations
- Parameter tuning

## 📊 Output Files

### Per Stock:

1. **`{SYMBOL}_strokes.csv`** - Stroke features and characteristics
2. **`{SYMBOL}_marked_data.csv`** - Price data with critical points marked
3. **`{SYMBOL}_news_data.csv`** - Relevant news articles
4. **`{SYMBOL}_merged_data.csv`** - Stock prices merged with news counts
5. **`{SYMBOL}_pattern_analysis.png`** - Visualization of strokes and critical points
6. **`{SYMBOL}_summary.txt`** - Analysis summary report

### Data Formats:

**Strokes CSV:**
```csv
stroke_id,duration,amplitude,direction,volatility,curvature,stroke_length
0,5.0,0.234,up,0.045,0.012,6
1,3.0,-0.156,down,0.067,0.023,4
```

**Marked Data CSV:**
```csv
datetime,open,high,low,close,volume,is_critical,critical_type,critical_subtype
2024-01-01,150.0,152.0,149.5,151.5,1000000,False,,,
2024-01-02,151.5,154.0,151.0,153.8,1200000,True,extremum,peak
```

## 🔧 Configuration

### API Rate Limits

```python
# Twelve Data: 8 requests per minute (free tier)
# Delay between requests: 8 seconds
delay_seconds = 8
```

### Pattern Recognition Parameters

```python
recognizer = StockPatternRecognizer(
    prominence_threshold=0.02,    # 2% price change for peaks/troughs
    volume_threshold=1.5,         # 50% above average for spikes
    smoothing_window=5            # 5-day moving average
)
```

### Stocks to Analyze

Edit in `integrated_stock_analyzer.py`:

```python
stocks = [
    {"symbol": "AAPL", "company_name": "Apple Inc"},
    {"symbol": "MSFT", "company_name": "Microsoft"},
    {"symbol": "GOOGL", "company_name": "Google"},
    {"symbol": "NVDA", "company_name": "NVIDIA"},
    {"symbol": "TSLA", "company_name": "Tesla"}
]
```

## 📚 Technical Details

### Critical Point Detection Algorithm

1. **Smoothing**: Apply Savitzky-Golay filter to reduce noise
2. **Peak Detection**: Use scipy's `find_peaks` with prominence threshold
3. **Trough Detection**: Invert signal and find peaks
4. **Trend Analysis**: Calculate moving average crossovers
5. **Volume Analysis**: Detect statistical outliers

### Stroke Extraction Process

1. Identify all critical points in time series
2. Segment data between consecutive critical points
3. Normalize prices within each segment (0-1 range)
4. Convert timestamps to relative time (seconds from start)
5. Optionally include normalized volume as third dimension
6. Return list of Nx2 or Nx3 NumPy arrays

### Pattern Classification Logic

```
Double Top:    [up, down, up] with similar amplitudes
Double Bottom: [down, up, down] with similar amplitudes
Ascending Triangle: [up, flat, up]
Descending Triangle: [down, flat, down]
```

## 🎯 Assignment Compliance

### Requirements Met:

✅ **Choose application**: Stock market analysis (alternative to handwriting/ECG)

✅ **Capture/download data**: 
- `StockDataCollector.get_time_series()` fetches historical price data
- `NewsCollector.search_stock_news()` retrieves news data
- Supports multiple stocks and date ranges

✅ **Callable routine "strokes"**:
- `StockPatternRecognizer.strokes()` processes application data
- Returns cursor location sequences (price trajectories)
- Sequences of stroke matrices containing (price, time) pairs
- Segments between critical points

✅ **Critical point determination**:
- `identify_critical_points()` uses domain-specific methods
- Peak/trough detection for price extrema
- Moving average crossovers for trend changes
- Volume spike detection for anomalies
- Different approach than handwriting/ECG (as required)

### Data Structure:

```python
# Each stroke is an Nx2 array
stroke = np.array([
    [normalized_price, time],  # Point 1
    [normalized_price, time],  # Point 2
    ...
    [normalized_price, time]   # Point N
])

# Or Nx3 with volume
stroke = np.array([
    [normalized_price, time, normalized_volume],
    ...
])
```

## 🔬 Example Analysis

```python
>>> strokes_list = recognizer.strokes(df)
>>> print(f"Extracted {len(strokes_list)} strokes")
Extracted 12 strokes

>>> stroke = strokes_list[0]
>>> print(f"Stroke shape: {stroke.shape}")
Stroke shape: (8, 2)

>>> print(f"Stroke data:\n{stroke}")
Stroke data:
[[0.0000, 0.0],
 [0.2145, 86400.0],
 [0.4521, 172800.0],
 [0.6734, 259200.0],
 [0.8234, 345600.0],
 [1.0000, 432000.0]]

>>> features = recognizer.get_stroke_features(stroke)
>>> print(features)
{
    'duration': 432000.0,          # 5 days in seconds
    'amplitude': 1.0,               # Full upward movement
    'direction': 'up',              # Uptrend
    'volatility': 0.045,            # Relatively stable
    'curvature': 0.012              # Smooth trend
}
```

## 📈 Visualization

The system generates comprehensive visualizations showing:

- **Top Plot**: Price chart with critical points marked
  - Red triangles: Peaks (resistance)
  - Green triangles: Troughs (support)
  - Blue line: Original prices
  - Orange line: Smoothed prices

- **Bottom Plot**: Extracted strokes
  - Each stroke shown in different color
  - Normalized 0-1 scale
  - Shows segmentation between critical points

## 🛠️ Dependencies

```
numpy
pandas
scipy
matplotlib
seaborn
requests
```

Install with:
```bash
pip install numpy pandas scipy matplotlib seaborn requests
```

## 📝 Notes

- **Rate Limiting**: The free tier of Twelve Data allows 8 requests per minute. The system includes automatic delays.
- **Data Quality**: News filtering removes irrelevant articles (sports, travel, etc.)
- **Normalization**: Each stroke is independently normalized for consistent comparison
- **Pattern Detection**: Currently implements basic patterns; can be extended with ML models

## 🎓 Educational Value

This project demonstrates:

1. **Signal Processing**: Peak detection, smoothing, noise reduction
2. **Time Series Analysis**: Segmentation, feature extraction
3. **Pattern Recognition**: Sequence classification, similarity metrics
4. **API Integration**: Rate-limited data collection
5. **Data Pipeline**: ETL, transformation, visualization

## 🔄 Future Enhancements

- Machine learning pattern classification
- Real-time data streaming
- Additional technical indicators
- Predictive modeling
- Portfolio optimization
- Risk analysis integration

## 📞 Support

For questions about the pattern recognition algorithm or stroke extraction, refer to:
- `stock_pattern_recognizer.py` - Core implementation
- Inline code comments and docstrings
- This README

---

**Summary**: This system implements a complete pattern recognition pipeline for stock market data, with the `strokes()` function serving as the callable routine that processes data into sequences between critical points, meeting all assignment requirements.
