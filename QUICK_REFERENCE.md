# Quick Reference Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install numpy pandas scipy matplotlib seaborn requests
```

### Step 2: Set Your API Keys

Edit the files and replace with your keys:
- **Twelve Data**: https://twelvedata.com/ (free tier: 8 req/min)
- **NewsAPI.ai**: https://newsapi.ai/ (optional for news)

### Step 3: Run the Analysis

**Option A: Quick Single Stock Analysis**
```bash
python stock_pattern_recognizer.py
```

**Option B: Full Pipeline (Multiple Stocks)**
```bash
python integrated_stock_analyzer.py
```

**Option C: Interactive Notebook**
```bash
jupyter notebook stock_analysis.ipynb
```

---

## 📋 Core Function: `strokes()`

This is the **main callable routine** required by the assignment.

### Basic Usage

```python
from stock_pattern_recognizer import StockPatternRecognizer
from stock_news_collector import StockDataCollector

# 1. Get data
collector = StockDataCollector("YOUR_API_KEY")
df = collector.get_time_series("AAPL", interval="1day", outputsize=60)

# 2. Extract strokes
recognizer = StockPatternRecognizer()
strokes_list = recognizer.strokes(df, include_time=True)

# 3. Analyze
print(f"Found {len(strokes_list)} strokes")
for i, stroke in enumerate(strokes_list):
    print(f"Stroke {i+1} shape: {stroke.shape}")
```

### What `strokes()` Returns

A **list of NumPy arrays**, where each array is a stroke:

```python
# Example output:
strokes_list = [
    array([[0.0000, 0.0],        # Stroke 1 (6 points)
           [0.2543, 86400.0],
           [0.5124, 172800.0],
           [0.8765, 259200.0],
           [1.0000, 345600.0]]),
           
    array([[0.0000, 0.0],        # Stroke 2 (4 points)
           [0.6543, 86400.0],
           [0.3421, 172800.0],
           [0.0000, 259200.0]]),
    ...
]
```

**Column meanings:**
- Column 0: Normalized price (0-1 within stroke)
- Column 1: Time in seconds from stroke start
- Column 2: (Optional) Normalized volume

---

## 🎯 Assignment Checklist

| Requirement | Implementation | File/Function |
|-------------|----------------|---------------|
| ✅ Choose application | Stock market analysis | All files |
| ✅ Capture/download data | Twelve Data API + NewsAPI | `stock_news_collector.py` |
| ✅ Callable "strokes" routine | `strokes()` function | `stock_pattern_recognizer.py:248` |
| ✅ Process application data | Stock prices → sequences | `strokes()` |
| ✅ Return cursor locations | Price trajectories | NumPy arrays (Nx2) |
| ✅ Sequence of matrices | List of arrays | `List[np.ndarray]` |
| ✅ (x,y) or (signal,time) pairs | (price, time) pairs | Column 0 & 1 |
| ✅ Between critical points | Peaks, troughs, reversals | `identify_critical_points()` |
| ✅ Domain-specific critical points | Stock market methods | Peak detection, MA crossovers |

---

## 🔧 Common Customizations

### Change Stocks to Analyze

Edit `integrated_stock_analyzer.py`:

```python
stocks = [
    {"symbol": "YOUR_SYMBOL", "company_name": "Company Name"},
    # Add more...
]
```

### Adjust Critical Point Sensitivity

```python
recognizer = StockPatternRecognizer(
    prominence_threshold=0.01,    # Lower = more sensitive (more critical points)
    volume_threshold=2.0,         # Higher = less sensitive (fewer volume spikes)
    smoothing_window=7            # Higher = smoother (fewer noise peaks)
)
```

### Change Time Period

```python
df = collector.get_time_series(
    symbol="AAPL",
    interval="1day",    # Options: 1min, 5min, 1h, 1day, 1week
    outputsize=90       # Number of periods
)
```

---

## 📊 Understanding the Output

### Stroke Features

Each stroke has these computed features:

```python
features = {
    'duration': 432000.0,        # Length in seconds
    'amplitude': 0.234,          # Price change (-1 to 1)
    'direction': 'up',           # 'up', 'down', or 'flat'
    'volatility': 0.045,         # Price variance
    'curvature': 0.012,          # Trend smoothness
    'price_range': 0.156,        # High-low within stroke
    'volume_change': 0.23        # Volume change (if available)
}
```

### Critical Point Types

```python
'extremum' → 'peak'              # Local maximum (resistance)
'extremum' → 'trough'            # Local minimum (support)
'trend_reversal' → 'bullish_cross'   # Upward trend start
'trend_reversal' → 'bearish_cross'   # Downward trend start
'volume_anomaly' → 'spike'           # Unusual trading volume
```

### Pattern Classifications

```python
'double_top'                # Resistance test pattern
'double_bottom'             # Support test pattern
'ascending_triangle'        # Bullish continuation
'descending_triangle'       # Bearish continuation
```

---

## 📁 Output Files Reference

| File | Contents | Use Case |
|------|----------|----------|
| `{SYMBOL}_strokes.csv` | Stroke features | ML training, statistics |
| `{SYMBOL}_marked_data.csv` | Prices + critical points | Verification, analysis |
| `{SYMBOL}_pattern_analysis.png` | Visualizations | Presentations, reports |
| `{SYMBOL}_summary.txt` | Text report | Quick overview |
| `{SYMBOL}_news_data.csv` | News articles | Sentiment analysis |

---

## 🐛 Troubleshooting

### "No critical points found"
- **Cause**: Data too smooth or threshold too high
- **Fix**: Lower `prominence_threshold` to 0.01 or less
- **Fix**: Use more data points (increase `outputsize`)

### "HTTP Error: 429"
- **Cause**: API rate limit exceeded
- **Fix**: Increase `delay_seconds` in the script
- **Fix**: Upgrade API plan

### "Empty stroke list"
- **Cause**: Not enough data or no valid segments
- **Fix**: Collect more data (increase `outputsize` to 100+)
- **Fix**: Check if data is valid (not all NaN values)

### ImportError
- **Fix**: Install missing package: `pip install [package_name]`
- **Fix**: Ensure all files in same directory

---

## 💡 Tips for Best Results

1. **Data Quantity**: Use at least 60 days for meaningful patterns
2. **Multiple Stocks**: Analyze 5-10 stocks for comparative analysis
3. **Rate Limiting**: Respect API limits (8/min free tier)
4. **Parameter Tuning**: Adjust thresholds based on stock volatility
5. **Visualization**: Always check plots to validate critical points

---

## 📞 Quick Help

**Question**: How do I access individual stroke data?
```python
stroke = strokes_list[0]           # First stroke
prices = stroke[:, 0]              # Price column
times = stroke[:, 1]               # Time column
```

**Question**: How do I count critical points?
```python
df_marked = recognizer.identify_critical_points(df)
num_critical = len(df_marked[df_marked['is_critical']])
```

**Question**: How do I get just peaks or troughs?
```python
peaks = df_marked[df_marked['critical_subtype'] == 'peak']
troughs = df_marked[df_marked['critical_subtype'] == 'trough']
```

---

## 🔗 File Dependencies

```
integrated_stock_analyzer.py
├── imports stock_news_collector.py
│   ├── StockDataCollector
│   ├── NewsCollector
│   └── TimeSeriesFormatter
└── imports stock_pattern_recognizer.py
    └── StockPatternRecognizer
        ├── identify_critical_points()
        ├── strokes()  ⭐ THE KEY FUNCTION
        ├── get_stroke_features()
        ├── classify_pattern()
        └── visualize_strokes()
```

---

## ⚡ Quick Commands

```bash
# Run main analysis
python integrated_stock_analyzer.py

# Run pattern recognizer only  
python stock_pattern_recognizer.py

# Install all requirements
pip install -r requirements.txt

# Generate requirements file
pip freeze > requirements.txt
```

---

## 📖 Further Reading

- **README.md** - Complete documentation
- **Code comments** - Inline explanations
- **Docstrings** - Function documentation

---

**Remember**: The `strokes()` function is the core requirement. Everything else supports it!
