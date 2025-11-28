"""
Integrated Stock Analysis Pipeline
Combines data collection with pattern recognition (stroke extraction)

This script:
1. Collects stock data using StockDataCollector
2. Processes data into strokes using StockPatternRecognizer
3. Extracts features and classifies patterns
4. Generates comprehensive analysis reports
"""

import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Import our custom modules
from stock_news_collector import StockDataCollector, NewsCollector, TimeSeriesFormatter
from stock_pattern_recognizer import StockPatternRecognizer


class IntegratedStockAnalyzer:
    """
    Integrated pipeline for stock data collection and pattern recognition.
    """
    
    def __init__(self, twelve_data_key: str, news_api_key: str):
        """
        Initialize the integrated analyzer with API keys.
        
        Parameters:
        - twelve_data_key: API key for Twelve Data
        - news_api_key: API key for NewsAPI.ai
        """
        self.stock_collector = StockDataCollector(twelve_data_key)
        self.news_collector = NewsCollector(news_api_key)
        self.formatter = TimeSeriesFormatter()
        self.pattern_recognizer = StockPatternRecognizer(
            prominence_threshold=0.02,
            volume_threshold=1.5,
            smoothing_window=5
        )
    
    def analyze_stock(self, symbol: str, company_name: str,
                     days: int = 60, collect_news: bool = True) -> Dict:
        """
        Complete analysis pipeline for a single stock.
        
        Parameters:
        - symbol: Stock ticker (e.g., 'AAPL')
        - company_name: Full company name (e.g., 'Apple Inc')
        - days: Number of days of historical data
        - collect_news: Whether to collect news data
        
        Returns:
        - Dictionary containing all analysis results
        """
        results = {
            'symbol': symbol,
            'company_name': company_name,
            'timestamp': datetime.now(),
            'success': False
        }
        
        print(f"\n{'='*70}")
        print(f"ANALYZING: {symbol} ({company_name})")
        print(f"{'='*70}")
        
        # Step 1: Collect stock data
        print(f"\n[1/5] 📊 Collecting stock price data...")
        stock_df = self.stock_collector.get_time_series(
            symbol=symbol,
            interval="1day",
            outputsize=days
        )
        
        if stock_df.empty:
            print(f"❌ Failed to collect stock data for {symbol}")
            return results
        
        print(f"✓ Retrieved {len(stock_df)} data points")
        print(f"  Date range: {stock_df['datetime'].min()} to {stock_df['datetime'].max()}")
        results['stock_data'] = stock_df
        
        # Step 2: Collect news data (optional)
        news_df = pd.DataFrame()
        if collect_news:
            print(f"\n[2/5] 📰 Collecting news articles...")
            date_end = datetime.now().strftime("%Y-%m-%d")
            date_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            news_df = self.news_collector.search_stock_news(
                symbol=symbol,
                company_name=company_name,
                max_items=50,
                date_start=date_start,
                date_end=date_end
            )
            
            if not news_df.empty:
                print(f"✓ Retrieved {len(news_df)} relevant news articles")
                results['news_data'] = news_df
                
                # Merge stock and news
                merged_df = self.formatter.merge_stock_news(stock_df, news_df)
                results['merged_data'] = merged_df
            else:
                print("⚠ No news data retrieved")
        else:
            print(f"\n[2/5] ⏭️  Skipping news collection")
        
        # Step 3: Identify critical points
        print(f"\n[3/5] 📍 Identifying critical points...")
        df_marked = self.pattern_recognizer.identify_critical_points(stock_df)
        critical_points = df_marked[df_marked['is_critical']]
        
        print(f"✓ Found {len(critical_points)} critical points:")
        peaks = critical_points[critical_points['critical_subtype'] == 'peak']
        troughs = critical_points[critical_points['critical_subtype'] == 'trough']
        print(f"  - Peaks (resistance): {len(peaks)}")
        print(f"  - Troughs (support): {len(troughs)}")
        print(f"  - Other signals: {len(critical_points) - len(peaks) - len(troughs)}")
        
        results['critical_points'] = critical_points
        results['marked_data'] = df_marked
        
        # Step 4: Extract strokes
        print(f"\n[4/5] ✂️  Extracting stroke sequences...")
        strokes_list = self.pattern_recognizer.strokes(stock_df, include_time=True)
        print(f"✓ Extracted {len(strokes_list)} strokes")
        
        # Analyze each stroke
        stroke_features = []
        for i, stroke in enumerate(strokes_list):
            features = self.pattern_recognizer.get_stroke_features(stroke)
            features['stroke_id'] = i
            features['stroke_length'] = len(stroke)
            features['symbol'] = symbol
            stroke_features.append(features)
        
        stroke_df = pd.DataFrame(stroke_features)
        results['strokes'] = strokes_list
        results['stroke_features'] = stroke_df
        
        # Show sample strokes
        print("\n  Sample stroke analysis:")
        for i in range(min(3, len(strokes_list))):
            feat = stroke_features[i]
            print(f"    Stroke {i+1}: {feat['direction']} movement, "
                  f"amplitude={feat['amplitude']:.3f}, "
                  f"volatility={feat['volatility']:.3f}")
        
        if len(strokes_list) > 3:
            print(f"    ... and {len(strokes_list) - 3} more strokes")
        
        # Step 5: Classify patterns
        print(f"\n[5/5] 🎯 Classifying chart patterns...")
        patterns = self.pattern_recognizer.classify_pattern(strokes_list, window_size=3)
        
        if patterns:
            print(f"✓ Detected {len(patterns)} pattern(s):")
            for pattern in patterns:
                print(f"  - {pattern['pattern_type']} "
                      f"(confidence: {pattern['confidence']:.1%})")
        else:
            print("  No standard patterns detected")
        
        results['patterns'] = patterns
        results['success'] = True
        
        return results
    
    def analyze_multiple_stocks(self, stock_list: List[Dict], 
                               collect_news: bool = True,
                               delay_seconds: int = 8) -> Dict[str, Dict]:
        """
        Analyze multiple stocks with rate limiting.
        
        Parameters:
        - stock_list: List of dicts with 'symbol' and 'company_name'
        - collect_news: Whether to collect news for each stock
        - delay_seconds: Delay between stocks (for API rate limiting)
        
        Returns:
        - Dictionary mapping symbol to analysis results
        """
        all_results = {}
        
        print("\n" + "="*70)
        print(f"MULTI-STOCK PATTERN ANALYSIS")
        print(f"Analyzing {len(stock_list)} stocks")
        print("="*70)
        
        for idx, stock_info in enumerate(stock_list, 1):
            symbol = stock_info['symbol']
            company_name = stock_info['company_name']
            
            print(f"\n[Stock {idx}/{len(stock_list)}]")
            
            results = self.analyze_stock(
                symbol=symbol,
                company_name=company_name,
                collect_news=collect_news
            )
            
            all_results[symbol] = results
            
            # Rate limiting
            if idx < len(stock_list):
                print(f"\n⏳ Waiting {delay_seconds} seconds (API rate limit)...")
                time.sleep(delay_seconds)
        
        return all_results
    
    def generate_report(self, results: Dict, output_dir: str = ".") -> str:
        """
        Generate comprehensive analysis report with visualizations.
        
        Parameters:
        - results: Analysis results from analyze_stock()
        - output_dir: Directory to save output files
        
        Returns:
        - Path to the generated report
        """
        if not results.get('success'):
            print("❌ Cannot generate report - analysis failed")
            return None
        
        symbol = results['symbol']
        print(f"\n📝 Generating report for {symbol}...")
        
        # Save stroke features
        if 'stroke_features' in results:
            stroke_path = f"{output_dir}/{symbol}_strokes.csv"
            results['stroke_features'].to_csv(stroke_path, index=False)
            print(f"✓ Saved stroke features: {stroke_path}")
        
        # Save stock data with critical points
        if 'marked_data' in results:
            marked_path = f"{output_dir}/{symbol}_marked_data.csv"
            results['marked_data'].to_csv(marked_path, index=False)
            print(f"✓ Saved marked data: {marked_path}")
        
        # Save news data if available
        if 'news_data' in results:
            news_path = f"{output_dir}/{symbol}_news_data.csv"
            results['news_data'].to_csv(news_path, index=False)
            print(f"✓ Saved news data: {news_path}")
        
        # Save merged data if available
        if 'merged_data' in results:
            merged_path = f"{output_dir}/{symbol}_merged_data.csv"
            results['merged_data'].to_csv(merged_path, index=False)
            print(f"✓ Saved merged data: {merged_path}")
        
        # Generate visualization
        if 'stock_data' in results and 'strokes' in results:
            viz_path = f"{output_dir}/{symbol}_pattern_analysis.png"
            self.pattern_recognizer.visualize_strokes(
                results['stock_data'],
                results['strokes'],
                symbol=symbol,
                save_path=viz_path
            )
        
        # Create summary report
        summary_path = f"{output_dir}/{symbol}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"STOCK PATTERN ANALYSIS REPORT\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Company: {results['company_name']}\n")
            f.write(f"Analysis Date: {results['timestamp']}\n\n")
            
            f.write(f"DATA SUMMARY\n")
            f.write(f"{'-'*70}\n")
            f.write(f"Stock data points: {len(results['stock_data'])}\n")
            if 'news_data' in results:
                f.write(f"News articles: {len(results['news_data'])}\n")
            f.write(f"Critical points: {len(results['critical_points'])}\n")
            f.write(f"Strokes extracted: {len(results['strokes'])}\n\n")
            
            f.write(f"PATTERN DETECTION\n")
            f.write(f"{'-'*70}\n")
            if results['patterns']:
                for pattern in results['patterns']:
                    f.write(f"- {pattern['pattern_type']} "
                           f"(confidence: {pattern['confidence']:.1%})\n")
            else:
                f.write("No standard patterns detected\n")
            
            f.write(f"\n{'='*70}\n")
        
        print(f"✓ Saved summary report: {summary_path}")
        
        return summary_path


def main():
    """
    Main execution function - runs the integrated pipeline.
    """
    # API Configuration
    TWELVE_DATA_API_KEY = "ac3aa4f4061b4c4f91a9e3636dbde84a"
    NEWSAPI_KEY = "ae825b1e-233a-4559-a551-9d59adffc00f"
    
    # Stocks to analyze
    stocks = [
        {"symbol": "AAPL", "company_name": "Apple Inc"},
        {"symbol": "MSFT", "company_name": "Microsoft"},
        {"symbol": "GOOGL", "company_name": "Google"},
        {"symbol": "NVDA", "company_name": "NVIDIA"},
        {"symbol": "TSLA", "company_name": "Tesla"}
    ]
    
    print("\n" + "="*70)
    print("INTEGRATED STOCK PATTERN RECOGNITION SYSTEM")
    print("="*70)
    print("\nThis system:")
    print("  1. Collects stock price and news data")
    print("  2. Identifies critical points (peaks, troughs, reversals)")
    print("  3. Extracts 'stroke' sequences between critical points")
    print("  4. Classifies chart patterns")
    print("  5. Generates comprehensive reports")
    print("="*70)
    
    # Initialize analyzer
    analyzer = IntegratedStockAnalyzer(TWELVE_DATA_API_KEY, NEWSAPI_KEY)
    
    # Analyze all stocks
    all_results = analyzer.analyze_multiple_stocks(
        stock_list=stocks,
        collect_news=True,
        delay_seconds=8  # Respect 8 requests/minute rate limit
    )
    
    # Generate reports
    print("\n" + "="*70)
    print("GENERATING REPORTS")
    print("="*70)
    
    for symbol, results in all_results.items():
        if results.get('success'):
            analyzer.generate_report(results)
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(stocks)} stocks:")
    for symbol, results in all_results.items():
        status = "✓" if results.get('success') else "✗"
        print(f"  {status} {symbol}")
        if results.get('patterns'):
            print(f"      Patterns: {len(results['patterns'])} detected")
    
    print("\n📁 Output files generated in current directory")
    print("   - *_strokes.csv: Stroke features for each stock")
    print("   - *_marked_data.csv: Price data with critical points marked")
    print("   - *_pattern_analysis.png: Visual analysis charts")
    print("   - *_summary.txt: Text summary reports")


if __name__ == "__main__":
    main()
