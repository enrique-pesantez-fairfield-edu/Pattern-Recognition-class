"""
Stock Data and News Collection System
Integrates Twelve Data API for stock information and NewsAPI for related news
Formats data as time series sequences for analysis
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import time


class StockDataCollector:
    """Collects stock market data from Twelve Data API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        
    def get_time_series(self, symbol: str, interval: str = "1day", 
                       outputsize: int = 30, timezone: str = "America/New_York") -> pd.DataFrame:
        """
        Fetch time series stock data
        
        Parameters:
        - symbol: Stock ticker (e.g., 'AAPL', 'GOOGL')
        - interval: Time interval (1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week, 1month)
        - outputsize: Number of data points to return (default 30, max 5000)
        - timezone: Timezone for the data
        """
        endpoint = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "timezone": timezone,
            "apikey": self.api_key
        }
        
        response = requests.get(endpoint, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if "values" in data:
                df = pd.DataFrame(data["values"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime")
                
                # Convert numeric columns
                numeric_cols = ["open", "high", "low", "close", "volume"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                
                return df
            else:
                print(f"Error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
        else:
            print(f"HTTP Error: {response.status_code}")
            return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a stock"""
        endpoint = f"{self.base_url}/quote"
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        response = requests.get(endpoint, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    
    def get_multiple_stocks(self, symbols: List[str], interval: str = "1day", 
                           outputsize: int = 30) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        stock_data = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            df = self.get_time_series(symbol, interval, outputsize)
            if not df.empty:
                stock_data[symbol] = df
            time.sleep(8)  # Rate limiting: 8 requests per minute
            
        return stock_data


class NewsCollector:
    """Collects news articles from NewsAPI.ai"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.ai/api/v1"
        
    def search_articles(self, keyword: str, max_items: int = 50, 
                       lang: str = "eng", date_start: Optional[str] = None,
                       date_end: Optional[str] = None) -> pd.DataFrame:
        """
        Search for news articles related to a keyword (e.g., stock symbol or company name)
        
        Parameters:
        - keyword: Search term (stock ticker or company name)
        - max_items: Maximum number of articles to return
        - lang: Language code (eng for English)
        - date_start: Start date in YYYY-MM-DD format
        - date_end: End date in YYYY-MM-DD format
        """
        endpoint = f"{self.base_url}/article/getArticles"
        
        # Build query
        query = {
            "$query": {
                "$and": [
                    {"keyword": keyword, "lang": lang}
                ]
            },
            "resultType": "articles",
            "articlesSortBy": "date",
            "articlesCount": max_items,
            "apiKey": self.api_key
        }
        
        if date_start:
            query["dateStart"] = date_start
        if date_end:
            query["dateEnd"] = date_end
        
        response = requests.post(endpoint, json=query)
        
        if response.status_code == 200:
            data = response.json()
            
            if "articles" in data and "results" in data["articles"]:
                articles = data["articles"]["results"]
                
                # Extract relevant fields
                articles_list = []
                for article in articles:
                    articles_list.append({
                        "date": article.get("date"),
                        "datetime": article.get("dateTime"),
                        "title": article.get("title"),
                        "body": article.get("body"),
                        "url": article.get("url"),
                        "source": article.get("source", {}).get("title"),
                        "sentiment": article.get("sentiment"),  # If available
                        "relevance": article.get("relevance")  # If available
                    })
                
                df = pd.DataFrame(articles_list)
                if not df.empty and "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.sort_values("datetime")
                
                return df
            else:
                print(f"No articles found or error: {data}")
                return pd.DataFrame()
        else:
            print(f"HTTP Error: {response.status_code}")
            return pd.DataFrame()
    
    def search_stock_news(self, symbol: str, company_name: str, max_items: int = 50,
                         date_start: Optional[str] = None, date_end: Optional[str] = None) -> pd.DataFrame:
        """
        Search for stock-specific news with better filtering
        
        Parameters:
        - symbol: Stock ticker (e.g., 'AAPL')
        - company_name: Full company name (e.g., 'Apple Inc')
        - max_items: Maximum number of articles to return
        - date_start: Start date in YYYY-MM-DD format
        - date_end: End date in YYYY-MM-DD format
        """
        # Fetch maximum allowed articles (100) to have enough for filtering
        fetch_count = min(100, max_items * 3)
        df = self.search_articles(company_name, fetch_count, date_start=date_start, date_end=date_end)
        
        if df.empty:
            return df
        
        # Stock-related keywords that should be present
        stock_keywords = ['stock', 'shares', 'trading', 'market', 'investor', 'earnings', 
                         'revenue', 'profit', 'quarter', 'Q1', 'Q2', 'Q3', 'Q4', 
                         'CEO', 'CFO', symbol.upper(), 'wall street', 'nasdaq', 'price',
                         'guidance', 'forecast', 'analysts', 'equity', 'dividend',
                         'financial', 'results', 'report']
        
        # Topics to exclude (not stock-related)
        exclude_keywords = ['football', 'soccer', 'basketball', 'baseball', 'sports',
                           'tennis', 'golf', 'hockey', 'tickets', 'game', 'poll',
                           'tourism', 'travel', 'vacation', 'hotel', 'resort',
                           'restaurant', 'food', 'recipe', 'concert', 'music',
                           'movie', 'film', 'tv show', 'streaming', 'parliament',
                           'election', 'vote', 'political', 'military', 'war',
                           'hurricane', 'weather', 'climate', 'symphony', 'orchestra']
        
        def is_stock_related(title, body):
            """Check if article is genuinely stock-related"""
            if pd.isna(title):
                return False
            
            title_lower = str(title).lower()
            body_lower = str(body).lower() if not pd.isna(body) else ""
            combined_text = title_lower + " " + body_lower
            
            # Check for exclusion keywords - if found, reject article
            for exclude in exclude_keywords:
                if exclude.lower() in title_lower:
                    return False
            
            # Must contain company name or symbol in title
            company_words = company_name.lower().split()
            has_company = any(word in title_lower for word in company_words) or symbol.lower() in title_lower
            
            if not has_company:
                return False
            
            # Must contain at least one stock-related keyword
            has_stock_keyword = any(keyword.lower() in combined_text for keyword in stock_keywords)
            
            return has_stock_keyword
        
        # Filter articles
        df['is_relevant'] = df.apply(lambda row: is_stock_related(row['title'], row['body']), axis=1)
        filtered_df = df[df['is_relevant']].copy()
        filtered_df = filtered_df.drop('is_relevant', axis=1)
        
        print(f"Filtered from {len(df)} to {len(filtered_df)} stock-relevant articles")
        
        # Return top max_items results
        return filtered_df.head(max_items)


class TimeSeriesFormatter:
    """Format collected data as sequences for time series analysis"""
    
    @staticmethod
    def create_sequences(data: pd.DataFrame, sequence_length: int = 10,
                        target_column: str = "close") -> tuple:
        """
        Create sequences for time series analysis
        
        Parameters:
        - data: DataFrame with time series data
        - sequence_length: Number of time steps in each sequence
        - target_column: Column to predict
        
        Returns:
        - X: Input sequences (features)
        - y: Target values
        """
        if data.empty or target_column not in data.columns:
            return np.array([]), np.array([])
        
        values = data[target_column].values
        X, y = [], []
        
        for i in range(len(values) - sequence_length):
            X.append(values[i:i + sequence_length])
            y.append(values[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def normalize_data(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize specified columns to 0-1 range"""
        normalized_data = data.copy()
        
        for col in columns:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val - min_val != 0:
                    normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
        
        return normalized_data
    
    @staticmethod
    def merge_stock_news(stock_df: pd.DataFrame, news_df: pd.DataFrame,
                        time_window: str = "1D") -> pd.DataFrame:
        """
        Merge stock data with news data based on time proximity
        
        Parameters:
        - stock_df: DataFrame with stock data
        - news_df: DataFrame with news data
        - time_window: Time window for matching news to stock data
        """
        if stock_df.empty or news_df.empty:
            return stock_df
        
        # Ensure datetime columns exist
        if "datetime" not in stock_df.columns or "datetime" not in news_df.columns:
            return stock_df
        
        # Create copies to avoid modifying original dataframes
        stock_df = stock_df.copy()
        news_df = news_df.copy()
        
        # Normalize timezones - convert both to UTC timezone-aware
        if stock_df["datetime"].dt.tz is None:
            stock_df["datetime"] = pd.to_datetime(stock_df["datetime"]).dt.tz_localize('UTC')
        else:
            stock_df["datetime"] = pd.to_datetime(stock_df["datetime"]).dt.tz_convert('UTC')
        
        if news_df["datetime"].dt.tz is None:
            news_df["datetime"] = pd.to_datetime(news_df["datetime"]).dt.tz_localize('UTC')
        else:
            news_df["datetime"] = pd.to_datetime(news_df["datetime"]).dt.tz_convert('UTC')
        
        merged_df = stock_df.copy()
        merged_df["news_count"] = 0
        merged_df["news_titles"] = ""
        
        for idx, row in stock_df.iterrows():
            stock_time = row["datetime"]
            
            # Find news within time window
            time_mask = (news_df["datetime"] >= stock_time - pd.Timedelta(time_window)) & \
                       (news_df["datetime"] <= stock_time + pd.Timedelta(time_window))
            
            relevant_news = news_df[time_mask]
            
            merged_df.at[idx, "news_count"] = len(relevant_news)
            if not relevant_news.empty:
                merged_df.at[idx, "news_titles"] = " | ".join(relevant_news["title"].head(3).tolist())
        
        return merged_df


def main():
    """Example usage of the data collection system
    
    Note: API Rate Limits
    - Twelve Data: 8 requests per minute on free tier
    - NewsAPI.ai: varies by plan
    - Script includes 8-second delays between stocks to stay within limits
    - Processing 10 stocks will take approximately 80 seconds (1.3 minutes)
    """
    
    # API Keys (replace with your actual keys)
    TWELVE_DATA_API_KEY = "ac3aa4f4061b4c4f91a9e3636dbde84a"
    NEWSAPI_KEY = "ae825b1e-233a-4559-a551-9d59adffc00f"
    
    # Initialize collectors
    stock_collector = StockDataCollector(TWELVE_DATA_API_KEY)
    news_collector = NewsCollector(NEWSAPI_KEY)
    formatter = TimeSeriesFormatter()
    
    # Define multiple major stocks to analyze
    stocks = [
        {"symbol": "AAPL", "company_name": "Apple Inc"},
        {"symbol": "MSFT", "company_name": "Microsoft"},
        {"symbol": "GOOGL", "company_name": "Google"},
        {"symbol": "AMZN", "company_name": "Amazon"},
        {"symbol": "NVDA", "company_name": "NVIDIA"},
        {"symbol": "META", "company_name": "Meta"},
        {"symbol": "TSLA", "company_name": "Tesla"},
        {"symbol": "JPM", "company_name": "JPMorgan Chase"},
        {"symbol": "V", "company_name": "Visa"},
        {"symbol": "WMT", "company_name": "Walmart"}
    ]
    
    print(f"\n{'='*50}")
    print(f"Collecting data for {len(stocks)} major stocks")
    print(f"{'='*50}\n")
    
    # Calculate date range (last 30 days)
    date_end = datetime.now().strftime("%Y-%m-%d")
    date_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    all_stock_data = {}
    all_news_data = {}
    all_merged_data = {}
    
    # Loop through each stock
    for idx, stock_info in enumerate(stocks, 1):
        symbol = stock_info["symbol"]
        company_name = stock_info["company_name"]
        
        print(f"\n{'='*60}")
        print(f"Processing {idx}/{len(stocks)}: {symbol} ({company_name})")
        print(f"{'='*60}")
        
        # Get stock data
        print(f"Fetching stock data for {symbol}...")
        stock_data = stock_collector.get_time_series(
            symbol=symbol,
            interval="1day",
            outputsize=60  # Last 60 days
        )
        
        if not stock_data.empty:
            print(f"✓ Retrieved {len(stock_data)} stock data points")
            all_stock_data[symbol] = stock_data
            
            # Get news data
            print(f"Fetching news articles for {company_name}...")
            
            # Use the stock-specific search method
            news_data = news_collector.search_stock_news(
                symbol=symbol,
                company_name=company_name,
                max_items=50,
                date_start=date_start,
                date_end=date_end
            )
            
            if not news_data.empty:
                print(f"✓ Retrieved {len(news_data)} news articles")
                all_news_data[symbol] = news_data
                
                # Merge stock and news data
                print(f"Merging stock and news data for {symbol}...")
                merged_data = formatter.merge_stock_news(stock_data, news_data)
                all_merged_data[symbol] = merged_data
                
                # Create sequences for time series analysis
                X, y = formatter.create_sequences(stock_data, sequence_length=10, target_column="close")
                print(f"✓ Created {len(X)} time series sequences")
                
                # Save data
                stock_data.to_csv(f"{symbol}_stock_data.csv", index=False)
                news_data.to_csv(f"{symbol}_news_data.csv", index=False)
                merged_data.to_csv(f"{symbol}_merged_data.csv", index=False)
                print(f"✓ Saved CSV files for {symbol}")
            else:
                print(f"⚠ No news data retrieved for {symbol}")
                all_stock_data[symbol] = stock_data
                stock_data.to_csv(f"{symbol}_stock_data.csv", index=False)
        else:
            print(f"✗ No stock data retrieved for {symbol}")
        
        
        # Rate limiting: 8 requests per minute = one request every 7.5 seconds
        # Adding buffer for safety
        if idx < len(stocks):  # Don't wait after the last stock
            print(f"\n⏳ Waiting 8 seconds for API rate limit (8 requests/minute)...")
            time.sleep(8)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DATA COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total stocks processed: {len(stocks)}")
    print(f"Successfully retrieved stock data: {len(all_stock_data)}")
    print(f"Successfully retrieved news data: {len(all_news_data)}")
    print(f"Successfully merged datasets: {len(all_merged_data)}")
    print(f"\nFiles saved in current directory:")
    for symbol in all_stock_data.keys():
        print(f"  - {symbol}_stock_data.csv")
        if symbol in all_news_data:
            print(f"  - {symbol}_news_data.csv")
            print(f"  - {symbol}_merged_data.csv")
    print(f"\n✓ Data collection complete!")



if __name__ == "__main__":
    main()