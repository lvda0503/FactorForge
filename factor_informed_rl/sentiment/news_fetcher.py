"""
News Fetcher — pull daily financial news from multiple free sources.

Sources:
  - AKShare (primary): stock_news_em — 东方财富个股新闻
  - AKShare: stock_news_global_em — 全球财经新闻
  - Sina Finance (fallback): direct HTTP parse

Output: dict[date][stock_code] = list of news text
"""
import akshare as ak
import pandas as pd
import time
import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def fetch_stock_news_akshare(code: str, days: int = 3) -> pd.DataFrame:
    """
    Fetch news for a single stock via AKShare (东方财富).
    Returns DataFrame with columns: title, content, date, source
    """
    try:
        df = ak.stock_news_em(symbol=code)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        # Column positions (avoid Chinese name encoding issues):
        # [0]=code, [1]=title, [2]=content, [3]=datetime, [4]=source, [5]=url
        result = pd.DataFrame({
            'title': df.iloc[:, 1],
            'content': df.iloc[:, 2],
            'date': pd.to_datetime(df.iloc[:, 3]).dt.date
        })
        cutoff = datetime.now().date() - timedelta(days=days)
        return result[result['date'] >= cutoff]
    except Exception as e:
        print(f"  AKShare news fetch failed for {code}: {e}")
        return pd.DataFrame()


def fetch_market_news_akshare(days: int = 1) -> pd.DataFrame:
    """
    Fetch general market/financial news (财新头条).
    """
    try:
        df = ak.stock_news_main_cx()
        if df is None or len(df) == 0:
            return pd.DataFrame()
        # Cols: [0]=source, [1]=title, [2]=url (no date column)
        result = pd.DataFrame({
            'title': df.iloc[:, 1],
            'content': '',
            'date': datetime.now().date()
        })
        return result
    except Exception as e:
        print(f"  Market news fetch failed: {e}")
        return pd.DataFrame()


def fetch_sina_news(code: str, days: int = 3) -> List[Dict]:
    """
    Fallback: fetch news from Sina Finance.
    """
    # Convert code to Sina format
    if code.startswith('6'):
        sina_code = 'sh' + code
    else:
        sina_code = 'sz' + code

    news_list = []
    try:
        # Sina news API (unofficial, may break)
        url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{sina_code}.phtml"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'gb2312'
        # Simple regex parse (Sina news page is HTML)
        pattern = r'<a href="(.*?)".*?target="_blank">(.*?)</a>.*?<span>(\\d{4}-\\d{2}-\\d{2})</span>'
        matches = re.findall(pattern, resp.text)
        cutoff = datetime.now().date() - timedelta(days=days)
        for href, title, date_str in matches:
            try:
                d = datetime.strptime(date_str, '%Y-%m-%d').date()
            except:
                continue
            if d >= cutoff:
                news_list.append({'title': title.strip(), 'content': '', 'date': d})
    except Exception as e:
        print(f"  Sina news fetch failed for {code}: {e}")

    return news_list


def fetch_daily_news(stock_codes: List[str], days: int = 1) -> Dict[str, List[str]]:
    """
    Main entry: fetch news for all stocks, return {stock_code: [news_text]}.
    Each news_text = title + content (concatenated for FinBERT).
    """
    result = {}
    today = datetime.now().date()

    # 1. AKShare per-stock news
    for code in stock_codes:
        df = fetch_stock_news_akshare(code, days=days)
        texts = []
        for _, row in df.iterrows():
            text = str(row.get('title', '')) + ' ' + str(row.get('content', ''))
            if len(text) > 20:  # Skip empty/short
                texts.append(text)
        if texts:
            result[code] = texts
        time.sleep(0.3)  # Rate limit

    # 2. Market-level news (shared across all stocks)
    df_market = fetch_market_news_akshare(days=days)
    market_texts = []
    for _, row in df_market.iterrows():
        text = str(row.get('title', '')) + ' ' + str(row.get('content', ''))
        if len(text) > 20:
            market_texts.append(text)

    if market_texts:
        result['_MARKET'] = market_texts

    print(f"[NewsFetcher] Fetched news for {len(result)} stocks + market")
    return result


if __name__ == '__main__':
    # Quick test
    test_codes = ['000858', '600519']
    news = fetch_daily_news(test_codes, days=3)
    for code, texts in news.items():
        print(f"\n{code}: {len(texts)} articles")
        for t in texts[:2]:
            print(f"  - {t[:100]}...")
