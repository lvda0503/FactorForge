"""
Daily Sentiment Factor Generator
=================================
Run once per day to generate sentiment scores for all CSI300 stocks.

Usage:
  cd D:\JoinQuant\quant_env
  python -m factor_informed_rl.sentiment.run_daily

Output:
  D:\data\sentiment_factor.csv     — latest daily sentiment scores
  D:\data\sentiment_history.csv    — accumulated history (append mode)

Integration:
  The sentiment score becomes an extra factor in the stock selection pipeline.
  Add "sentiment_Nd" (N=1,5,20) to factor lists in config.py.
"""
import sys, os, json, time
sys.path.insert(0, r'D:\JoinQuant\quant_env')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from factor_informed_rl.sentiment.news_fetcher import fetch_daily_news
from factor_informed_rl.sentiment.sentiment_scorer import score_all_stocks

OUTPUT = r"D:\data\sentiment_factor_v2.csv"
HISTORY = r"D:\data\sentiment_history_v2.csv"
CSI300_CACHE = r"D:\JoinQuant\quant_env\data_cache\csi300"


def get_csi300_codes() -> list:
    """Get list of CSI300 stock codes from local cache."""
    codes = []
    if os.path.exists(CSI300_CACHE):
        for f in os.listdir(CSI300_CACHE):
            if f.endswith('.pkl'):
                codes.append(f.replace('.pkl', ''))
    return codes


def main():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"[Sentiment] Daily run for {today}")

    # Step 1: Get stock codes (limit to 30 for speed, or all 300)
    all_codes = get_csi300_codes()
    # Start with a subset for daily testing; expand to full CSI300 later
    sample_codes = all_codes[:50]  # First 50 stocks (top by market cap)
    print(f"[Sentiment] Processing {len(sample_codes)} stocks (of {len(all_codes)} total)")

    # Step 2: Fetch news
    print("[Sentiment] Fetching news...")
    t0 = time.time()
    news_dict = fetch_daily_news(sample_codes, days=1)
    n_articles = sum(len(v) for v in news_dict.values())
    print(f"[Sentiment] Fetched {n_articles} articles in {time.time()-t0:.0f}s")

    # Step 3: Score sentiment
    print("[Sentiment] Scoring sentiment...")
    t0 = time.time()
    scores = score_all_stocks(news_dict)
    print(f"[Sentiment] Scored {len(scores)} stocks in {time.time()-t0:.0f}s")

    if not scores:
        print("[Sentiment] WARNING: no scores generated. Using fallback (zeros).")
        scores = {c: 0.0 for c in sample_codes}

    # Step 4: Save
    df = pd.DataFrame([
        {'date': today, 'stock': code, 'sentiment': score}
        for code, score in scores.items()
    ])
    df.to_csv(OUTPUT, index=False)
    print(f"[Sentiment] Saved to {OUTPUT}")

    # Append to history
    if os.path.exists(HISTORY):
        hist = pd.read_csv(HISTORY)
        hist = pd.concat([hist, df], ignore_index=True)
    else:
        hist = df
    hist.to_csv(HISTORY, index=False)
    print(f"[Sentiment] History: {len(hist)} records")

    # Summary
    s = df['sentiment']
    print(f"[Sentiment] Summary: mean={s.mean():+.4f} std={s.std():.4f} "
          f"pos={int(100*(s>0).mean())}% stocks positive")
    print("[Sentiment] Done.")


if __name__ == '__main__':
    main()
