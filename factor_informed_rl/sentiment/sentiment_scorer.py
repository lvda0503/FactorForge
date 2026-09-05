"""
Sentiment Scorer — Chinese financial BERT + SnowNLP fallback.

Model:
  - Primary: bardsai/finance-sentiment-zh-base (97.3% accuracy, Chinese financial news)
  - Fallback: SnowNLP (fast, <1MB, no GPU needed)

Scoring:
  - positive → +1, negative → -1, neutral → 0
  - Daily stock sentiment = mean(article scores)
"""
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ── Lazy load ──
_zh_finance_pipeline = None


def _load_model():
    """Load Chinese financial sentiment model (first call only)."""
    global _zh_finance_pipeline
    if _zh_finance_pipeline is not None:
        return True
    try:
        from transformers import pipeline
        _zh_finance_pipeline = pipeline(
            "text-classification",
            model="bardsai/finance-sentiment-zh-base",
            max_length=512,
            truncation=True
        )
        print("[Sentiment] Loaded bardsai/finance-sentiment-zh-base (433MB)")
        return True
    except Exception as e:
        print(f"[Sentiment] Model load failed: {e}, using SnowNLP fallback")
        return False


def _score_snownlp(text: str) -> float:
    """Chinese sentiment via SnowNLP. Map [0,1] → [-1,+1]."""
    try:
        from snownlp import SnowNLP
        return (SnowNLP(text).sentiments - 0.5) * 2
    except Exception:
        return 0.0


def score_text(text: str) -> float:
    """
    Score a single news text. Returns score in [-1, +1].
    +1 = positive, -1 = negative, 0 = neutral.
    """
    if _load_model():
        try:
            result = _zh_finance_pipeline(text[:2000])[0]
            label = result['label'].lower()
            score = result['score']
            if 'positive' in label:
                return score
            elif 'negative' in label:
                return -score
            else:
                return 0.0  # neutral
        except Exception:
            pass
    return _score_snownlp(text)


def score_stock_news(news_texts: List[str]) -> Optional[float]:
    """
    Aggregate sentiment for a single stock from multiple news articles.

    Args:
        news_texts: list of news article texts (title + content)

    Returns:
        mean sentiment score in [-1, +1], or None if no valid scores
    """
    if not news_texts:
        return None

    scores = []
    for text in news_texts:
        s = score_text(text)
        scores.append(s)

    if not scores:
        return None

    # Weighted: more extreme scores get higher weight (surprise factor)
    arr = np.array(scores)
    weights = np.abs(arr) + 0.5  # Base weight 0.5 + sentiment magnitude
    weighted = np.average(arr, weights=weights)
    return float(np.clip(weighted, -1.0, 1.0))


def score_all_stocks(news_dict: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Score all stocks from news dictionary.

    Input: {stock_code: [news_text, ...], '_MARKET': [market_news, ...]}
    Output: {stock_code: sentiment_score}

    Each stock's score = 0.7 * stock_news_sentiment + 0.3 * market_sentiment
    """
    market_news = news_dict.get('_MARKET', [])
    market_score = score_stock_news(market_news) if market_news else None

    result = {}
    for code, texts in news_dict.items():
        if code == '_MARKET':
            continue
        stock_score = score_stock_news(texts)
        if stock_score is None and market_score is None:
            continue

        # Blend
        if stock_score is not None and market_score is not None:
            final = 0.7 * stock_score + 0.3 * market_score
        elif stock_score is not None:
            final = stock_score
        else:
            final = market_score

        result[code] = round(final, 4)

    return result


if __name__ == '__main__':
    # Quick test
    test = {"000858": ["五粮液业绩大幅增长超出市场预期",
                       "白酒板块整体回调行业面临压力"],
            "600519": ["贵州茅台提价预期强烈机构看好"]}
    results = score_all_stocks(test)
    for code, score in results.items():
        print(f"  {code}: {score:+.4f}")
