#!/usr/bin/env python
"""
build_watchlist_with_llm.py

End-to-end pipeline for the DSC 670 project:

1. Read Chance_Watchlist.csv
2. Normalize symbols and pull OHLCV from Yahoo Finance
3. Compute technical metrics (RSI, MACD, Bollinger, 52w hi/lo, support/resistance, TD Sequential)
4. Build tag string + rule-based local_score
5. (Optional) Call fine-tuned OpenAI stance model (using m3_artifacts/inference_config.json)
   to get llm_stance, llm_score, llm_summary
6. Blend local_score + llm_score into final_score + final_decision
7. Save final CSV for use in Streamlit

Usage:

    python build_watchlist_with_llm.py --with-llm

User can omit --with-llm if only want the technical metrics.
"""

import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import os

import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

WATCHLIST_CSV_DEFAULT = "Chance_Watchlist.csv"
TICKER_COL_DEFAULT = "Symbol"  # column in watchlist CSV
LOOKBACK_DAYS_DEFAULT = 420    # ~20 months of daily bars
INTERVAL_DEFAULT = "1d"
BATCH_SIZE_DEFAULT = 25        # yfinance batch size

# Default location of the milestone 3 inference config
INFERENCE_CONFIG_DEFAULT = "m3_artifacts/inference_config.json"

EXCHANGE_SUFFIXES = {
    "O", "N", "K", "AS", "L", "T", "KS", "KQ", "SW",
    "PA", "DE", "MI", "BR", "TO", "V"
}

# ---------------------------------------------------------------------
# Helper to load OPENAI_API_KEY from .env if needed
# ---------------------------------------------------------------------

def load_env_api_key(
    dotenv_path: str = ".env",
    key: str = "OPENAI_API_KEY",
) -> None:
    """
    Load OPENAI_API_KEY from a .env file in the working directory if it's not
    already set in the environment.

    Expected format in .env:
        OPENAI_API_KEY=sk-...

    Lines starting with # or blank lines are ignored.
    """
    # If it's already set (e.g., from the shell), do nothing
    if os.getenv(key):
        return

    path = Path(dotenv_path)
    if not path.exists():
        print(f"[WARN] .env file not found at {path.resolve()}, expecting {key} in environment.")
        return

    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                # Strip quotes if present
                os.environ[key] = v.strip().strip('"').strip("'")
                return

        print(f"[WARN] {key} not found in {dotenv_path}.")
    except Exception as e:
        print(f"[WARN] Failed to read {dotenv_path}: {e}")

# ---------------------------------------------------------------------
# Symbol normalization + watchlist loading
# ---------------------------------------------------------------------

def normalize_symbol_for_yahoo(sym: str) -> str:
    """
    Normalize symbols coming from the watchlist into something
    Yahoo Finance will understand (e.g., strip exchange codes, fix BRK.B).
    """
    s = str(sym).strip().upper()
    if not s:
        return ""

    # If the symbol contains a dot, decide whether it's an exchange code or share-class suffix
    if "." in s:
        base, suffix = s.rsplit(".", 1)
        # Suffix looks like exchange code -> drop it
        if suffix in EXCHANGE_SUFFIXES:
            s = base
        # Single-letter suffix for US share classes, convert to dash
        elif len(suffix) == 1 and suffix.isalpha():
            s = f"{base}-{suffix}"
        else:
            # Fallback: drop suffix
            s = base

    # Special fixes
    if s == "BRK.B":
        return "BRK-B"
    if s == "BRK.A":
        return "BRK-A"

    return s


def load_watchlist(csv_path: str, ticker_col: str | None = None) -> pd.DataFrame:
    """
    Read the watchlist CSV and return a DataFrame with:
        symbol_original, symbol, and (optionally) Name

    - symbol_original: exactly as in the CSV
    - symbol: normalized for Yahoo Finance
    - Name: company name (if a suitable column exists)
    """
    df = pd.read_csv(csv_path)

    # Determine ticker column
    if ticker_col and ticker_col in df.columns:
        col = ticker_col
    else:
        # Heuristic: choose the first column that "looks like" a ticker
        candidates = [
            c for c in df.columns
            if df[c].astype(str).str.match(r"^[A-Za-z0-9.\-]+$").mean() > 0.5
        ]
        col = candidates[0] if candidates else df.columns[0]

    # Try to find a name-like column
    name_candidates = [
        c for c in df.columns
        if c.lower() in ("name", "company", "company name", "company_name")
    ]
    keep_cols = [col]
    if name_candidates:
        # Keep the first plausible name column; we preserve its original name
        keep_cols.append(name_candidates[0])

    df = df[keep_cols].copy()
    df.rename(columns={col: "symbol_original"}, inplace=True)
    df["symbol"] = df["symbol_original"].astype(str).apply(normalize_symbol_for_yahoo)

    # Drop empties + duplicates
    df = (
        df[df["symbol"].str.len() > 0]
        .drop_duplicates(subset=["symbol"])
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------
# Technical indicators and helpers
# ---------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=close.index)
    dn = pd.Series(np.where(delta < 0, -delta, 0.0), index=close.index)
    roll_up = up.rolling(period).mean()
    roll_dn = dn.rolling(period).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    pctb = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / ma
    return ma, upper, lower, pctb, bandwidth


def fifty_two_week_metrics(close: pd.Series) -> Tuple[float, float]:
    last_252 = close.tail(252)
    return float(last_252.max()), float(last_252.min())


def local_extrema_support_resistance(
    close: pd.Series,
    lookback: int = 180
) -> Tuple[float, float]:
    """
    Very simple local minima/maxima based support/resistance heuristic.
    """
    s = close.tail(lookback)
    if s.empty:
        return np.nan, np.nan

    last_px = s.iloc[-1]
    mins = s[(s.shift(1) > s) & (s.shift(-1) > s)]
    maxs = s[(s.shift(1) < s) & (s.shift(-1) < s)]

    support = mins[mins < last_px].max() if not mins.empty else np.nan
    resistance = maxs[maxs > last_px].min() if not maxs.empty else np.nan

    support_val = float(support) if pd.notna(support) else np.nan
    resistance_val = float(resistance) if pd.notna(resistance) else np.nan
    return support_val, resistance_val


def td_sequential_basic(df: pd.DataFrame) -> Dict[str, object]:
    """
    Minimal TD Sequential-style count:
      - Compare close vs close 4 bars prior.
      - Track 9-counts for BUY and SELL.
      - Mark "perfected" if last 2 lows (for BUY) or highs (for SELL)
        are more extreme than bars 6/7.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    n = len(df)

    buy_count = 0
    sell_count = 0
    buy_perfected = False
    sell_perfected = False
    last_buy9_idx = None
    last_sell9_idx = None

    # Full TD counts
    for i in range(n):
        if i >= 4:
            if close.iloc[i] < close.iloc[i - 4]:
                buy_count += 1
                sell_count = 0
            elif close.iloc[i] > close.iloc[i - 4]:
                sell_count += 1
                buy_count = 0
            else:
                buy_count = 0
                sell_count = 0

        if buy_count == 9:
            last_buy9_idx = i
            if i >= 8:
                l8, l9 = low.iloc[i - 1], low.iloc[i]
                l6, l7 = low.iloc[i - 3], low.iloc[i - 2]
                buy_perfected = (l8 <= min(l6, l7)) or (l9 <= min(l6, l7))
            buy_count = 0

        if sell_count == 9:
            last_sell9_idx = i
            if i >= 8:
                h8, h9 = high.iloc[i - 1], high.iloc[i]
                h6, h7 = high.iloc[i - 3], high.iloc[i - 2]
                sell_perfected = (h8 >= max(h6, h7)) or (h9 >= max(h6, h7))
            sell_count = 0

    # Current run (up to last ~8 bars)
    curr_buy = 0
    curr_sell = 0
    start_idx = max(4, n - 8)
    for i in range(start_idx, n):
        if close.iloc[i] < close.iloc[i - 4]:
            curr_buy += 1
            curr_sell = 0
        elif close.iloc[i] > close.iloc[i - 4]:
            curr_sell += 1
            curr_buy = 0
        else:
            curr_buy = 0
            curr_sell = 0

    if last_buy9_idx is not None and (
        last_sell9_idx is None or last_buy9_idx > last_sell9_idx
    ):
        last_signal = "BUY9"
    elif last_sell9_idx is not None and (
        last_buy9_idx is None or last_sell9_idx > last_buy9_idx
    ):
        last_signal = "SELL9"
    else:
        last_signal = "NONE"

    return {
        "last_signal": last_signal,
        "last_buy9_date": (
            str(close.index[last_buy9_idx].date()) if last_buy9_idx is not None else None
        ),
        "last_sell9_date": (
            str(close.index[last_sell9_idx].date()) if last_sell9_idx is not None else None
        ),
        "buy_perfected": bool(buy_perfected),
        "sell_perfected": bool(sell_perfected),
        "curr_buy_count": int(curr_buy),
        "curr_sell_count": int(curr_sell),
    }


# ---------------------------------------------------------------------
# Price download
# ---------------------------------------------------------------------

def fetch_prices_yf(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> Dict[str, pd.DataFrame]:
    """
    Download price history with yfinance in polite batches.
    Returns a dict: {symbol: OHLCV DataFrame}
    """
    data: Dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            df = yf.download(
                tickers=" ".join(batch),
                start=start,
                end=end,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"[WARN] batch {i}-{i+len(batch)} failed: {e}")
            df = pd.DataFrame()

        # Multi-ticker frame => MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            for t in batch:
                if t in df.columns.levels[0]:
                    sub = df[t].copy()
                    sub.columns = [c.capitalize() for c in sub.columns]
                    sub = sub.dropna(how="all")
                    if not sub.empty:
                        data[t] = sub
        else:
            # Single ticker fallback
            sub = df.copy()
            if not sub.empty:
                sub.columns = [c.capitalize() for c in sub.columns]
                data[batch[0]] = sub

        time.sleep(0.2)  # polite pause

    return data


# ---------------------------------------------------------------------
# Metrics / tags / local score
# ---------------------------------------------------------------------

def compute_metrics(price_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute per-ticker metrics:
        close, 1d % change, RSI, MACD, Bollinger stats,
        52w high/low, support, resistance, TD Sequential fields.
    """
    rows: list[dict] = []

    for t, df in price_dict.items():
        df = df.dropna()
        if df.empty or {"Open", "High", "Low", "Close"}.difference(df.columns):
            continue

        close = df["Close"]

        rsi_ser = rsi(close)
        macd_line, macd_signal, macd_hist = macd(close)
        bb_mid, bb_up, bb_lo, bb_pctb, bb_bw = bollinger(close)
        hi52, lo52 = fifty_two_week_metrics(close)
        support, resistance = local_extrema_support_resistance(close, lookback=180)
        td = td_sequential_basic(df)

        last = df.iloc[-1]
        rows.append(
            dict(
                symbol=t,
                date=str(df.index[-1].date()),
                close=float(last["Close"]),
                change_1d=float((close.pct_change().iloc[-1] or 0) * 100),
                rsi=float(rsi_ser.iloc[-1])
                if pd.notna(rsi_ser.iloc[-1])
                else np.nan,
                macd=float(macd_line.iloc[-1])
                if pd.notna(macd_line.iloc[-1])
                else np.nan,
                macd_signal=float(macd_signal.iloc[-1])
                if pd.notna(macd_signal.iloc[-1])
                else np.nan,
                macd_hist=float(macd_hist.iloc[-1])
                if pd.notna(macd_hist.iloc[-1])
                else np.nan,
                bb_mid=float(bb_mid.iloc[-1])
                if pd.notna(bb_mid.iloc[-1])
                else np.nan,
                bb_upper=float(bb_up.iloc[-1])
                if pd.notna(bb_up.iloc[-1])
                else np.nan,
                bb_lower=float(bb_lo.iloc[-1])
                if pd.notna(bb_lo.iloc[-1])
                else np.nan,
                bb_pctb=float(bb_pctb.iloc[-1])
                if pd.notna(bb_pctb.iloc[-1])
                else np.nan,
                bb_bandwidth=float(bb_bw.iloc[-1])
                if pd.notna(bb_bw.iloc[-1])
                else np.nan,
                hi_52w=float(hi52),
                lo_52w=float(lo52),
                support=float(support) if pd.notna(support) else np.nan,
                resistance=float(resistance) if pd.notna(resistance) else np.nan,
                td_last_signal=td["last_signal"],
                td_last_buy9_date=td["last_buy9_date"],
                td_last_sell9_date=td["last_sell9_date"],
                td_buy_perfected=td["buy_perfected"],
                td_sell_perfected=td["sell_perfected"],
                td_curr_buy_count=td["curr_buy_count"],
                td_curr_sell_count=td["curr_sell_count"],
            )
        )

    return pd.DataFrame(rows)


def build_tags(r: pd.Series) -> str:
    """
    Build a space-separated string of boolean-ish tags
    from the metrics row (RSI/MACD/Bollinger/52w/TD/etc.).
    """
    tags: list[str] = []

    # RSI zones
    if pd.notna(r.rsi):
        if r.rsi >= 70:
            tags.append("RSI_overbought")
        elif r.rsi <= 30:
            tags.append("RSI_oversold")
        else:
            tags.append("RSI_neutral")

    # MACD momentum
    if pd.notna(r.macd_hist) and pd.notna(r.macd) and pd.notna(r.macd_signal):
        if r.macd_hist > 0 and r.macd >= r.macd_signal:
            tags.append("MACD_bullish")
        elif r.macd_hist < 0 and r.macd <= r.macd_signal:
            tags.append("MACD_bearish")
        else:
            tags.append("MACD_mixed")

    # Bollinger position & width
    if pd.notna(r.bb_pctb):
        if r.bb_pctb >= 0.9:
            tags.append("near_upper_band")
        elif r.bb_pctb <= 0.1:
            tags.append("near_lower_band")
        else:
            tags.append("inside_bands")
    if pd.notna(r.bb_bandwidth):
        if r.bb_bandwidth <= 0.05:
            tags.append("bollinger_squeeze")
        elif r.bb_bandwidth >= 0.25:
            tags.append("high_volatility_bandwidth")

    # 52w placement
    if pd.notna(r.hi_52w) and pd.notna(r.lo_52w) and pd.notna(r.close):
        rng = r.hi_52w - r.lo_52w
        pos = (r.close - r.lo_52w) / rng if rng > 0 else np.nan
        if pd.notna(pos):
            if pos >= 0.95:
                tags.append("near_52w_high")
            elif pos <= 0.05:
                tags.append("near_52w_low")

    # Support / Resistance proximity
    if pd.notna(r.support) and r.support > 0:
        if 0 < (r.close - r.support) / r.close <= 0.02:
            tags.append("on_support")
    if pd.notna(r.resistance) and r.resistance > 0:
        if 0 < (r.resistance - r.close) / r.close <= 0.02:
            tags.append("near_resistance")

    # TD Sequential
    if r.td_last_signal == "BUY9":
        tags.append("TD_recent_buy9")
    if r.td_last_signal == "SELL9":
        tags.append("TD_recent_sell9")
    if bool(r.td_buy_perfected):
        tags.append("TD_buy_perfected")
    if bool(r.td_sell_perfected):
        tags.append("TD_sell_perfected")
    if r.td_curr_buy_count >= 5:
        tags.append(f"TD_buy_count_{int(r.td_curr_buy_count)}")
    if r.td_curr_sell_count >= 5:
        tags.append(f"TD_sell_count_{int(r.td_curr_sell_count)}")

    return " ".join(tags)


def local_score_from_tags(tags: str) -> int:
    """
    Turn tags into a simple integer local_score, roughly matching
    the notebook heuristic (positive = bullish tilt, negative = bearish).
    """
    s = 0
    if not isinstance(tags, str):
        return s

    parts = set(tags.split())

    # MACD
    if "MACD_bullish" in parts:
        s += 2
    elif "MACD_bearish" in parts:
        s -= 2

    # RSI extremes
    if "RSI_oversold" in parts:
        s += 1
    elif "RSI_overbought" in parts:
        s -= 1

    # Bands
    if "near_lower_band" in parts:
        s += 1
    elif "near_upper_band" in parts:
        s -= 1

    # 52w placement
    if "near_52w_low" in parts:
        s += 1
    elif "near_52w_high" in parts:
        s -= 1

    # Support / resistance
    if "on_support" in parts:
        s += 1
    if "near_resistance" in parts:
        s -= 1

    # TD Sequential
    if "TD_recent_buy9" in parts:
        s += 2
    if "TD_recent_sell9" in parts:
        s -= 2
    if "TD_buy_perfected" in parts:
        s += 1
    if "TD_sell_perfected" in parts:
        s -= 1

    return int(s)


# ---------------------------------------------------------------------
# Fine-tuned model: config, prompt, parsing, blending
# ---------------------------------------------------------------------

def load_inference_config(cfg_path: str | Path) -> dict:
    cfg_path = Path(cfg_path)
    assert cfg_path.exists(), f"Inference config not found: {cfg_path}"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def format_features_for_prompt(row: pd.Series, cfg: dict) -> str:
    """
    Turn a metrics row into the string the fine-tuned model expects.
    Uses the feature_cols from inference_config.json.

    Header example:
        symbol: AAPL (as of 2025-11-22)

    Followed by one feature per line.
    """
    feature_cols: list[str] = cfg["feature_cols"]
    symbol_col: str = cfg["symbol_col"]
    date_col = cfg.get("date_col")

    lines: list[str] = []

    # --- Nicer symbol/date header ---
    symbol_val = row.get(symbol_col, "UNKNOWN")

    date_str = None
    if date_col and date_col in row.index and pd.notna(row[date_col]):
        date_str = str(row[date_col])

    if date_str:
        lines.append(f"symbol: {symbol_val} (as of {date_str})")
    else:
        lines.append(f"symbol: {symbol_val}")

    # Optional blank line before features (cosmetic)
    lines.append("")

    # --- Feature lines ---
    for col in feature_cols:
        if col not in row.index:
            continue
        val = row[col]
        if isinstance(val, float):
            lines.append(f"{col}: {val:.4f}")
        else:
            lines.append(f"{col}: {val}")

    return "\n".join(lines)


def parse_llm_output(text: str) -> Tuple[str, float, str]:
    """
    Parse the fine-tuned model output:

        stance: BULLISH|NEUTRAL|BEARISH
        score: <float>
        summary: <text>

    Returns (stance, score, summary).
    """
    stance_match = re.search(r"stance\s*:\s*(BULLISH|NEUTRAL|BEARISH)", text, re.I)
    score_match = re.search(r"score\s*:\s*([-+]?\d+(\.\d+)?)", text, re.I)
    summary_match = re.search(r"summary\s*:\s*(.*)", text, re.I | re.S)

    stance = stance_match.group(1).upper() if stance_match else "NEUTRAL"
    score = float(score_match.group(1)) if score_match else 0.0
    summary = summary_match.group(1).strip() if summary_match else text.strip()

    # Bound score to the intended range [-2, 2]
    if score > 2:
        score = 2.0
    if score < -2:
        score = -2.0

    return stance, score, summary


def blend_local_and_llm(local_score: float | int, llm_score: float) -> float:
    """
    Blend local_score and llm_score into a single final_score.
    This mirrors the "Final decision blending (Local + GenAI)" idea:

        - local_score is roughly in [-3, 3]
        - llm_score is in [-2, 2] per the fine-tuning system prompt

    We normalize each and then blend with a slight overweight on the LLM.
    """
    # Normalize
    local_norm = max(min(local_score, 3), -3) / 3.0 if local_score is not None else 0.0
    llm_norm = max(min(llm_score, 2), -2) / 2.0

    # Weights: 0.4 local, 0.6 LLM (can be adjusted)
    final_score = 0.4 * local_norm + 0.6 * llm_norm
    return float(final_score)


def decision_from_final_score(final_score: float) -> str:
    """
    Map final_score into an action-style decision string.

        >=  0.75  -> Strong Buy
        0.25–0.75 -> Buy
        -0.25–0.25 -> Hold
        -0.75– -0.25 -> Sell
        <= -0.75  -> Strong Sell
    """
    if final_score >= 0.75:
        return "Strong Buy"
    if final_score >= 0.25:
        return "Buy"
    if final_score <= -0.75:
        return "Strong Sell"
    if final_score <= -0.25:
        return "Sell"
    return "Hold"


def enrich_with_llm(
    df: pd.DataFrame,
    cfg_path: str | Path = INFERENCE_CONFIG_DEFAULT,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    For each row in df, call the fine-tuned stance model and add:

        - llm_stance (BULLISH/NEUTRAL/BEARISH)
        - llm_score  (float in [-2, 2])
        - llm_summary (short explanation)
        - final_score (blended local + llm)
        - final_decision (Strong Buy/Buy/Hold/Sell/Strong Sell)

    Only rows without llm_stance/llm_score will be processed, so user can
    safely re-run this on an existing CSV without duplicating work.
    """
    cfg = load_inference_config(cfg_path)
    system_prompt_path = Path(cfg["system_prompt_path"])
    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    # Ensure OPENAI_API_KEY is available (from env or .env)
    load_env_api_key()

    client = OpenAI()

    # Ensure columns exist with appropriate dtypes
    for col in ["llm_stance", "llm_score", "llm_summary", "final_score", "final_decision"]:
        if col not in df.columns:
            if col in ["llm_score", "final_score"]:
                df[col] = np.nan          # numeric
            else:
                df[col] = None            # object (strings / labels)

    # Decide which rows to process
    mask = df["llm_stance"].isna() | df["llm_score"].isna()
    idx_to_process = df[mask].index.tolist()
    if max_rows is not None:
        idx_to_process = idx_to_process[:max_rows]

    print(f"Calling fine-tuned model for {len(idx_to_process)} rows...")

    for idx in idx_to_process:
        row = df.loc[idx]
        user_content = format_features_for_prompt(row, cfg)

        try:
            resp = client.chat.completions.create(
                model=cfg["fine_tuned_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
            )
            text = resp.choices[0].message.content
        except Exception as e:
            print(f"[WARN] LLM call failed for {row.get('symbol', 'UNKNOWN')}: {e}")
            continue

        stance, score, summary = parse_llm_output(text)
        df.at[idx, "llm_stance"] = stance
        df.at[idx, "llm_score"] = score
        df.at[idx, "llm_summary"] = summary

        # Blend with local_score
        local = df.at[idx, "local_score"] if "local_score" in df.columns else 0
        final_score = blend_local_and_llm(local, score)
        df.at[idx, "final_score"] = final_score
        df.at[idx, "final_decision"] = decision_from_final_score(final_score)

        # Light rate limiting
        time.sleep(0.15)

    return df


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def run_pipeline(
    watchlist_csv: str = WATCHLIST_CSV_DEFAULT,
    ticker_col: str = TICKER_COL_DEFAULT,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    interval: str = INTERVAL_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
    out_csv: str | None = None,
    with_llm: bool = False,
    inference_config: str | None = None,
    max_llm_rows: int | None = None,
) -> pd.DataFrame:
    """
    Full watchlist -> metrics -> (optional) LLM pipeline.
    Returns the final DataFrame and optionally writes to CSV.
    """
    # Load + normalize watchlist
    watch_df = load_watchlist(watchlist_csv, ticker_col)
    tickers = watch_df["symbol"].tolist()
    print(f"Loaded {len(tickers)} normalized tickers. Sample: {tickers[:10]}")

    # Time window
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_days)

    # Download prices
    price_data = fetch_prices_yf(
        tickers,
        start=str(start_date),
        end=str(end_date + timedelta(days=1)),
        interval=interval,
        batch_size=batch_size,
    )
    print(f"Fetched price history for {len(price_data)}/{len(tickers)} tickers.")

    # Compute metrics
    metrics_df = (
        compute_metrics(price_data)
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    print(f"Computed metrics for {len(metrics_df)} tickers.")

    # Tags + local score
    metrics_df["tags"] = metrics_df.apply(build_tags, axis=1)
    metrics_df["local_score"] = metrics_df["tags"].apply(local_score_from_tags)

    # Merge original symbol + Name for reference
    merge_cols = ["symbol", "symbol_original"]
    if "Name" in watch_df.columns:
        merge_cols.append("Name")

    metrics_df = metrics_df.merge(
        watch_df[merge_cols],
        on="symbol",
        how="left",
    )

    # Optionally call fine-tuned model
    if with_llm:
        cfg_path = inference_config or INFERENCE_CONFIG_DEFAULT
        metrics_df = enrich_with_llm(
            metrics_df,
            cfg_path=cfg_path,
            max_rows=max_llm_rows,
        )

    # Drop symbol_original from final output; it's only for internal mapping
    if "symbol_original" in metrics_df.columns:
        metrics_df = metrics_df.drop(columns=["symbol_original"])

    # Reorder some "nice" columns toward the front
    cols_preferred = [
        "symbol",
        "Name",  # if not present, will be ignored
        "date",
        "close",
        "change_1d",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        # Bollinger group (in derivation order)
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "bb_pctb",
        "bb_bandwidth",
        "support",
        "resistance",
        "hi_52w",
        "lo_52w",
        # TD group: dates + perfected flags + counts, then signal
        "td_last_buy9_date",
        "td_last_sell9_date",
        "td_buy_perfected",
        "td_sell_perfected",
        "td_curr_buy_count",
        "td_curr_sell_count",
        "td_last_signal",
        "tags",
        "local_score",
        "llm_stance",
        "llm_score",
        "llm_summary",
        "final_score",
        "final_decision",
    ]
    cols_preferred = [c for c in cols_preferred if c in metrics_df.columns]
    other_cols = [c for c in metrics_df.columns if c not in cols_preferred]
    output_df = metrics_df[cols_preferred + other_cols]

    # Round float columns for a cleaner CSV (e.g., 12.95 instead of 12.946999...)
    float_cols = output_df.select_dtypes(include=["float32", "float64"]).columns
    output_df[float_cols] = output_df[float_cols].round(2)

    # Choose default output path if not provided
    if out_csv is None:
        suffix = "with_llm" if with_llm else "metrics"
        out_csv = f"./watchlist_{suffix}_{datetime.now().strftime('%Y%m%d')}.csv"

    output_df.to_csv(out_csv, index=False)
    print(f"Saved CSV ⇒ {out_csv}")

    return output_df


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process Chance_Watchlist.csv into a metrics CSV and optionally call the fine-tuned stance model."
    )
    parser.add_argument(
        "--watchlist",
        default=WATCHLIST_CSV_DEFAULT,
        help=f"Path to watchlist CSV (default: {WATCHLIST_CSV_DEFAULT})",
    )
    parser.add_argument(
        "--ticker-col",
        default=TICKER_COL_DEFAULT,
        help=f"Column name containing tickers (default: {TICKER_COL_DEFAULT})",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=LOOKBACK_DAYS_DEFAULT,
        help=f"History lookback in days (default: {LOOKBACK_DAYS_DEFAULT})",
    )
    parser.add_argument(
        "--interval",
        default=INTERVAL_DEFAULT,
        help=f"yfinance interval string (default: {INTERVAL_DEFAULT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=f"yfinance batch size (default: {BATCH_SIZE_DEFAULT})",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional explicit output CSV path (default: auto-named)",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="If set, call the fine-tuned stance model using inference_config.json.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to inference_config.json (default: {INFERENCE_CONFIG_DEFAULT})",
    )
    parser.add_argument(
        "--max-llm-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows to send to the LLM (useful for testing).",
    )

    args = parser.parse_args()

    df = run_pipeline(
        watchlist_csv=args.watchlist,
        ticker_col=args.ticker_col,
        lookback_days=args.lookback_days,
        interval=args.interval,
        batch_size=args.batch_size,
        out_csv=args.out_csv,
        with_llm=args.with_llm,
        inference_config=args.config,
        max_llm_rows=args.max_llm_rows,
    )

    # Show a small preview in the terminal
    with pd.option_context("display.max_columns", None):
        print(df.head(10))


if __name__ == "__main__":
    main()
