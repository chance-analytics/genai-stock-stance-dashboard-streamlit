# GenAI Stock Stance Dashboard
*Fine-Tuning + Technical Indicators + Streamlit App*

## Overview
This project builds an end-to-end pipeline that scores a personal stock watchlist using **technical indicators** plus a **fine-tuned LLM “stance head”**. The goal is to convert noisy chart signals into a consistent, explainable output for each ticker:

- **LLM stance**: BULLISH / NEUTRAL / BEARISH  
- **LLM score**: −2 to +2  
- **1–2 sentence summary** explaining the setup  
- A blended **final_score** mapped to **Strong Buy / Buy / Hold / Sell / Strong Sell**

The final results are published to an **interactive Streamlit dashboard** for filtering, sorting, and per-ticker drilldowns.

---

## Inputs
- **Watchlist:** `Chance_Watchlist.csv` (~129 tickers) exported from investing.com (symbols normalized for Yahoo Finance, e.g., `BRK.B → BRK-B`)  
- **Market data:** ~420 trading days of daily OHLCV per ticker downloaded in batches via `yfinance`  
- **Model:** fine-tuned OpenAI model (stance head) that takes structured features and returns strict JSON outputs

---

## Feature engineering (per ticker)
Computed locally from OHLCV:
- **Price context:** 1D change, 52-week high/low proximity
- **RSI**
- **MACD** (line, signal, histogram)
- **Bollinger Bands** (mid/upper/lower, %B, bandwidth)
- **Support / resistance** via local extrema
- **TD Sequential** setup counts + “perfected” flags

These metrics are also transformed into human-readable **tags** (e.g., `RSI_oversold`, `MACD_bullish`, `near_lower_band`, `TD_buy_perfected`) to keep reasoning transparent.

---

## Scoring logic
### 1) Local rule-based score (technical-only)
A simple heuristic converts tags into an integer **local_score** (roughly in the range −3 to +3):
- bullish tags add points (oversold RSI, bullish MACD, support, TD buy signals)
- bearish tags subtract points (overbought RSI, bearish MACD, resistance, TD sell signals)

### 2) Fine-tuned LLM stance head
For each ticker row, the pipeline sends structured features to the fine-tuned model. The model returns:
- `llm_stance` ∈ {BULLISH, NEUTRAL, BEARISH}
- `llm_score` ∈ [−2, 2]
- `llm_summary` (1–2 sentences)

### 3) Blended final decision
Scores are normalized and blended:
- local_score → normalized to [−1, 1]
- llm_score → normalized to [−1, 1]
- **final_score = 0.4 × local_norm + 0.6 × llm_norm**

Decision buckets:
- ≥ 0.75 → **Strong Buy**
- 0.25–0.75 → **Buy**
- −0.25–0.25 → **Hold**
- −0.75–−0.25 → **Sell**
- ≤ −0.75 → **Strong Sell**

---

## Outputs
- **Scored CSV**: one row per ticker with metrics, tags, local_score, llm fields, final_score, and final_decision
- **Streamlit dashboard**:
  - sortable table + filters (symbol, stance, decision, score ranges)
  - summary KPIs (count, average score, most common decision)
  - decision distribution chart
  - per-ticker drilldown (metrics + tags + LLM summary)

---

## How to run
### 1) Clone
```bash
git clone https://github.com/chance-analytics/genai-stock-stance-dashboard-streamlit.git
cd genai-stock-stance-dashboard-streamlit
```

### 2) Environment (example)
```bash
conda create -n stance-dashboard python=3.11 -y
conda activate stance-dashboard
```

### 3) Install dependencies (typical)
```bash
pip install pandas numpy yfinance tqdm streamlit python-dotenv
```

### 4) Configure credentials
Set your OpenAI API key as an environment variable (recommended):
```bash
export OPENAI_API_KEY="YOUR_KEY"
```

If your repo uses a JSON config (e.g., `m3_artifacts/inference_config.json`), confirm:
- `fine_tuned_model` (your model name)
- feature columns used for inference
- symbol/date column names
- prompt paths (system/user)

### 5) Run the pipeline
Run the script/notebook that:
- loads the watchlist
- downloads OHLCV
- computes indicators + tags + local_score
- calls the fine-tuned model for `llm_*`
- writes the final scored CSV

### 6) Launch Streamlit
```bash
streamlit run app.py
```

---

## Limitations
- Uses **daily technical indicators only** (no fundamentals or news sentiment yet)
- Fine-tuning dataset size is limited to the project scope
- Not a backtested trading strategy (no execution logic; educational analytics tool)

---

## Next improvements
- Add fundamentals (valuation + growth) as extra features
- Integrate recent news/sentiment (RAG) into the stance head
- Backtest decision buckets vs. future returns and calibrate thresholds
- Automate daily refresh and optional hosted deployment

---

## Author
**Chance Xu**  
GitHub: https://github.com/chance-analytics
