# 📈 Markowitz Portfolio Theory Strategy

A Python-based backtesting framework implementing Markowitz mean-variance portfolio optimization for systematic paper trading.

It supports:
- ✅ **Minimum variance** and **tangency portfolio** strategies  
- ✅ **Momentum screening** and **market regime filters**  
- ✅ Full performance analytics with **Monte Carlo simulations**, risk metrics, and plots

---

## ⚙️ Assumptions

This backtest framework operates under the following assumptions:
- **Long-only:** Short-selling is not allowed
- **Whole shares only:** Partial shares are not supported
- **Risk-free rate:** Fixed at 3% annualized
- **Zero transaction fees:** No commissions or costs
- **No bid-ask spread or slippage:** Perfect execution at daily close prices
- **Past returns are representative:** Historical return distributions are assumed valid for the future
- **Instant execution:** Orders execute at the daily close price with no delay
- **Fallback:** If the optimizer fails to converge, it defaults to equal weighting

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. **Create and activate your Python virtual environment (recommended)**

```bash
python  --m venv venv
source venv/bin/activate     # on Linux/macOS
.\venv\Scripts\Activate      # on Windows PowerShell
```

### 3. **Install required libraries**

```bash
pip install -r requirements.txt
```

### 4. **Adjust parameters**

Open the `markowitz.py` file and look for the `if __name__ == __main__` section at the bottom

You can adjust the following parameters

- Tickers : modify the `tickers` list to include the stock symbols you want to trade, e.g.:

```
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
```

- Data Range : set the backtest start and end dates (YYYY-MM-DD)

```
start_date = "2020-01-01"
end_date = "2025-07-25"
```

- Initial Capital : starting cash amount

```
initial_capital = 100000
```

- Rebalance Period : change how often the portfolio rebalances (in trading days)

```
rebalance_period = 21 # monthly rebalancing
```

- Lookback Period : change how often the strategy looks back to calculate momentum (in trading days)

```
lookback_period = 252 # 1 year lookback
```

After making changes, save the file and re-run the script.

### 5. **Run the backtester**

```bash
python markowitz.py
```

### 6. **Logs and Files**

The backtesting framework generates and saves files for full transparency and reproducibility:

- **`./data/`**  
  Contains the historical daily **adjusted closing prices** (`prices.csv`) and computed **daily returns** (`returns.csv`) for all tickers used during the backtest period. Useful for debugging, audit trails, or custom analysis.

- **`./logs/`**  
  - **`./logs/transactions/`**: CSV files logging each **buy** and **sell** transaction, including date, ticker, side, quantity, and price.  
  - **`./logs/weights/`**: CSV files tracking **daily portfolio weights** for each stock, cash position, stock position, and total portfolio value over time.

- **`./plots/`**  
  Contains all generated **visualizations** for performance analysis, including:
  - `equity_curve.png` — Equity curve
  - `weights_over_time.png` — Portfolio weights over time
  - `drawdown.png` — Portfolio drawdown over time
  - `daily_returns_histogram.png` — Histogram of daily returns
  - `cumulative_returns.png` — Cumulative returns curve
  - `rolling_volatility.png` — Rolling annualized volatility
  - `rolling_sharpe_ratio.png` — Rolling Sharpe ratio
  - `rolling_var.png` — Rolling Value at Risk (VaR)
  - `monthy_returns_heatmap/` — Monthly returns heatmaps for each ticker
  - `monte_carlo_simulation.png` — Monte Carlo simulation output

All plots are organized for easy inspection and can be reused in reports, presentations, or further research.