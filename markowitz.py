from datetime import datetime
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import yfinance as yf

class MinimumVariancePortfolio:
    """
    Markowitz mean-variance portfolio optimizer.
    This class finds the minimum variance allocation for given historical returns.

    Assumptions
    - risk-free rate of 3%
    - target return of 7%
    - no short-selling
    """
    def __init__(self, rebalance_period=21, lookback_period=126, target_return=0):
        """
        rebalance_period : how often to rebalance (in trading days)
        target_return : specify a target return (increases drawdown, VaR)
        use_target_return : indicate use of target_return (increases drawdown, VaR)
        """
        self.rebalance_period = rebalance_period
        self.lookback_period = lookback_period
        self.target_return = target_return

    def optimize_portfolio(self, returns):
        """
        Optimize the portfolio weights to minimize variance under the constraint that 
        sum of weights = 1 and are >= 0
        """
        n_assets = returns.shape[1]
        
        def portfolio_variance(weights):
            """
            Computes portfolio variance = W.TxCxW
            """
            cov = returns.cov() * 252
            return np.dot(weights.T, np.dot(cov, weights))
        
        def constraint_sum(weights):
            return np.sum(weights) - 1
        
        def constraint_return(weights):
            mean_returns = returns.mean() * 252
            return np.dot(weights, mean_returns) - self.target_return
        
        # Initial Guess : equal weightage
        x0 = np.ones(n_assets)/n_assets

        # Define constraints and bounds (long-only)
        if self.target_return > 0:
            constraints = [{
                "type": "eq",
                "fun": constraint_sum
            }, {
                "type": "eq",
                "fun": constraint_return
            }]
        else:
            constraints = {
                "type": "eq",
                "fun": constraint_sum
            }
        bounds = tuple((0, 1) for _ in range(n_assets))

        result = minimize(portfolio_variance, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            print("Optimization failed! Using equal weights instead.")
        return result.x if result.success else x0
    
class TangencyPortfolio:
    def __init__(self, rebalance_period=21, lookback_period=126, risk_free_rate=0.03):
        self.rebalance_period = rebalance_period
        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate

    def optimize_portfolio(self, returns):
        """
        Optimize the portfolio weights to maximise Sharpe Ratio (minimize the negative Sharpe Ratio) 
        under the constraint that  sum of weights = 1 and are >= 0
        """
        n_assets = returns.shape[1]
        mean_returns = returns.mean() * 252
        cov = returns.cov() * 252
        rf = self.risk_free_rate

        def negative_sharpe(weights):
            """
            Computes negative of Sharpe ratio
            """
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            return -(port_return - rf) / port_vol  # negative Sharpe to minimize

        def constraint_sum(weights):
            return np.sum(weights) - 1

        # Initial Guess : equal weightage
        x0 = np.ones(n_assets) / n_assets

        # Define constraints and bounds (long-only)
        constraints = {
            "type": "eq", 
            "fun": constraint_sum
        }
        bounds = tuple((0, 1) for _ in range(n_assets))

        result = minimize(negative_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        
        if not result.success:
            print("Optimization failed! Using equal weights instead.")
        return result.x if result.success else x0

class MonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio return paths.
    Uses bootstrapped daily returns from backtest.
    """

    def __init__(self, daily_returns, num_simulations=1000, num_days=None, seed=42):
        """
        daily_returns: pd.Series or pd.DataFrame with daily returns (e.g. backtester equity curve pct_change())
        num_simulations: number of MC paths
        num_days: simulation horizon; default = length of input daily returns
        """
        self.daily_returns = daily_returns.dropna()
        self.num_simulations = num_simulations
        self.num_days = num_days or len(self.daily_returns)
        self.seed = seed
        np.random.seed(self.seed)

    def run(self):
        """
        Runs the Monte Carlo bootstrap simulation.
        Returns a DataFrame with shape (num_days, num_simulations)
        """
        returns = self.daily_returns.values
        simulations = np.zeros((self.num_days, self.num_simulations))

        for i in range(self.num_simulations):
            sampled = np.random.choice(returns, size=self.num_days, replace=True)
            path = np.cumprod(1 + sampled)
            simulations[:, i] = path

        self.simulations = pd.DataFrame(simulations)
        return self.simulations

    def plot(self, initial_value=1.0, percentile_bands=(5, 95)):
        """
        Plots Monte Carlo paths with percentile bands.
        """
        if not hasattr(self, "simulations"):
            raise ValueError("Run the simulation first!")

        paths = self.simulations * initial_value

        # Compute percentiles
        lower = paths.quantile(percentile_bands[0]/100, axis=1)
        upper = paths.quantile(percentile_bands[1]/100, axis=1)
        median = paths.median(axis=1)

        plt.figure(figsize=(12, 6))
        plt.plot(paths, color="grey", alpha=0.05)
        plt.plot(median, color="blue", label="Median Path")
        plt.fill_between(lower.index, lower, upper, color="blue", alpha=0.2, label=f"{percentile_bands[0]}-{percentile_bands[1]}% Interval")

        plt.title("Monte Carlo Simulated Portfolio Paths")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./plots/monte_carlo_simulation.png")
        plt.show()

class Backtester:
    """
    Runs the backtest loop:
    - Download data from yfiannce
    - Runs strategy
    - Logs transactions and daily weights
    - Generates graphs
    """
    def __init__(self, tickers, strategy, start_date, end_date, initial_capital=100000, risk_free_rate=0.03):
        """
        tickers: list of stock symbols
        strategy: an instance of a Strategy class
        start_date, end_date: date range for backtest period
        initial_capital: starting cash
        """
        self.tickers = tickers
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Price Data
        self.data = {}

        # Returns (pct_change)
        self.returns = pd.DataFrame()

        # Investable Cash
        self.cash = initial_capital

        # Number of shares held in each stock
        self.positions = {ticker: 0 for ticker in self.tickers}

        # Equity Curve
        self.daily_value = []
        self.daily_dates = []
        
        # Logs
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        self.date_time_str = date_time_str
        self.txn_file = f"./logs/transactions/{self.date_time_str}.csv"
        self.weights_file = f"./logs/weights/{self.date_time_str}.csv"

        os.makedirs("./plots", exist_ok=True)
        os.makedirs("./plots/monthy_returns_heatmap", exist_ok=True)
        os.makedirs("./logs/transactions", exist_ok=True)
        os.makedirs("./logs/weights", exist_ok=True)


        # Write CSV headers
        with open(self.txn_file, "w") as f:
            f.write("DATE,STOCK,BUY/SELL,QUANTITY,PRICE\n")

        with open(self.weights_file, "w") as f:
            f.write("DATE," + ",".join(self.tickers) + ",CASH_POSITION,STOCK_POSITION,TOTAL_AMOUNT\n")

    def fetch_data(self):
        """
        Download historical adjusted close prices for the tickers from Yahoo Finance
        """
        print(f"Fetching data for {self.tickers}...")

        df = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True)["Close"]
        df = df.dropna()
        self.data = df
        self.returns = df.pct_change().dropna()
        
        # SPY S&P500 benchmark 
        self.spy_prices = yf.download("SPY", start=self.start_date, end=self.end_date, auto_adjust=True)["Close"]
        self.spy_ma200 = self.spy_prices.rolling(200).mean()

        # save data
        os.makedirs("./data", exist_ok=True)
        self.data.to_csv("./data/prices.csv")
        print("Saved data to ./data/prices.csv")
        self.returns.to_csv("./data/returns.csv")
        print("Saved returns to ./data/returns.csv")

    def accrue_cash_interest(self):
        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
        self.cash *= (1+ daily_rf)

    def calculate_momentum(self, date, lookback=126):
        """
        Calculate simple 6-month momentum for each stock.
        """
        if self.data.index.get_loc(date) < lookback:
            return pd.Series([0] * len(self.tickers), index=self.tickers)
        prices_now = self.data.loc[date]
        prices_past = self.data.loc[self.data.index[self.data.index.get_loc(date) - lookback]]
        momentum = (prices_now / prices_past) - 1
        return momentum
    

    def get_market_regime(self, date):
        """
        Robust regime filter
        - Uses SPY 200-day MA.
        - Returns False if insufficient data.
        - Always returns a scalar bool.
        """
        price = self.spy_prices.loc[date]["SPY"]
        ma200 = self.spy_ma200.loc[date]["SPY"]
        if pd.isna(ma200):
            return False
        else:
            return bool(price < ma200)

    def rebalance_portfolio(self, date, target_weights):
        """
        Rebalance the portfolio by selling overweight positions, then buying underweight positions.
        """

        n_stocks = len(self.tickers)
        stock_weights = target_weights[:n_stocks]
        
        # Compute total portfolio value (cash + market value)
        total_value = self.cash + sum(self.positions[t] * self.data.loc[date, t] for t in self.tickers)

        current_values = {ticker: self.positions[ticker] * self.data.loc[date, ticker] for ticker in self.tickers}
        target_values = {ticker: stock_weights[i] * total_value for i, ticker in enumerate(self.tickers)}

        # Sell overweighted stocks
        for ticker in self.tickers:
            price = self.data.loc[date, ticker]
            diff = current_values[ticker] - target_values[ticker]

            if diff > price:
                qty_to_sell = int(diff//price)
                qty_to_sell = min(qty_to_sell, self.positions[ticker])
                # sell only when we have shares
                if qty_to_sell > 0:
                    # sell the shares
                    self.positions[ticker] -= qty_to_sell
                    # add to cash
                    self.cash += qty_to_sell * price
                    # log
                    self.log_transaction(date, ticker, "SELL", qty_to_sell, price)

        # Buy underweighted stocks
        for ticker in self.tickers:
            price = self.data.loc[date, ticker]
            diff = target_values[ticker] - current_values[ticker]

            if diff > price and self.cash > price:
                qty_to_buy = int(min(diff // price, self.cash // price))
                # buy only if we need to
                if qty_to_buy > 0:
                    # buy the shares
                    self.positions[ticker] += qty_to_buy
                    # deduct from cash
                    self.cash -= qty_to_buy * price
                    # log
                    self.log_transaction(date, ticker, "BUY", qty_to_buy, price)
    
    def log_transaction(self, date, ticker, side, qty, price):
        """
        Writes a single transaction (buy/sell side) to CSV
        """
        with open(self.txn_file, "a") as f:
            f.write(f"{date},{ticker},{side},{qty},{price:.2f}\n")
        
    def log_weights(self, date):
        """
        Compute and log:
        - Stock Weights
        - Cash Position
        - Stock Position
        - Total Portfolio Value
        Daily PnL for equity curve plot
        """
        total_value = self.cash + sum(self.positions[t] * self.data.loc[date, t] for t in self.tickers)
        stock_value = total_value - self.cash
        stock_weights = []
        for ticker in self.tickers:
            val = self.positions[ticker] * self.data.loc[date, ticker]
            w = val/total_value if total_value != 0 else 0
            stock_weights.append(w)

        cash_weight = self.cash / total_value if total_value != 0 else 0
        stock_position = stock_value/total_value if total_value != 0 else 0

        line = f"{date}," + ",".join([f"{w:.4f}" for w in stock_weights]) + f",{cash_weight:.4f},{stock_position:.4f},{total_value:.2f}\n"

        with open(self.weights_file, "a") as f:
            f.write(line)
        
        self.daily_dates.append(date)
        self.daily_value.append(total_value)

    def plot_equity_curve(self):
        weights_df = pd.read_csv(self.weights_file, parse_dates=["DATE"])
        weights_df.set_index("DATE", inplace=True)

        plt.figure(figsize=(12, 6))
        plt.plot(self.daily_dates, self.daily_value, label="Portfolio Value")
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./plots/equity_curve.png")
        plt.show()

    def plot_weights_over_time(self):
        weights_df = pd.read_csv(self.weights_file, parse_dates=["DATE"])
        weights_df.set_index("DATE", inplace=True)

        plt.figure(figsize=(12, 6))
        for ticker in self.tickers:
            plt.plot(weights_df.index, weights_df[ticker], label=ticker)
        plt.plot(weights_df.index, weights_df["CASH_POSITION"], label="Cash", linestyle="--")
        plt.title("Potfolio Weights")
        plt.xlabel("Date")
        plt.ylabel("Weight")
        plt.legend()
        plt.tight_layout()
        plt.savefig("./plots/weights_over_time.png")
        plt.show()
    
    def plot_drawdown(self):
        portfolio_values = pd.Series(self.daily_value, index=self.daily_dates)
        cum_returns = portfolio_values/portfolio_values.iloc[0]
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max

        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, drawdown.values * 100, color="red", alpha=0.5)
        plt.title("Drawdown Over Time")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./plots/drawdown.png")
        plt.show()

    def plot_daily_returns_histogram(self):
        portfolio_values = pd.Series(self.daily_value, index=self.daily_dates)
        daily_returns = portfolio_values.pct_change().dropna()

        plt.figure(figsize=(12, 6))
        plt.hist(daily_returns, bins=50, color="blue", alpha=0.7)
        plt.title("Histogram of Daily Returns")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./plots/daily_returns_histogram.png")
        plt.show()

    def plot_cumulative_returns(self):
        portfolio_values = pd.Series(self.daily_value, index=self.daily_dates)
        daily_returns = portfolio_values.pct_change().dropna()
        cum_returns = (1 + daily_returns).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(cum_returns, label="Cumulative Return")
        plt.title("Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./plots/cumulative_returns.png")
        plt.show()

    def plot_rolling_volatility(self, window=21):
        portfolio_values = pd.Series(self.daily_value, index=self.daily_dates)
        daily_returns = portfolio_values.pct_change().dropna()
        rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(252)

        plt.figure(figsize=(12, 6))
        plt.plot(rolling_vol, label=f"Rolling Annualized Volatility ({window}-day)")
        plt.title("Rolling Volatility Over Time")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./plots/rolling_volatility.png")
        plt.show()

    def plot_rolling_sharpe(self, window=21, risk_free_rate=0.0):
        portfolio_values = pd.Series(self.daily_value, index=self.daily_dates)
        daily_returns = portfolio_values.pct_change().dropna()
        excess_returns = daily_returns - risk_free_rate/252
        rolling_sharpe = (excess_returns.rolling(window=window).mean() / excess_returns.rolling(window=window).std()) * np.sqrt(252)

        plt.figure(figsize=(12, 6))
        plt.plot(rolling_sharpe, label=f"Rolling Sharpe Ratio ({window}-day)")
        plt.title("Rolling Sharpe Ratio Over Time")
        plt.xlabel("Date")
        plt.ylabel("Sharpe Ratio")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./plots/rolling_sharpe_ratio.png")
        plt.show()
    
    def plot_rolling_var(self, window=21, confidence_level=0.95):
        """
        Calculates Value at Risk at 95% confidence level based on the rolling portfolio returns in the specified window.

        Describes maximum percentage of portfolio loss with 95% confidence (not absolute value).
        """
        def calculate_historical_var(daily_returns, confidence_level=0.95):
            sorted_returns = daily_returns.sort_values()
            index = int((1 - confidence_level) * len(sorted_returns))
            var = sorted_returns.iloc[index]
            return var

        portfolio_values = pd.Series(self.daily_value, index=self.daily_dates)
        daily_returns = portfolio_values.pct_change().dropna()
        rolling_var = daily_returns.rolling(window=window).apply(
            lambda x : calculate_historical_var(x, confidence_level)
        )

        plt.figure(figsize=(12, 6))
        plt.plot(rolling_var, label=f"Rolling VaR ({window}-day, {confidence_level*100}% Confidence Level)")
        plt.title("Rolling Value at Risk Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value at Risk")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./plots/rolling_var.png")
        plt.show()

    def plot_data_monthly_returns_heatmap(self):
        """
        Plot heatmap of average monthly returns for each ticker.
        Shows how each stock performed across calendar months.
        """
        # Use daily returns DataFrame
        df = self.returns.copy()
        df.index = pd.to_datetime(df.index)

        # Add Year & Month
        df["Year"] = df.index.year
        df["Month"] = df.index.month

        # Compute monthly returns: product of daily returns per month - 1
        monthly_returns = df.groupby(["Year", "Month"])[self.tickers].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()

        # Pivot for Heatmap
        pivot_df = monthly_returns.pivot(index="Year", columns="Month")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        for i, ticker in enumerate(self.tickers):
            data = pivot_df[ticker]
            data.columns = month_names[:data.shape[1]]

            plt.figure(figsize=(12, 6))
            sns.heatmap(
                data * 100,
                annot=True, fmt=".2f",
                cmap="RdYlGn", center=0,
                linewidths=0.5
            )
            plt.title(f"Monthly Returns Heatmap: {ticker}")
            plt.xlabel("Month")
            plt.ylabel("Year")
            plt.tight_layout()
            plt.savefig(f"./plots/monthy_returns_heatmap/{ticker}.png")
    
    def print_performance_summary(self):
        """
        Calculate and print key portfolio performance statistics:
        - Total Return
        - Annualized Return
        - Annualized Volatility
        - Sharpe Ratio
        - Max Drawdown
        """
        portfolio_values = pd.Series(self.daily_value, index=self.daily_dates)
        daily_returns = portfolio_values.pct_change().dropna()

        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        num_days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
        annualized_vol = daily_returns.std() * np.sqrt(252)
        risk_free_rate = self.risk_free_rate

        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol != 0 else np.nan

        # Max Drawdown Calculation
        cumulative_returns = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        print("\n--- Portfolio Performance Summary ---")
        print(f"Total Return.          : {total_return:.2%}")
        print(f"Annualized Return      : {annualized_return:.2%}")
        print(f"Annualized Volatility  : {annualized_vol:.2%}")
        print(f"Sharpe Ratio (RF={risk_free_rate*100:.1f}%) : {sharpe_ratio:.2f}")
        print(f"Max Drawdown           : {max_drawdown:.2%}")

    def run_backtest(self):
        rebalance_frequency = self.strategy.rebalance_period
        lookback_period = self.strategy.lookback_period

        for i, date in enumerate(self.returns.index):
            self.accrue_cash_interest()

            if i % rebalance_frequency == 0:
                window_returns = self.returns.loc[:date].tail(lookback_period)
                if window_returns.shape[0] < 21:
                    continue

                # Momentum Filter : use only top 5 momentum tickers
                momentum = self.calculate_momentum(date, lookback=126)
                top_n = 10
                selected = momentum.nlargest(top_n).index.tolist()

                # Optimize with selected stocks only
                window_returns = window_returns[selected]
                weights = self.strategy.optimize_portfolio(window_returns)

                # Expand back to full tickers list with zeros for dropped tickers
                weights_full = []
                for t in self.tickers:
                    if t in selected:
                        weights_full.append(weights[selected.index(t)])
                    else:
                        weights_full.append(0.0)

                weights_full = np.array(weights_full)

                # Regime Filter : risk-off if SPY < 200MA - reduce stock exposure by 50% (put to cash)
                if self.get_market_regime(date):
                    weights_full *= 0.5

                # print(f"{date} : Rebalancing | Momentum: {selected} | Regime Risk-off: {self.get_market_regime(date)} | Weights: {dict(zip(self.tickers, [round(w, 2) for w in weights_full]))}\n")

                self.rebalance_portfolio(date, weights_full)

            self.log_weights(date)

        final_value = self.cash + sum(self.positions[t] * self.data.iloc[-1][t] for t in self.tickers)

        # Performance Statistics
        print(f"Backtest complete. Final portfolio value: ${final_value:.2f}")
        self.print_performance_summary()

        # Monte Carlo Simulator (1 year horizon)
        portfolio_values = pd.Series(self.daily_value, index=self.daily_dates)
        daily_returns = portfolio_values.pct_change().dropna()

        mc = MonteCarloSimulator(daily_returns, num_simulations=500, num_days=252)
        mc.run()
        mc.plot(initial_value=final_value)

        # Plot Graphs
        self.plot_equity_curve()
        self.plot_weights_over_time()
        self.plot_drawdown()
        self.plot_daily_returns_histogram()
        self.plot_cumulative_returns()
        self.plot_rolling_volatility(window=rebalance_frequency)
        self.plot_rolling_sharpe(window=rebalance_frequency)
        self.plot_rolling_var(window=rebalance_frequency)
        self.plot_data_monthly_returns_heatmap()

# Run Backtest
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "MA", "XOM"]

    start_date = "2020-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    initial_capital = 100000

    # Minimum Variance Portfolio Strategy
    # rebalance_period = 21       # rebalance every month
    # lookback_period = 126       # use previous 6 months data
    # target_return = 0           # set target return (increase portfolio risk)
    # strategy = MinimumVariancePortfolio(rebalance_period=rebalance_period, target_return=target_return)

    # Tangency Portfolio Strategy
    rebalance_period = 21       # rebalance every month
    lookback_period = 252       # use previous year data
    risk_free_rate = 0.03
    strategy = TangencyPortfolio(rebalance_period=rebalance_period, lookback_period=lookback_period, risk_free_rate=risk_free_rate)

    backtester = Backtester(
        tickers=tickers,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate
    )

    backtester.fetch_data()
    backtester.run_backtest()