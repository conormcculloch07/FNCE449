# FNCE449
My name is Conor McCulloch
# CODE FOR PORTFOLIO OPTIMIZATION WITH NO ESG FACTOR
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers = ["BIR.TO", "POU.TO", "TOU.TO", "CNQ.TO", "IMO.TO"]
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']

esg_scores = np.array([41.2, 72.1, 40.0, 49.7, 17.6])  # ESG scores on a 0-100 scale

# Calculate daily returns
returns = data.ffill().bfill().pct_change()

# Calculate mean returns and covariance matrix
mean_returns = returns.mean() * 252  # Annualize returns
cov_matrix = returns.cov() * 252     # Annualize covariance

# Portfolio optimization functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    """ Params
    weights: randomized weights
    mean_returns: average returns
    cov_matrix: cov of assets
    
    Calculate portfolio return and volatility.
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Objective function: Sharpe Ratio maximization (negative for minimization)
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    """
    This function returns - sharpe ratio
    """
    
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

# Optimize portfolio for maximum Sharpe Ratio
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(len(tickers)))
initial_weights = [1 / len(tickers)] * len(tickers)
result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Extract optimal weights and calculate performance metrics
optimal_weights = result.x
optimal_return, optimal_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
risk_free_rate = 0.04  # Assuming a 4% risk-free rate
optimal_sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility

# Print optimal portfolio metrics
print(tickers)
print("Optimal Weights:", optimal_weights)
print("Expected Portfolio Return:", optimal_return)
print("Expected Portfolio Volatility (Risk):", optimal_volatility)
print("Optimal Sharpe Ratio:", optimal_sharpe_ratio)

# Plot Efficient Frontier
def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=50000, risk_free_rate=0.04):
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(mean_returns)), size=1).flatten()
        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio
    plt.figure(figsize=(10, 6),facecolor='lightblue')
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(optimal_volatility, optimal_return, marker='*', color='black', s=150, label='Optimal Portfolio')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Return')
    plt.title('Efficient Frontier for Portfolio')
    plt.gca().set_facecolor('lightgreen')
    plt.legend()
    plt.show()

plot_efficient_frontier(mean_returns, cov_matrix)













# CODE FOR OPTIMAL PORTFOLIO WITH ESG PENALTY FACTOR

# Penalized objective function: Maximizing Sharpe Ratio with ESG penalties
def negative_sharpe_with_esg_penalty(weights, mean_returns, cov_matrix, esg_scores, esg_penalty_factor=1.5, risk_free_rate=0.04):
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Amplified ESG penalty: penalize low ESG scores using squared penalty
    esg_penalty = np.sum(weights * ((1 - esg_scores / 100) ** 2) * esg_penalty_factor)  # Higher penalty for lower scores
    return -(sharpe_ratio - esg_penalty)

# Constraints and bounds
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(len(tickers)))
initial_weights = [1 / len(tickers)] * len(tickers)

# Optimize portfolio for maximum Sharpe Ratio with enhanced ESG penalty
esg_penalty_factor = 1.5  # Stronger impact of ESG penalty
result = minimize(negative_sharpe_with_esg_penalty, initial_weights, args=(mean_returns, cov_matrix, esg_scores, esg_penalty_factor),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights and performance
optimal_weights = result.x
optimal_return, optimal_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
optimal_esg_score = np.dot(optimal_weights, esg_scores)
optimal_sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility

print(tickers)
print("Optimal Weights:", optimal_weights)
print("Expected Portfolio Return:", optimal_return)
print("Expected Portfolio Volatility (Risk):", optimal_volatility)
print("Optimal ESG Score:", optimal_esg_score)
print("Optimal Sharpe Ratio (with Enhanced ESG Penalty):", optimal_sharpe_ratio)

# Plot Efficient Frontier with ESG-Penalized Portfolio
def plot_efficient_frontier_esg(mean_returns, cov_matrix, esg_scores, num_portfolios=50000, risk_free_rate=0.04, esg_penalty_factor=1.5):
    results = np.zeros((4, num_portfolios))  # Adding ESG score for the fourth result
    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(mean_returns)), size=1).flatten()
        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Apply enhanced ESG penalty
        esg_penalty = np.sum(weights * ((1 - esg_scores / 100) ** 2) * esg_penalty_factor)
        adjusted_sharpe = sharpe_ratio - esg_penalty
        weighted_esg_score = np.dot(weights, esg_scores)
        
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = adjusted_sharpe
        results[3, i] = weighted_esg_score  # Track ESG score

    plt.figure(figsize=(10, 6),facecolor='lightblue')
    plt.scatter(results[0, :], results[1, :], c=results[3, :], cmap='cool', marker='o', alpha=0.6)
    plt.colorbar(label='Weighted ESG Score')
    plt.scatter(optimal_volatility, optimal_return, marker='*', color='black', s=150, label='ESG-Penalized Portfolio')
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Return")
    plt.title("Efficient Frontier with ESG-Penalty Factor Portfolio")
    plt.gca().set_facecolor('lightgreen')
    plt.legend()
    plt.show()

plot_efficient_frontier_esg(mean_returns, cov_matrix, esg_scores)
