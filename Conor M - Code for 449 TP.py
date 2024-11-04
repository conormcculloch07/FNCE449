# Import necessary libraries for data handling, financial analysis, and optimization
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the stock tickers for the selected assets in the portfolio
tickers = ["BIR.TO", "POU.TO", "TOU.TO", "CNQ.TO", "IMO.TO"]

# Download adjusted closing price data for each ticker from Yahoo Finance
# The date range is set to capture data from January 1, 2020, to January 1, 2023
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']

# Define ESG scores for each asset on a 0-100 scale
esg_scores = np.array([41.2, 72.1, 40.0, 49.7, 17.6]) 

# Calculate daily returns for each asset
# Use forward and backward fill to handle any missing data points
returns = data.ffill().bfill().pct_change()

# Calculate the mean of the returns and the covariance matrix
# Annualize both by multiplying by 252 (approximate trading days per year)
mean_returns = returns.mean() * 252  # Annualize returns
cov_matrix = returns.cov() * 252     # Annualize covariance

# Define a function to calculate portfolio performance
# This function computes the portfolio return and portfolio volatility (risk)
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

# Objective function for Sharpe Ratio maximization, used in optimization
# Returns the negative Sharpe Ratio to enable maximization via minimization techniques
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    """
    This function calculates the negative Sharpe Ratio for the given portfolio.
    By returning the negative Sharpe Ratio, it allows us to maximize the Sharpe Ratio
    using a minimization function.
    """

    # Calculate portfolio return and volatility using the given weights
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    # Compute the Sharpe Ratio; subtracts risk-free rate from portfolio return,
    # then divides by portfolio volatility (risk)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    # Return the negative Sharpe Ratio (to enable maximization in a minimization context)
    return -sharpe_ratio

# Optimize portfolio for maximum Sharpe Ratio
# Set up constraints and parameters for optimizing the portfolio to maximize the Sharpe Ratio

# Constraint: Ensure that the sum of the portfolio weights equals 1 (fully invested portfolio)
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
# Bounds: Restrict each weight to be between 0 and 1, preventing short-selling and leveraging
bounds = tuple((0, 1) for _ in range(len(tickers)))
# Initial weights: Start with equal allocation across all assets
initial_weights = [1 / len(tickers)] * len(tickers)
# Run the optimization using Sequential Least Squares Programming (SLSQP) method
# The goal is to minimize the negative Sharpe Ratio to maximize the Sharpe Ratio
# Arguments passed include initial weights, mean returns, and covariance matrix
result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Retrieve the optimized asset weights from the result
optimal_weights = result.x
# Calculate the portfolio's expected return and volatility with the optimal weights
optimal_return, optimal_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
# Set the risk-free rate for Sharpe Ratio calculation (assumed to be 4%)
risk_free_rate = 0.04 
# Calculate the Sharpe Ratio for the optimized portfolio using the optimal return and volatility
optimal_sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility

# Print optimal portfolio metrics
print(tickers)
print("Optimal Weights:", np.round(optimal_weights,4))
print("Expected Portfolio Return:", optimal_return)
print("Expected Portfolio Volatility (Risk):", optimal_volatility)
print("Optimal Sharpe Ratio:", optimal_sharpe_ratio)

# Function to plot the Efficient Frontier for a given set of assets
def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=50000, risk_free_rate=0.04):
    # Initialize an array to store portfolio risk, return, and Sharpe Ratio for each simulated portfolio
    results = np.zeros((3, num_portfolios))
    # Generate random portfolios by sampling weights and calculate their performance metrics
    for i in range(num_portfolios):
        # Generate random weights that sum to 1 using the Dirichlet distribution
        weights = np.random.dirichlet(np.ones(len(mean_returns)), size=1).flatten()
        # Calculate the expected return and volatility for the portfolio
        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        # Calculate the Sharpe Ratio for the portfolio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        # Store the portfolio volatility, return, and Sharpe Ratio in the results array
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

# Objective function for maximizing Sharpe Ratio with an ESG penalty factor applied
def negative_sharpe_with_esg_penalty(weights, mean_returns, cov_matrix, esg_scores, esg_penalty_factor=1.5, risk_free_rate=0.04):
    # Calculate portfolio return and volatility based on current weights
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    # Calculate the Sharpe Ratio using the risk-free rate
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Calculate an ESG penalty: penalize low ESG scores more heavily with squared terms and penalty factor
    esg_penalty = np.sum(weights * ((1 - esg_scores / 100) ** 2) * esg_penalty_factor)  # Higher penalty for lower scores
    # Return the negative of the adjusted Sharpe Ratio (to enable maximization through minimization)
    return -(sharpe_ratio - esg_penalty)

# Set constraints and bounds for portfolio optimization
# Constraint: Ensure the sum of portfolio weights equals 1
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
# Bounds: Ensure each weight is between 0 and 1 (no short-selling or leveraging)
bounds = tuple((0, 1) for _ in range(len(tickers)))
# Initial weights: Start with equal allocation across all assets
initial_weights = [1 / len(tickers)] * len(tickers)

# Run the optimization to maximize the Sharpe Ratio, factoring in the ESG penalty
# Pass in ESG penalty factor and other arguments
esg_penalty_factor = 1.5  # Stronger impact of ESG penalty
result = minimize(negative_sharpe_with_esg_penalty, initial_weights, args=(mean_returns, cov_matrix, esg_scores, esg_penalty_factor),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Extract optimal weights and calculate performance metrics
optimal_weights = result.x  # Retrieve the optimized weights for the assets
optimal_return, optimal_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)  # Calculate return and volatility for optimal portfolio
optimal_esg_score = np.dot(optimal_weights, esg_scores)  # Calculate the weighted ESG score of the optimal portfolio
optimal_sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility  # Calculate the Sharpe Ratio

# Display the results
print(tickers)
print("Optimal Weights:", np.round(optimal_weights,4))
print("Expected Portfolio Return:", optimal_return)
print("Expected Portfolio Volatility (Risk):", optimal_volatility)
print("Optimal ESG Score:", optimal_esg_score)
print("Optimal Sharpe Ratio (with Enhanced ESG Penalty):", optimal_sharpe_ratio)

# Function to plot the Efficient Frontier with the ESG-Penalized Portfolio
def plot_efficient_frontier_esg(mean_returns, cov_matrix, esg_scores, num_portfolios=50000, risk_free_rate=0.04, esg_penalty_factor=1.5):
    # Initialize array to store portfolio metrics: volatility, return, Sharpe Ratio, and ESG score
    results = np.zeros((4, num_portfolios))  # Adding ESG score for the fourth result
    # Generate multiple random portfolios to simulate the efficient frontier
    for i in range(num_portfolios):
        # Generate random weights summing to 1 using the Dirichlet distribution
        weights = np.random.dirichlet(np.ones(len(mean_returns)), size=1).flatten()
        # Calculate portfolio return and volatility
        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        # Calculate the Sharpe Ratio for the portfolio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
         # Apply the ESG penalty to adjust the Sharpe Ratio based on ESG scores
        esg_penalty = np.sum(weights * ((1 - esg_scores / 100) ** 2) * esg_penalty_factor)
        adjusted_sharpe = sharpe_ratio - esg_penalty  # Adjusted Sharpe Ratio with ESG penalty
        # Calculate the weighted ESG score for the portfolio
        weighted_esg_score = np.dot(weights, esg_scores)  
        
        # Store the portfolio volatility, return, adjusted Sharpe Ratio, and ESG score
        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = adjusted_sharpe
        results[3, i] = weighted_esg_score  # To track ESG score

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
