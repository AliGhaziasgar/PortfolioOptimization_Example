'''
Implement Mean Variance Portfolio Optimisation using Lagrange Multipliers to find the optimal weights for a portfolio of stocks
'''

import logging
import numpy as np
from datetime import datetime
import pandas as pd
import scipy.stats as ss
import scipy.optimize as sco
import datetime
import tabulate
import plotly.express as px

import Constants
from DataHandling import StockData


# LOGGING CONFIGURATION
logging.basicConfig(level = logging.INFO, format=Constants.LOGGING_FORMAT, datefmt=Constants.LOGGING_DATE_FORMAT)


class PortfolioOptimisation:
    def __init__(self, stocks: list[str], mean_returns_vector: np.ndarray, volatility_vector: np.ndarray, covariance_matrix: np.ndarray, risk_free_rate: float = 0.0):
        self.stocks = stocks
        self.initial_Weights = np.array([1 / len(stocks)] * len(stocks))
        self.mean_Returns = mean_returns_vector
        self.volatility_vector = volatility_vector
        self.covariance_Matrix = covariance_matrix
        self.risk_Free_Rate = risk_free_rate
    
    def annualised_Volatility(self, weights: np.ndarray):
        return np.sqrt(np.dot(weights.T, np.dot(self.covariance_Matrix, weights))) * np.sqrt(Constants.TRADING_DAYS)

    def annualised_Return(self, weights: np.ndarray):
        return np.sum(self.mean_Returns * weights) * Constants.TRADING_DAYS
    
    def max_Sharpe_Ratio(self, weights: np.ndarray):
        return self.annualised_Return(weights) / self.annualised_Volatility(weights)

    def negative_Max_Sharpe_Ratio(self, weights: np.ndarray):
        return -self.max_Sharpe_Ratio(weights)

    def calculate_Portfolio_Performance(self, weights: np.ndarray):
        annualised_return = self.annualised_Return(weights)
        annualised_volatility = self.annualised_Volatility(weights)
        sharpe_ratio = self.max_Sharpe_Ratio(weights)

        return pd.DataFrame({
            'Return': annualised_return,
            'Volatility': annualised_volatility,
            'Sharpe Ratio': sharpe_ratio
        }, index=[0])
    
    def monte_Carlo_Portfolio_Simulation(self):
        '''Monte Carlo simulation to generate random portfolio weight and calculate the expected portfolio return, variance and sharpe ratio for every simulated allocation. '''
        # Random seed for reproducibility
        np.random.seed(Constants.RANDOM_SEED)

        # Simulate all portfolios
        portfolio_weights = np.random.random((Constants.NUMBER_OF_SIMULATIONS, len(self.stocks)))

        # Normalise the weights to sum to 1
        portfolio_weights /= np.sum(portfolio_weights, axis=1)[:, np.newaxis]
        print(portfolio_weights)

        # Annualised portfolio returns, μ = w'.μ
        portfolio_returns = np.dot(portfolio_weights, self.mean_Returns) * Constants.TRADING_DAYS

        # Annualised portfolio volatility, σ = sqrt(w'.Σ.w)
        portfolio_volatility = np.sqrt(np.sum((np.dot(portfolio_weights, self.covariance_Matrix * 260) * portfolio_weights), axis=1))

        # Sharpe Ratio, SR = μ/σ
        sharpe_ratio = portfolio_returns / portfolio_volatility

        # Create a dataframe to store the portfolio weights, returns, volatility and sharpe ratio
        simulation_df = pd.DataFrame({
            'Portfolio Return': portfolio_returns,
            'Portfolio Volatility': portfolio_volatility,
            'Portfolio Sharpe Ratio': sharpe_ratio
        })
    
        for i, stock in enumerate(self.stocks):
            simulation_df[stock + ' Weight'] = portfolio_weights[:, i]
        
        simulation_df = simulation_df.round(4)

        # Get the max sharpe ratio portfolio
        max_sharpe_ratio_portfolio = simulation_df.iloc[simulation_df['Portfolio Sharpe Ratio'].idxmax()]

        return simulation_df, max_sharpe_ratio_portfolio
    
    def plot_Monte_Carlo_Simulation(self):
        ''' Plot the Monte Carlo simulation of the portfolio weights, returns, volatility and sharpe ratio '''

        # Run the Monte Carlo simulation
        simulation_df, max_sharpe_ratio_portfolio = self.monte_Carlo_Portfolio_Simulation()

        fig = px.scatter(
            simulation_df, x='Portfolio Volatility', y='Portfolio Return', color='Portfolio Sharpe Ratio',
            labels={'Portfolio Volatility': 'Volatility', 'Portfolio Return': 'Return', 'Portfolio Sharpe Ratio': 'Sharpe Ratio'},
            color_continuous_scale=px.colors.sequential.Teal,
            title='Monte Carlo Simulated Portfolios'
            ).update_traces(mode='markers', marker=dict(symbol='circle', size=5))

        # Plot the max sharpe portfolio
        fig.add_scatter(
            mode='markers',
            x=[max_sharpe_ratio_portfolio['Portfolio Volatility']],
            y=[max_sharpe_ratio_portfolio['Portfolio Return']],
            marker=dict(color='red', size=6, symbol='circle', line=dict(color='red', width=1)),
            name='Max Sharpe Ratio Portfolio'
        ).update_layout(showlegend=False)

        # Show spikes
        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)
        fig.show()
    
    def lagrangian_Multipliers_Optimisation(self, target_return: float, covariance_matrix: np.ndarray, mean_returns_vector: np.ndarray):
        ''' Calculate the Markowitz Optimal Portfolio weights using Lagrangian Multipliers closed form solution
        Args:
            target_return (float) = The minimum or target return required for the portfolio
            covariance_matrix (np.array) = The covariance matrix of the portfolio
            mean_returns_vector (np.array) = The mean returns vector of the portfolio
        Returns:
            optimal_weights (np.array) = The optimal weights for the portfolio
        '''
        # Initiliase the 1s vector
        ones_vector = np.ones(mean_returns_vector.shape[0])

        # Calculate the inverse of the covariance matrix (Σ^-1)
        inverse_covariance = np.linalg.inv(covariance_matrix)

        # Get scalar values to solve the quadratic equation for the two constaint equations -> 1. Sum of weights = 1 and 2. Minimum return
        # A = 1^.(Σ^-1).1^
        A = ones_vector.dot(inverse_covariance.dot(ones_vector))

        # B = 1^.(Σ^-1).μ
        B = ones_vector.dot(inverse_covariance.dot(mean_returns_vector))

        # C = μ'.(Σ^-1).μ
        C = mean_returns_vector.T.dot(inverse_covariance.dot(mean_returns_vector))

        # Use the quadratic formula to solve for the two lagrangian multipliers
        # λ = (A*target_return - B) / (A*C - B^2)
        lagrangian_lambda = (A*target_return - B) / (A*C - B**2)

        # γ = (C - B*target_return) / (A*C - B^2)
        lagrangian_gamma = (C - B*target_return) / (A*C - B**2)

        # Calculate the optimal weights, w* for the portfolio
        # w* = (Σ^-1).(λ.μ + γ.1^)
        optimal_weights = inverse_covariance.dot(lagrangian_lambda * mean_returns_vector + lagrangian_gamma * ones_vector) / 100

        # print(f'A: {A}')
        # print(f'B: {B}')
        # print(f'C: {C} \n')
        # print(f'λ: {lagrangian_lambda}')
        # print(f'γ: {lagrangian_gamma} \n')

        return optimal_weights


def calculate_Covariance_Matrix(volatility_vector: np.ndarray, correlation_matrix: np.ndarray):
    ''' Calculates the covariance matrix from the standard deviation vector and correlation matrix'''
    return np.outer(volatility_vector, volatility_vector) * correlation_matrix

def main():
    # INITIALISE STOCK DATA #
    stock_data = StockData(Constants.START_DATE, Constants.END_DATE, Constants.PORTFOLIO_ASSETS)
    print(stock_data.mean_Returns)
    # stock_data.plot_Returns()
    # stock_data.plot_Correlation()

    # INITIALISE PORTFOLIO OPTIMISATION #
    optimisation = PortfolioOptimisation(stock_data.stock_List, stock_data.mean_Returns, stock_data.volatility_Vector, stock_data.covariance_Matrix)

    # CALCULATE PERFORMANCE OF INITIAL PORTFOLIO #
    # intial_portfolio_df = optimisation.calculate_Portfolio_Performance(optimisation.initial_Weights)
    # print(f'Initial Portfolio: \n', intial_portfolio_df)

    # OPTIMISE THE PORTFOLIO #
    # lagrangian_weights = optimisation.lagrangian_Multipliers_Optimisation(Constants.TARGET_RETURN, stock_data.covariance_Matrix, stock_data.mean_Returns)

    # # Lagrangian portfolio performance
    # lagrangian_portfolio_df = optimisation.calculate_Portfolio_Performance(lagrangian_weights)
    # print(f'Lagrangian Portfolio: \n', lagrangian_portfolio_df)

    #  Monte Carlo portfolio performance
    optimisation.plot_Monte_Carlo_Simulation()
    

if __name__ == '__main__':
    main()