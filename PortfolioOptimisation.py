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

from DataHandling import StockData

# LOGGING CONFIGURATION
format = '[%(levelname)s] - %(message)s - [%(module)s] - %(asctime)s'
date_format = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level = logging.INFO, format=format, datefmt=date_format)


class PortfolioOptimisation:
    def __init__(self, stocks: list[str], mean_returns_vector: np.ndarray, std_deviation_matrix: np.ndarray, covariance_matrix: np.ndarray, risk_free_rate: float = 0.0):
        self.stocks = stocks
        self.returns_Vector = mean_returns_vector
        self.std_Deviation_Matrix = std_deviation_matrix
        self.covariance_Matrix = covariance_matrix
        self.risk_Free_Rate = risk_free_rate

    def calculate_Portfolio_Performance(self, weights: np.ndarray):
        annualised_return = np.sum(self.returns_Vector * weights) * 252
        annualised_std = np.sqrt(np.dot(weights.T, np.dot(self.covariance_Matrix, weights))) * np.sqrt(252)

        # Weights and stock dictionary
        formated_weights = [f'{weight:.2%}' for weight in weights]
        portfolio = dict(zip(self.stocks, formated_weights))

        print(f'Portfolio return: {annualised_return:.2%}')
        print(f'Portoflio standard deviation: {annualised_std:.2%}')
        
        # Print the portfolio using tabulate
        print(tabulate.tabulate(portfolio.items(), headers=['Stock', 'Weight'], tablefmt='fancy_grid')) 

        return annualised_return, annualised_std, portfolio
    
    def lagrangian_Multipliers_Optimisation(self, min_return: float, covariance_matrix: np.ndarray, mean_returns_vector: np.ndarray):
        ''' Calculate the Markowitz Optimal Portfolio weights using Lagrangian Multipliers closed form solution
        Args:
            min_return (float) = The minimum or 'target' return required for the portfolio
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
        # λ = (A*min_return - B) / (A*C - B^2)
        lagrangian_lambda = (A*min_return - B) / (A*C - B**2)

        # γ = (C - B*min_return) / (A*C - B^2)
        lagrangian_gamma = (C - B*min_return) / (A*C - B**2)

        # Calculate the optimal weights, w* for the portfolio
        # w* = (Σ^-1).(λ.μ + γ.1^)
        optimal_weights = inverse_covariance.dot(lagrangian_lambda * mean_returns_vector + lagrangian_gamma * ones_vector)

        print(f'A: {A}')
        print(f'B: {B}')
        print(f'C: {C} \n')
        print(f'λ: {lagrangian_lambda}')
        print(f'γ: {lagrangian_gamma} \n')

        formatted_weights = [f'{weight:.2%}' for weight in optimal_weights]
        portfolio = dict(zip(self.stocks, formatted_weights))

        print(tabulate.tabulate(portfolio.items(), headers=['Stock', 'Weight'], tablefmt='fancy_grid')) 

        return optimal_weights
        

def calculate_Covariance_Matrix(std_deviation_vector: np.ndarray, correlation_matrix: np.ndarray):
    ''' Calculates the covariance matrix from the standard deviation vector and correlation matrix'''
    return np.outer(std_deviation_vector, std_deviation_vector) * correlation_matrix

def main():
    # INPUT DATA #
    # Get the start and end dates for the stock data and list of stocks
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    stocks = ['CBA', 'BHP', 'TLS', 'MSFT', 'GOOG']

    # TEST DATA #
    test_stocks = ['A', 'B', 'C', 'D']
    test_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    test_correlation = np.array([[1.0, 0.0, 0.0, 0.0], 
                                 [0.0, 1.0, 0.0, 0.0], 
                                 [0.0, 0.0, 1.0, 0.0], 
                                 [0.0, 0.0, 0.0, 1.0]])
    test_mean_returns = np.array([0.08, 0.1, 0.1, 0.14])
    test_std_deviation = np.array([0.12, 0.12, 0.15, 0.2])

    # Calculate covariance matrix for test data
    test_covariance = calculate_Covariance_Matrix(test_std_deviation, test_correlation)
    
    # INITIALISE STOCK DATA #
    # stock_data = StockData(start_date, end_date, stocks)
    # stock_data.plot_Returns()
    # stock_data.plot_Correlation()

    # INITIALISE PORTFOLIO OPTIMISATION #
    test_optimisation = PortfolioOptimisation(test_stocks, test_mean_returns, test_std_deviation, test_covariance)
    optimal_weights = test_optimisation.lagrangian_Multipliers_Optimisation(0.1, test_covariance, test_mean_returns)
    # optimisation = PortfolioOptimisation(stock_data.stock_List, stock_data.returns_Vector, stock_data.std_Deviation_Vector, stock_data.covariance_Matrix)

    # CALCULATE PORTFOLIO PERFORMANCE #
    # annualised_return, annualised_std, weights = optimisation.calculate_Portfolio_Performance(test_weights)

if __name__ == '__main__':
    main()