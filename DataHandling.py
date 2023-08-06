'''
This module is used to handle data I/O and data preprocessing.
'''

import numpy as np
import pandas as pd
import logging
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.express as px

# LOGGING CONFIGURATION
format = '[%(levelname)s] - %(message)s - [%(module)s] - %(asctime)s'
date_format = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level = logging.INFO, format=format, datefmt=date_format)

class StockData:
    ''' Class for allowing the user to retrieve historic stock data across a given time period
    Args:
    start_date = The start date to retrieve stock data from in the format YYYY-MM-DD
    end_date = The end date to retrieve stock data from in the format YYYY-MM-DD
    '''
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime, stock_list: list[str]):
        self.start_Date = start_date
        self.end_Date = end_date
        self.stock_List = stock_list
        self.returns = self.calculate_Log_Returns()
        self.mean_Returns, self.volatility_Vector, self.correlation_Matrix, self.covariance_Matrix = self.get_Matrices_From_Returns()
    
    def calculate_Log_Returns(self):
        logging.info(f'Getting close prices for stocks: {self.stock_List} and calculating returns')
        # Get close prices
        df = yf.download(self.stock_List, start=self.start_Date, end=self.end_Date, progress=False)['Adj Close']

        return np.log(df) - np.log(df.shift(1))

    def get_Matrices_From_Returns(self):
        ''' Calculates the mean returns and standard deviation vectors and correlation and covariance matrices for a given list of stocks between a given time period
        Args:
            None
        Returns:
            mean returns vector (pd.Series) = A vector of mean returns for each stock
            volatility vector (pd.Series) = A vector of standard deviations for each stock
            correlation_matrix (pd.DataFrame) = A correlation matrix of returns for each stock
            covariance_matrix (pd.DataFrame) = A covariance matrix of returns for each stock
        '''
        # Mean returns vector
        mean_returns = self.returns.mean().to_numpy()

        # Standard deviation vector
        std_deviation = self.returns.std().to_numpy()

        # Correlation matrix
        correlation_matrix = self.returns.corr().to_numpy()

        # Covariance matrix
        covariance_matrix = self.returns.cov().to_numpy()

        return mean_returns, std_deviation, correlation_matrix, covariance_matrix

    def plot_Returns(self):
        ''' Plots the returns of the portfolio
        Args:
            None
        Returns:
            None
        '''
        fig = px.line(self.returns, x=self.returns.index, y=self.returns.columns, title='Historical Daily Returns of Portfolio', labels={'value':'Return (%)', 'variable':'Stock', 'index':'Date'}, template='plotly_white')

        fig.show()
    
    def plot_Correlation(self):
        ''' Plots the correlation of the portfolio
        Args:
            None
        Returns:
            None
        '''
        fig = px.imshow(self.returns.corr(), title='Correlation Matrix of Portfolio', labels={'x':'Stock', 'y':'Stock', 'color':'Correlation'}, color_continuous_scale='sunset', template='plotly_white')

        fig.show()


def main():
    # Input data
    stocks = ['CBA', 'BHP', 'TLS', 'MSFT', 'GOOG']
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)

    # Create StockData object
    stock_data = StockData(start_date, end_date, stocks)
    

if __name__ == '__main__':
    main()