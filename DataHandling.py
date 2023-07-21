'''
This module is used to handle data I/O and data preprocessing.
'''

import pandas as pd
import logging
import yfinance as yf
import datetime

# LOGGING CONFIGURATION
format = '%(asctime)s - %(levelname)s - %(message)s - %(name)s'
logging.basicConfig(level = logging.INFO, format=format)

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
    self.close_Prices = self.get_Close_Prices()

  def get_Close_Prices(self):
    ''' Loop through the list of stocks and retrieve the historic data for each stock, storing the data in a dataframe'''
    df = yf.download(self.stock_List, start=self.start_Date, end=self.end_Date, progress=False)['Adj Close']

    return df
  
  def get_Returns(self):
    ''' Calculate returns for a given dataframe of stock prices'''   
    
    # Calculate returns
    df = self.close_Prices.pct_change().fillna(0)  

    return df
  

def main():
    # Input data
    stocks = ['CBA', 'BHP', 'TLS'] 
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)

    # Create StockData object
    stock_data = StockData(start_date, end_date, stocks)
    stock_df = stock_data.get_Returns()
    print(stock_df.head())

if __name__ == '__main__':
    main()