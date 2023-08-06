import datetime

# Dates
TRADING_DAYS = 252
END_DATE = datetime.datetime(2018, 12, 31)
START_DATE = END_DATE - datetime.timedelta(days=365)

# Financial Constraints
PORTFOLIO_ASSETS = ['AAPL', 'MSFT', 'AMZN', 'GOOG']
TARGET_RETURN = 0.1
RISK_FREE_RATE = 0.02

# Config Constants
LOGGING_FORMAT = '[%(levelname)s] - %(message)s - [%(module)s] - %(asctime)s'
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
PLOT_DIMENSIONS = (1000, 600)
PLOT_STYLE = 'plotly_white'

# Monte Carlo Optimisation
RANDOM_SEED = 0
NUMBER_OF_SIMULATIONS = 10000