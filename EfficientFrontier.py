'''
Implement Modern Potfolio Theory (MPT) to find the optimal portfolio and plot the efficient frontier
'''

import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.optimize as sco


from DataHandling import StockData

# 