"""
Created on Tues June 2 11:10:09 2020

@author: David A. Nash
"""
import numpy as np
import pandas as pd
import pylab as p
import matplotlib.pyplot as plot
from collections import Counter
import re
##additional packages for prediction of time-series data
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

fileName = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
train = pd.read_csv(fileName)
##keep only desired columns
train = train[['Timestamp','Open','High', 'Low', 'Close']]
##eliminate timestamps with no trades
train = train.dropna()
##convert time from UNIX time to date/time
train['Timestamp']= pd.to_datetime(train['Timestamp'],unit='s')


