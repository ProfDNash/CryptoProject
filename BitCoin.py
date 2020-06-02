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
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings("ignore")

fileName = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
data = pd.read_csv(fileName)
##keep only desired columns
data = data[['Timestamp','Open','High', 'Low', 'Close']]
##eliminate timestamps with no trades
data = data.dropna()
##convert time from UNIX time to date/time
data['Timestamp'] = pd.to_datetime(data['Timestamp'],unit='s')
##Truncate Timestamp to the Day
data['Timestamp'] = data['Timestamp'].dt.floor('d')
##Eliminate duplicates by keeping only the last entry on each day
data = data.drop_duplicates(subset=['Timestamp'], keep='last')

##Visualize Closing Prices Over Time
closing = data[['Timestamp','Close']].set_index('Timestamp')
plot.plot(closing)
plot.xlabel('Date', fontsize=12)
plot.ylabel('Price in USD', fontsize=12)
plot.title("Closing price distribution of bitcoin", fontsize=15)
plot.show()

##Test closing data for stationarity using Augmented Dicky Fuller Test
##Null hypothesis is that the series is non-stationary
##Cannot apply ARIMA model unless stationarity holds for the series
def testStationarity(X):
    print('Running ADF Test... this may take awhile...')
    result = smt.adfuller(X, autolag='AIC')
    print('Done!')
    print(f'ADF Statistic: {result[0]}')
    print(f'n_lags: {result[2]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

testStationarity(closing)
##Optimal number of lags was 29, so calculate rolling mean with 29 lags for comparison
rollmean = closing.rolling(window=29, center=False).mean()
rollstd = closing.rolling(window=29, center=False).std()
#Plot rolling statistics:
orig = plot.plot(closing, color='blue',label='Original')
mean = plot.plot(rollmean, color='red', label='Rolling Mean')
std = plot.plot(rollstd, color='black', label = 'Rolling Std')
plot.legend(loc='best')
plot.title('Rolling Mean & Standard Deviation')
plot.show(block=False)

input('Press enter to explore the log transform of the series')
##Log Transform the Series and retest for non-stationarity
closingLog = np.log(closing)
testStationarity(closingLog)
##Still non-stationary, but optimal number of lags was 17, so again compare
logrollmean = closingLog.rolling(window=17, center=False).mean()
logrollstd = closingLog.rolling(window=17, center=False).std()
#Plot rolling statistics:
origLog = plot.plot(closingLog, color='blue',label='Log')
meanLog = plot.plot(logrollmean, color='red', label='Rolling Mean of Log')
stdLog = plot.plot(logrollstd, color='black', label = 'Rolling Std of Log')
plot.legend(loc='best')
plot.title('Rolling Mean & Standard Deviation of Log Transform')
plot.show(block=False)

input('Press enter to apply differencing to the log transform')
##Difference the Log Transform and retest
closingLogDiff = closingLog - closingLog.shift()
##Drop NAs
closingLogDiff.dropna(inplace=True)
testStationarity(closingLogDiff)
##It no longer suffers from non-stationarity
##Optimal number of lags was 16, again visualize
logdiffrollmean = closingLogDiff.rolling(window=16, center=False).mean()
logdiffrollstd = closingLogDiff.rolling(window=16, center=False).std()
#Plot rolling statistics:
origLogDiff = plot.plot(closingLogDiff, color='blue',label='Log Diff')
meanLogDiff = plot.plot(logdiffrollmean, color='red', label='Rolling Mean of Log Diff')
stdLogDiff = plot.plot(logdiffrollstd, color='black', label = 'Rolling Std of Log Diff')
plot.legend(loc='best')
plot.title('Rolling Mean & Standard Deviation of Log Transform Once Differenced')
plot.show(block=False)


input('Press enter to visualize an auto-regressive model')
##Apply auto-regressive model first (i.e. follow lag as we assume each value depends on the previous)
model = ARIMA(closingLog, order=(1,1,0))  
results_ARIMA = model.fit(disp=-1)  
plot.plot(closingLogDiff)
plot.plot(results_ARIMA.fittedvalues, color='red')
plot.title('RSS: %.7f'% sum((results_ARIMA.fittedvalues-closingLogDiff.Close)**2))
plot.show()
print('RSS: %.7f'% sum((results_ARIMA.fittedvalues-closingLogDiff.Close)**2))

input('Press enter to visualize a moving average model')
##Apply moving average model next (i.e. use error in previous terms to predict)
model2 = ARIMA(closingLog, order=(0,1,1))  
results_ARIMA2 = model2.fit(disp=-1)  
plot.plot(closingLogDiff)
plot.plot(results_ARIMA2.fittedvalues, color='red')
plot.title('RSS: %.7f'% sum((results_ARIMA2.fittedvalues-closingLogDiff.Close)**2))
plot.show()
print('RSS: %.7f'% sum((results_ARIMA2.fittedvalues-closingLogDiff.Close)**2))

input('Press enter to visualize a full ARIMA model')
##Apply full ARIMA model next (i.e. combination of AR and MA models)
model3 = ARIMA(closingLog, order=(2,1,0))  
results_ARIMA3 = model3.fit(disp=-1)  
plot.plot(closingLogDiff)
plot.plot(results_ARIMA3.fittedvalues, color='red')
plot.title('RSS: %.7f'% sum((results_ARIMA3.fittedvalues-closingLogDiff.Close)**2))
plot.show()
print('RSS: %.7f'% sum((results_ARIMA3.fittedvalues-closingLogDiff.Close)**2))
##Full ARIMA is slightly better in RSS than the other two.

input('Press enter to plot predictions using ARIMA model over last 100 days of data')
# Divide into train and test
train_arima, test_arima = closingLog[0:-100], closingLog[-100:]
history = train_arima
predictions = list()
originals = list()
error_list = list()

print('Printing Predicted vs Expected Values...')
print('\n')
''' We go over each value in the test set and then apply ARIMA model and 
calculate the predicted value. We have the expected value in the test set 
therefore we calculate the error between predicted and expected value'''
for t in range(len(test_arima)):
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit(disp=-1)
    output = model_fit.forecast()
    pred_value = output[0][0]
    #pred_history = test_arima.iloc[[t]]
    #pred_history.Close = pred_value ##create a prediction for the next day to use.
    original_value = test_arima.iloc[[t]]
    history = history.append(original_value)
    pred_value = np.exp(pred_value)
    #history = history.append(pred_history)
    original_value = np.exp(original_value.Close[0])
    # Calculating the error
    error = ((abs(pred_value - original_value)) / original_value) * 100
    error_list.append(error)
    print('predicted = %f,   expected = %f,   error = %f ' % (pred_value, original_value, error), '%')
    predictions.append(float(pred_value))
    originals.append(float(original_value))
    
# After iterating over whole test set the overall mean error is calculated.   
print('\n Mean Error in Predicting Test Case Articles : %f ' % (sum(error_list)/float(len(error_list))), '%')
plot.figure(figsize=(8, 6))
test_day = [t for t in range(len(test_arima))]
labels={'Orginal','Predicted'}
plot.plot(test_day, originals, color = 'orange')
plot.plot(test_day, predictions, color= 'green')
plot.title('Expected Vs Predicted Views Forecasting')
plot.xlabel('Day')
plot.ylabel('Closing Price')
plot.legend(labels)
plot.show()

