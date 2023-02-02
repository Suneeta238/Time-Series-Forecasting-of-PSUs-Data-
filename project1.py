import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
#%matplotlib inline

#Analyzing Data
df=pd.read_excel(r'C:\Users\Sunita-pc\Desktop\mtech material\DSS\DSS Model project\Book1.xlsx')
df.head()
print(df.head())

df.columns=['Date','PSU Data']
df.head()
print(df.head())

df['Date']=pd.to_datetime(df['Date'])
df.head()

df.set_index('Date',inplace=True)
df.head()
print(df.head())

#visualization of data
df.plot()
plt.title("Origional Graph")
plt.show()


df['PSU Data'].rolling(12).mean().plot(label='12 SMA',figsize=(16,8))
df['PSU Data'].rolling(12).std().plot(label='12 STD')
df['PSU Data'].plot()
plt.legend()
plt.show()

#Decomposition of the time series data into its trend,seasonality and residual components
from statsmodels.tsa.seasonal import seasonal_decompose  
decomp = seasonal_decompose(df['PSU Data'])
fig=decomp.plot()
fig.set_size_inches(14,7)
plt.show()

#ckeck the data is stationary form or not by Dicky fuller test
#Dicky fuller test
from statsmodels.tsa.stattools import adfuller 
fuller_test = adfuller (df['PSU Data'])
print(fuller_test)           

#Test p value
def test_p_value(data): 
   fuller_test = adfuller(data)
print( ' p-value: ',fuller_test[1])
if fuller_test[1] <= 0.05:
  print( 'Reject null hypothesis,data is stationary')
else:
    print('Do not reject null hypothesis, data is not stationary')

#First order difference
df['First_diff'] = df['PSU Data']-df['PSU Data'].shift(1)
df['First_diff'].plot()
plt.title("First order Difference")
plt.show()
test_p_value(df['First_diff'].dropna())
result= adfuller(df['First_diff'].dropna())
print('p-value:', result[1])
if result[1] <= 0.05:
  print( 'Reject null hypothesis,data is stationary')
else:
    print('Do not reject null hypothesis, data is not stationary')

#Second order difference
df['Second_diff'] = df['First_diff'] - df['First_diff'].shift(1)
df['Second_diff'].plot()
plt.title("Second order Difference")
plt.show()
test_p_value(df['Second_diff'].dropna())
result= adfuller(df['Second_diff'].dropna())
print('p-value:',result[1])
if result[1] <= 0.05:
  print( 'Reject null hypothesis,data is stationary')
else:
    print('Do not reject null hypothesis, data is not stationary')

#Seasonal Difference
df['Seasonal_diff']=df['PSU Data'] - df['PSU Data'].shift(12)
df['Seasonal_diff'].plot()
plt.title("Seasonal Difference")
plt.show()
test_p_value(df['Seasonal_diff'].dropna())
result= adfuller(df['Seasonal_diff'].dropna())
print('p-value :',result[1])
if result[1] <= 0.05:
  print( 'Reject null hypothesis,data is stationary')
else:
    print('Do not reject null hypothesis, data is not stationary')

#plotting the ACF and PACF plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
first_diff = plot_acf(df['First_diff'].dropna())
plt.title("First Difference ACF")

second_diff = plot_pacf(df['Second_diff'].dropna())
#plt.title("Second Difference PACF")

p1 = plot_acf(df['Seasonal_diff'].dropna())
plt.title("Autocorrelation Seasonal Difference")
plt.show()
p2 = plot_pacf(df['Seasonal_diff'].dropna())
plt.title("Partial Autocorrelation Seasonal Difference")
plt.show()

#Constructing the ARIMA Model
df.index.freq = 'D'
from statsmodels.tsa.arima_model import ARIMA
model = sm.tsa.statespace.SARIMAX(df['PSU Data'],order = (0,1,0),seasonal_order = (1,1,1,12))

#Fit the model
results = model.fit()
print(results.summary())

#know about the residuals value or error
print(results.resid)
results.resid.plot()
plt.show()
results.resid.plot(kind='kde')

#predicting or forecasting
df['prediction'] = results.predict()
df[['PSU Data','prediction']].plot(figsize = (12,8))
plt.title("Forecasting")
plt.show()

from pandas.tseries.offsets import DateOffset
extra_dates = [df.index[-1] + DateOffset(months = m) for m in range (1,24)]

print(extra_dates)

#now another dataframe is created with these extra future date values
forecast_df = pd.DataFrame(index=extra_dates,columns=df.columns)
forecast_df.head()
print(forecast_df.head())

#the origional dataframe and this forecast dataframe is concatenated into a single one for forecasting
final_df = pd.concat([df,forecast_df])

#we can predict the values for end data points
final_df['prediction'] = results.predict(start=1460, end=1660)
final_df[['PSU Data','prediction']].plot(figsize=(12,8))
plt.show()
