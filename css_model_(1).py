# -*- coding: utf-8 -*-
"""CSS_model (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HIeKGIA3nv9C5TLuelVWQCtUJhT9Ajk9

# Importing a library that is not in Colaboratory

To import a library that's not in Colaboratory by default, you can use `!pip install` or `!apt-get install`.
"""

!pip install matplotlib-venn

!apt-get -qq install -y libfluidsynth1

"""# Install 7zip reader [libarchive](https://pypi.python.org/pypi/libarchive) """

# https://pypi.python.org/pypi/libarchive
!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive

"""# Install GraphViz & [PyDot](https://pypi.python.org/pypi/pydot)"""

# https://pypi.python.org/pypi/pydot
!apt-get -qq install -y graphviz && pip install pydot
import pydot

"""# Install [cartopy](http://scitools.org.uk/cartopy/docs/latest/)"""

!pip install cartopy
import cartopy



# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import pandas.util.testing as tm
import statsmodels.api as sm
import matplotlib.pyplot as plt
# %matplotlib inline

#Analyzing Data
df=pd.read_excel('Book1.xlsx')
df.head()

df.columns=['Date','PSU Data']
df.head()

df['Date']=pd.to_datetime(df['Date'])
df.head()

df.set_index('Date',inplace=True)
df.head()

#Visualization of Data
df.plot()

df['PSU Data'].rolling(12).mean().plot(label='12 SMA',figsize=(16,8))
df['PSU Data'].rolling(12).std().plot(label='12 STD')
df['PSU Data'].plot()
plt.legend()

#Decomposition of Time Series Data to its trend,seasonality and residual components
from statsmodels.tsa.seasonal import seasonal_decompose  
decomp = seasonal_decompose(df['PSU Data'])
fig=decomp.plot()
fig.set_size_inches(14,7)

#converting the data into stationary form
#dicky fuller test
from statsmodels.tsa.stattools import adfuller 
fuller_test = adfuller (df['PSU Data'])
fuller_test

#Test p value
def test_p_value(data): 
   fuller_test = adfuller(data)
print( ' p-value: ',fuller_test[1])
if fuller_test[1] <= 0.05:
  print( 'Reject null hypothesis,data is stationary')
else:
    print('Do not reject null hypothesis, data is not stationary')
test_p_value(df['PSU Data'])

#First order difference
test_p_value(df['PSU Data'])
df['First_diff'] = df['PSU Data']-df['PSU Data'].shift(1)
df['First_diff'].plot()

test_p_value(df['First_diff'].dropna())

#Second order difference 
df['Second_diff']=df['First_diff'] - df['First_diff'].shift(1)
df['Second_diff'].plot()

test_p_value(df['Second_diff'].dropna())

#Seasonal difference
df['Seasonal_diff']=df['PSU Data'] - df['PSU Data'].shift(12)
df['Seasonal_diff'].plot()

test_p_value(df['Seasonal_diff'].dropna())

#plotting the ACF and PACF plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
first_diff = plot_acf(df['First_diff'].dropna())

sec_diff = plot_pacf(df['Second_diff'].dropna())

p1 = plot_acf(df['Seasonal_diff'].dropna())
p2 = plot_pacf(df['Seasonal_diff'].dropna())

#Constructing the ARIMA Model
df.index.freq = 'D'
from statsmodels.tsa.arima_model import ARIMA
model = sm.tsa.statespace.SARIMAX(df['PSU Data'],order = (0,1,0),seasonal_order = (1,1,1,12))

#Fit the model
results = model.fit()
results.summary()

#Know about the residuals value or error
results.resid

results.resid.plot()

results.resid.plot(kind='kde')

#predicting or forecasting
df['prediction'] = results.predict()
df[['PSU Data','prediction']].plot(figsize = (12,8))

from pandas.tseries.offsets import DateOffset
extra_dates = [df.index[-1] + DateOffset(months = m) for m in range (1,24)]

extra_dates

#now another datafame is created with these extra future date values
forecast_df = pd.DataFrame(index=extra_dates,columns=df.columns)
forecast_df.head()

#now the origional dataframe and this forecast dataframe is concatenated into a single one for forecasting
final_df = pd.concat([df,forecast_df])

#we can predict the values for end data points
final_df['prediction'] = results.predict(start=1460, end=1660)
final_df[['PSU Data','prediction']].plot(figsize=(12,8))