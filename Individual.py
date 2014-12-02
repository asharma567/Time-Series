
# coding: utf-8

# In[1]:

#read the file in
#data/total-volume.csv
#data/tag-volume.csv
import pandas as pd
import datetime
get_ipython().magic(u'pylab inline')

def dateparser(datestring):
    return datetime.datetime.strptime(datestring,'%Y%m%d%H')

df_vol = pd.read_csv(
'data/total-volume.csv', 
parse_dates={'dater':[0]}, 
date_parser=dateparser,
index_col='dater'
)
df_vol


# In[2]:

df_vol.plot()


# In[3]:

df_vol.describe()


# In[4]:

df_vol.ix[0]


# In[5]:

#extract the first 2 months 
first_2mos_df = df_vol.ix['2012-01':'2012-02']

#add a marker for each data point
first_2mos_df.plot(marker = '*', markerfacecolor= 'r', markersize=10)

#compute cumsum()
first_2mos_df.cumsum().plot()
df_vol.cumsum().plot()


# In[6]:

df_hour = df_vol.resample('h', how='mean')
df_day = df_vol.resample('D', how='mean')
df_mon = df_vol.resample('m', how='mean')

df_hour.plot()
df_day.plot()
df_mon.plot()


# In[7]:

#What is the count, mean, std, and quantiles?

print df_vol.describe().ix['25%']
df_vol.describe()


# In[8]:

print df_vol.count(), df_vol.max(), df_vol.min(), df_vol.std(), df_vol.mean()
print df_vol.quantile([.25,.75])


# In[9]:

pd.rolling_mean(df_vol, 7).plot(color='c')
pd.rolling_mean(df_vol, 30).plot(style='k--')
df_vol.plot(style='k')


# In[14]:

# import matplotlib.pyplot as plt
import seaborn as sns
pylab.rcParams['figure.figsize'] = (16.0, 10.0)

pd.rolling_mean(df_day, 7).plot(color='c', label= '7-day')
pd.rolling_mean(df_day, 30).plot(style='k--', label= '30-day')
df_day.plot(style='k', label= 'raw')

plt.tight_layout()


# In[16]:

pd.rolling_median(df_vol, 60).plot()


# In[26]:

df_7_rolling = pd.rolling_mean(df_day, 7)
df_7_rolling.index = df_7_rolling.index - pd.offsets.Day(3.5)
df_7_rolling.plot()
print pd.rolling_mean(df_day, 7)
df_7_rolling


# In[28]:

pd.ewma(df_day, span=30).plot(label='EWMA 30')


# In[35]:

#what is seasonality?
#what is cyclical?
first_400_tweets_df = df_vol[:400]


# In[38]:

#what does a log transformation do?
first_400_tweets_df_logged = log(first_400_tweets_df)
first_400_tweets_df_logged.plot()

# Are you able to find any seasonality within the dataset?
# yes it's cyclical very day-over-day i.e. intraday


# In[45]:

# Apply a Hodrick-Prescott filter to the data and decompose the cycle and trend elements. 
# Plot the decomposition.

import statsmodels as sm

tsa = sm.tsa
tweet_cycle, tweet_trend = tsa.filters.hpfilter(first_400_tweets_df, 1600)

tweet_decomp = pd.DataFrame(first_400_tweets_df)
tweet_decomp['cycle'] = tweet_cycle
tweet_decomp['trend'] = tweet_trend

Create a stationary distribution by removing the trend from the raw data, do you detect any seasonality? Smooth the data and use .describe() to interpret your results.tweet_decomp.plot()


# In[53]:

# Create a stationary distribution by removing the trend from the raw data, do you detect any seasonality? 
# Smooth the data and use .describe() to interpret your results.
pd.ewma(tweet_decomp['cycle'],span=30).plot()
pd.ewma(tweet_decomp['cycle'],span=30).describe()


# In[ ]:

# Another approach to detect a trend is to fit a polynomial. 
# Using numpy's polyfit try to fit various polynomials to the 
# daily count data and decompose the seasonality and cyclical components, 
# if observed.

#look @ Uber take home for this

