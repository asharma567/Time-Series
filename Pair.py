
# coding: utf-8

# In[32]:

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
get_ipython().magic(u'pylab inline')


# In[14]:

import datetime
def dateparser(datestring):
    return datetime.datetime.strptime(datestring,'%Y%m%d%H')

time_series = pd.read_json(
'data/logins.json', 
typ='series',
)

time_series_df = pd.read_json(
'data/logins.json', 
typ='frame', 
)

# time_series_df = time_series_df.set_index([0])
s= pd.to_datetime(time_series_df.index,unit='s')
# time_series_df[0].apply(dateparser)

# time_series_df.resample('H')
tstamps = pd.to_datetime(time_series.index)
s
time_series_df['time']= pd.to_datetime(time_series_df[0], unit = 's')
time_series_df = time_series_df.set_index('time')
time_series_df.pop(0)


# In[22]:

pylab.rcParams['figure.figsize']=(15,10)
ts = time_series_df
ts['count'] = 1

hourly_time_series = ts.resample('H', how='sum')
hourly_time_series.plot(style='g', lw=.6)


# In[28]:

hourly_time_series[(hourly_time_series.index >= '3/10/2012') & (hourly_time_series.index < '3/12/2012')].plot();


# In[41]:

# time_series.reindex(index = [], method='ffill')
hourly_time_series = hourly_time_series.dropna()

p_orders = [0,1]
q_orders = [0,1,2,3,4,5]

AICs = []

for p in p_orders:
    for q in q_orders:
        arma_model = sm.tsa.ARMA(hourly_time_series, order=(p,q))
        arma_res = arma_model.fit(trend = 'c', disp=-1)
        AICs.append([p,q,arma_res.aic])
            


# In[45]:

AICs_trans = np.transpose(AICs)
ax = plt.plot(AICs_trans[1,0:6],AICs_trans[2,0:6],label="p=0")
plt.plot(AICs_trans[1,6:],AICs_trans[2,6:],label="p=1")
xlabel("qvalue")
ylabel("AICvalue")
plt.legend();


# In[52]:

arma_model = sm.tsa.ARMA(hourly_time_series, order=(1,2))
arma_res = arma_model.fit(trend='c', disp=1)
arma_pred = arma_res.predict()
plt.figure(figsize=(20, 5))
plot(arma_pred, 'b', lw=3, label='ARMA')
plot(hourly_time_series, 'r', lw=1, label='logins')
xlabel('hour')
ylabel('number of logins');
legend();
arma_res


# In[53]:

print 'p-values'
arma_res.pvalues


# In[55]:

arma_res.resid.plot();


# In[61]:

cycle, trend = sm.tsa.filters.hpfilter(hourly_time_series, 16000000)

decomposition = pd.DataFrame(hourly_time_series)

decomposition['cycle'] = cycle
decomposition['trend'] = trend

decomposition.plot();


# In[63]:

decomposition.trend.plot(label = 'hourly logins trend')
ylabel('number of logins')
plt.legend(loc=2);


# In[68]:

daily_time_series = hourly_time_series.resample('D', how='sum')
daily_time_series.cycle.plot(label='daily cycle')
plt.legend(loc=2);


# In[ ]:



