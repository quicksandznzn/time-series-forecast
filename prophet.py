
import numpy as np
import pandas as pd
from prophet import Prophet

csv_data = pd.read_csv('./data.csv',header=0,sep=',')
csv_data.head()

model = Prophet(weekly_seasonality=False)
# seasonality_mode='multiplicative' ,interval_width=0.8
# yearly_seasonality/
# help(Prophet)
model.fit(csv_data)
future = model.make_future_dataframe(periods=24,freq='H')
forecast = model.predict(future)
forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']].tail(10)

model.plot(forecast)