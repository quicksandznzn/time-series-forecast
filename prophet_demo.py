import numpy as np
import pandas as pd
from prophet import Prophet

csv_data = pd.read_csv('data.csv', header=0, sep=',')
csv_data.head()
# model = Prophet(weekly_seasonality=False,growth='flat')
model = Prophet(weekly_seasonality=False,changepoint_range=0.5)
# model = Prophet(weekly_seasonality=False,changepoints=['2021-10-11 19:00:00','2021-10-11 20:00:00','2021-10-11 21:00:00'])
# changepoint_range 突变点百分比 n_changepoints=25 变点个数 changepoint_prior_scale 突变点增长率的分布
# changepoints=None(能否设置每天小时时段？)
# changepoint_range 太小容易过于拟合
# seasonality_mode='multiplicative' ,interval_width=0.8
# changepoint_range
# yearly_seasonality/
# help(Prophet)
model.fit(csv_data)
future = model.make_future_dataframe(periods=5, freq='H')
forecast = model.predict(future)
forecast_yhat = forecast[['ds', 'yhat','yhat_lower','yhat_upper']].tail(20)
# rename(columns={'yhat':'y'})

print(forecast_yhat)
# 05  115  	117(-2)	143(-28)	335 	34%
# 06  194  	206(-12)	198(-4)	502 	38%
# 07  252  	323(-71)	297(-45)	807 	31%
# 08  443  	422(21)	412(31)	1221	36%
# 09  410  	557(-147)	496(-86)	1358	30%
# 10  497  	538(-41)	525(-28)	1639	30%
# 11  538  	586(-48)	610(-72)	1618	33%
# 12  580  	653(-73)	712(-132)	1612	35%
# 13  529  	606(-77)	571(-42)	1561	33%
# 14  453  	531(-78)	594(-141)	1204	37%
# 15  549
# model.plot(forecast).show()
