
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

csv_data = pd.read_csv('./data.csv',header=0,sep=',')
csv_data.head()

train_data = csv_data['y']
plt.plot(train_data)

ets = ExponentialSmoothing(np.asarray(train_data), trend='add', seasonal='add',
                           seasonal_periods=24)
fit = ets.fit()
fit.predict(start=0, end=len(train_data)-1)
