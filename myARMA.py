import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import statsmodels
from statsmodels.tsa.arima_model import ARIMA

n = 1000
y = [0]*n

y[0] = 0
noise = np.random.normal(0, 1, size=n)
y = [-0.9*y[i-1] + 0.5*noise[i-1] + noise[i] for i in range(1, n-1)]

arma_11 = sm.tsa.ARMA(y, (1, 1)).fit()
print(arma_11.summary())

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
fig1 = plt.plot(y, linewidth=0.5)
plt.show()
