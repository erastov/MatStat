from __future__ import print_function
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot


mean = 0
std = 1
n = 300
samples = np.random.normal(mean, std, size=n)

info_criteria = sm.tsa.stattools.arma_order_select_ic(
                    samples, ic=['aic', 'bic'])

print('AIC' + str(info_criteria.aic_min_order))
print('BIC' + str(info_criteria.bic_min_order))

arma_10 = sm.tsa.ARMA(samples, (1, 0)).fit()
arma_11 = sm.tsa.ARMA(samples, (1, 1)).fit()

print('ARMA(1,0): AIC = %.2f' % arma_10.aic)
print('ARMA(1,1): AIC = %.2f' % arma_11.aic)

arma_10_inpred = arma_10.predict(start=1, end=n)
arma_11_inpred = arma_11.predict(start=1, end=n)

# out-of-sample predict
arma_10_outpred = arma_10.predict(start=1, end=n)
arma_11_outpred = arma_11.predict(start=1, end=n)

def plot_ARMA_results(origdata, pred10in, pred10out, pred11in, pred11out):
    px = [i for i in range(0, n, 1)]
    py1 = origdata
    plt.plot(px, py1, 'b:', label='White noise')

    px_in = [i for i in range(0, n, 1)]
    plt.plot(px_in, pred10in, 'g')
    plt.plot(px_in, pred11in, 'c')

    px_out = [i for i in range(0, n, 1)]
    plt.plot(px_out, pred10out, 'g', label='ARMA(1,0)')
    plt.plot(px_out, pred11out, 'c', label='ARMA(1,1)')

    plt.legend()
    plt.grid(True)


# Residue (mse for train)
arma_10_mse_tr = np.array([r ** 2 for r in arma_10.resid]).mean()
arma_11_mse_tr = np.array([r ** 2 for r in arma_11.resid]).mean()

# Residue (mse for test)
arma_10_mse_te = np.array([pow(samples[i] - arma_10_outpred[i], 2) for i in range(30)]).mean()
arma_11_mse_te = np.array([pow(samples[i] - arma_11_outpred[i], 2) for i in range(30)]).mean()

print('RMSE(1,0): ' + str(pow(arma_10_mse_te / n, 1/2)))
print('RMSE(1,1): ' + str(pow(arma_11_mse_te / n, 1/2)))

print(arma_11.summary())

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(samples, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(samples, lags=40, ax=ax2)

fig2 = plt.figure()
ax3 = fig2.add_subplot(1, 1, 1)
fig2 = plt.plot(samples, linewidth=0.5)

fig3 = plt.figure()
ax4 = fig3.add_subplot(1, 1, 1)
fig3 = plot_ARMA_results(samples, arma_10_inpred, arma_10_outpred, arma_11_inpred, arma_11_outpred)

plt.show()
