# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:02:12 2026

@author: Pc_Lenovo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import pareto
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde as kde
from scipy.stats import jarque_bera
import warnings
from scipy.stats import t as student_t

warnings.filterwarnings("ignore", category=UserWarning)

# Inseriscilo subito dopo gli import

def autocorrelation(series, max_lag):
    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            acf.append(np.corrcoef(series[:-lag], series[lag:])[0,1])
    return np.array(acf)


allprices = pd.read_csv(r'C:\Users\Pc_Lenovo\Desktop\TTF\ttf_forward1.csv')
business_days = pd.bdate_range(start='2018-01-02', periods=len(allprices), freq='B')
allprices['Date'] = business_days

#Insert here the preferred period

chunk = allprices[allprices['Date'].between('2018-01-02', '2025-09-01')]

log_prices = np.log(chunk['Prices'].values)
log_ret = np.diff(np.log(chunk['Prices'].values))
mean_price = chunk['Prices'].mean()
median_price = chunk['Prices'].median()
std_price  = chunk['Prices'].std()
max_price= chunk['Prices'].max()
min_price  = chunk['Prices'].min()
variance = chunk['Prices'].var()
skew_price = chunk['Prices'].skew()
kurt_price = chunk['Prices'].kurtosis()
mean_ret = log_ret.mean() * 252
std_ret = log_ret.std() * np.sqrt(252)    #annualized
std_ret_daily=log_ret.std()
skew_ret = pd.Series(log_ret).skew()
kurt_ret = pd.Series(log_ret).kurtosis()
kurt_lognormal = np.exp(4*std_ret**2) + 2*np.exp(3*std_ret**2) + 3*np.exp(2*std_ret**2) - 6   #expression of kurtosis for log-normal distrib
kurt_log_abs = np.exp(4*std_ret_daily**2) + 2*np.exp(3*std_ret_daily**2) + 3*np.exp(2*std_ret_daily**2) - 6

#Possible Bi-modal distribution for Log-prices

def bimodal_pdf(x, mu1, sigma1, mu2, sigma2, w):
    return w * norm.pdf(x, mu1, sigma1) + (1 - w) * norm.pdf(x, mu2, sigma2)

from sklearn.mixture import GaussianMixture
import numpy as np

data = log_prices.reshape(-1, 1)

gmm = GaussianMixture(n_components=2)
gmm.fit(data)
mu1, mu2 = gmm.means_.flatten()
s1, s2 = np.sqrt(gmm.covariances_.flatten())
w1, w2 = gmm.weights_      #weights used later to compute the bimodal pdf



#normality test

jb, pvalue= jarque_bera(log_ret)

print(f"\nJB test result for normality of log-returns (refute if less than 0.05): {pvalue}")

print("\nStatistics on given data chunk")

print(f"Osservazioni: {len(chunk)}")
print(f"Period: {chunk['Date'].iloc[0].date()} → {chunk['Date'].iloc[-1].date()}")
print(f"\n[Prices]")
print(f"  Mean: {mean_price:.2f} €/MWh")
print(f"  Median: {median_price:.2f} €/MWh")
print(f"  Std: {std_price:.2f} €/MWh")
print(f"  Min: {min_price:.2f} €/MWh")
print(f"  Max: {max_price:.2f} €/MWh")
print(f"  Skewness: {skew_price:.4f}")
print(f"  Kurtosis: {kurt_price:.4f}")
print(f"\n[Log-Returns]")
print(f"  Mean: {mean_ret:.4f}  ({mean_ret*100:.2f}%)")
print(f"  Std: {std_ret:.4f}  ({std_ret*100:.2f}%)")
print(f"  Skewness: {skew_ret:.4f}")
print(f"  Kurtosis: {kurt_ret:.4f}")
print(f"annualized Log-normal kurtosis on data: {kurt_lognormal:.2f}")
print(f"absolute Log-normal kurtosis on daily std: {kurt_log_abs:.2f}")

period_label = (f"{chunk['Date'].iloc[0].date()} → {chunk['Date'].iloc[-1].date()}")



fig, axes = plt.subplots(4, 1, figsize=(12, 18))
fig.suptitle(f"Statistics for: {period_label}", fontsize=16, fontweight='bold', y=1)
axes[0].plot(chunk['Date'], chunk['Prices'], color='steelblue', linewidth=1)
axes[0].set_title('Price History')
axes[0].set_ylabel('€/MWh')
axes[0].grid(alpha=0.3)
axes[1].hist(log_ret, bins='auto', density=True, color='steelblue', edgecolor='white', alpha=0.7)
kurt_excess = kurt_ret  # già calcolata sopra
df_approx = max(6/kurt_excess + 4, 2.5) if kurt_excess > 0 else 10.0

x = np.linspace(-0.15, 0.15, 300)
scale_t = log_ret.std() * np.sqrt((df_approx-2)/df_approx)

axes[1].hist(log_ret, bins='auto', density=True,
             color='steelblue', edgecolor='white', alpha=0.7)
axes[1].plot(x, norm.pdf(x, log_ret.mean(), log_ret.std()),
             color='red', lw=2, label='Gaussian')
axes[1].plot(x, student_t.pdf(x, df=df_approx, loc=log_ret.mean(), scale=scale_t),
             color='green', lw=2, label=f't-Student (df≈{df_approx:.1f})')
axes[1].set_title('Log-Return Distribution')
axes[1].set_ylabel('Density')
axes[1].set_xlim(-0.15, 0.15)
axes[1].legend()
axes[1].grid(alpha=0.3)





max_lag = 40
acf_vals = autocorrelation(log_ret, max_lag)
lags = np.arange(max_lag + 1)
conf = 1.96 / np.sqrt(len(log_ret))
axes[2].stem(lags[1:], acf_vals[1:], basefmt=' ', linefmt='blue', markerfmt='bo')
axes[2].axhline(y=conf, color='red', linestyle='--', linewidth=1, label='CI 95%')
axes[2].axhline(y=-conf, color='red', linestyle='--', linewidth=1)
axes[2].axhline(y=0, color='black', linewidth=0.5)
axes[2].set_title('ACF Log-Returns')
axes[2].set_xlabel('Lag')
axes[2].set_ylabel('ACF')
axes[2].legend()
axes[2].grid(alpha=0.3)

acf_sq = autocorrelation(log_ret**2, max_lag)
axes[3].stem(lags[1:], acf_sq[1:], basefmt=' ', linefmt='darkorange', markerfmt='bo')
axes[3].axhline(y=conf, color='red', linestyle='--', linewidth=1, label='CI 95%')
axes[3].axhline(y=-conf, color='red', linestyle='--', linewidth=1)
axes[3].axhline(y=0, color='black', linewidth=0.5)
axes[3].set_title('ACF squared Log-Returns  (vol-clustering)')
axes[3].set_xlabel('Lag')
axes[3].set_ylabel('ACF')
axes[3].legend()
axes[3].grid(alpha=0.3)
axes[1].set_xlim(-0.15, 0.15)




plt.tight_layout()
plt.show()