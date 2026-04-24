
#  TTF MODELLING — 6 MODELS, TWO FAMILIES
#
#  OU family   : OU M1 | OU M2 | OU M2 + GARCH
#  GARCH family: GARCH-Gaussian | GARCH-t | EGARCH
#
#  Metrics
#  -------
#  CRPS one-step (log-returns) — PRIMARY, always comparable across families
#  RMSE / MAE / Coverage       — SECONDARY, path-based, comparable only for
#                                short test horizons (< ~60 days)


# CONFIGURATION insert the period here, data starts from 
TRAIN_END = '2025-01-01'
TEST_START = '2025-01-01'
TEST_END = '2025-06-01'
CSV_PRICES = r'C:\Users\Pc_Lenovo\Desktop\TTF\ttf_forward1.csv'     #save the csv file and specify the path here
N_PATHS = 1000
SEED = 42


import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import probplot


#  SHARED UTILITIES


def crps_gaussian(mu, sigma, y):
    """Closed-form CRPS — Gneiting & Raftery (2007) eq. 21."""
    z = (y - mu) / sigma
    return sigma * (z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))


def crps_ensemble(r_sim, y_obs):
    """Energy-score CRPS from sorted ensemble."""
    ens= np.sort(r_sim)
    n = len(ens)
    mae = np.mean(np.abs(ens - y_obs))
    i_arr = np.arange(1, n+1)
    gini  = np.sum((2*i_arr - n - 1) * ens) / (n*n)
    return mae - gini


def path_metrics(S_paths, prices_test):
    """RMSE, MAE, Coverage on median path — comparable for short horizons only."""
    p5  = np.percentile(S_paths,  5, axis=0)
    p50 = np.percentile(S_paths, 50, axis=0)
    p95 = np.percentile(S_paths, 95, axis=0)
    rmse = np.sqrt(np.mean((prices_test - p50[1:])**2))
    mae = np.mean(np.abs(prices_test - p50[1:]))
    cov= np.mean((prices_test >= p5[1:]) & (prices_test <= p95[1:]))
    return {'p5':p5, 'p50':p50, 'p95':p95, 'rmse':rmse, 'mae':mae, 'coverage':cov}


def fan_plot(ax, t_axis, prices_train, prices_test, S_paths, m,
             crps_arr, label, color):
    ax.plot(t_axis,
            np.concatenate([[prices_train[-1]], prices_test]),
            color='black', lw=2.0, label='Observed', zorder=5)
    for i in range(min(100, S_paths.shape[0])):
        ax.plot(t_axis, S_paths[i, :], color=color, lw=0.3, alpha=0.15)
    ax.plot(t_axis, m['p50'], color=color, lw=1.8, label='Median')
    ax.fill_between(t_axis, m['p5'], m['p95'],
                    alpha=0.2, color=color, label='5–95th pct')
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.set_ylim(min(prices_test.min(), m['p5'][1:].min()) * 0.88,
                max(prices_test.max(), m['p95'][1:].max()) * 1.12)
    crps_str = f"CRPS={crps_arr.mean():.5f}  " if crps_arr is not None else ''
    ax.set_title(f"{label}:  {crps_str}"
                 f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  "
                 f"Cov={m['coverage']*100:.0f}%")
    ax.set_ylabel('Price €/MWh')
    ax.set_xlabel('Business days')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def styled_table(ax, rows, columns):
    #for results
    table = ax.table(cellText=rows, colLabels=columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    n_rows = len(rows)
    for j in range(len(columns)):
        table[0, j].set_facecolor('#2d2d2d')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, n_rows+1):
        bg = '#f5f5f5' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[i, j].set_facecolor(bg)
    # highlight best per metric (skip col 0 = name, last col = coverage → max)
    best_fns = [min] * (len(columns)-2) + [max]
    for col_idx, best_fn in zip(range(1, len(columns)), best_fns):
        try:
            vals = [float(rows[r][col_idx].replace('%','').replace('—','nan'))
                    for r in range(n_rows)]
            finite = [v for v in vals if not np.isnan(v)]
            if not finite:
                continue
            best = best_fn(finite)
            for r_idx, v in enumerate(vals):
                if not np.isnan(v) and v == best:
                    table[r_idx+1, col_idx].set_facecolor('#c8f0c8')
                    table[r_idx+1, col_idx].set_text_props(fontweight='bold')
        except Exception:
            pass
    return table


#  OU M1  —  constant mean

def fit_ou_m1(prices, dt):
    Y = np.log(prices)
    Y_t, Y_lag = Y[1:], Y[:-1]
    b, a, r_value, _, _ = stats.linregress(Y_lag, Y_t)
    if not (0 < b < 1):
        return None
    kappa = -np.log(b) / dt
    mu    = a / (1-b)
    resid = Y_t - (a + b*Y_lag)
    s2    = np.var(resid, ddof=2)
    sigma = np.sqrt(s2 * 2*kappa / (1-np.exp(-2*kappa*dt)))
    k, n  = 3, len(resid)
    # log-likelihood under Gaussian AR(1)
    sigma_r = np.sqrt(sigma**2/(2*kappa)*(1-np.exp(-2*kappa*dt)))
    ll = np.sum(norm.logpdf(resid, 0, sigma_r))
    return {'kappa':kappa, 'mu_log':mu, 'mu_price':np.exp(mu),
            'sigma':sigma, 'alpha':b,
            'half_life_days':np.log(2)/kappa*252,
            'aic':-2*ll+2*k, 'bic':-2*ll+k*np.log(n),
            'log_returns': np.diff(np.log(prices))}


def crps_onestep_ou_m1(ou, prices_train, prices_test, dt):
    """
    One-step CRPS on log-returns.
    Predictive distribution: N(mu_r, sigma_r^2) where both are
    F_{t-1}-measurable (OU transition, constant parameters).
    Comparable with GARCH CRPS — same space, same anchoring.
    """
    alpha_ou = np.exp(-ou['kappa']*dt)
    sigma_r  = np.sqrt(ou['sigma']**2/(2*ou['kappa'])
                       *(1-np.exp(-2*ou['kappa']*dt)))
    T = len(prices_test)
    crps = np.zeros(T)
    for t in range(T):
        lp = np.log(prices_train[-1]) if t==0 else np.log(prices_test[t-1])
        mu_r = (alpha_ou-1)*lp + ou['mu_log']*(1-alpha_ou)
        y_obs = np.log(prices_test[t]) - lp
        crps[t] = crps_gaussian(mu_r, sigma_r, y_obs)
    return crps


def simulate_ou_m1(ou, S0, T_days, n_paths, dt, seed=SEED):
    rng = np.random.default_rng(seed)
    alpha_ou = np.exp(-ou['kappa']*dt)
    std_eps = np.sqrt(ou['sigma']**2/(2*ou['kappa'])
                       *(1-np.exp(-2*ou['kappa']*dt)))
    Y = np.zeros((n_paths, T_days+1))
    Y[:, 0] = np.log(S0)
    for t in range(1, T_days+1):
        eps = rng.normal(0, std_eps, size=n_paths)
        Y[:, t] = ou['mu_log']*(1-alpha_ou) + alpha_ou*Y[:, t-1] + eps
    return np.exp(Y)



#  OU M2  —  sinusoidal seasonal mean

def mu_seasonal(t_index, mu0, A, phi):
    return mu0 + A*np.sin(2*np.pi*t_index/252 + phi)


def fit_ou_m2(prices, t_index, dt):
    ou_m1 = fit_ou_m1(prices, dt)
    if ou_m1 is None:
        return None
    alpha = ou_m1['alpha']
    Y= np.log(prices)
    Y_t, Y_lag = Y[1:], Y[:-1]
    t   = t_index[1:]
    r_t = Y_t - alpha*Y_lag
    X = np.column_stack([np.ones(len(t)),
                           np.sin(2*np.pi*t/252),
                           np.cos(2*np.pi*t/252)])
    c0, c1, c2 = np.linalg.lstsq(X, r_t, rcond=None)[0]
    mu0 = c0/(1-alpha)
    A = np.sqrt(c1**2+c2**2)/(1-alpha)
    phi = np.arctan2(c2, c1)
    mu_t = mu_seasonal(t_index[1:], mu0, A, phi)
    resid = r_t - mu_t*(1-alpha)
    s2= np.var(resid, ddof=4)
    kappa = -np.log(alpha)/dt
    sigma = np.sqrt(s2*2*kappa/(1-np.exp(-2*kappa*dt)))
    k, n = 5, len(resid)
    sigma_r = np.sqrt(sigma**2/(2*kappa)*(1-np.exp(-2*kappa*dt)))
    ll = np.sum(norm.logpdf(resid, 0, sigma_r))
    return {'kappa':kappa, 'mu0_log':mu0, 'mu0_price':np.exp(mu0),
            'A':A, 'phi':phi, 'sigma':sigma, 'alpha':alpha,
            'half_life_days':np.log(2)/kappa*252,
            'residuals':resid,
            'aic':-2*ll+2*k, 'bic':-2*ll+k*np.log(n),
            'mu_t_func': lambda t: mu_seasonal(t, mu0, A, phi)}


def crps_onestep_ou_m2(ou, prices_train, prices_test, t_start, dt):
    """
    One-step CRPS on log-returns with time-varying mean mu(t).
    sigma_r constant (OU transition variance), mu_r time-varying but
    F_{t-1}-measurable. Directly comparable with GARCH CRPS.
    """
    alpha_ou = ou['alpha']
    sigma_r  = np.sqrt(ou['sigma']**2/(2*ou['kappa'])
                       *(1-np.exp(-2*ou['kappa']*dt)))
    mu_func=ou['mu_t_func']
    T = len(prices_test)
    crps = np.zeros(T)
    for t in range(T):
        lp = np.log(prices_train[-1]) if t==0 else np.log(prices_test[t-1])
        mu_r = (alpha_ou-1)*lp + mu_func(t_start+t+1)*(1-alpha_ou)
        y_obs = np.log(prices_test[t]) - lp
        crps[t] = crps_gaussian(mu_r, sigma_r, y_obs)
    return crps


def simulate_ou_m2(ou, S0, T_days, t_start, n_paths, dt, seed=SEED):
    rng = np.random.default_rng(seed)
    alpha_ou = ou['alpha']
    std_eps = np.sqrt(ou['sigma']**2/(2*ou['kappa'])
                       *(1-np.exp(-2*ou['kappa']*dt)))
    mu_func = ou['mu_t_func']
    Y = np.zeros((n_paths, T_days+1))
    Y[:, 0] = np.log(S0)
    for t in range(1, T_days+1):
        mu_t = mu_func(t_start+t)
        eps = rng.normal(0, std_eps, size=n_paths)
        Y[:, t] = mu_t*(1-alpha_ou) + alpha_ou*Y[:, t-1] + eps
    return np.exp(Y)



#  OU M2 + GARCH(1,1) on residuals

def fit_garch_on_residuals(residuals):
    T = len(residuals)

    def neg_ll(params):
        omega, alpha_g, beta_g, mu_g = params
        if omega<=0 or alpha_g<=0 or beta_g<=0 or alpha_g+beta_g>=1:
            return 1e10
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(residuals)
        for t in range(1, T):
            eps_prev  = residuals[t-1] - mu_g
            sigma2[t] = omega + alpha_g*eps_prev**2 + beta_g*sigma2[t-1]
        ll = -0.5*np.sum(np.log(2*np.pi*sigma2)
                         + (residuals-mu_g)**2/sigma2)
        return -ll

    x0= [np.var(residuals)*0.05, 0.1, 0.85, np.mean(residuals)]
    bounds = [(1e-8,1),(1e-4,1),(1e-4,1),(-1,1)]
    result = minimize(neg_ll, x0, bounds=bounds, method='L-BFGS-B')
    omega, alpha_g, beta_g, mu_g = result.x
    ll  = -result.fun
    k   = 4
    cur = np.var(residuals)
    for t in range(1, T):
        cur = omega + alpha_g*(residuals[t-1]-mu_g)**2 + beta_g*cur
    last_sigma2 = omega + alpha_g*(residuals[-1]-mu_g)**2 + beta_g*cur
    return {'omega':omega, 'alpha':alpha_g, 'beta':beta_g, 'mu':mu_g,
            'log_likelihood':ll, 'aic':-2*ll+2*k, 'bic':-2*ll+k*np.log(T),
            'last_sigma2':last_sigma2, 'success':result.success}


def crps_onestep_ou_m2_garch(ou, garch, prices_train, prices_test, t_start, dt):
    """
    One-step CRPS on log-returns.
    sigma_t is time-varying (GARCH filter on OU residuals) and
    F_{t-1}-measurable. mu_r is F_{t-1}-measurable via mu(t).
    Predictive: N(mu_r, sigma_t^2). Comparable with plain GARCH CRPS.
    """
    alpha_ou = ou['alpha']
    mu_func  = ou['mu_t_func']
    omega, alpha_g, beta_g, mu_g = (garch['omega'], garch['alpha'],
                                     garch['beta'], garch['mu'])
    train_resid = ou['residuals']
    sigma2 = np.var(train_resid)
    for i in range(1, len(train_resid)):
        sigma2 = omega + alpha_g*(train_resid[i-1]-mu_g)**2 + beta_g*sigma2
    sigma2 = omega + alpha_g*(train_resid[-1]-mu_g)**2 + beta_g*sigma2
    T = len(prices_test)
    crps = np.zeros(T)
    for t in range(T):
        lp = np.log(prices_train[-1]) if t==0 else np.log(prices_test[t-1])
        t_abs = t_start+t+1
        mu_r = (alpha_ou-1)*lp + mu_func(t_abs)*(1-alpha_ou) + mu_g
        y_obs = np.log(prices_test[t]) - lp
        crps[t] = crps_gaussian(mu_r, np.sqrt(sigma2), y_obs)
        resid_obs = y_obs - ((alpha_ou-1)*lp + mu_func(t_abs)*(1-alpha_ou))
        sigma2 = omega + alpha_g*(resid_obs-mu_g)**2 + beta_g*sigma2
    return crps


def simulate_ou_m2_garch(ou, garch, S0, T_days, t_start, n_paths, dt, seed=SEED):
    rng = np.random.default_rng(seed)
    alpha_ou = ou['alpha']
    mu_func = ou['mu_t_func']
    omega, alpha_g, beta_g, mu_g = (garch['omega'], garch['alpha'],
                                     garch['beta'], garch['mu'])
    log_S = np.zeros((n_paths, T_days+1))
    log_S[:, 0] = np.log(S0)
    for i in range(n_paths):
        sigma2 = garch['last_sigma2']
        for t in range(1, T_days+1):
            mu_t = mu_func(t_start+t)
            eps = mu_g + np.sqrt(sigma2)*rng.normal()
            log_S[i, t] = mu_t*(1-alpha_ou) + alpha_ou*log_S[i, t-1] + eps
            sigma2 = omega + alpha_g*eps**2 + beta_g*sigma2
    return np.exp(log_S)



#  GARCH(1,1) Gaussian


def fit_garch_gaussian(prices, dt):
    log_returns = np.diff(np.log(prices))
    T = len(log_returns)

    def neg_ll(params):
        omega, alpha, beta, mu = params
        if omega<=0 or alpha<=0 or beta<=0 or alpha+beta>=1:
            return 1e10
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(log_returns)
        for t in range(1, T):
            sigma2[t] = omega + alpha*(log_returns[t-1]-mu)**2 + beta*sigma2[t-1]
        return 0.5*np.sum(np.log(2*np.pi*sigma2) + (log_returns-mu)**2/sigma2)

    x0 = [np.var(log_returns)*0.05, 0.1, 0.85, np.mean(log_returns)]
    bounds = [(1e-8,1),(1e-4,1),(1e-4,1),(-1,1)]
    result = minimize(neg_ll, x0, bounds=bounds, method='L-BFGS-B')
    omega, alpha, beta, mu = result.x
    ll  = -result.fun
    k = 4
    cur = np.var(log_returns)
    for t in range(1, T):
        cur = omega + alpha*(log_returns[t-1]-mu)**2 + beta*cur
    last_sigma2 = omega + alpha*(log_returns[-1]-mu)**2 + beta*cur
    return {'omega':omega, 'alpha':alpha, 'beta':beta, 'mu':mu,
            'log_likelihood':ll, 'aic':-2*ll+2*k, 'bic':-2*ll+k*np.log(T),
            'last_sigma2':last_sigma2, 'success':result.success,
            'log_returns':log_returns}


def crps_onestep_garch_gaussian(params, prices_train, prices_test, dt):
    """
    Closed-form CRPS. sigma_t^2 is F_{t-1}-measurable via GARCH filter.
    Predictive: N(mu, sigma_t^2). Directly comparable with OU CRPS.
    """
    omega, alpha, beta, mu = (params['omega'], params['alpha'],
                               params['beta'], params['mu'])
    lrt = params['log_returns']
    sigma2 = np.var(lrt)
    for i in range(1, len(lrt)):
        sigma2 = omega + alpha*(lrt[i-1]-mu)**2 + beta*sigma2
    sigma2 = omega + alpha*(lrt[-1]-mu)**2 + beta*sigma2
    T = len(prices_test)
    crps = np.zeros(T)
    for t in range(T):
        lp = np.log(prices_train[-1]) if t==0 else np.log(prices_test[t-1])
        y_obs = np.log(prices_test[t]) - lp
        crps[t] = crps_gaussian(mu, np.sqrt(sigma2), y_obs)
        sigma2  = omega + alpha*(y_obs-mu)**2 + beta*sigma2
    return crps


def simulate_garch_gaussian(params, S0, T_days, n_paths, dt, seed=SEED):
    rng = np.random.default_rng(seed)
    omega, alpha, beta, mu = (params['omega'], params['alpha'],
                               params['beta'], params['mu'])
    log_S = np.zeros((n_paths, T_days+1))
    log_S[:, 0] = np.log(S0)
    for i in range(n_paths):
        sigma2 = params['last_sigma2']
        for t in range(1, T_days+1):
            eps = np.sqrt(sigma2)*rng.normal()
            log_S[i, t] = log_S[i, t-1] + mu + eps
            sigma2 = omega + alpha*eps**2 + beta*sigma2
    return np.exp(log_S)



#  GARCH(1,1) t-Student


def fit_garch_t(prices, dt):
    log_returns = np.diff(np.log(prices))
    T = len(log_returns)

    def neg_ll(params):
        omega, alpha, beta, mu, df = params
        if omega<=1e-12 or alpha<0 or beta<0 or alpha+beta>=1 or df<=2.01:
            return 1e10
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(log_returns)
        for t in range(1, T):
            sigma2[t] = omega + alpha*(log_returns[t-1]-mu)**2 + beta*sigma2[t-1]
        scale = np.sqrt(sigma2*(df-2)/df)
        return -np.sum(student_t.logpdf(log_returns, df=df, loc=mu, scale=scale))

    x0 = [np.var(log_returns)*0.05, 0.1, 0.8, np.mean(log_returns), 6.0]
    bounds = [(1e-15,1),(1e-4,1),(1e-4,1),(-1,1),(2.1,50)]
    result = minimize(neg_ll, x0, bounds=bounds, method='L-BFGS-B')
    omega, alpha, beta, mu, df = result.x
    ll  = -result.fun
    k = 5
    cur = np.var(log_returns)
    for t in range(1, T):
        cur = omega + alpha*(log_returns[t-1]-mu)**2 + beta*cur
    last_sigma2 = omega + alpha*(log_returns[-1]-mu)**2 + beta*cur
    return {'omega':omega, 'alpha':alpha, 'beta':beta, 'mu':mu, 'df':df,
            'log_likelihood':ll, 'aic':-2*ll+2*k, 'bic':-2*ll+k*np.log(T),
            'last_sigma2':last_sigma2, 'success':result.success,
            'log_returns':log_returns}


def crps_onestep_garch_t(params, prices_train, prices_test, dt, n_draws=10000):
    
    omega, alpha, beta, mu, df = (params['omega'], params['alpha'], params['beta'],
                                   params['mu'], params['df'])
    lrt = params['log_returns']
    rng = np.random.default_rng(0)
    t_scale = np.sqrt((df-2)/df)
    sigma2 = np.var(lrt)
    for i in range(1, len(lrt)):
        sigma2 = omega + alpha*(lrt[i-1]-mu)**2 + beta*sigma2
    sigma2 = omega + alpha*(lrt[-1]-mu)**2 + beta*sigma2
    T = len(prices_test)
    crps = np.zeros(T)
    for t in range(T):
        lp = np.log(prices_train[-1]) if t==0 else np.log(prices_test[t-1])
        y_obs = np.log(prices_test[t]) - lp
        z = rng.standard_t(df, size=n_draws)*t_scale
        crps[t] = crps_ensemble(mu + np.sqrt(sigma2)*z, y_obs)
        sigma2 = omega + alpha*(y_obs-mu)**2 + beta*sigma2
    return crps


def simulate_garch_t(params, S0, T_days, n_paths, dt, seed=SEED):
    rng   = np.random.default_rng(seed)
    omega, alpha, beta, mu, df = (params['omega'], params['alpha'], params['beta'],
                                   params['mu'], params['df'])
    t_scale = np.sqrt((df-2)/df)
    log_S = np.zeros((n_paths, T_days+1))
    log_S[:, 0] = np.log(S0)
    for i in range(n_paths):
        sigma2 = params['last_sigma2']
        for t in range(1, T_days+1):
            eps = np.sqrt(sigma2)*rng.standard_t(df)*t_scale
            log_S[i, t] = log_S[i, t-1] + mu + eps
            sigma2 = omega + alpha*eps**2 + beta*sigma2
    return np.exp(log_S)



#  EGARCH(1,1)  (asymmetric for negative returns)


def fit_egarch(prices, dt):
    log_returns = np.diff(np.log(prices))
    T = len(log_returns)
    E_abs = np.sqrt(2/np.pi)

    def neg_ll(params):
        omega, alpha, gamma, beta, mu = params
        if beta >= 1 or beta <= -1:
            return 1e10
        log_var    = np.zeros(T)
        log_var[0] = np.log(np.var(log_returns))
        for t in range(1, T):
            eps_prev   = (log_returns[t-1]-mu)/np.exp(0.5*log_var[t-1])
            log_var[t] = (omega + alpha*(np.abs(eps_prev)-E_abs)
                          + gamma*eps_prev + beta*log_var[t-1])
        sigma2 = np.exp(log_var)
        return 0.5*np.sum(np.log(2*np.pi*sigma2) + (log_returns-mu)**2/sigma2)

    x0 = [np.log(np.var(log_returns))*0.1, 0.1, -0.05, 0.9, np.mean(log_returns)]
    bounds = [(-5,5),(0,2),(-2,2),(-0.999,0.999),(-1,1)]
    result = minimize(neg_ll, x0, bounds=bounds, method='Nelder-Mead')
    omega, alpha, gamma, beta, mu = result.x
    ll = -result.fun
    k  = 5
    return {'omega':omega, 'alpha':alpha, 'gamma':gamma, 'beta':beta, 'mu':mu,
            'log_likelihood':ll, 'aic':-2*ll+2*k, 'bic':-2*ll+k*np.log(T),
            'success':result.success, 'log_returns':log_returns}


def crps_onestep_egarch(params, prices_train, prices_test, dt):
    
    omega, alpha, gamma, beta, mu = (params['omega'], params['alpha'], params['gamma'],
                                      params['beta'], params['mu'])
    lrt   = params['log_returns']
    E_abs = np.sqrt(2/np.pi)
    log_var = np.log(np.var(lrt))
    for i in range(1, len(lrt)):
        eps_prev = (lrt[i-1]-mu)/np.exp(0.5*log_var)
        log_var = (omega + alpha*(np.abs(eps_prev)-E_abs)
                    + gamma*eps_prev + beta*log_var)
    eps_last = (lrt[-1]-mu)/np.exp(0.5*log_var)
    log_var  = (omega + alpha*(np.abs(eps_last)-E_abs)
                + gamma*eps_last + beta*log_var)
    T = len(prices_test)
    crps = np.zeros(T)
    for t in range(T):
        lp = np.log(prices_train[-1]) if t==0 else np.log(prices_test[t-1])
        y_obs = np.log(prices_test[t]) - lp
        sigma_t = np.exp(0.5*log_var)
        crps[t] = crps_gaussian(mu, sigma_t, y_obs)
        eps_obs = (y_obs-mu)/sigma_t
        log_var = (omega + alpha*(np.abs(eps_obs)-E_abs)
                   + gamma*eps_obs + beta*log_var)
    return crps


def simulate_egarch(params, S0, T_days, n_paths, dt, seed=SEED):
    rng = np.random.default_rng(seed)
    omega, alpha, gamma, beta, mu = (params['omega'], params['alpha'], params['gamma'],
                                      params['beta'], params['mu'])
    E_abs = np.sqrt(2/np.pi)
    log_S = np.zeros((n_paths, T_days+1))
    log_S[:, 0] = np.log(S0)
    for i in range(n_paths):
        log_var = (omega/(1-beta) if abs(beta)<1
                   else np.log(np.var(params['log_returns'])))
        for t in range(1, T_days+1):
            sigma_t = np.exp(0.5*log_var)
            eps = rng.normal()
            log_S[i, t] = log_S[i, t-1] + mu + sigma_t*eps
            log_var = (omega + alpha*(np.abs(eps)-E_abs)
                           + gamma*eps + beta*log_var)
    return np.exp(log_S)

def plot_qq_comparison(log_returns, ga_t, chunk_label):

    n = len(log_returns)
# quantili empirici standardizzati
    r_std = (log_returns - log_returns.mean()) / log_returns.std()
    r_sorted = np.sort(r_std)
# probabilità teoriche (Filliben formula)
    probs = (np.arange(1, n+1) - 0.3175) / (n + 0.365)

# quantili teorici per le tre distribuzioni
    q_norm = norm.ppf(probs)
    q_t14 = student_t.ppf(probs, df=14)  / np.sqrt(14/(14-2)) # standardizzata
    df_fit = ga_t['df']
    q_tfit = student_t.ppf(probs, df=df_fit) / np.sqrt(df_fit/(df_fit-2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{chunk_label} — QQ-plot: log-returns vs theoretical distributions",
             fontsize=13, fontweight='bold')

    configs = [
    (axes[0], q_norm,"Gaussian N(0,1)", "steelblue"),
    (axes[1], q_tfit,  f"t-Student (df={df_fit:.1f}, stimato)", "darkorange"),
    (axes[2], q_t14,  "t-Student (df=14)", "firebrick"),
]

    for ax, q_theor, label, color in configs:
        ax.scatter(q_theor, r_sorted, color=color, alpha=0.4, s=8, label='Empirici')
    # retta di riferimento 45°
        lim = max(abs(q_theor.min()), abs(q_theor.max()),
              abs(r_sorted.min()), abs(r_sorted.max())) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color='black', lw=1.2,
            linestyle='--', label='y = x')
    # evidenzia le code (ultimi/primi 2.5%)
        tail_mask = (probs < 0.025) | (probs > 0.975)
        ax.scatter(q_theor[tail_mask], r_sorted[tail_mask],
               color='red', s=20, zorder=5, label='Code (2.5%)')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(label)
        ax.set_xlabel('Quantiles')

#  MAIN section, define the train and test period here


if __name__ == '__main__':

    dt = 1/252

    #load data 
    allprices = pd.read_csv(CSV_PRICES)
    business_days = pd.bdate_range(start='2018-01-02',
                                    periods=len(allprices), freq='B')
    allprices['Date'] = business_days

    chunk_train = allprices[allprices['Date'] < TRAIN_END]
    chunk_test = allprices[allprices['Date'].between(TEST_START, TEST_END)]

    prices_train = chunk_train['Prices'].values
    prices_test  = chunk_test['Prices'].values
    idx_train = np.arange(len(prices_train))
    T_test = len(prices_test)
    n_train = len(prices_train)
    t_test_axis = np.arange(n_train-1, n_train+T_test)

    chunk_label = (f"{pd.Timestamp(chunk_train['Date'].iloc[0]).date()} → "
                   f"{pd.Timestamp(chunk_test['Date'].iloc[-1]).date()}")
    print(f"Train: {n_train}  Test: {T_test}  [{chunk_label}]")

    # fitting 
    
    ou_m1 = fit_ou_m1(prices_train, dt)
    ou_m2 = fit_ou_m2(prices_train, idx_train, dt)
    garch_r = fit_garch_on_residuals(ou_m2['residuals'])
    ga_g  = fit_garch_gaussian(prices_train, dt)
    ga_t = fit_garch_t(prices_train, dt)
    eg = fit_egarch(prices_train, dt)

    print(f"\n OU family")
    print(f"OU M1  : kappa={ou_m1['kappa']:.4f}  mu={ou_m1['mu_price']:.2f} €/MWh  "
          f"sigma={ou_m1['sigma']:.4f}  half-life={ou_m1['half_life_days']:.1f}d  "
          f"AIC={ou_m1['aic']:.1f}")
    print(f"OU M2  : kappa={ou_m2['kappa']:.4f}  mu0={ou_m2['mu0_price']:.2f} €/MWh  "
          f"A={ou_m2['A']:.4f}  phi={ou_m2['phi']:.4f}  "
          f"AIC={ou_m2['aic']:.1f}")
    print(f"GARCHr: omega={garch_r['omega']:.2e}  alpha={garch_r['alpha']:.4f}  "
          f"beta={garch_r['beta']:.4f}  "
          f"pers={garch_r['alpha']+garch_r['beta']:.4f}  "
          f"AIC={garch_r['aic']:.1f}")

    print(f"\n GARCH family ")
    print(f"GARCH-G: omega={ga_g['omega']:.2e}  alpha={ga_g['alpha']:.4f}  "
          f"beta={ga_g['beta']:.4f}  "
          f"pers={ga_g['alpha']+ga_g['beta']:.4f}  "
          f"AIC={ga_g['aic']:.1f}")
    print(f"GARCH-t: omega={ga_t['omega']:.2e}  alpha={ga_t['alpha']:.4f}  "
          f"beta={ga_t['beta']:.4f}  df={ga_t['df']:.2f}  "
          f"AIC={ga_t['aic']:.1f}")
    print(f"EGARCH : omega={eg['omega']:.4f}  alpha={eg['alpha']:.4f}  "
          f"gamma={eg['gamma']:.4f}  beta={eg['beta']:.4f}  "
          f"AIC={eg['aic']:.1f}")

    # CRPS (one-step, log-returns — PRIMARY metric)
    crps_m1 = crps_onestep_ou_m1(ou_m1, prices_train, prices_test, dt)
    crps_m2 = crps_onestep_ou_m2(ou_m2, prices_train, prices_test,
                                   len(idx_train), dt)
    crps_m2g = crps_onestep_ou_m2_garch(ou_m2, garch_r, prices_train,
                                         prices_test, len(idx_train), dt)
    crps_gg = crps_onestep_garch_gaussian(ga_g, prices_train, prices_test, dt)
    crps_gt = crps_onestep_garch_t(ga_t, prices_train, prices_test, dt)
    crps_eg = crps_onestep_egarch(eg, prices_train, prices_test, dt)

    # simulations
    
    S_m1 = simulate_ou_m1(ou_m1, prices_train[-1], T_test, N_PATHS, dt)
    S_m2 = simulate_ou_m2(ou_m2, prices_train[-1], T_test,
                            len(idx_train), N_PATHS, dt)
    S_m2g = simulate_ou_m2_garch(ou_m2, garch_r, prices_train[-1],
                                   T_test, len(idx_train), N_PATHS, dt)
    S_gg = simulate_garch_gaussian(ga_g, prices_train[-1], T_test, N_PATHS, dt)
    S_gt = simulate_garch_t(ga_t, prices_train[-1], T_test, N_PATHS, dt)
    S_eg = simulate_egarch(eg, prices_train[-1], T_test, N_PATHS, dt)

    m_m1 = path_metrics(S_m1,  prices_test)
    m_m2 = path_metrics(S_m2,  prices_test)
    m_m2g = path_metrics(S_m2g, prices_test)
    m_gg = path_metrics(S_gg, prices_test)
    m_gt = path_metrics(S_gt,prices_test)
    m_eg = path_metrics(S_eg, prices_test)

    #print summary 
    print(f"\n{'Model':<20} {'CRPS':>9} {'RMSE':>8} {'MAE':>8} {'Cov':>7}")
    print("-"*56)
    for name, crps_arr, m in [
        ("OU M1",crps_m1,  m_m1),
        ("OU M2",crps_m2,  m_m2),
        ("OU M2+GARCH",   crps_m2g, m_m2g),
        ("GARCH-Gaussian",crps_gg,  m_gg),
        ("GARCH-t",       crps_gt,  m_gt),
        ("EGARCH",        crps_eg,  m_eg),
    ]:
        print(f"{name:<20} {crps_arr.mean():9.5f} {m['rmse']:8.3f} "
              f"{m['mae']:8.3f} {m['coverage']*100:7.1f}%")
    
    #  FIGURE 1 — Fan charts OU family  (1×3)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(f"Calibrated on {chunk_label} — OU family", fontsize=13, fontweight='bold')
    for ax, S, m, crps_arr, label, color in [
        (axes[0], S_m1,  m_m1,  crps_m1,  "OU M1",         "steelblue"),
        (axes[1], S_m2,  m_m2,  crps_m2,  "OU M2",         "seagreen"),
        (axes[2], S_m2g, m_m2g, crps_m2g, "OU M2 + GARCH", "firebrick"),
    ]:
        fan_plot(ax, t_test_axis, prices_train, prices_test,
                 S, m, crps_arr, label, color)
    plt.tight_layout()
    plt.savefig("fig1_ou_family.png", dpi=150, bbox_inches='tight')
    plt.show()

    #  FIGURE 2 — Fan charts GARCH family  (1×3)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle(f"{chunk_label} — GARCH family", fontsize=13, fontweight='bold')
    for ax, S, m, crps_arr, label, color in [
        (axes[0], S_gg, m_gg, crps_gg, "GARCH-Gaussian", "steelblue"),
        (axes[1], S_gt, m_gt, crps_gt, "GARCH-t",        "darkorange"),
        (axes[2], S_eg, m_eg, crps_eg, "EGARCH",         "firebrick"),
    ]:
        fan_plot(ax, t_test_axis, prices_train, prices_test,
                 S, m, crps_arr, label, color)
    plt.tight_layout()
    plt.savefig("fig2_garch_family.png", dpi=150, bbox_inches='tight')
    plt.show()

    
    #  FIGURE 3 — CRPS over time  (all 6 models)
    
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f"{chunk_label} — One-step CRPS (log-returns)",
                 fontsize=13, fontweight='bold')
    t_axis = np.arange(T_test)

    axes[0].set_title("OU family")
    for crps_arr, label, color in [
        (crps_m1,  "OU M1",         "steelblue"),
        (crps_m2,  "OU M2",         "seagreen"),
        (crps_m2g, "OU M2 + GARCH", "firebrick"),
    ]:
        axes[0].plot(t_axis, crps_arr, label=label, color=color, lw=1.5)
    axes[0].set_ylabel("CRPS")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("GARCH family")
    for crps_arr, label, color in [
        (crps_gg, "GARCH-Gaussian", "steelblue"),
        (crps_gt, "GARCH-t",        "darkorange"),
        (crps_eg, "EGARCH",         "firebrick"),
    ]:
        axes[1].plot(t_axis, crps_arr, label=label, color=color, lw=1.5)
    axes[1].set_ylabel("CRPS")
    axes[1].set_xlabel("Test day")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig3_crps_time.png", dpi=150, bbox_inches='tight')
    plt.show()

   
    #  FIGURE 4 — Results table + params table
    
    # results
    columns_r = ['Model', 'CRPS ', 'RMSE', 'MAE', 'Coverage²']
    rows_r = [
        ["OU M1",          f"{crps_m1.mean():.5f}",
         f"{m_m1['rmse']:.3f}",  f"{m_m1['mae']:.3f}",  f"{m_m1['coverage']*100:.1f}%"],
        ["OU M2",  f"{crps_m2.mean():.5f}",
         f"{m_m2['rmse']:.3f}",  f"{m_m2['mae']:.3f}",  f"{m_m2['coverage']*100:.1f}%"],
        ["OU M2+GARCH",    f"{crps_m2g.mean():.5f}",
         f"{m_m2g['rmse']:.3f}", f"{m_m2g['mae']:.3f}", f"{m_m2g['coverage']*100:.1f}%"],
        ["GARCH-Gaussian", f"{crps_gg.mean():.5f}",
         f"{m_gg['rmse']:.3f}",  f"{m_gg['mae']:.3f}",  f"{m_gg['coverage']*100:.1f}%"],
        ["GARCH-t",        f"{crps_gt.mean():.5f}",
         f"{m_gt['rmse']:.3f}",  f"{m_gt['mae']:.3f}",  f"{m_gt['coverage']*100:.1f}%"],
        ["EGARCH",         f"{crps_eg.mean():.5f}",
         f"{m_eg['rmse']:.3f}",  f"{m_eg['mae']:.3f}",  f"{m_eg['coverage']*100:.1f}%"],
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13, 6))
    axes[0].axis('off')
    styled_table(axes[0], rows_r, columns_r)
    axes[0].set_title("Performance metrics\n",
                       fontsize=9, loc='left', pad=4)

    # params
    columns_p = ['Model', 'κ / pers', 'σ / ω', 'α', 'β', 'γ - degrees of freedom', 'AIC', 'BIC']
    rows_p = [
        ["OU M1",
         f"κ={ou_m1['kappa']:.3f}",
         f"σ={ou_m1['sigma']:.4f}", '—', '—', '—',
         f"{ou_m1['aic']:.1f}", f"{ou_m1['bic']:.1f}"],
        ["OU M2",
         f"κ={ou_m2['kappa']:.3f}  A={ou_m2['A']:.3f}",
         f"σ={ou_m2['sigma']:.4f}", '—', '—', '—',
         f"{ou_m2['aic']:.1f}", f"{ou_m2['bic']:.1f}"],
        ["OU M2+GARCH",
         f"κ={ou_m2['kappa']:.3f}",
         f"ω={garch_r['omega']:.2e}",
         f"{garch_r['alpha']:.4f}", f"{garch_r['beta']:.4f}", '—',
         f"{garch_r['aic']:.1f}", f"{garch_r['bic']:.1f}"],
        ["GARCH-Gaussian",
         f"pers={ga_g['alpha']+ga_g['beta']:.4f}",
         f"ω={ga_g['omega']:.2e}",
         f"{ga_g['alpha']:.4f}", f"{ga_g['beta']:.4f}", '—',
         f"{ga_g['aic']:.1f}", f"{ga_g['bic']:.1f}"],
        ["GARCH-t",
         f"pers={ga_t['alpha']+ga_t['beta']:.4f}",
         f"ω={ga_t['omega']:.2e}",
         f"{ga_t['alpha']:.4f}", f"{ga_t['beta']:.4f}",
         f"df={ga_t['df']:.2f}",
         f"{ga_t['aic']:.1f}", f"{ga_t['bic']:.1f}"],
        ["EGARCH",
         f"pers≈{eg['beta']:.4f}",
         f"ω={eg['omega']:.4f}",
         f"{eg['alpha']:.4f}", f"{eg['beta']:.4f}",
         f"γ={eg['gamma']:.4f}",
         f"{eg['aic']:.1f}", f"{eg['bic']:.1f}"],
    ]
    
    axes[1].axis('off')
    styled_table(axes[1], rows_p, columns_p)
    axes[1].set_title("Model parameters", fontsize=9, loc='left', pad=4)

    fig.suptitle(f"Results summary — Train period: {chunk_label}",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("fig4_tables.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # FIGURE 5 — QQ-plot
    plot_qq_comparison(np.diff(np.log(prices_train)), ga_t, chunk_label)
    print("\nDone. Saved: fig1_ou_family | fig2_garch_family | "
          "fig3_crps_time | fig4_tables")
