#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 
from scipy.optimize import minimize

def moment(series,degree):
    n = len(series)
    moment = sum(x ** degree for x in series) / n
    return moment

def skewness(x):
    return(moment(x,3)/(np.std(x)**3))

def kurtosis(x):
    return(moment(x,4)/(np.std(x)**4))
    
def annualized_returns(df):

    num_years = len(df)/12 
    cumulative_returns = (df + 1).cumprod()
    annualized_returns = cumulative_returns.iloc[-1] ** (1 / num_years) - 1

    return annualized_returns

def annualized_volatility(returns):
    
    returns = np.array(returns)
    n = 12 
    volatility = np.std(returns) * np.sqrt(n)

    return volatility
    

def sharpe_ratio(returns, risk_free_rate):
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_dev = np.std(excess_returns)

    if std_dev == 0:
        return 0.0

    sharpe_ratio = mean_return / std_dev
    return sharpe_ratio

def Jarque_Barque_test(x):
    s = skewness(x)
    k = kurtosis(x)
    return((len(x)/6)*(s**2+(k-3)**2)/4)

def drawdowns(returns):
    returns = np.array(returns)
    cumulative_returns = np.cumprod(1+returns)
    peaks = np.maximum.accumulate(cumulative_returns) #np.maximum.accumulate accumulates the result of applying the operator to all elements.
    drawdowns = (cumulative_returns - peaks) / peaks
    return drawdowns

def semi_deviation (returns):
    
    returns = np.array(returns)
    mean_return = np.mean(returns)
    negative_returns = returns[returns<mean_return]
    squared_negative_returns = np.square(negative_returns)
    semi_deviation = np.sqrt(np.mean(squared_negative_returns))

    return semi_deviation

def historical_VaR(returns,confidence_level):
    returns = np.array(returns)
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    VaR = abs(sorted_returns[index])

    return VaR

def historical_CVaR(returns,confidence_level):
    returns = np.array(returns)
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    VaR = abs(sorted_returns[index])
    CVaR = np.mean(sorted_returns[:index])

    return CVaR

def gaussian_VaR(returns,confidence_level):
    returns = np.array(returns)
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    z_score = np.abs(np.percentile(returns, (1 - confidence_level) * 100))
    var = mean_return - z_score * std_dev

    return var

def gaussian_CVaR(returns,confidence_level):

    returns = np.array(returns)
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    z_score = np.abs(np.percentile(returns, (1 - confidence_level) * 100))
    cvar = mean_return - (z_score * std_dev) / (1 - confidence_level)

    return cvar

def cornish_fisher_var(returns, confidence_level):
    
    returns = np.array(returns)
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    z_score = np.abs(np.percentile(returns, (1 - confidence_level) * 100))
    
    skewness = np.mean(((returns - mean_return) / std_dev) ** 3)
    kurtosis = np.mean(((returns - mean_return) / std_dev) ** 4)

    z_score_cornish_fisher = z_score + (z_score**2 - 1) * skewness / 6 + (z_score**3 - 3 * z_score) * kurtosis / 24 - (2 * z_score**3 - 5 * z_score) * skewness**2 / 36

    var = mean_return - z_score_cornish_fisher * std_dev

    return var


def minimize_vol(target_return, er, cov):
    
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

def optimal_weights(n_points, er, cov):
    """
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov):
    
    weights = optimal_weights(n_points, er, cov) 
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style='.-')

def msr(riskfree_rate, er, cov):
    
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def portfolio_return(weights, returns):
    return weights.T@returns

def portfolio_vol(weights, cov):
    return weights.T@cov@weights

def portfolio_sharpe(portf_return,riskfree_rate,portf_vol):
    return (portf_return-riskfree_rate)/portf_vol
    
def cppi(risky_r,safe_r=None,m=3,start=100000,floor=0.85,rfr=0.03,drawdown=None):
    dates=risky_r.index
    n_steps=len(dates)
    account_value=start
    floor_value=floor*start
    
    if isinstance(risky_r,pd.Series):
        risky_r=pd.DataFrame(risky_r,columns=["R"])
        
    if safe_r is None:
        safe_r=pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:]=rfr/12
    
    account_history=pd.DataFrame().reindex_like(risky_r)
    floor_history=pd.DataFrame().reindex_like(risky_r)
    cushion_history=pd.DataFrame().reindex_like(risky_r)
    risky_w_history=pd.DataFrame().reindex_like(risky_r)
    peak_history=pd.DataFrame().reindex_like(risky_r)
    peak=0
    for step in range(len(dates)):
        cushion=(account_value-floor_value)/account_value
        risky_w=m*cushion
        risky_w=np.minimum(risky_w,1)
        risky_w=np.maximum(risky_w,0)
        safe_w=1-risky_w
        risky_alloc=account_value*risky_w
        safe_alloc=account_value*safe_w
        account_value=risky_alloc*(1+risky_r.iloc[step])+safe_alloc*(1+safe_r.iloc[step])
        if drawdown is not None:
            peak=np.maximum(peak,account_value)
            floor_value=peak*(1-drawdown)
        cushion_history.iloc[step]=cushion
        risky_w_history.iloc[step]=risky_w
        account_history.iloc[step]=account_value
        floor_history.iloc[step]=floor_value
        peak_history.iloc[step]=peak
            
            
    risky_wealth=start*(1+risky_r).cumprod()
    
    
    
    backtest={
        "Wealth":account_history,
        "Risky Wealth":risky_wealth,
        "Risk Budget":cushion_history,
        "Risky Allocation":risky_w_history,
        "Multiplier":m,
        "Start":start,
        "Floor":floor,
        "Risky Asset Returns":risky_r,
        "Safe Asset Returns":safe_r,
        "Peak Value":peak_history,
        "Drawdown":drawdown,
        "Floor History":floor_history
    }
    return backtest    
