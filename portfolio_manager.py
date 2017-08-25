import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.optimize as spo


def get_path(symbol) :
    return os.path.join('{}.csv'.format(str(symbol)))

def get_data(symbols,dates):
    df=pd.DataFrame(index=dates)

    if 'SPY' not in symbols:
        symbols.insert(0,'SPY')

    for s in symbols:
        df_temp=pd.read_csv(get_path(s),usecols=['Date','Adj Close'],index_col='Date',parse_dates=True,na_values=['nan'])
        df_temp=df_temp.rename(columns={'Adj Close':s})
        df=df.join(df_temp)
        if s=='SPY':
            df=df.dropna(subset=['SPY'])
    return df

def compute_daily_returns(df):
    daily_returns=df.copy()
    daily_returns[1:] = (df[1:]/df[:-1].values)-1
    daily_returns.ix[0,:]=0
    return daily_returns


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def compute_cummulative_return(df):
    cummulative_return=df.copy()
    cummulative_return=(df/df.ix[0,:])-1
    return cummulative_return

def portfolio(alloc,total_portfolio):
    dates = pd.date_range('2015-12-15', '2016-12-14')
    symbols = ['YHOO', 'APPL', 'FB', 'SPY']
    df = get_data(symbols, dates)
    df = df.loc[:,['YHOO', 'APPL', 'FB']]
    normed = df/df.ix[0]
    allocated_portfolio = normed*alloc
    portfolio_val = (allocated_portfolio*total_portfolio)
    df['PORTFOLIO'] = portfolio_val.sum(1)
    print(df.head())
    '''
    temp = compute_daily_returns(df).loc[:,['SPY' ,'PORTFOLIO']]
    ax = temp.plot(title="Daily Returns", label='SPY')
    ax.set_xlabel("DATE")
    ax.set_ylabel("RETURN")
    plt.show()
    '''
    return df


def plotter():
    dates=pd.date_range('2015-12-15','2016-12-14')
    symbols=['YHOO','APPL','FB','SPY']
    df=get_data(symbols,dates)
    print(compute_cummulative_return((df)).head())

    plot_data(df)
    daily_returns = compute_daily_returns(df)

    daily_returns['SPY'].hist(label="SPY", bins=20)
    daily_returns['YHOO'].hist(label="YHOO", bins=20)
    daily_returns['APPL'].hist(label="APPL", bins=20)
    daily_returns['FB'].hist(label="FB", bins=20)
    plt.legend(loc='upper left')
    plt.show()

    daily_returns.plot(kind='scatter', x='SPY', y='YHOO')
    beta_YHOO, alpha_YHOO= np.polyfit(daily_returns['SPY'], daily_returns['YHOO'], 1)
    print("alpha value of YHOO:{}".format(alpha_YHOO))
    print("beta value of YHOO:{}".format(beta_YHOO))
    plt.plot(daily_returns['SPY'], beta_YHOO*daily_returns['SPY']+ alpha_YHOO, '-', color='r')
    plt.show()
    daily_returns.plot(kind='scatter', x='SPY', y='APPL')
    beta_APPL, alpha_APPL = np.polyfit(daily_returns['SPY'], daily_returns['APPL'], 1)
    plt.plot(daily_returns['SPY'], daily_returns['SPY'] * beta_APPL + alpha_APPL, '-', color='r')
    plt.show()

    plot_data(daily_returns, "Daily Returns", "Date","Fraction")
    plot_data(compute_cummulative_return(df), "Cummulative Return", "Date", "Fraction")

def func(X):
    Y = (X-2)**2+5
    print("X={} Y={}".format(X ,Y))
    return Y

def optimizer():
    x_guess=0.0
    min = spo.minimize(func, x_guess, method='SLSQP',options={'disp':True})
    print("X_min={} Y_min={}".format(min.x,min.fun))

def error_func(C ,data):
    print(np.polyval(C, data[:,0]))
    err = np.sum((data[:,1]-np.polyval(C ,data[:,0]))**2)
    print(err)
    return err


def poly_optimizer(data, error_poly, degree):
    guess = np.poly1d(np.ones(degree+1, dtype=np.float32))
    c = spo.minimize(error_poly, guess, args=(data,), method='SLSQP', options={'disp': True})
    return np.poly1d(c.x)


def sharpe_ratio(alloc, total_portfolio, risk_free_rate):
    df = portfolio(alloc, total_portfolio)
    daily_return = compute_daily_returns(df).loc[:, ['PORTFOLIO']]
    return (daily_return.mean()-risk_free_rate*total_portfolio)/daily_return.std()


def optimize_by_sharpe_ratio(total_portfolio, risk_free_return):
    alloc = [0.33, 0.33, 0.34]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = ((0, 1), (0, 1), (0, 1))
    result = spo.minimize(sharpe_ratio, alloc, args=(total_portfolio, risk_free_return), method='SLSQP', bounds=bounds, constraints=constraints, options={'disp':True})
    return result
'''
degree = 4
data=[]
poly_coef=np.poly1d([1.5, -10, -5, 10, 50])
for x in range(-6, 6, 1):
    data.append([x, poly_coef(x)])
d = np.asarray(data, dtype=np.float32)
result = poly_optimizer(d, error_func, degree)
print(poly_coef)
print(result)

x=np.linspace(-6,6,21)
plt.plot(x ,np.polyval(poly_coef ,x) ,'m--' ,linewidth=2.0)
plt.plot(x ,np.polyval(result ,x) ,'m--' ,linewidth=2.0)
plt.show()
print(np.polyfit(d[:,0], d[:,1],4))
'''
'''
total_portfolio = 10000
d=optimize_by_sharpe_ratio(total_portfolio, 0.4)
print(d)
dates = pd.date_range('2015-12-15', '2016-12-14')
symbols = ['SPY']
df = get_data(symbols, dates)
d = portfolio(d.x, total_portfolio)
d = d.join(df)
temp = compute_daily_returns(d).loc[:,['PORTFOLIO']]
temp = temp.join(compute_daily_returns(df).loc[:, ['SPY']])
ax = temp.plot(title="Daily Returns", label='SPY')
ax.set_xlabel("DATE")
ax.set_ylabel("RETURN")
plt.show()
'''
