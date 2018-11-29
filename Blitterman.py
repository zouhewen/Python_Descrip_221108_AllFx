# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 08:57:41 2018

@author: Natali Blondeau
Black Litterman model
#https://github.com/omartinsky/QuantAndFinancial/blob/master/black_litterman/black_litterman.ipynb
"""



import scipy.optimize


import numpy as np
import math as math
import pandas as pd
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from empyrical import max_drawdown, alpha_beta
import scipy.stats as stats
from datetime import datetime
from dateutil.parser import parse
import array as arr


from fpdf import FPDF
import matplotlib.backends.backend_pdf

# Calculates portfolio mean return
def port_mean(W, R):
    return sum(R * W)

# Calculates portfolio variance of returns
def port_var(W, C):
    return np.dot(np.dot(W, C), W)

# Combination of the two functions above - mean and variance of returns calculation
def port_mean_var(W, R, C):
    return port_mean(W, R), port_var(W, C)


# Given risk-free rate, assets returns and covariances, this function calculates
# mean-variance frontier and returns its [x,y] points in two arrays
def solve_frontier(R, C, rf):
    def fitness(W, R, C, r):
        # For given level of return r, find weights which minimizes portfolio variance.
        mean, var = port_mean_var(W, R, C)
        penalty = 100 * abs(
            mean - r)  # Big penalty for not meeting stated portfolio return effectively serves as optimization constraint
        return var + penalty

    frontier_mean, frontier_var = [], []
    n = len(R)  # Number of assets in the portfolio
    for r in np.linspace(min(R), max(R), num=20):  # Iterate through the range of returns on Y axis
        W = np.ones([n]) / n  # start optimization with equal weights
        b_ = [(0, 1) for i in range(n)]
        c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
        optimized = scipy.optimize.minimize(fitness, W, (R, C, r), method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success:
            raise BaseException(optimized.message)
        # add point to the efficient frontier [x,y] = [optimized.x, r]
        frontier_mean.append(r)
        frontier_var.append(port_var(optimized.x, C))
    return np.array(frontier_mean), np.array(frontier_var)

# Given risk-free rate, assets returns and covariances, this function calculates
# weights of tangency portfolio with respect to sharpe ratio maximization
def solve_weights(R, C, rf):
    def fitness(W, R, C, rf):
        mean, var = port_mean_var(W, R, C)  # calculate mean/variance of the portfolio
        util = (mean - rf) / np.sqrt(var)  # utility = Sharpe ratio
        return 1 / util  # maximize the utility, minimize its inverse value
    n = len(R)
    W = np.ones([n]) / n  # start optimization with equal weights
    b_ = [(0., 1.) for i in range(n)]  # weights for boundaries between 0%..100%. No leverage, no shorting
    c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # Sum of weights must be 100%
    optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success: raise BaseException(optimized.message)
    return optimized.x



def compute_historical_performance(daily_returns ,W):
    ret_p = np.log(1+W@daily_returns)
    cum_p = np.exp(np.cumsum(ret_p))
    return cum_p

def max_drawdown(X):
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd

class Result:
    def __init__(self, W, tan_mean, tan_var, front_mean, front_var):
        self.W=W
        self.tan_mean=tan_mean
        self.tan_var=tan_var
        self.front_mean=front_mean
        self.front_var=front_var

def optimize_frontier(R, C, rf):
    W = solve_weights(R, C, rf)
    tan_mean, tan_var = port_mean_var(W, R, C)  # calculate tangency portfolio
    front_mean, front_var = solve_frontier(R, C, rf)  # calculate efficient frontier
    # Weights, Tangency portfolio asset means and variances, Efficient frontier means and variances
    return Result(W, tan_mean, tan_var, front_mean, front_var)

def display_assets(names, R, C, color='black'):
    plt.scatter([C[i, i] ** .5 for i in range(n)], R, marker='x', color=color), plt.grid(True)  # draw assets
    for i in range(n):
        plt.text(C[i, i] ** .5, R[i], '  %s' % names[i], verticalalignment='center', color=color) # draw labels

def display_frontier(result, label=None, color='black'):
    plt.text(result.tan_var ** .5, result.tan_mean, '   tangent', verticalalignment='center', color=color)
    plt.scatter(result.tan_var ** .5, result.tan_mean, marker='o', color=color), plt.grid(True)
    plt.plot(result.front_var ** .5, result.front_mean, label=label, color=color), plt.grid(True)  # draw efficient frontier

# Call function to get mean and covar

# Function loads historical stock prices of nine major S&P companies and returns them together
# with their market capitalizations, as of 2013-07-01
def load_data():
    symbols = ['BRL  Curncy', 'PLN Curncy', 'TRY Curncy', 'ZAR Curncy', 'MXN Curncy']
    cap = {'BRL  Curncy': 403.02e9, 'PLN Curncy': 392.90e9, 'TRY Curncy': 283.60e9, 'ZAR Curncy': 243.17e9, 'MXN Curncy': 236.79e9}
    n = len(symbols)
    prices_out, caps_out = [], []

    prices_out = pd.read_excel("FX_MarketData.xlsx",sheet="Tier1_FX",header=0)
    for s in symbols:

        #prices = list(dataframe['close'])[-500:] # trailing window 500 days
       # prices_out.append(prices)
       caps_out.append(cap[s])
    return symbols, prices_out, caps_out

names, prices, caps = load_data()
n = len(names)

# Function takes historical stock prices together with market capitalizations and
# calculates weights, historical returns and historical covariances
def assets_historical_returns_and_covariances(subprices):

    M_prices = np.matrix(subprices)  # create numpy matrix from prices
    # create matrix of historical returns
    rows, cols = M_prices.shape
    returns = np.zeros([rows-1, cols])

    for c in range(cols):
        for r in range(rows-1):
            p0, p1 = M_prices[r, c], M_prices[r+1, c]
            returns[r, c] = (p1 / p0) - 1


    # calculate returns
  #  expreturns = np.array([])

  #  for x in range(cols):
  #     expreturns = np.append(expreturns, np.mean(returns[x]))
    # calculate covariances
    expreturns=returns.mean(0)*100
    dailyreturns=pd.DataFrame(returns)
    covars = dailyreturns.cov()*10000
    covars1 = covars.values

     #Statistics
    symb =pd.Series(names)
    mean = dailyreturns.mean(0)*100
    stdev = dailyreturns.std(0)*100
    var = np.var(dailyreturns,axis=0)*10000
    skew= dailyreturns.skew(0)
    kurtosis=dailyreturns.kurtosis(0)
    descrip = pd.concat([symb,mean,stdev,var,skew,kurtosis],axis=1)
    descrip.columns=["Instrument","Mean","Standard Deviation","VAR","Skew","Kurtosis"]

    corr=dailyreturns.corr()
    ax=sns.heatmap(corr)
    fig = ax.get_figure()
    fig.savefig("HeatMapCorrelations.png")
    plt.close()

    writer=pd.ExcelWriter('Descriptive.xlsx',engine='xlsxwriter')
    descrip.to_excel(writer, 'Statistics')
    covars_write = pd.DataFrame(covars1,columns=names,index=names)
    corr_write = pd.DataFrame(corr.values,columns=names,index=names)
    covars_write.to_excel(writer, 'Covariance')
    corr_write.to_excel(writer, 'Correlation')
    writer.save()

  #  expreturns = (1 + expreturns) ** 250 - 1  # Annualize returns
   # covars = covars * 250  # Annualize covariances
    return expreturns, covars1

W = np.array(caps) / sum(caps) # calculate market weights from capitalizations
subprices = pd.DataFrame(prices)
subprices = subprices.drop("Dates",axis=1)
R, C = assets_historical_returns_and_covariances(subprices)
rf = .015  # Risk-free rate

display(pd.DataFrame({'Return': R, 'Weight (based on market cap)': W}, index=names).T)
display(pd.DataFrame(C, columns=names, index=names))

# Creating a pdf file to save the graphs
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Black Litterman model", ln=1, align="C")
pdf.output("simple_demo.pdf")


#Mean-Variance Optimization (based on historical returns)
res1 = optimize_frontier(R, C, rf)

display_assets(names, R, C, color='blue')
display_frontier(res1, color='blue')
plt.xlabel('variance $\sigma$'), plt.ylabel('mean $\mu$'),plt.savefig("MVO based on historical returns.png"), plt.show()

display(pd.DataFrame({'Weight': res1.W}, index=names).T)
print(res1.W)

plt.close
#plt.savefig("MVO based on historical returns.jpeg")


#Black-litterman reverse optimization
# Calculate portfolio historical return and variance
mean, var = port_mean_var(W, R, C)

lmb = (mean - rf) / var  # Calculate risk aversion
Pi = np.dot(np.dot(lmb, C), W)  # Calculate equilibrium excess returns

#Mean-variance Optimization (based on equilibrium returns)
res2 = optimize_frontier(Pi+rf, C, rf)

display_assets(names, R, C, color='red')
display_frontier(res1, label='Historical returns', color='red')
display_assets(names, Pi+rf, C, color='green')
display_frontier(res2, label='Implied returns', color='green')
plt.xlabel('variance $\sigma$'), plt.ylabel('mean $\mu$'), plt.legend(),plt.savefig("MVO based on equilibrium returns.png"), plt.show()
display(pd.DataFrame({'Weight': res2.W}, index=names).T)
plt.close

#Determine views to the equilibrium returns and prepare views (Q) and link (P) matrices

def create_views_and_link_matrix(names, views):
    r, c = len(views), len(names)
    Q = [views[i][3] for i in range(r)]  # view matrix
    P = np.zeros([r, c])
    nameToIndex = dict()
    for i, n in enumerate(names):
        nameToIndex[n] = i
    for i, v in enumerate(views):
        name1, name2 = views[i][0], views[i][2]
        P[i, nameToIndex[name1]] = +1 if views[i][1] == '>' else -1
        P[i, nameToIndex[name2]] = -1 if views[i][1] == '>' else +1
    return np.array(Q), P

views = [('BRL  Curncy', '>', 'PLN Curncy', 0.02),
         ('MXN Curncy', '<', 'TRY Curncy', 0.02)]

Q, P = create_views_and_link_matrix(names, views)
print('Views Matrix')
display(pd.DataFrame({'Views':Q}))
print('Link Matrix')
display(pd.DataFrame(P))

#Optimization based on Equilibrium returns with adjusted views
tau = .025  # scaling factor

# Calculate omega - uncertainty matrix about views
omega = np.dot(np.dot(np.dot(tau, P), C), np.transpose(P))  # 0.025 * P * C * transpose(P)
# Calculate equilibrium excess returns with views incorporated
sub_a = np.linalg.inv(np.dot(tau, C))
sub_b = np.dot(np.dot(np.transpose(P), np.linalg.inv(omega)), P)
sub_c = np.dot(np.linalg.inv(np.dot(tau, C)), Pi)
sub_d = np.dot(np.dot(np.transpose(P), np.linalg.inv(omega)), Q)
Pi_adj = np.dot(np.linalg.inv(sub_a + sub_b), (sub_c + sub_d))

res3 = optimize_frontier(Pi_adj + rf, C, rf)

display_assets(names, Pi+rf, C, color='green')
display_frontier(res2, label='Implied returns', color='green')
display_assets(names, Pi_adj+rf, C, color='blue')
display_frontier(res3, label='Implied returns (adjusted views)', color='blue')
plt.xlabel('variance $\sigma$'), plt.ylabel('mean $\mu$'), plt.legend(), plt.savefig("MVO based on equilibrium excess returns with views.png"), plt.show()
display(pd.DataFrame({'Weight': res3.W}, index=names).T)
plt.close



