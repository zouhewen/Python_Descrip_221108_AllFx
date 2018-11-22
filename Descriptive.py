# -*- codinpg: utf-8 -*-
"""
Spyder Editor
#Implementing Descriptive Statistics

"""


import numpy as np
import pandas as pd
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from empyrical import max_drawdown, alpha_beta
import scipy.stats as stats
from datetime import datetime
from dateutil.parser import parse

#Descriptive statistics
def descriptive(file="FX_MarketData.xlsx",sheet="All_FX",start_date='22-10-2013', end_date='22-11-2018'):
 
   # os.chdir('C:\Users\Natali & Pierre\Documents\Natali\EM Strategy\Python\Nat\Python_Descriptive')
    
   # input_file=file
    
        
    prices = pd.read_excel(file, sheet_name = sheet,header=0)
    prices.drop(prices.columns[[0]], axis=1) 
    fund_num = len(prices.columns)  - 1

    mask=(prices['Dates'] > start_date) & (prices['Dates'] <= end_date)
   
    subprices = prices.loc[mask]
    #print(prices.loc[mask])
    
    # Calculating Daily Returns
    dates=subprices.iloc[:,0]
    dates = pd.DataFrame(dates)
    dates=dates.drop(dates.index[0])
    test=subprices.copy()
    test=test.drop("Dates",axis=1)
    daily_pc= test/test.shift(1) -1
    daily_pc=daily_pc.drop(daily_pc.index[0])
    daily_pc_returns=pd.concat([dates,daily_pc],axis=1)
    daily_pc_returns=daily_pc_returns.drop(daily_pc_returns.index[-1])
    
    #Statistics
    mean = daily_pc_returns.mean(0)*100
    stdev = daily_pc_returns.std(0)*100
    var = np.var(daily_pc_returns,axis=0)
    skew= daily_pc_returns.skew(0)
    kurtosis=daily_pc_returns.kurtosis(0)
    descrip = pd.concat([mean,stdev,var,skew,kurtosis],axis=1)
    descrip.columns=["Mean","Standard Deviation","VAR","Skew","Kurtosis"]
    
    covar = daily_pc_returns.cov()*10000
    corr=daily_pc_returns.corr()
    ax=sns.heatmap(corr)
    fig = ax.get_figure()
    fig.savefig("HeatMapCorrelations.png")
    
    writer=pd.ExcelWriter('Descriptive.xlsx',engine='xlsxwriter')
    descrip.to_excel(writer, 'Statistics')
    covar.to_excel(writer, 'Covariance')
    corr.to_excel(writer, 'Correlation')
    writer.save()


    return;
    
#



descriptive()

