#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:38:03 2025

@author: juanpablomadrigalcianci
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timezone
from functools import reduce   # for a clean one-liner with many dfs



df1=pd.read_csv('data/Commit_Pledge_per_32GiB_QAP.csv')
df2=pd.read_csv('data/FIL_Protocol_Circulating_Supply.csv')
df3=pd.read_csv('data/Network_Storage_Capacity.csv')
#df4=pd.read_csv('data/Daily_Active_Miners.csv')
#df5=pd.read_csv('data/Daily_Capacity_Added_by_Sector_RBP.csv')
df6=pd.read_csv('data/Daily_Capacity_Added_by_Sector_QAP.csv')
df7=pd.read_csv('data/Daily_Active_Faults.csv')
df8=pd.read_csv('data/Network_Block_Rewards_per_WinCount.csv')

def set_idx(df):
    return df.assign(stateTime=pd.to_datetime(df['stateTime'])).set_index('stateTime')

dfs = [df1,df2,df3,df6,df7,df8]          # list of your individual frames
df = reduce(lambda left, right:   # outer join on the index
              pd.concat([left, right], axis=1, join='outer'),
              map(set_idx, dfs))

df = df.sort_index() 
df['stateTime']=df.index
df.index=np.arange(len(df))


#%%

markets=['FIL','BTC','ETH']
for m in markets:
    
    START_DATE = df.stateTime.min()                      # inclusive
    END_DATE   = datetime.now(timezone.utc).date()  # today (UTC)
    
    # ----------------------------------------------------------------------
    # 1. Download historical data
    prices = yf.download(
        tickers=f"{m}-USD",      # Yahoo Finance symbol for Filecoin in USD :contentReference[oaicite:0]{index=0}
        start=START_DATE,
        end=str(END_DATE + pd.Timedelta(days=1)),  # yfinance's 'end' is exclusive
        interval="1d",
        progress=False,
        auto_adjust=False      # keep raw Close/Adj Close
    )['Close']
    
    prices['stateTime']=prices.index
    prices['stateTime']=pd.to_datetime(prices['stateTime'])
    df['stateTime']=pd.to_datetime(df['stateTime'])
    df=pd.merge(df,prices,on='stateTime')

'''
df.columns
Out[21]: 
Index(['Commit Pledge per 32GiB QAP', 'Protocol Circulating Supply',
       'Total FIL Mined', 'Total FIL Vested', 'Fil Reserve Disbursed',
       'Total FIL Burned', 'Total FIL Locked', 'Network QA Power',
       'Network RB Power', 'Baseline Power', '32 GiB', '64 GiB',
       'Active Faults', 'Block Rewards per WinCount', 'stateTime', 'FIL-USD',
       'BTC-USD', 'ETH-USD'],
'''

df=df.dropna()
#%%
df['onboard_32G'] = df['32 GiB']
df['onboard_64G'] = df['64 GiB']
df.drop(columns=['32 GiB','64 GiB'], inplace=True)

# daily onboarding totals
df['daily_onboard_QAP'] = (df['onboard_32G'] + df['onboard_64G']) * 32 / 1024  # PiB→EiB tweak as needed

# FIL price log-returns for stationarity
df['dlog_FILUSD'] = np.log(df['FIL-USD']).diff()
df['dlog_QAP']    = np.log(df['Network QA Power']).diff()

# storage-pledge and consensus-pledge terms (theory)
reward_per_day   = df['Total FIL Mined'].diff()
df['est_20-day_reward'] = 20 * reward_per_day / df['Network QA Power']
df['consensus_term'] = (0.30 * df['Protocol Circulating Supply'] /
                        df[['Network QA Power','Baseline Power']].max(axis=1))
df['Pledge_theory'] = df['est_20-day_reward'] + df['consensus_term']

# residual gap we’re modelling
df['gap'] = df['Commit Pledge per 32GiB QAP'] - df['Pledge_theory']

df.to_csv('filecoin_data.csv')
