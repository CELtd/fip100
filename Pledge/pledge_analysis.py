#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:41:36 2025

@author: juanpablomadrigalcianci
"""

#!/usr/bin/env python
"""
fil_pledge_causality.py  —  Causal diagnostics for Filecoin pledge decline
Author: <your-name>        Date: 2025-05-27
Python 3.11; tested on pandas-2.2, statsmodels-0.15, linearmodels-5.4
---------------------------------------------------------------------
"""

# --------------------------------------------------------------------
# 0.  Imports & constants
# --------------------------------------------------------------------
import os, warnings, pathlib, typing, itertools as it
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from packaging import version
import statsmodels as sm_pkg

# optional (synthetic control)
try:
    from syntheticcontrol.method import Synth
except ImportError:
    Synth = None

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH   = 'filecoin_data.csv'       # <- update
SAVE_DIR    = pathlib.Path("results")
EVENT_DATE  = pd.Timestamp("2024-10-15")   # v19 pledge-bug fix height
HAC_LAGS    = 21                           # ~ three-week kernel for NW


# --------------------------------------------------------------------
# 1.  Helpers
# --------------------------------------------------------------------
def mkdir(p: pathlib.Path):
    if not p.exists(): p.mkdir(parents=True, exist_ok=True)

def hac_ols(y: pd.Series, X: pd.DataFrame, lags: int = HAC_LAGS):
    mod = sm.OLS(y, sm.add_constant(X))
    res = mod.fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return res

def adf_kpss(series: pd.Series, name: str):
    adf = adfuller(series.dropna(), autolag='AIC')
    kps = kpss(series.dropna(), nlags="auto")
    return {
        "series": name,
        "adf_stat": adf[0], "adf_p": adf[1],
        "kpss_stat": kps[0], "kpss_p": kps[1]
    }

def save_regression(res, fname: str):
    SAVE_DIR.joinpath(fname).with_suffix(".txt").write_text(res.summary().as_text())


# --------------------------------------------------------------------
# 2.  Load & engineer variables
# --------------------------------------------------------------------
    
    
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- robust datetime parsing ------------------------------------------
    if np.issubdtype(df['stateTime'].dtype, np.number):
        # numeric → treat as epoch-seconds
        df['stateTime'] = pd.to_datetime(df['stateTime'], unit='s')
    else:
        # string → let pandas parse ISO / RFC dates; fall back to epoch-secs
        dt = pd.to_datetime(df['stateTime'], errors='coerce', utc=False, infer_datetime_format=True)
        # any NaT left? try interpreting those entries as epoch-seconds too
        nat_mask = dt.isna()
        if nat_mask.any():
            dt.loc[nat_mask] = pd.to_datetime(df.loc[nat_mask, 'stateTime'].astype(float), unit='s', errors='coerce')
        df['stateTime'] = dt
    # -----------------------------------------------------------------------

    df = df.sort_values('stateTime').set_index('stateTime')

    # Block reward flow
    df['BR']      = df['Total FIL Mined'].diff().fillna(0)

    # Expected 20-day reward per 32 GiB
    df['ER']      = 20 * df['BR'] / df['Network QA Power']

    # Consensus pledge term
    df['CP_th']   = (0.30 * df['Protocol Circulating Supply'] /
                    df[['Network QA Power','Baseline Power']].max(axis=1))

    # Theory pledge
    df['P_th']    = df['ER'] + df['CP_th']

    # Locked stake ratio
    df['locked_ratio'] = df['Total FIL Locked'] / df['Protocol Circulating Supply']

    # Observed & residual
    df = df.rename(columns={'Commit Pledge per 32GiB QAP':'P_obs',
                            'FIL-USD':'FILUSD'})
    df['gap']     = df['P_obs'] - df['P_th']

    # Event dummies for ITS
    df['post_event']   = (df.index >= EVENT_DATE).astype(int)
    df['trend_post']   = df['post_event'] * (np.arange(len(df)) -
                        np.where(df['post_event']==1)[0][0])

    return df


# --------------------------------------------------------------------
# 3.  Analyses
# --------------------------------------------------------------------
def elasticity_ols(df: pd.DataFrame):
    y = np.log(df['P_obs'])
    lagvars = {
        'dlog_price'  : np.log(df['FILUSD']).diff(),
        'dlog_Q'      : np.log(df['Network QA Power']).diff(),
        'dlog_supply' : np.log(df['Protocol Circulating Supply']).diff()
    }
    X = pd.DataFrame(lagvars).dropna()
    res = hac_ols(y.loc[X.index], X)
    save_regression(res, "elasticity_ols")
    print(res.summary())


def gap_regression(df: pd.DataFrame):
    X = sm.add_constant(df[['FILUSD','locked_ratio']].dropna())
    res = hac_ols(df['gap'].loc[X.index], X)
    save_regression(res, "gap_regression")
    print(res.summary())



def vecm_causality(df: pd.DataFrame):
    vec_df = df[['P_obs', 'FILUSD', 'locked_ratio']].pct_change().dropna()

    # 1. pick cointegration rank (5 % level)
    if version.parse(sm_pkg.__version__) >= version.parse("0.14"):
        from statsmodels.tsa.vector_ar.vecm import select_coint_rank
        rank = select_coint_rank(vec_df, det_order=0, k_ar_diff=2, signif=0.05).rank
    else:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        joh = coint_johansen(vec_df, det_order=0, k_ar_diff=2)
        rank = (joh.lr1 > joh.cvt[:, 1]).sum()   # 5 % critical column

    # 2. fit VECM
    from statsmodels.tsa.vector_ar.vecm import VECM
    vecm_res = VECM(vec_df, k_ar_diff=2, coint_rank=rank).fit()

    # 3. *Always* get a VARResults object
    try:
        var_res = vecm_res.vecm_to_var()            # >=0.12
    except AttributeError:
        # very old versions: re-estimate VAR on same data
        var_res = VAR(vec_df).fit(maxlags=2, ic='aic')

    # 4. Granger causality tests
    tests = [
        ('P_obs',  ['FILUSD']),        # does price → pledge?
        ('P_obs',  ['locked_ratio']),  # does locked stake → pledge?
    ]
    for caused, causing in tests:
        res = var_res.test_causality(caused, causing, kind='f')
        print(f"\nGranger test: {causing} ⇒ {caused}")
        print(res.summary())

    # 5. save summary
    (SAVE_DIR / "vecm_summary.txt").write_text(vecm_res.summary().as_text())
    
    
def impulse_response(df: pd.DataFrame):
    var_df = df[['P_obs','FILUSD','locked_ratio']].pct_change().dropna()

    # keep real dates for modelling
    mod = VAR(var_df).fit(maxlags=2, ic='aic')
    irf = mod.irf(30)

    # build a date index for the horizon
    start_date = var_df.index[0]              # e.g. 2020-10-15
    horizon    = pd.date_range(start_date, periods=31, freq='D')

    fig = irf.plot(orth=True)
    for ax in fig.axes:
        ax.set_xticks(range(31))
        ax.set_xticklabels(horizon.strftime('%Y-%m-%d'), rotation=45, ha='right')
        ax.set_xlabel("Calendar day")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "plots" / "irf_calendar.png", dpi=300)
    
    
def interrupted_time_series(df: pd.DataFrame):
    X = sm.add_constant(df[['post_event','trend_post','FILUSD']])
    res = hac_ols(df['P_obs'], X)
    save_regression(res, "its_v19")
    print(res.summary())


def rolling_ols(df: pd.DataFrame, window: int = 180):
    slopes = []
    for i in range(window, len(df)):
        sub = df.iloc[i-window:i]
        res = hac_ols(np.log(sub['P_obs']),
                      np.log(sub[['FILUSD','Network QA Power']]))
        slopes.append(res.params['FILUSD'])
    df.loc[df.index[window:], 'beta_price'] = slopes
    df['beta_price'].plot(figsize=(10,3), title='Rolling β_price (180-day)')
    plt.tight_layout(); plt.savefig(SAVE_DIR/"plots"/"rolling_beta.png", dpi=300)


def synthetic_control_placeholder():
    if Synth is None:
        print(">>> synthetic-control skipped (package not installed)")
        return
    # Skeleton: user must load peer-chain stake ratios into `donor_df`
    # donor_df columns: ['date','fil','polygon','cosmos', ...]
    # treat unit = 'fil', donors = others
    pass


# --------------------------------------------------------------------
# 4.  Main driver
# --------------------------------------------------------------------
def main():
    mkdir(SAVE_DIR); mkdir(SAVE_DIR/"plots")

    df = pd.read_csv(DATA_PATH)
    df = engineer(df)

    # --- Diagnostics ---
    stats = pd.DataFrame([adf_kpss(df['P_obs'],'P_obs'),
                          adf_kpss(df['FILUSD'],'FILUSD')])
    stats.to_csv(SAVE_DIR/"unit_root_tests.csv", index=False)

    # --- Analyses ---
    elasticity_ols(df)
    gap_regression(df)
    vecm_causality(df)
    impulse_response(df)
    interrupted_time_series(df)
    rolling_ols(df)

    synthetic_control_placeholder()

    print("\n[✓] All analyses complete. Results in ./results/")

if __name__ == "__main__":
    main()
