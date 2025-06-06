import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- 0. CONSTANTS AND UNIT CONVERSIONS ---
# Conversion factor for a 32 GiB sector to EiB, as B and Q are in EiB.
# 1 EiB = 2^30 GiB
GIB_32_IN_EIB = 32.0 / (2**30)
MILLION = 1_000_000


# --- 1. LOAD AND PREPARE HISTORICAL DATA ---

def prepare_simulation_data(df_raw):
    """
    Takes the raw historical dataframe and calculates the implied daily
    rates for p (locking), r (unlocking), and M (minting).
    """
    print("Preparing historical data and calculating implied daily rates...")
    df = df_raw.copy()

    # Rename columns for easier access
    if 'Network RB Power' in df.columns:
        df = df.rename(columns={'Network RB Power': 'B_hist'})
    if 'Q' in df.columns:
        df = df.rename(columns={'Q': 'Q_hist'})
    if 'C' in df.columns:
        df = df.rename(columns={'C': 'C_hist'})
    if 'L' in df.columns:
        df = df.rename(columns={'L': 'L_hist'})

    df = df.sort_index()
    df = df.interpolate(method='time')
    df = df.dropna(subset=['L_hist', 'C_hist', 'B_hist', 'Q_hist', 'Q_dot_new'])

    dL_hist = df['L_hist'].diff()
    dC_hist = df['C_hist'].diff()

    p_natural = (0.3 * df['C_hist'].shift(1) * df['Q_dot_new']) / np.maximum(df['B_hist'], df['Q_hist'])
    
    r_natural = p_natural - dL_hist
    M_natural = dC_hist - r_natural + p_natural

    df['p_natural'] = p_natural
    df['r_natural'] = r_natural
    df['M_natural'] = M_natural

    df = df.dropna()
    print("Data preparation complete.")
    return df

# --- 2. THE SIMULATION FUNCTION (Counterfactual Analysis) ---

def run_counterfactual_simulation(df_prepared, k):
    """
    Runs a counterfactual simulation on top of prepared historical data.
    """
    print(f"Running counterfactual simulation for k = {k}...")
    sim_df = df_prepared.copy()

    sim_df['L_sim'] = 0.0
    sim_df['C_sim'] = 0.0
    sim_df['L_sim'].iloc[0] = sim_df['L_hist'].iloc[0]
    sim_df['C_sim'].iloc[0] = sim_df['C_hist'].iloc[0]

    for i in range(len(sim_df) - 1):
        L_sim_i = sim_df['L_sim'].iloc[i]
        C_sim_i = sim_df['C_sim'].iloc[i]
        
        p_natural_i = sim_df['p_natural'].iloc[i+1]
        r_natural_i = sim_df['r_natural'].iloc[i+1]
        M_natural_i = sim_df['M_natural'].iloc[i+1]
        
        error_i = (0.3 * C_sim_i) - L_sim_i
        p_ctrl_i = p_natural_i + k * error_i

        dL_dt = p_ctrl_i - r_natural_i
        dC_dt = M_natural_i + r_natural_i - p_ctrl_i

        sim_df['L_sim'].iloc[i+1] = L_sim_i + dL_dt
        sim_df['C_sim'].iloc[i+1] = C_sim_i + dC_dt

    sim_df['Target_sim'] = 0.3 * sim_df['C_sim']
    sim_df['Error_sim'] = sim_df['Target_sim'] - sim_df['L_sim']
    
    sim_df['Pledge_32GiB_sim'] = (0.3 * sim_df['C_sim'] * MILLION * GIB_32_IN_EIB) / np.maximum(sim_df['B_hist'], sim_df['Q_hist'])
    
    print("Simulation finished.")
    return sim_df

# --- 3. NEW PLOTTING FUNCTION using Plotly (One by One) ---

def plot_all_figures(df_prepped, results, k_values):
    """
    Creates and displays four separate, interactive plots using Plotly.
    """
    colors = px.colors.qualitative.Plotly
    
    # === PLOT 1: System Evolution Over Time ===
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_prepped.index, y=df_prepped['L_hist'], name='Historical L(t)',
        mode='lines', line=dict(color='black', dash='dot', width=2)
    ))
    for i, k in enumerate(k_values):
        res_df = results[k]
        fig1.add_trace(go.Scatter(
            x=res_df.index, y=res_df['L_sim'], name=f'Controlled L(t) [k={k}]',
            mode='lines', line=dict(color=colors[i], width=2)
        ))
    fig1.update_layout(
        title='System Evolution: Total Locked (L) Over Time',
        xaxis_title='Date',
        yaxis_title='Value (Millions of FIL)',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig1.show()

    # === PLOT 2: L/C Ratio Over Time ===
    fig2 = go.Figure()
    fig2.add_hline(y=0.3, line_width=2, line_dash="dash", line_color="red", name='Target Ratio (0.3)')
    fig2.add_trace(go.Scatter(
        x=df_prepped.index, y=df_prepped['L_hist'] / df_prepped['C_hist'], name='Historical L/C Ratio',
        mode='lines', line=dict(color='black', dash='dot', width=2)
    ))
    for i, k in enumerate(k_values):
        res_df = results[k]
        fig2.add_trace(go.Scatter(
            x=res_df.index, y=res_df['L_sim'] / res_df['C_sim'], name=f'Controlled L/C Ratio [k={k}]',
            mode='lines', line=dict(color=colors[i], width=2)
        ))
    fig2.update_layout(
        title='Pledge Ratio (L/C) Over Time',
        xaxis_title='Date',
        yaxis_title='Ratio (Total Locked / Circulating Supply)',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig2.show()

    # === PLOT 3: Error Over Time ===
    fig3 = go.Figure()
    fig3.add_hline(y=0, line_width=2, line_dash="dash", line_color="red", name='Target (Error = 0)')
    for i, k in enumerate(k_values):
        res_df = results[k]
        fig3.add_trace(go.Scatter(
            x=res_df.index, y=res_df['Error_sim'], name=f'Error for k={k}',
            mode='lines', line=dict(color=colors[i], width=2)
        ))
    fig3.update_layout(
        title='Error [Target - L(t)] Over Time',
        xaxis_title='Date',
        yaxis_title='Error (Millions of FIL)',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig3.show()
    
    # === PLOT 4: Pledge per 32GiB Sector ===
    fig4 = go.Figure()
    if 'Commit Pledge per 32GiB QAP' in df_prepped.columns:
        fig4.add_trace(go.Scatter(
            x=df_prepped.index, y=df_prepped['Commit Pledge per 32GiB QAP'], name='Historical Pledge',
            mode='lines', line=dict(color='black', dash='dot', width=2)
        ))
    for i, k in enumerate(k_values):
        res_df = results[k]
        fig4.add_trace(go.Scatter(
            x=res_df.index, y=res_df['Pledge_32GiB_sim'], name=f'Controlled Pledge [k={k}]',
            mode='lines', line=dict(color=colors[i], width=2)
        ))
    fig4.update_layout(
        title='Pledge per 32GiB Sector Over Time',
        xaxis_title='Date',
        yaxis_title='Pledge (FIL)',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig4.show()


# --- 4. MAIN EXECUTION ---

# Load your data
# IMPORTANT: Replace 'path/to/your/filecoin_data.csv' with your actual file path.
try:
    df_raw = pd.read_csv('filecoin_data.csv', index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
except FileNotFoundError:
    print("Error: Data file not found. Please update the file path.")
    print("Using randomly generated dataframe for demonstration purposes.")
    days = 365 * 2
    date_rng = pd.date_range(start='2023-01-01', periods=days, freq='D')
    df_raw = pd.DataFrame(index=date_rng)
    df_raw['Q'] = np.linspace(120, 480, days) + np.random.randn(days) * 20
    df_raw['Network RB Power'] = np.linspace(100, 500, days) + np.random.randn(days) * 10
    df_raw['C'] = np.linspace(50, 250, days) + np.sin(np.linspace(0, 20, days)) * 10
    df_raw['L'] = 0.2 * df_raw['C'] + np.random.randn(days) * 5 + 10
    df_raw['Q_dot_new'] = np.random.uniform(0.5, 2.0, size=days)
    df_raw['Commit Pledge per 32GiB QAP'] = (0.3 * df_raw['C'] * MILLION * GIB_32_IN_EIB) / np.maximum(df_raw['Network RB Power'], df_raw['Q']) + np.random.randn(days)*0.05

df_prepped = prepare_simulation_data(df_raw)
df_prepped.index=pd.to_datetime(df_prepped['t'])
k_values = [0.01, 0.05, 0.2]
results = {}

for k in k_values:
    results[k] = run_counterfactual_simulation(df_prepped, k)

# Call the new plotting function
plot_all_figures(df_prepped, results, k_values)