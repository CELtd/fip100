# dashboard.py

import streamlit as st
import os
import shutil
from datetime import date, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import jax.numpy as jnp
from diskcache import Cache
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Placeholder for User's Custom Modules ---
# In a real scenario, these would be in separate files. For this script to be self-contained,
# we will define dummy versions of them if they can't be imported.
try:
    import mechafil_jax.data as data
    import mechafil_jax.sim as sim
    import mechafil_jax.constants as C
    import mechafil_jax.date_utils as du
    import scenario_generator.utils as u
    from get_data import get_all # Assuming you have this file
except ImportError:
    st.error("Could not import custom modules (`mechafil_jax`, `get_data`, etc.). Using dummy data for demonstration.")
    # Create dummy classes and functions to allow the app to run
    class C:
        PIB_PER_SECTOR = 1 / 32768
        EIB_PER_SECTOR = (1 / 32768) / 1024
        
    PIB_PER_SECTOR=C.PIB_PER_SECTOR
    EIB_PER_SECTOR=C.EIB_PER_SECTOR
    
    class DummySim:
        def run_sim(*args, **kwargs):
            # Generate dummy results that have the same keys as the real simulation
            forecast_length = kwargs.get('forecast_length_days', 3650)
            n = forecast_length
            results = {
                'network_QAP_EIB': np.linspace(20, 100, n) + np.random.randn(n).cumsum(),
                'network_baseline_EIB': np.linspace(20, 250, n),
                'circ_supply': np.linspace(500e6, 800e6, n) + np.random.randn(n).cumsum() * 1e5,
                'day_rewards_per_sector': np.linspace(0.0002, 0.0001, n) + np.random.randn(n) * 1e-5,
                'day_pledge_per_QAP': np.linspace(5, 3, n) + np.random.randn(n) * 0.1,
                'network_locked': np.linspace(200e6, 400e6, n) + np.random.randn(n).cumsum() * 1e4,
                'day_network_reward': np.linspace(3e5, 2e5, n) + np.random.randn(n).cumsum(),
                'cum_network_reward': np.linspace(100e6, 1000e6, n),
                'cum_simple_reward': np.linspace(50e6, 500e6, n),
                '1y_sector_roi': np.linspace(0.3, 0.2, n) + np.random.randn(n) * 0.01,
                '1y_return_per_sector': np.linspace(1.5, 0.6, n) + np.random.randn(n) * 0.05,
                'rb_day_onboarded_power_pib': np.ones(n) * 10,
                'day_onboarded_power_QAP_PIB': np.ones(n) * 10 + np.random.rand(n),
                'day_renewed_power_QAP_PIB': np.ones(n) * 2 + np.random.rand(n),
            }
            return results

    class DummyData:
         def get_simulation_data(*args, **kwargs):
              return {'daily_burnt_fil': np.ones(3650) * 1e5}

    class DummyUtils:
        def get_t(start_date, end_date):
            return pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D')).tolist()
        def get_historical_daily_onboarded_power(*args): return None, np.ones(180) * 10
        def get_historical_renewal_rate(*args): return None, np.ones(180) * 0.6
        def get_historical_filplus_rate(*args): return None, np.ones(180) * 0.8
        
    sim = DummySim()
    data = DummyData()
    du = DummyUtils()
    u = DummyUtils()

    def get_all(start_date_str: str) -> pd.DataFrame:
        """Generates dummy data resembling the gas analysis source."""
        start_date = pd.to_datetime(start_date_str)
        end_date = date.today()
        dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'), utc=True)
        n = len(dates)
        update_date = pd.Timestamp('2025-04-15', tz='UTC')
        
        data_dict = { 'stat_date': dates }
        cols_to_gen = [
            'total_gas_used', 'provecommit_sector_gas_used', 'precommit_sector_gas_used',
            'provecommit_aggregate_gas_used', 'precommit_sector_batch_gas_used',
            'publish_storage_deals_gas_used', 'submit_windowed_post_gas_used',
            'burnt_fil', 'sector_rewards_360d_32gib', 'sector_provecommit_fee_32gib',
            'total_qa_bytes_power', 'prove_commit_sector_rbp'
        ]
        
        for col in cols_to_gen:
            pre_update_len = len(dates[dates < update_date])
            post_update_len = len(dates[dates >= update_date])
            series = np.random.randn(n).cumsum() * 1e6 + 1e8
            if post_update_len > 0 and pre_update_len > 0:
                shift_factor = 0.5 if 'commit' in col else 1.2
                series[pre_update_len:] += (np.random.randn(post_update_len).cumsum() * 1e5 + (series[pre_update_len-1] * shift_factor - series[pre_update_len-1]))
            data_dict[col] = series
            
        return pd.DataFrame(data_dict)

# --- Page and App Configuration ---
st.set_page_config(layout="wide", page_title="Filecoin Simulation Dashboard", page_icon="📈")

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "historical_days": 180,
    "smoothing_days": 30,
    "fip81_activation_date": date(2024, 11, 21),
    "auth_token": 'Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ',
    "cache_dir": "./cache_directory",
}

# --- Sidebar for User Inputs ---
st.sidebar.title("Controls")
st.sidebar.header("Simulation Parameters")
forecast_length = st.sidebar.slider("Forecast Length (Years)", 1, 20, 10)
lock_target = st.sidebar.slider("Lock Target (%)", 10, 50, 30) / 100.0
sector_duration = st.sidebar.slider("Sector Duration (Days)", 180, 1080, 540)

st.sidebar.header("Scenario Selection")
SCENARIO_MAPPING = {'0.5x': 0.5, '0.75x': 0.75, '1.0x': 1.0, '2.0x': 2.0}
selected_labels = st.sidebar.multiselect("Select Growth Scenarios", options=list(SCENARIO_MAPPING.keys()), default=['0.5x', '1.0x', '2.0x'])
selected_scaling_factors = [SCENARIO_MAPPING[label] for label in selected_labels]

st.sidebar.header("Fee Analysis Parameters")
FEE_FRAC_MAPPING = {'1% Fee': 0.01, '2% Fee': 0.02, '3% Fee': 0.03, '5% Fee': 0.05}
wp_fee_plateau = st.sidebar.slider("Fee Multiplier Plateau", 1, 50, 25)
wp_fee_increase_years = st.sidebar.slider("Fee Ramp-Up (Years)", 1, 10, 5)
wp_fee_increase_days = wp_fee_increase_years * 365
selected_fee_labels = st.sidebar.multiselect("Select Fee Fractions", options=list(FEE_FRAC_MAPPING.keys()), default=['1% Fee', '2% Fee'])
selected_fee_fracs = [FEE_FRAC_MAPPING[label] for label in selected_fee_labels]

st.sidebar.header("Gas Analysis Parameters")
gas_start_date = st.sidebar.date_input("Analysis Start Date", value=date(2025, 1, 1))
gas_update_date = st.sidebar.date_input("FIP Update Date", value=date(2025, 4, 15))


# --- Caching and Simulation Logic ---
@st.cache_data
def run_full_simulation(forecast_length_years, sector_duration_days, lock_target_fraction, rbp_scaling_factors):
    cache_dir = DEFAULT_CONFIG['cache_dir']
    if os.path.exists(cache_dir): shutil.rmtree(cache_dir)
    local_cache = Cache(cache_dir)
    current_date, mo_start = date.today() - timedelta(days=3), max(date.today().month, 1)
    start_date = date(current_date.year, mo_start, 1)
    forecast_length_days, end_date = int(forecast_length_years * 365), current_date + timedelta(days=int(forecast_length_years * 365))
    cache_key_data = f"offline_data_{start_date}_{current_date}_{end_date}"
    if cache_key_data in local_cache:
        offline_data, smoothed_rbp, smoothed_rr, smoothed_fpr = local_cache.get(cache_key_data)
    else:
        auth_token = DEFAULT_CONFIG.get("auth_token")
        offline_data = data.get_simulation_data(auth_token, start_date, current_date, end_date)
        hist_start_date = current_date - timedelta(days=DEFAULT_CONFIG["historical_days"])
        _, hist_rbp = u.get_historical_daily_onboarded_power(hist_start_date, current_date)
        _, hist_rr = u.get_historical_renewal_rate(hist_start_date, current_date)
        _, hist_fpr = u.get_historical_filplus_rate(hist_start_date, current_date)
        smoothed_rbp, smoothed_rr, smoothed_fpr = float(np.median(hist_rbp[-DEFAULT_CONFIG["smoothing_days"]:])), float(np.median(hist_rr[-DEFAULT_CONFIG["smoothing_days"]:])), float(np.median(hist_fpr[-DEFAULT_CONFIG["smoothing_days"]:]))
        local_cache.set(cache_key_data, (offline_data, smoothed_rbp, smoothed_rr, smoothed_fpr))
    all_results = []
    for scale_factor in rbp_scaling_factors:
        rbp = jnp.ones(forecast_length_days) * smoothed_rbp * scale_factor
        if scale_factor > 1.0: rr, fpr = (jnp.ones(forecast_length_days) * smoothed_rr, jnp.ones(forecast_length_days) * smoothed_fpr)
        else: rr, fpr = (jnp.ones(forecast_length_days) * min(1.0, smoothed_rr * scale_factor), jnp.ones(forecast_length_days) * min(1.0, smoothed_fpr * scale_factor))
        sim_results = sim.run_sim(rbp, rr, fpr, lock_target_fraction, start_date, current_date, forecast_length_days, sector_duration_days, offline_data, use_available_supply=False)
        all_results.append(sim_results)
    time_vector = du.get_t(start_date, end_date=end_date)
    return all_results, offline_data, time_vector, current_date

# --- Gas Analysis Functions ---
@st.cache_data
def load_and_process_gas_data(start_date, update_date):
    df = get_all(start_date.strftime('%Y-%m-%d'))
    df = df.sort_values(by='stat_date')
    df['stat_date'] = pd.to_datetime(df['stat_date'])
    df['onboarded_qa'] = df['total_qa_bytes_power'].diff()
    if not df.empty:
        df['onboarded_qa'].iloc[0] = np.copy(df['onboarded_qa'].iloc[1])
    update_ts = pd.Timestamp(update_date, tz='UTC')
    gas_cols = ['total_gas_used','provecommit_sector_gas_used','precommit_sector_gas_used','provecommit_aggregate_gas_used','precommit_sector_batch_gas_used','publish_storage_deals_gas_used','submit_windowed_post_gas_used']
    for c in gas_cols:
        with np.errstate(divide='ignore', invalid='ignore'):
            df[f'{c}_rel'] = np.nan_to_num(df[c] / df['total_gas_used'])
    df['burnt_fil'] = df['burnt_fil'] / 1e9
    return df, update_ts, gas_cols

def calculate_gas_summary(df, cols, update_ts, normalize_by_qap=False):
    pre_df = df[df['stat_date'] < update_ts]
    post_df = df[df['stat_date'] >= update_ts]
    summary_data = []
    for c in cols:
        with np.errstate(divide='ignore', invalid='ignore'):
            if normalize_by_qap and c != 'prove_commit_sector_rbp':
                mean_pre = (pre_df[c] / pre_df['prove_commit_sector_rbp'] / 10).mean()
                mean_post = (post_df[c] / post_df['prove_commit_sector_rbp'] / 10).mean()
            else:
                mean_pre, mean_post = pre_df[c].mean(), post_df[c].mean()
        pct_change = (mean_post - mean_pre) / mean_pre * 100 if mean_pre and not np.isnan(mean_pre) and mean_pre != 0 else 0
        summary_data.append({'Metric': c, 'Mean Before': mean_pre, 'Mean After': mean_post, '% Change': pct_change})
    return pd.DataFrame(summary_data)

# --- Plotting & Calculation Functions ---
def create_gas_timeseries_plot(df, gas_cols, update_ts, relative=False):
    fig = go.Figure()
    for col in gas_cols:
        col_to_plot = f"{col}_rel" if relative else col
        fig.add_trace(go.Scatter(x=df['stat_date'], y=df[col_to_plot], mode='lines', name=col))
    
    # CORRECTED: Use add_shape and add_annotation instead of add_vline
    fig.add_shape(
        type="line",
        x0=update_ts, y0=0, x1=update_ts, y1=1,
        line=dict(color="grey", width=2, dash="dash"),
        xref="x", yref="paper"
    )
    fig.add_annotation(
        x=update_ts, y=1.05, text="Update", showarrow=False,
        xref="x", yref="paper", font=dict(color="grey", size=12)
    )

    title = 'Relative Daily Gas Usage' if relative else 'Absolute Daily Gas Usage'
    yaxis_title = 'Relative Usage' if relative else 'Gas Used'
    fig.update_layout(title=title, template='plotly_white', yaxis_title=yaxis_title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_gas_barchart(summary_df, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Before', x=summary_df['Metric'], y=summary_df['Mean Before']))
    fig.add_trace(go.Bar(name='After', x=summary_df['Metric'], y=summary_df['Mean After']))
    fig.update_layout(barmode='group', title=title, template='plotly_white', yaxis_type="log", yaxis_title="Mean Value (Log Scale)")
    return fig

def create_network_metrics_plot(simulation_results, time_vector, labels, start_plot_date):
    plot_config = [
        {'key': 'network_QAP_EIB', 'title': 'Network QAP (EIB)', 'log_y': True, 'special_plot': 'baseline'},
        {'key': 'circ_supply', 'title': 'Circulating Supply (M-FIL)', 'scale': 1e6},
        {'key': 'day_rewards_per_sector', 'title': 'Daily Reward per Sector (FIL)'},
        {'key': 'day_pledge_per_QAP', 'title': 'Initial Pledge per QAP (FIL)'},
        {'key': 'network_locked', 'title': 'Network Locked Supply (M-FIL)', 'scale': 1e6},
        {'key': 'day_network_reward', 'title': 'Daily Network Reward (FIL)'},
        {'key': 'cum_network_reward', 'title': 'Cumulative Network Reward (M-FIL)', 'scale': 1e6},
        {'key': 'network_QAP_EIB', 'title': 'Daily QAP Growth (PiB/day)', 'transform': lambda x: np.diff(x) * 1024, 'special_plot': 'hline'},
        {'key': '1y_sector_roi', 'title': 'Standard 1-Year Sector ROI (%)', 'scale': 100, 'transform': lambda x: x[:-364]},
    ]
    fig = make_subplots(rows=3, cols=3, subplot_titles=[item['title'] for item in plot_config], vertical_spacing=0.1)
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']
    for i, item in enumerate(plot_config):
        row, col = (i // 3) + 1, (i % 3) + 1
        for sim_idx, sim_result in enumerate(simulation_results):
            y_data, t_data = sim_result[item['key']], time_vector
            if 'transform' in item:
                y_data, t_data = item['transform'](y_data), t_data[:len(item['transform'](y_data))] if len(item['transform'](y_data)) < len(t_data) else t_data
            if 'scale' in item: y_data = y_data / item.get('scale', 1.0)
            fig.add_trace(go.Scatter(x=t_data, y=y_data, mode='lines', name=labels[sim_idx], line=dict(color=colors[sim_idx]), legendgroup=f'group{sim_idx}', showlegend=(i == 0)), row=row, col=col)
        if item.get('special_plot') == 'baseline': fig.add_trace(go.Scatter(x=time_vector, y=simulation_results[0]['network_baseline_EIB'], mode='lines', name='Baseline', line=dict(color='grey', dash='dash'), legendgroup='baseline', showlegend=(i==0)), row=row, col=col)
        if item.get('special_plot') == 'hline': fig.add_hline(y=30, line_dash="dash", line_color="grey", row=row, col=col)
        if item.get('log_y'): fig.update_yaxes(type="log", row=row, col=col)
    fig.update_layout(height=800, template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(range=[start_plot_date, time_vector[-1]])
    return fig

def compute_windowpost_fee(day_rewards_per_sector, fee_frac, n_lookahead, n_increase_upto, plateau):
    x, ramp = np.array(day_rewards_per_sector), np.linspace(1, plateau, n_increase_upto)
    plat = np.ones(len(x) - n_increase_upto) * plateau
    fee_by_t = np.concatenate([ramp, plat])
    rolling_rewards = np.array([np.sum(x[i:min(i + n_lookahead, len(x))]) for i in range(len(x) - n_lookahead)])
    return rolling_rewards * fee_frac * fee_by_t[:len(rolling_rewards)]

def calculate_roi_with_fees(sim_results, offline_data, fee_per_sector):
    rps_1y = sim_results['1y_return_per_sector']
    onboarding_rate = sim_results['rb_day_onboarded_power_pib'][-1]
    prefip_multiplier, fee_regime_scaler, f_postfip = (0.5, 1, 1) if onboarding_rate < 2.5 else (0.95, 2, 0.0001)
    with np.errstate(divide='ignore', invalid='ignore'):
        power_onboarded_renewed = (sim_results['day_onboarded_power_QAP_PIB'] + sim_results['day_renewed_power_QAP_PIB']) / C.PIB_PER_SECTOR
        gas_fees = np.nan_to_num((offline_data['daily_burnt_fil'] * prefip_multiplier * fee_regime_scaler) / power_onboarded_renewed)
    gas_fees_after_fip = gas_fees * f_postfip
    max_len = min(len(rps_1y), len(fee_per_sector), len(gas_fees))
    return_1y_base = rps_1y[:max_len] - gas_fees[:max_len]
    return_1y_with_fee = return_1y_base - fee_per_sector[:max_len] - gas_fees_after_fip[:max_len]
    pledge = sim_results['day_pledge_per_QAP'][:max_len]
    with np.errstate(divide='ignore', invalid='ignore'):
        roi_base = np.nan_to_num(return_1y_base / pledge)
        roi_with_fee = np.nan_to_num(return_1y_with_fee / pledge)
    return roi_base, roi_with_fee, roi_with_fee - roi_base

def create_fee_panel_plot(simulation_results, offline_data, time_vector, labels, start_plot_date, fee_frac, wp_fee_increase_days, wp_fee_plateau):
    fee_titles = ["Fee / Sector", "Fee / Pledge (%)", "Fee / 540d Reward (%)", "Cumulative Fees (M-FIL)", "Base ROI (%)", "ROI w/ Fee (%)", "ROI Delta (%)"]
    fig = make_subplots(rows=1, cols=7, subplot_titles=fee_titles)
    colors, lookahead_days = ['#0072B2', '#009E73', '#D55E00', '#CC79A7'], 540
    for sim_idx, sim_result in enumerate(simulation_results):
        color, fee_per_sector = colors[sim_idx], compute_windowpost_fee(sim_result['day_rewards_per_sector'], fee_frac, lookahead_days, wp_fee_increase_days, wp_fee_plateau)
        roi_base, roi_with_fee, roi_delta = calculate_roi_with_fees(sim_result, offline_data, fee_per_sector)
        pledge, rewards_540d = sim_result['day_pledge_per_QAP'], np.array([np.sum(sim_result['day_rewards_per_sector'][i:i+lookahead_days]) for i in range(len(fee_per_sector))])
        with np.errstate(divide='ignore', invalid='ignore'):
            fee_per_pledge, fee_per_540d_reward = np.nan_to_num(fee_per_sector / pledge[:len(fee_per_sector)]), np.nan_to_num(fee_per_sector / rewards_540d)
        total_onboards = (sim_result['day_onboarded_power_QAP_PIB'] + sim_result['day_renewed_power_QAP_PIB']) / C.PIB_PER_SECTOR
        cum_fees = (total_onboards[:len(fee_per_sector)] * fee_per_sector).cumsum()
        traces = [fee_per_sector, fee_per_pledge * 100, fee_per_540d_reward * 100, cum_fees / 1e6, roi_base * 100, roi_with_fee * 100, roi_delta * 100]
        for i, y_data in enumerate(traces):
            fig.add_trace(go.Scatter(x=time_vector, y=y_data, name=labels[sim_idx], line=dict(color=color), showlegend=(i==4)), row=1, col=i+1)
    fig.update_layout(height=350, template='plotly_white', margin=dict(t=60), title_text=f"Fee Analysis (Fraction: {fee_frac:.0%})", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(range=[start_plot_date, time_vector[-1]])
    return fig

# --- Streamlit App Main ---
st.title("📈 Filecoin Network Dashboard")
st.markdown("An interactive dashboard to explore Filecoin network dynamics. Adjust parameters in the sidebar to re-run the simulation.")

simulation_tab, fee_tab, gas_tab = st.tabs(["📊 Network Simulation", "💰 Fee Mechanism Analysis", "⛽ Gas Analysis"])

with simulation_tab:
    st.header("Network Growth Scenarios")
    if not selected_labels:
        st.warning("Please select at least one growth scenario from the sidebar.")
    else:
        with st.spinner('Running simulations...'):
            simulation_results, _, time_vector, current_date = run_full_simulation(forecast_length, sector_duration, lock_target, selected_scaling_factors)
        
        if simulation_results:
            st.subheader("KPI Summary")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**At Forecast Start**")
                today_index = time_vector.index(current_date)
                for i, (sim_result, label) in enumerate(zip(simulation_results, selected_labels)):
                    st.metric(label=f"{label} QAP (EiB)", value=f"{sim_result['network_QAP_EIB'][today_index]:.2f}")
            with c2:
                st.write("**At End of Forecast**")
                for i, (sim_result, label) in enumerate(zip(simulation_results, selected_labels)):
                    st.metric(label=f"{label} QAP (EiB)", value=f"{sim_result['network_QAP_EIB'][-1]:.2f}")
            
            st.markdown("---")
            st.header("Network Metrics Over Time")
            fig_network = create_network_metrics_plot(simulation_results, time_vector, selected_labels, current_date)
            st.plotly_chart(fig_network, use_container_width=True)
        else:
            st.error("Simulation failed. Please check the console for errors.")

with fee_tab:
    st.header("WindowPoSt Scaling Fee Analysis")
    if not selected_labels:
        st.warning("Please select at least one growth scenario to perform fee analysis.")
    elif not selected_fee_fracs:
        st.warning("Please select at least one fee fraction to analyze.")
    else:
        with st.spinner('Running simulations for fee analysis...'):
            simulation_results, offline_data, time_vector, current_date = run_full_simulation(forecast_length, sector_duration, lock_target, selected_scaling_factors)
        
        if simulation_results:
            st.info("""
                **Note on ROI Calculations:** The "Standard ROI" in the General Metrics tab uses the direct ROI output from the main simulation. The "Base ROI" in this panel is calculated differently to establish a specific baseline for comparing the new fee model.
            """)
            for fee_frac in selected_fee_fracs:
                fig_fee = create_fee_panel_plot(simulation_results, offline_data, time_vector, selected_labels, current_date, fee_frac, wp_fee_increase_days, wp_fee_plateau)
                st.plotly_chart(fig_fee, use_container_width=True)
        else:
            st.error("Simulation failed. Cannot perform fee analysis.")

with gas_tab:
    st.header("FIP100 Gas Usage Analysis")
    st.markdown("Analysis of various gas metrics before and after a specific update date.")
    
    with st.spinner("Loading and processing gas data..."):
        df, update_ts, gas_cols = load_and_process_gas_data(gas_start_date, gas_update_date)
    
    st.subheader("Gas Usage Over Time")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_gas_timeseries_plot(df, gas_cols, update_ts, relative=False), use_container_width=True)
    with c2:
        st.plotly_chart(create_gas_timeseries_plot(df, gas_cols, update_ts, relative=True), use_container_width=True)

    st.subheader("Pre vs. Post Update Comparison")
    
    summary_cols_abs = ['burnt_fil', 'sector_rewards_360d_32gib', 'sector_provecommit_fee_32gib', 'total_gas_used', 'provecommit_sector_gas_used', 'precommit_sector_batch_gas_used', 'publish_storage_deals_gas_used', 'submit_windowed_post_gas_used']
    summary_df_abs = calculate_gas_summary(df, summary_cols_abs, update_ts)
    c3, c4 = st.columns([1, 2])
    with c3:
        st.markdown("#### Absolute Mean Values")
        st.dataframe(summary_df_abs.style.format({'Mean Before': '{:,.2f}', 'Mean After': '{:,.2f}', '% Change': '{:+.2f}%'}))
    with c4:
        st.plotly_chart(create_gas_barchart(summary_df_abs, "Mean Gas Usage Before vs After Update"), use_container_width=True)

    summary_cols_norm = ['burnt_fil', 'sector_rewards_360d_32gib', 'sector_provecommit_fee_32gib', 'total_gas_used', 'provecommit_sector_gas_used', 'precommit_sector_batch_gas_used', 'publish_storage_deals_gas_used', 'submit_windowed_post_gas_used', 'prove_commit_sector_rbp']
    summary_df_norm = calculate_gas_summary(df, summary_cols_norm, update_ts, normalize_by_qap=True)
    c5, c6 = st.columns([1, 2])
    with c5:
        st.markdown("#### Values Normalized by Onboarded QAP")
        st.dataframe(summary_df_norm.style.format({'Mean Before': '{:,.6f}', 'Mean After': '{:,.6f}', '% Change': '{:+.2f}%'}))
    with c6:
        st.plotly_chart(create_gas_barchart(summary_df_norm, "Mean Gas per Onboarded QAP Before vs After Update"), use_container_width=True)
