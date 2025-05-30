import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import warnings

# Suppress some common warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_FILE = 'filecoin_data.csv'  # <--- CHANGE THIS TO YOUR FILENAME
TIME_COLUMN_ORIGINAL = 'stateTime'

# --- IMPORTANT: Update this dictionary to match YOUR EXACT column names ---
COLUMN_MAPPING = {
    'stateTime': 'Date',
    'Commit Pledge per 32GiB QAP': 'Pledge_Per_32GiB_QAP',
    'Protocol Circulating Supply': 'Circulating_Supply',
    'Total FIL Mined': 'Total_FIL_Mined',
    'Total FIL Vested': 'Total_FIL_Vested',
    'Fil Reserve Disbursed': 'Fil_Reserve_Disbursed',
    'Total FIL Burned': 'Total_FIL_Burned',
    'Total FIL Locked': 'Total_Pledge_Collateral', # Renamed for clarity
    'Network QA Power': 'Network_QA_Power',
    'Network RB Power': 'Network_RB_Power',
    'Baseline Power': 'Baseline_Power',
    'FIL-USD': 'FIL_Price_USD'
    # Add any other columns you have and want to use/keep
}
# --- End of Important Update Section ---

# Define the standardized names we'll use in the script
TIME_COLUMN = COLUMN_MAPPING[TIME_COLUMN_ORIGINAL]
TARGET_VARIABLES = [COLUMN_MAPPING['Total FIL Locked'], COLUMN_MAPPING['Commit Pledge per 32GiB QAP']]
PRICE_COL = COLUMN_MAPPING['FIL-USD']
QAP_COL = COLUMN_MAPPING['Network QA Power']
BASELINE_COL = COLUMN_MAPPING['Baseline Power']
CIRC_SUPPLY_COL = COLUMN_MAPPING['Protocol Circulating Supply']

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess(filepath, column_mapping, time_col_original):
    """Loads and preprocesses the Filecoin data using a specific mapping."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        print(f"Original columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please update the DATA_FILE variable.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

    # Check if all keys from the mapping exist in the DataFrame columns
    missing_cols = [col for col in column_mapping.keys() if col not in df.columns]
    if missing_cols:
        print(f"Error: The following columns specified in COLUMN_MAPPING were not found in the CSV:")
        for col in missing_cols:
            print(f" - '{col}'")
        print("Please check your COLUMN_MAPPING dictionary or the CSV file.")
        return None

    # Keep only the columns we need and rename them
    df = df[list(column_mapping.keys())].copy()
    df.rename(columns=column_mapping, inplace=True)
    print(f"Renamed columns: {df.columns.tolist()}")

    # Convert time column to datetime and set as index
    try:
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
        df = df.set_index(TIME_COLUMN).sort_index()
    except Exception as e:
        print(f"Error converting time column '{TIME_COLUMN}': {e}")
        return None

    # Convert all other columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values - Forward fill is often suitable for time series
    print(f"\nMissing values before fill: \n{df.isnull().sum()[df.isnull().sum() > 0]}")
    df.ffill(inplace=True)
    df.bfill(inplace=True) # Backfill any remaining NaNs at the beginning
    print(f"Missing values after fill: \n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Drop rows if any critical NaNs remain
    df.dropna(inplace=True)

    print(f"\nData preprocessed. Final shape: {df.shape}")
    return df

# --- 2. Feature Engineering ---
def engineer_features(df):
    """Creates new features that might be useful for analysis."""
    df_feat = df.copy()

    # Calculate QAP vs Baseline factor (important for consensus pledge)
    # Ensure Baseline_Power is not zero to avoid division issues
    df_feat[BASELINE_COL] = df_feat[BASELINE_COL].replace(0, 1e-9) # Replace 0 with a very small number
    df_feat['QAP_vs_Baseline'] = df_feat[QAP_COL] / df_feat[[BASELINE_COL, QAP_COL]].max(axis=1)

    # Calculate Total Pledge as % of Circulating Supply
    df_feat['Pledge_Ratio'] = (df_feat[TARGET_VARIABLES[0]] / df_feat[CIRC_SUPPLY_COL]) * 100

    # Calculate an *estimate* of the Consensus Pledge portion factor
    df_feat['Estimated_Consensus_Pledge_Factor'] = 0.30 * df_feat[CIRC_SUPPLY_COL] / df_feat[[BASELINE_COL, QAP_COL]].max(axis=1)

    # Calculate daily changes or growth rates
    df_feat['FIL_USD_Change'] = df_feat[PRICE_COL].pct_change() * 100
    df_feat['QAP_Change'] = df_feat[QAP_COL].pct_change() * 100
    df_feat['Pledge_Change'] = df_feat[TARGET_VARIABLES[0]].pct_change() * 100

    # Clean up NaNs created by pct_change
    df_feat.fillna(0, inplace=True)
    
    print("New features engineered.")
    return df_feat

# --- 3. Exploratory Data Analysis (EDA) ---
def perform_eda(df):
    """Generates various plots for EDA."""
    print("\nPerforming Exploratory Data Analysis...")

    # Plot key time series
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 20), sharex=True)
    
    df[TARGET_VARIABLES[0]].plot(ax=axes[0], title='Total FIL Locked')
    axes[0].set_ylabel('FIL')
    df[TARGET_VARIABLES[1]].plot(ax=axes[1], title='Commit Pledge per 32GiB QAP')
    axes[1].set_ylabel('FIL / 32GiB')
    df[PRICE_COL].plot(ax=axes[2], title='FIL-USD Price')
    axes[2].set_ylabel('USD')
    df[QAP_COL].plot(ax=axes[3], title='Network Quality Adjusted Power')
    axes[3].set_ylabel('Power Units') # Adjust label based on your units (PiB?)
    
    for ax in axes:
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Filecoin Metrics')
    plt.show()

    # Plot distributions
    cols_to_plot = [col for col in TARGET_VARIABLES + [PRICE_COL, QAP_COL] if col in df.columns]
    if cols_to_plot:
        df[cols_to_plot].hist(bins=50, figsize=(15, 10))
        plt.suptitle('Distribution of Key Variables')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# --- 4. Time Series Analysis ---
def analyze_time_series(df, column):
    """Performs time series decomposition and stationarity test."""
    print(f"\n--- Time Series Analysis for: {column} ---")
    
    ts_data = df[column].dropna()
    
    if len(ts_data) < 60: # Need enough points for seasonal decomp (e.g., 2*period)
        print(f"Not enough data points ({len(ts_data)}) for seasonal decomposition.")
    else:
        try:
            decomposition = seasonal_decompose(ts_data, model='additive', period=30) # Assuming monthly pattern
            fig = decomposition.plot()
            fig.set_size_inches(12, 8)
            plt.suptitle(f'Time Series Decomposition of {column}', y=1.02)
            plt.show()
        except ValueError as e:
            print(f"Could not perform seasonal decomposition: {e}")

    # Stationarity Test (Augmented Dickey-Fuller)
    print(f"Performing Augmented Dickey-Fuller Test for {column}:")
    try:
        result = adfuller(ts_data)
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')
        if result[1] <= 0.05:
            print("Result: The series is likely stationary.")
        else:
            print("Result: The series is likely non-stationary (consider differencing for modeling).")
    except Exception as e:
        print(f"Could not perform ADF test: {e}")

# --- 5. Regression Analysis ---
def run_regression(df, target_var, feature_cols):
    """Runs a simple OLS regression to model the target variable."""
    print(f"\n--- Regression Analysis for: {target_var} ---")
    
    # Ensure all selected columns exist
    available_features = [f for f in feature_cols if f in df.columns]
    if not available_features:
        print("None of the specified feature columns are available. Skipping regression.")
        return
    
    print(f"Using features: {available_features}")
    
    df_reg = df.copy()
    df_reg = df_reg.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_var] + available_features)

    if df_reg.empty:
        print("Not enough data to run regression after handling NaNs/Infs.")
        return

    X = df_reg[available_features]
    y = df_reg[target_var]

    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
        print(model.summary())
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print("\nVariance Inflation Factor (VIF):")
        print(vif)
        print("(VIF > 10 often indicates problematic multicollinearity)")
        
    except Exception as e:
        print(f"An error occurred during regression analysis: {e}")


# --- 6. Main Execution ---
if __name__ == "__main__":
    filecoin_data = load_and_preprocess(DATA_FILE, COLUMN_MAPPING, TIME_COLUMN_ORIGINAL)

    if filecoin_data is not None:
        filecoin_data = engineer_features(filecoin_data)
        perform_eda(filecoin_data)

        for var in TARGET_VARIABLES + [PRICE_COL, QAP_COL]:
            if var in filecoin_data.columns:
                analyze_time_series(filecoin_data, var)

        # Define potential predictor variables using the NEW standardized names
        predictors = [
            CIRC_SUPPLY_COL,
            QAP_COL,
            BASELINE_COL,
            PRICE_COL,
            'QAP_vs_Baseline' 
            # Add other columns you want to test, e.g., 'Total_FIL_Burned'
        ]
        
        # Filter predictors to only include those present
        available_predictors = [p for p in predictors if p in filecoin_data.columns]
        
        if available_predictors:
            run_regression(filecoin_data, TARGET_VARIABLES[0], available_predictors)
            # You can uncomment this to run for the second target variable too
            # run_regression(filecoin_data, TARGET_VARIABLES[1], available_predictors)
        else:
            print("No suitable predictor variables found for regression.")

        print("\n--- Analysis Complete ---")
        print("Check the generated plots and regression summaries for insights.")
        
#%%

# --- Create Differenced Data ---
df_diff = filecoin_data.copy()
cols_to_diff = [
    'Total_Pledge_Collateral', 
    'Circulating_Supply', 
    'Network_QA_Power', 
    'Baseline_Power', 
    'FIL_Price_USD'
]

for col in cols_to_diff:
    if col in df_diff.columns:
        df_diff[f'{col}_diff'] = df_diff[col].diff()

df_diff.dropna(inplace=True) # Drop the first row with NaNs

# --- Define New Target and Predictors ---
target_diff = 'Total_Pledge_Collateral_diff'
predictors_diff = [
    'Circulating_Supply_diff', 
    'Network_QA_Power_diff', 
    'FIL_Price_USD_diff' 
    # Choose a smaller, less collinear set
]

# --- Run Regression on Differenced Data ---
if all(col in df_diff.columns for col in [target_diff] + predictors_diff):
    run_regression(df_diff, target_diff, predictors_diff)
else:
    print("Could not find all required differenced columns for regression.")