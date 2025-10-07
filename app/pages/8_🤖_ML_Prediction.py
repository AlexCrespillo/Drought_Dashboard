"""
ML Prediction Page - Time Series Forecasting and Drought Event Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Time series libraries
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from utils.data_loader import *
from utils.visualization import *

# Page config
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🤖 ML Prediction & Forecasting</h1>
        <p>Advanced machine learning for streamflow forecasting and drought event prediction</p>
    </div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data(ttl=CACHE_TTL)
def load_all_data():
    """Load all required datasets"""
    drought_df = load_drought_events()
    streamflow_df = load_streamflow_data()
    return drought_df, streamflow_df

with st.spinner("Loading data..."):
    drought_df, streamflow_df = load_all_data()

if streamflow_df.empty:
    st.error("No streamflow data available")
    st.stop()

# ==================== HELPER FUNCTIONS ====================

def check_stationarity(timeseries, title='Time Series'):
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test
    """
    result = adfuller(timeseries.dropna())

    output = {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Number of Observations': result[3],
        'Stationary': result[1] < 0.05
    }

    return output


def prepare_streamflow_data(df, station, start_date, end_date, fill_method='ffill'):
    """
    Prepare streamflow data for time series modeling

    Args:
        df: Streamflow dataframe
        station: Station ID
        start_date: Start date
        end_date: End date
        fill_method: Method to fill missing values ('ffill', 'interpolate', 'drop')

    Returns:
        pd.Series: Prepared time series
    """
    # Convert dates to pandas Timestamp for comparison
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Filter by station and date range
    station_data = df[df['indroea'] == station].copy()
    station_data = station_data[(station_data['fecha'] >= start_date) &
                                (station_data['fecha'] <= end_date)]

    # Set date as index
    station_data = station_data.set_index('fecha').sort_index()

    # Create complete date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    station_data = station_data.reindex(date_range)

    # Handle missing values
    if fill_method == 'ffill':
        station_data['caudal'] = station_data['caudal'].ffill()
    elif fill_method == 'interpolate':
        station_data['caudal'] = station_data['caudal'].interpolate(method='linear')
    elif fill_method == 'drop':
        station_data = station_data.dropna(subset=['caudal'])

    return station_data['caudal']


def create_event_features(drought_df, streamflow_df, lookback_days=30):
    """
    Create features for drought event prediction

    Args:
        drought_df: Drought events dataframe
        streamflow_df: Streamflow dataframe
        lookback_days: Days before event to calculate mean flow

    Returns:
        pd.DataFrame: Feature dataframe (cleaned of NaN values)
    """
    features_list = []

    for idx, event in drought_df.iterrows():
        station = event['indroea']
        start_date = pd.Timestamp(event['inicio'])

        # Get streamflow data before event
        lookback_start = start_date - timedelta(days=lookback_days)
        station_flow = streamflow_df[
            (streamflow_df['indroea'] == station) &
            (streamflow_df['fecha'] >= lookback_start) &
            (streamflow_df['fecha'] < start_date)
        ]

        # Only include events with sufficient lookback data
        if len(station_flow) >= lookback_days * 0.5:  # At least 50% of lookback period
            # Calculate features with NaN handling
            mean_flow = station_flow['caudal'].mean()
            min_flow = station_flow['caudal'].min()
            max_flow = station_flow['caudal'].max()
            std_flow = station_flow['caudal'].std()

            # Handle potential NaN in std (if all values are the same)
            if pd.isna(std_flow):
                std_flow = 0.0

            # Calculate trend
            if len(station_flow) > 1:
                trend = (station_flow['caudal'].iloc[-1] - station_flow['caudal'].iloc[0]) / lookback_days
            else:
                trend = 0.0

            features = {
                'event_id': event['event_id'],
                'station': station,
                'duration': event['duracion'],
                'deficit': event['deficit_hm3'],
                'mean_flow_before': mean_flow,
                'min_flow_before': min_flow,
                'max_flow_before': max_flow,
                'std_flow_before': std_flow,
                'trend_flow_before': trend,
                'month': start_date.month,
                'year': start_date.year
            }
            features_list.append(features)

    # Create DataFrame and drop any remaining NaN values
    features_df = pd.DataFrame(features_list)

    # Drop rows with any NaN values
    features_df = features_df.dropna()

    return features_df


def train_arima_model(data, order=(1, 1, 1), seasonal_order=None):
    """
    Train ARIMA/SARIMA model

    Args:
        data: Time series data
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s) or None

    Returns:
        Fitted model
    """
    try:
        if seasonal_order:
            model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(data, order=order)

        # Fit the model (disp parameter removed in newer statsmodels versions)
        fitted_model = model.fit()
        return fitted_model
    except Exception as e:
        # If model fitting fails, raise with informative message
        raise ValueError(f"Model fitting failed: {str(e)}. Try adjusting parameters (p, d, q) or check data quality.")


def forecast_arima(model, steps=1):
    """
    Generate forecast from ARIMA model

    Args:
        model: Fitted ARIMA model
        steps: Number of steps to forecast

    Returns:
        pd.DataFrame: Forecast with confidence intervals
    """
    forecast = model.get_forecast(steps=steps)
    forecast_df = pd.DataFrame({
        'forecast': forecast.predicted_mean,
        'lower_ci': forecast.conf_int().iloc[:, 0],
        'upper_ci': forecast.conf_int().iloc[:, 1]
    })

    return forecast_df


def train_regression_model(X_train, y_train, model_type='linear', **kwargs):
    """
    Train regression model

    Args:
        X_train: Training features
        y_train: Training target
        model_type: 'linear' or 'rf' (Random Forest)
        **kwargs: Model parameters

    Returns:
        Fitted model and scaler
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'rf':
        n_estimators = kwargs.get('n_estimators', 100)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    model.fit(X_train_scaled, y_train)

    return model, scaler


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        dict: Metrics dictionary
    """
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    }

    return metrics


def create_baseline_forecast(data, steps=1):
    """
    Create persistence baseline (tomorrow = today)

    Args:
        data: Time series data
        steps: Number of steps

    Returns:
        np.array: Baseline predictions
    """
    return np.full(steps, data.iloc[-1])


# ==================== MAIN APP ====================

st.markdown("""
This section provides machine learning tools for:
1. **Time Series Forecasting**: ARIMA/SARIMA models for predicting daily streamflow
2. **Event Prediction**: Regression models for predicting drought event characteristics
""")

st.markdown("---")

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs([
    "📈 Time Series Forecasting (ARIMA)",
    "🎯 Event Prediction (Regression)",
    "📚 Methodology & Help"
])

# ==================== TAB 1: ARIMA FORECASTING ====================
with tab1:
    st.markdown("### 📈 ARIMA/SARIMA Time Series Forecasting")

    st.info("""
    **ARIMA** (AutoRegressive Integrated Moving Average) models are used for time series forecasting.
    This tool predicts future daily streamflow values based on historical patterns.
    """)

    # Sidebar controls for ARIMA
    st.sidebar.header("🔧 ARIMA Configuration")

    # Station selection
    available_stations = sorted(streamflow_df['indroea'].unique())
    selected_station = st.sidebar.selectbox(
        "Select Station",
        options=available_stations,
        key='arima_station'
    )

    # Date range
    min_date = streamflow_df['fecha'].min()
    max_date = streamflow_df['fecha'].max()

    date_range = st.sidebar.date_input(
        "Training Period",
        value=(min_date, max_date - timedelta(days=365)),
        min_value=min_date,
        max_value=max_date,
        key='arima_dates'
    )

    # Test size
    test_size = st.sidebar.slider(
        "Test Set Size (days)",
        min_value=30,
        max_value=365,
        value=90,
        step=10,
        key='arima_test_size'
    )

    # Fill method
    fill_method = st.sidebar.selectbox(
        "Missing Value Handling",
        options=['ffill', 'interpolate', 'drop'],
        index=0,
        help="ffill: Forward fill, interpolate: Linear interpolation, drop: Remove missing"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**ARIMA Parameters**")
    st.sidebar.info("""
    🔍 **What are these?** ARIMA uses three numbers (p, d, q) to predict the future:
    - **p**: How many past days to look at
    - **d**: How much to smooth the data
    - **q**: How much to adjust for past errors

    💡 **Start with (1,1,1)** and adjust if needed!
    """)

    # ARIMA order
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        p = st.number_input("p (AR)", min_value=0, max_value=5, value=1, key='arima_p',
                           help="AutoRegressive order: How many previous days influence tomorrow. Higher = more historical days considered.")
    with col2:
        d = st.number_input("d (I)", min_value=0, max_value=2, value=1, key='arima_d',
                           help="Differencing order: Makes data stationary. 0=no smoothing, 1=remove trend, 2=remove curved trend.")
    with col3:
        q = st.number_input("q (MA)", min_value=0, max_value=5, value=1, key='arima_q',
                           help="Moving Average order: How much to learn from past prediction errors. Higher = more error correction.")

    # Seasonal ARIMA
    use_seasonal = st.sidebar.checkbox("Use Seasonal ARIMA (SARIMA)", value=False,
                                       help="Enable if data has repeating patterns (e.g., summer droughts every year)")

    if use_seasonal:
        st.sidebar.markdown("**Seasonal Parameters**")
        st.sidebar.info("""
        🔄 **Seasonality**: Repeating patterns over time
        - **P, D, Q**: Same as p, d, q but for seasonal patterns
        - **s**: Season length (e.g., 12 for monthly, 365 for yearly)

        💡 **Tip**: Use s=365 for annual patterns in daily data
        """)
        col1, col2, col3, col4 = st.sidebar.columns(4)
        with col1:
            P = st.number_input("P", min_value=0, max_value=2, value=1, key='sarima_P',
                               help="Seasonal AR: Patterns from same season last year")
        with col2:
            D = st.number_input("D", min_value=0, max_value=1, value=1, key='sarima_D',
                               help="Seasonal differencing: Remove seasonal trend")
        with col3:
            Q = st.number_input("Q", min_value=0, max_value=2, value=1, key='sarima_Q',
                               help="Seasonal MA: Errors from same season last year")
        with col4:
            s = st.number_input("s", min_value=2, max_value=365, value=12, key='sarima_s',
                               help="Season length: 12=monthly patterns, 365=yearly patterns")

    # Forecast horizon
    forecast_steps = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=1,
        max_value=30,
        value=7,
        key='forecast_steps'
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Data Preparation")

    with col2:
        train_arima_button = st.button("🚀 Train ARIMA Model", type="primary", use_container_width=True)

    if train_arima_button:
        with st.spinner("Preparing data and training model..."):
            try:
                # Prepare data
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    # Convert to pandas Timestamps
                    start_date = pd.Timestamp(start_date)
                    end_date = pd.Timestamp(end_date)
                elif hasattr(date_range, '__len__') and len(date_range) == 2:
                    start_date, end_date = date_range[0], date_range[1]
                    start_date = pd.Timestamp(start_date)
                    end_date = pd.Timestamp(end_date)
                else:
                    st.error("Please select a valid date range with both start and end dates")
                    st.stop()

                if start_date >= end_date:
                    st.error("Start date must be before end date")
                    st.stop()

                # Load and prepare time series
                ts_data = prepare_streamflow_data(
                    streamflow_df,
                    selected_station,
                    start_date,
                    end_date + timedelta(days=test_size),
                    fill_method
                )

                if len(ts_data) < 100:
                    st.error(f"⚠️ Insufficient data for station {selected_station}. Need at least 100 observations, found {len(ts_data)}.")
                    st.stop()

                # Split train/test
                train_data = ts_data[:-test_size]
                test_data = ts_data[-test_size:]

                # Display data info
                st.success(f"✅ Data prepared: {len(train_data)} training samples, {len(test_data)} test samples")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Training Samples", len(train_data))
                with col2:
                    st.metric("Test Samples", len(test_data))
                with col3:
                    st.metric("Mean Flow", f"{train_data.mean():.2f} m³/s")
                with col4:
                    missing_pct = (ts_data.isna().sum() / len(ts_data)) * 100
                    st.metric("Missing Data", f"{missing_pct:.1f}%")

                st.markdown("---")
                st.markdown("#### Stationarity Check")

                # Check stationarity
                stationarity_result = check_stationarity(train_data)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Augmented Dickey-Fuller Test:**")
                    st.write(f"Test Statistic: {stationarity_result['Test Statistic']:.4f}")
                    st.write(f"p-value: {stationarity_result['p-value']:.4f}")

                    if stationarity_result['Stationary']:
                        st.success("✅ Series is stationary")
                    else:
                        st.warning("⚠️ Series is non-stationary (differencing recommended)")

                with col2:
                    st.markdown("**Interpretation:**")
                    st.write("""
                    - p-value < 0.05: Stationary (good for ARIMA)
                    - p-value ≥ 0.05: Non-stationary (use d > 0)
                    """)

                st.markdown("---")
                st.markdown("#### Model Training")

                # Train ARIMA model
                order = (p, d, q)
                seasonal_order = (P, D, Q, s) if use_seasonal else None

                with st.spinner("Training ARIMA model..."):
                    model = train_arima_model(train_data, order=order, seasonal_order=seasonal_order)

                    # Store model in session state (use different keys to avoid widget conflict)
                    st.session_state['trained_arima_model'] = model
                    st.session_state['trained_arima_train_data'] = train_data
                    st.session_state['trained_arima_test_data'] = test_data
                    st.session_state['trained_arima_station_id'] = selected_station

                st.success("✅ Model trained successfully!")

                # Model summary
                with st.expander("📊 Model Summary", expanded=False):
                    st.text(str(model.summary()))

                st.markdown("---")
                st.markdown("#### Forecasting & Evaluation")

                # In-sample predictions
                in_sample_pred = model.fittedvalues

                # Out-of-sample predictions
                forecast_df = forecast_arima(model, steps=len(test_data))

                # Baseline
                baseline_pred = create_baseline_forecast(train_data, steps=len(test_data))

                # Calculate metrics
                train_metrics = calculate_metrics(train_data[len(train_data)-len(in_sample_pred):], in_sample_pred)
                test_metrics = calculate_metrics(test_data, forecast_df['forecast'])
                baseline_metrics = calculate_metrics(test_data, baseline_pred)

                # Display metrics
                st.markdown("**Performance Metrics:**")

                metrics_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE (%)'],
                    'Training': [train_metrics['RMSE'], train_metrics['MAE'], train_metrics['R²'], train_metrics['MAPE']],
                    'Test': [test_metrics['RMSE'], test_metrics['MAE'], test_metrics['R²'], test_metrics['MAPE']],
                    'Baseline (Persistence)': [baseline_metrics['RMSE'], baseline_metrics['MAE'], baseline_metrics['R²'], baseline_metrics['MAPE']]
                })

                st.dataframe(
                    metrics_df.style.format({
                        'Training': '{:.4f}',
                        'Test': '{:.4f}',
                        'Baseline (Persistence)': '{:.4f}'
                    }).background_gradient(subset=['Test'], cmap='RdYlGn_r'),
                    use_container_width=True
                )

                # Interpretation
                if test_metrics['RMSE'] < baseline_metrics['RMSE']:
                    improvement_pct = ((baseline_metrics['RMSE'] - test_metrics['RMSE']) / baseline_metrics['RMSE'] * 100)
                    st.success(f"✅ Model outperforms baseline by {improvement_pct:.1f}%")

                    st.info(f"""
                    📊 **What does this mean?**

                    Your ARIMA model is **{improvement_pct:.1f}% more accurate** than simply predicting "tomorrow will be like today" (the baseline method).

                    - **RMSE (Root Mean Squared Error)**: Average prediction error = **{test_metrics['RMSE']:.2f} m³/s**
                      - This means predictions are typically off by about {test_metrics['RMSE']:.2f} cubic meters per second
                      - Lower is better! Your model: {test_metrics['RMSE']:.2f} vs Baseline: {baseline_metrics['RMSE']:.2f}

                    - **R² (R-squared)**: **{test_metrics['R²']:.2%}** of the flow variation is explained by the model
                      - 100% = perfect predictions, 0% = random guessing
                      - Your score: **{test_metrics['R²']:.2%}** {'🌟 Excellent!' if test_metrics['R²'] > 0.7 else '✅ Good!' if test_metrics['R²'] > 0.5 else '⚠️ Could be better'}

                    - **MAPE (Mean Absolute Percentage Error)**: **{test_metrics['MAPE']:.1f}%** average error
                      - On average, predictions are {test_metrics['MAPE']:.1f}% off from the actual values
                      - Below 10% is excellent, below 20% is good
                    """)
                else:
                    st.warning("⚠️ Model does not outperform persistence baseline. Consider adjusting parameters or adding more features.")

                    st.info("""
                    💡 **What can you do?**

                    The model isn't beating the simple "tomorrow = today" prediction. Try these:

                    1. **Increase p or q**: Try p=2 or q=2 to consider more history
                    2. **Check seasonality**: Enable SARIMA if you see yearly patterns
                    3. **More data**: Select a longer training period
                    4. **Different station**: Some stations are harder to predict than others
                    """)

                st.markdown("---")
                st.markdown("#### Visualization")

                # Plot results
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Observed vs Predicted (Test Set)', 'Residuals'),
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0.12
                )

                # Test data plot
                test_dates = test_data.index

                fig.add_trace(
                    go.Scatter(x=test_dates, y=test_data, name='Observed', line=dict(color='blue')),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=test_dates, y=forecast_df['forecast'], name='ARIMA Forecast', line=dict(color='red', dash='dash')),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=test_dates,
                        y=forecast_df['upper_ci'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=test_dates,
                        y=forecast_df['lower_ci'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        name='95% CI',
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=test_dates, y=baseline_pred, name='Baseline (Persistence)',
                               line=dict(color='gray', dash='dot')),
                    row=1, col=1
                )

                # Residuals
                residuals = test_data.values - forecast_df['forecast'].values

                fig.add_trace(
                    go.Scatter(x=test_dates, y=residuals, mode='markers', name='Residuals',
                               marker=dict(color='purple', size=4)),
                    row=2, col=1
                )

                fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Flow (m³/s)", row=1, col=1)
                fig.update_yaxes(title_text="Residuals", row=2, col=1)

                fig.update_layout(height=700, hovermode='x unified', template='plotly_white')

                st.plotly_chart(fig, use_container_width=True)

                # Future forecast
                st.markdown("---")
                st.markdown(f"#### Future Forecast ({forecast_steps} days ahead)")

                future_forecast = forecast_arima(model, steps=forecast_steps)
                future_dates = pd.date_range(start=ts_data.index[-1] + timedelta(days=1), periods=forecast_steps)
                future_forecast.index = future_dates

                # Plot future forecast
                fig_future = go.Figure()

                # Last 30 days of training
                last_days = min(30, len(ts_data))
                fig_future.add_trace(
                    go.Scatter(x=ts_data.index[-last_days:], y=ts_data.values[-last_days:],
                               name='Historical', line=dict(color='blue'))
                )

                fig_future.add_trace(
                    go.Scatter(x=future_dates, y=future_forecast['forecast'],
                               name='Forecast', line=dict(color='red', dash='dash'))
                )

                fig_future.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=future_forecast['upper_ci'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    )
                )

                fig_future.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=future_forecast['lower_ci'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        name='95% CI'
                    )
                )

                fig_future.update_layout(
                    title=f"{forecast_steps}-Day Ahead Forecast",
                    xaxis_title="Date",
                    yaxis_title="Flow (m³/s)",
                    template='plotly_white',
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig_future, use_container_width=True)

                # Forecast table
                forecast_table = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast (m³/s)': future_forecast['forecast'].values,
                    'Lower 95% CI': future_forecast['lower_ci'].values,
                    'Upper 95% CI': future_forecast['upper_ci'].values
                })

                st.dataframe(
                    forecast_table.style.format({
                        'Forecast (m³/s)': '{:.2f}',
                        'Lower 95% CI': '{:.2f}',
                        'Upper 95% CI': '{:.2f}'
                    }),
                    use_container_width=True
                )

                # Export options
                st.markdown("---")
                st.markdown("#### Export Results")

                col1, col2 = st.columns(2)

                with col1:
                    # Export predictions
                    predictions_df = pd.DataFrame({
                        'Date': test_dates,
                        'Observed': test_data.values,
                        'Predicted': forecast_df['forecast'].values,
                        'Lower_CI': forecast_df['lower_ci'].values,
                        'Upper_CI': forecast_df['upper_ci'].values
                    })

                    csv = predictions_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Test Predictions (CSV)",
                        data=csv,
                        file_name=f"arima_predictions_station_{selected_station}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    # Export model
                    model_path = Path(f"arima_model_station_{selected_station}.pkl")
                    joblib.dump(model, model_path)

                    with open(model_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Trained Model (.pkl)",
                            data=f,
                            file_name=f"arima_model_station_{selected_station}.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )

                # Recommendations
                st.markdown("---")
                st.markdown("#### 💡 Recommendations")

                recommendations = []

                if test_metrics['R²'] > 0.7:
                    recommendations.append("✅ Model shows strong predictive performance (R² > 0.7)")
                elif test_metrics['R²'] > 0.5:
                    recommendations.append("⚠️ Model shows moderate predictive performance (0.5 < R² < 0.7)")
                else:
                    recommendations.append("❌ Model shows weak predictive performance (R² < 0.5). Consider:")
                    recommendations.append("  - Adjusting ARIMA parameters (p, d, q)")
                    recommendations.append("  - Using seasonal SARIMA if data shows seasonality")
                    recommendations.append("  - Including exogenous variables (temperature, precipitation)")

                if test_metrics['MAPE'] < 10:
                    recommendations.append("✅ Low prediction error (MAPE < 10%)")
                elif test_metrics['MAPE'] < 20:
                    recommendations.append("⚠️ Moderate prediction error (10% < MAPE < 20%)")
                else:
                    recommendations.append("❌ High prediction error (MAPE > 20%)")

                if len(train_data) < 365:
                    recommendations.append("⚠️ Small training dataset. Consider using more historical data for better predictions.")

                for rec in recommendations:
                    st.markdown(f"- {rec}")

            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")
                st.exception(e)

# ==================== TAB 2: REGRESSION ====================
with tab2:
    st.markdown("### 🎯 Drought Event Deficit Prediction")

    st.info("""
    **Regression models** predict drought event deficit (water shortage) based on event characteristics
    and pre-event flow conditions. Features include event duration, mean flow before event, and temporal factors.
    """)

    if drought_df.empty:
        st.warning("No drought event data available for regression modeling")
    else:
        # Sidebar controls for regression
        st.sidebar.header("🔧 Regression Configuration")

        st.sidebar.info("""
        🎯 **Goal**: Predict how severe a drought will be based on conditions before it starts

        💡 **How it works**: The model looks at flow levels before a drought begins and learns patterns to estimate water deficit
        """)

        # Lookback period
        lookback_days = st.sidebar.slider(
            "Lookback Period (days)",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="How many days before drought to analyze. 30 days = look at last month's flow patterns"
        )

        st.sidebar.caption("""
        📅 **Lookback Period**: How far back to look before a drought starts
        - **7 days**: Very recent conditions only
        - **30 days**: Last month (recommended)
        - **90 days**: Last 3 months of patterns
        """)

        # Model selection
        model_type = st.sidebar.selectbox(
            "Model Type",
            options=['linear', 'rf'],
            format_func=lambda x: 'Linear Regression' if x == 'linear' else 'Random Forest',
            index=0,
            help="Linear = simple & fast, Random Forest = complex but more accurate"
        )

        st.sidebar.caption("""
        🤖 **Model Types**:
        - **Linear Regression**: Simple, interpretable, assumes straight-line relationships
        - **Random Forest**: Complex, handles curved patterns, usually more accurate
        """)


        if model_type == 'rf':
            n_estimators = st.sidebar.slider(
                "Number of Trees",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
        else:
            n_estimators = None

        # Train/test split
        test_split = st.sidebar.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5
        )

        # Feature selection
        st.sidebar.markdown("**Feature Selection**")
        use_duration = st.sidebar.checkbox("Duration", value=True)
        use_mean_flow = st.sidebar.checkbox("Mean Flow Before Event", value=True)
        use_min_flow = st.sidebar.checkbox("Min Flow Before Event", value=True)
        use_std_flow = st.sidebar.checkbox("Std Flow Before Event", value=True)
        use_trend = st.sidebar.checkbox("Flow Trend Before Event", value=True)
        use_temporal = st.sidebar.checkbox("Temporal Features (month, year)", value=True)

        # Main content
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Feature Engineering")

        with col2:
            train_regression_button = st.button("🚀 Train Regression Model", type="primary", use_container_width=True)

        if train_regression_button:
            with st.spinner("Creating features and training model..."):
                try:
                    # Create features
                    features_df = create_event_features(drought_df, streamflow_df, lookback_days)

                    if len(features_df) < 20:
                        st.error(f"⚠️ Insufficient events with valid features. Found {len(features_df)} events. Need at least 20.")
                        st.info("💡 Try reducing the lookback period or selecting stations with more historical data.")
                        st.stop()

                    st.success(f"✅ Features created for {len(features_df)} drought events (NaN values removed)")

                    # Display feature statistics
                    st.markdown("**Feature Statistics:**")
                    st.dataframe(features_df.describe().T.style.format('{:.2f}'), use_container_width=True)

                    # Correlation Analysis
                    st.markdown("---")
                    st.markdown("#### 🔗 Feature Correlation Analysis")

                    st.info("""
                    📊 **What is correlation?**

                    Correlation shows how features relate to each other:
                    - **+1.0**: Perfect positive relationship (when one goes up, the other goes up)
                    - **0.0**: No relationship
                    - **-1.0**: Perfect negative relationship (when one goes up, the other goes down)

                    **Why it matters:**
                    - Highly correlated features (>0.8) may be redundant
                    - Understanding relationships helps interpret the model
                    - Can identify which conditions tend to occur together
                    """)

                    # Calculate correlation matrix for numeric features
                    numeric_cols = ['duration', 'mean_flow_before', 'min_flow_before', 'max_flow_before',
                                   'std_flow_before', 'trend_flow_before', 'month', 'year', 'deficit']
                    available_cols = [col for col in numeric_cols if col in features_df.columns]

                    if len(available_cols) > 1:
                        corr_matrix = features_df[available_cols].corr()

                        # Create correlation heatmap
                        fig_corr = px.imshow(
                            corr_matrix,
                            labels=dict(x="Features", y="Features", color="Correlation"),
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1,
                            aspect="auto",
                            title="Feature Correlation Matrix"
                        )

                        fig_corr.update_layout(
                            height=600,
                            template='plotly_white'
                        )

                        st.plotly_chart(fig_corr, use_container_width=True)

                        # Analyze correlations with target (deficit)
                        if 'deficit' in corr_matrix.columns:
                            st.markdown("**📌 Correlation with Drought Severity (Deficit):**")

                            deficit_corr = corr_matrix['deficit'].drop('deficit').sort_values(ascending=False)

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Strongest Positive Relationships:**")
                                positive_corr = deficit_corr[deficit_corr > 0].head(3)
                                for feature, corr in positive_corr.items():
                                    strength = "🔴 Strong" if abs(corr) > 0.7 else "🟠 Moderate" if abs(corr) > 0.4 else "🟡 Weak"
                                    st.markdown(f"- **{feature}**: {corr:.3f} {strength}")
                                    if abs(corr) > 0.4:
                                        st.caption(f"  → Higher {feature} tends to mean higher deficit")

                            with col2:
                                st.markdown("**Strongest Negative Relationships:**")
                                negative_corr = deficit_corr[deficit_corr < 0].tail(3)
                                for feature, corr in negative_corr.items():
                                    strength = "🔴 Strong" if abs(corr) > 0.7 else "🟠 Moderate" if abs(corr) > 0.4 else "🟡 Weak"
                                    st.markdown(f"- **{feature}**: {corr:.3f} {strength}")
                                    if abs(corr) > 0.4:
                                        st.caption(f"  → Higher {feature} tends to mean lower deficit")

                        # Check for multicollinearity (high correlation between predictors)
                        st.markdown("---")
                        st.markdown("**⚠️ Multicollinearity Check:**")

                        high_corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                col1_name = corr_matrix.columns[i]
                                col2_name = corr_matrix.columns[j]
                                corr_val = corr_matrix.iloc[i, j]

                                # Skip correlations with deficit (target variable)
                                if col1_name != 'deficit' and col2_name != 'deficit' and abs(corr_val) > 0.8:
                                    high_corr_pairs.append((col1_name, col2_name, corr_val))

                        if high_corr_pairs:
                            st.warning(f"⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8)")
                            st.caption("""
                            **Multicollinearity** means features are very similar. This can cause:
                            - Unstable model coefficients
                            - Difficulty interpreting which feature is really important
                            - Overfitting

                            **Solution:** Consider removing one feature from each pair or using Random Forest (less affected by multicollinearity)
                            """)

                            for feat1, feat2, corr_val in high_corr_pairs:
                                st.markdown(f"- **{feat1}** ↔ **{feat2}**: {corr_val:.3f}")
                        else:
                            st.success("✅ No high multicollinearity detected (all pairwise correlations < 0.8)")
                            st.caption("Your features are reasonably independent, which is good for modeling!")

                    else:
                        st.warning("Not enough numeric features for correlation analysis")

                    # Select features
                    feature_cols = []
                    if use_duration:
                        feature_cols.append('duration')
                    if use_mean_flow:
                        feature_cols.append('mean_flow_before')
                    if use_min_flow:
                        feature_cols.append('min_flow_before')
                    if use_std_flow:
                        feature_cols.append('std_flow_before')
                    if use_trend:
                        feature_cols.append('trend_flow_before')
                    if use_temporal:
                        feature_cols.extend(['month', 'year'])

                    if len(feature_cols) == 0:
                        st.error("⚠️ Please select at least one feature")
                        st.stop()

                    st.markdown(f"**Selected Features:** {', '.join(feature_cols)}")

                    # Prepare data
                    X = features_df[feature_cols].values
                    y = features_df['deficit'].values

                    # Final check for NaN values
                    if np.isnan(X).any() or np.isnan(y).any():
                        st.error("⚠️ NaN values detected in features. This should not happen - please report this issue.")
                        st.write("Features with NaN:")
                        st.write(features_df[feature_cols].isna().sum())
                        st.stop()

                    # Train/test split
                    test_size_ratio = test_split / 100
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_ratio, random_state=42
                    )

                    st.markdown("---")
                    st.markdown("#### Model Training")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training Samples", len(X_train))
                    with col2:
                        st.metric("Test Samples", len(X_test))
                    with col3:
                        st.metric("Features", X_train.shape[1])

                    # Train model
                    with st.spinner("Training model..."):
                        if model_type == 'rf':
                            model, scaler = train_regression_model(X_train, y_train, 'rf', n_estimators=n_estimators)
                        else:
                            model, scaler = train_regression_model(X_train, y_train, 'linear')

                        # Store in session state (use different keys to avoid widget conflict)
                        st.session_state['trained_regression_model'] = model
                        st.session_state['trained_regression_scaler'] = scaler
                        st.session_state['trained_regression_features'] = feature_cols

                    st.success("✅ Model trained successfully!")

                    # Model info
                    st.markdown("---")
                    st.markdown("#### 🎯 Feature Importance & Model Interpretation")

                    st.info("""
                    📊 **Understanding Feature Importance:**

                    Shows which variables have the most influence on drought severity predictions.
                    - **High importance**: This variable strongly affects the prediction
                    - **Low importance**: This variable has little effect
                    """)

                    if model_type == 'linear':
                        st.markdown("**📈 Linear Regression Coefficients**")

                        coef_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Coefficient': model.coef_,
                            'Abs_Coefficient': np.abs(model.coef_)
                        }).sort_values('Abs_Coefficient', ascending=False)

                        # Create bar chart for coefficients
                        fig_coef = go.Figure()

                        colors = ['red' if c < 0 else 'green' for c in coef_df['Coefficient']]

                        fig_coef.add_trace(go.Bar(
                            y=coef_df['Feature'],
                            x=coef_df['Coefficient'],
                            orientation='h',
                            marker=dict(color=colors),
                            text=coef_df['Coefficient'].round(4),
                            textposition='outside'
                        ))

                        fig_coef.update_layout(
                            title='Feature Coefficients (Impact on Deficit)',
                            xaxis_title='Coefficient Value',
                            yaxis_title='Feature',
                            template='plotly_white',
                            height=400,
                            showlegend=False
                        )

                        st.plotly_chart(fig_coef, use_container_width=True)

                        st.write(f"**Model Intercept (baseline):** {model.intercept_:.4f} hm³")

                        # Interpret top coefficients
                        st.markdown("**🔍 What This Means:**")

                        top_positive = coef_df[coef_df['Coefficient'] > 0].head(2)
                        top_negative = coef_df[coef_df['Coefficient'] < 0].head(2)

                        if not top_positive.empty:
                            st.markdown("**Factors that INCREASE deficit:**")
                            for _, row in top_positive.iterrows():
                                st.markdown(f"- **{row['Feature']}**: Each unit increase adds **{row['Coefficient']:.2f} hm³** to deficit")

                        if not top_negative.empty:
                            st.markdown("**Factors that DECREASE deficit:**")
                            for _, row in top_negative.iterrows():
                                st.markdown(f"- **{row['Feature']}**: Each unit increase reduces deficit by **{abs(row['Coefficient']):.2f} hm³**")

                        with st.expander("📋 Full Coefficient Table", expanded=False):
                            st.dataframe(
                                coef_df.style.format({'Coefficient': '{:.4f}', 'Abs_Coefficient': '{:.4f}'}),
                                use_container_width=True
                            )

                    elif model_type == 'rf':
                        st.markdown("**🌲 Random Forest Feature Importance**")

                        importance_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': model.feature_importances_,
                            'Importance_Pct': model.feature_importances_ * 100
                        }).sort_values('Importance', ascending=False)

                        # Create bar chart
                        fig_importance = px.bar(
                            importance_df,
                            x='Importance_Pct',
                            y='Feature',
                            orientation='h',
                            title='Feature Importance (%)',
                            color='Importance_Pct',
                            color_continuous_scale='Viridis',
                            text='Importance_Pct'
                        )

                        fig_importance.update_traces(
                            texttemplate='%{text:.1f}%',
                            textposition='outside'
                        )

                        fig_importance.update_layout(
                            template='plotly_white',
                            height=400,
                            xaxis_title='Importance (%)',
                            yaxis_title='Feature',
                            showlegend=False
                        )

                        st.plotly_chart(fig_importance, use_container_width=True)

                        # Interpret importance
                        st.markdown("**🔍 What This Means:**")

                        total_importance = importance_df['Importance_Pct'].sum()
                        top3 = importance_df.head(3)
                        top3_pct = top3['Importance_Pct'].sum()

                        st.markdown(f"**Top 3 features explain {top3_pct:.1f}% of the model's decisions:**")
                        for idx, row in top3.iterrows():
                            st.markdown(f"- **{row['Feature']}**: {row['Importance_Pct']:.1f}% importance")

                            # Add interpretation based on feature name
                            if 'duration' in row['Feature'].lower():
                                st.caption("  → Longer droughts tend to have more severe water deficits")
                            elif 'mean_flow' in row['Feature'].lower():
                                st.caption("  → Average flow before drought is a strong indicator of severity")
                            elif 'min_flow' in row['Feature'].lower():
                                st.caption("  → Minimum flow conditions signal potential drought intensity")
                            elif 'std_flow' in row['Feature'].lower():
                                st.caption("  → Flow variability affects drought predictability")
                            elif 'trend' in row['Feature'].lower():
                                st.caption("  → Whether flow is rising or falling before drought matters")
                            elif 'month' in row['Feature'].lower():
                                st.caption("  → Season of drought onset affects its severity")

                        # Show cumulative importance
                        importance_df['Cumulative_Pct'] = importance_df['Importance_Pct'].cumsum()

                        features_for_80pct = len(importance_df[importance_df['Cumulative_Pct'] <= 80])
                        if features_for_80pct > 0:
                            st.info(f"💡 **Key Insight:** Just **{features_for_80pct} out of {len(importance_df)} features** explain 80% of predictions!")

                        with st.expander("📋 Full Importance Table", expanded=False):
                            st.dataframe(
                                importance_df.style.format({
                                    'Importance': '{:.4f}',
                                    'Importance_Pct': '{:.2f}%',
                                    'Cumulative_Pct': '{:.2f}%'
                                }),
                                use_container_width=True
                            )

                    st.markdown("---")
                    st.markdown("#### Model Evaluation")

                    # Predictions
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)

                    # Metrics
                    train_metrics = calculate_metrics(y_train, y_train_pred)
                    test_metrics = calculate_metrics(y_test, y_test_pred)

                    # Display metrics
                    st.markdown("**Performance Metrics:**")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Test RMSE", f"{test_metrics['RMSE']:.2f}")
                    with col2:
                        st.metric("Test MAE", f"{test_metrics['MAE']:.2f}")
                    with col3:
                        st.metric("Test R²", f"{test_metrics['R²']:.3f}")
                    with col4:
                        st.metric("Test MAPE", f"{test_metrics['MAPE']:.1f}%")

                    metrics_comparison = pd.DataFrame({
                        'Metric': ['RMSE', 'MAE', 'R²', 'MAPE (%)'],
                        'Training': [train_metrics['RMSE'], train_metrics['MAE'], train_metrics['R²'], train_metrics['MAPE']],
                        'Test': [test_metrics['RMSE'], test_metrics['MAE'], test_metrics['R²'], test_metrics['MAPE']]
                    })

                    st.dataframe(
                        metrics_comparison.style.format({'Training': '{:.4f}', 'Test': '{:.4f}'})
                        .background_gradient(subset=['Test'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )

                    # Check overfitting
                    if train_metrics['R²'] - test_metrics['R²'] > 0.2:
                        st.warning("⚠️ Significant difference between train and test R². Model may be overfitting.")
                        st.caption("""
                        **Overfitting** means the model memorized the training data too well but doesn't work well on new data.
                        Try using fewer features or simpler model (Linear instead of Random Forest).
                        """)

                    # Add interpretation
                    st.markdown("---")
                    st.markdown("#### 📊 What Do These Results Mean?")

                    if test_metrics['R²'] > 0.7:
                        st.success(f"""
                        🌟 **Excellent Performance!** (R² = {test_metrics['R²']:.2%})

                        Your model is **highly accurate** at predicting drought water deficit!

                        **In simple terms:**
                        - The model explains **{test_metrics['R²']:.0%}** of why droughts have different severities
                        - On average, predictions are off by **{test_metrics['MAE']:.2f} hm³** (that's about **{test_metrics['MAPE']:.1f}%** error)
                        - For example, if actual deficit is 100 hm³, prediction is typically around {100 - test_metrics['MAPE']:.0f}-{100 + test_metrics['MAPE']:.0f} hm³

                        **Practical value:**
                        - You can use this model to estimate drought severity early
                        - Helps plan water allocation before drought gets severe
                        - Can trigger early warning systems based on pre-drought conditions
                        """)
                    elif test_metrics['R²'] > 0.5:
                        st.info(f"""
                        ✅ **Good Performance** (R² = {test_metrics['R²']:.2%})

                        Your model has **moderate accuracy** at predicting drought severity.

                        **In simple terms:**
                        - The model explains **{test_metrics['R²']:.0%}** of the variation in drought severity
                        - Average prediction error: **{test_metrics['MAE']:.2f} hm³** (about **{test_metrics['MAPE']:.1f}%**)
                        - Useful for rough estimates, but not precise enough for critical decisions

                        **Ways to improve:**
                        - Try Random Forest if using Linear (or vice versa)
                        - Adjust lookback period (try 60 or 90 days)
                        - Add more features if you have climate data (rainfall, temperature)
                        """)
                    else:
                        st.warning(f"""
                        ⚠️ **Room for Improvement** (R² = {test_metrics['R²']:.2%})

                        The model has **limited accuracy** in predicting drought severity.

                        **What this means:**
                        - Only **{test_metrics['R²']:.0%}** of drought severity variation is predicted correctly
                        - Predictions have high error: **{test_metrics['MAPE']:.1f}%** on average
                        - May not be reliable enough for operational use yet

                        **Recommendations:**
                        1. **Try Random Forest**: Better at capturing complex patterns
                        2. **Increase lookback**: Try 60-90 days instead of 30
                        3. **Check your data**: Make sure stations have enough historical events
                        4. **Add features**: Temperature, rainfall, or soil moisture could help
                        """)

                    st.markdown("---")
                    st.markdown("#### Visualization")

                    # Observed vs Predicted scatter plot
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Training Set', 'Test Set'),
                        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
                    )

                    # Training
                    fig.add_trace(
                        go.Scatter(x=y_train, y=y_train_pred, mode='markers',
                                   name='Train', marker=dict(color='blue', opacity=0.6)),
                        row=1, col=1
                    )

                    # 1:1 line
                    min_val = min(y_train.min(), y_train_pred.min())
                    max_val = max(y_train.max(), y_train_pred.max())
                    fig.add_trace(
                        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode='lines', name='1:1 Line',
                                   line=dict(color='red', dash='dash')),
                        row=1, col=1
                    )

                    # Test
                    fig.add_trace(
                        go.Scatter(x=y_test, y=y_test_pred, mode='markers',
                                   name='Test', marker=dict(color='green', opacity=0.6)),
                        row=1, col=2
                    )

                    # 1:1 line
                    min_val = min(y_test.min(), y_test_pred.min())
                    max_val = max(y_test.max(), y_test_pred.max())
                    fig.add_trace(
                        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode='lines', name='1:1 Line',
                                   line=dict(color='red', dash='dash'),
                                   showlegend=False),
                        row=1, col=2
                    )

                    fig.update_xaxes(title_text="Observed Deficit (hm³)", row=1, col=1)
                    fig.update_xaxes(title_text="Observed Deficit (hm³)", row=1, col=2)
                    fig.update_yaxes(title_text="Predicted Deficit (hm³)", row=1, col=1)
                    fig.update_yaxes(title_text="Predicted Deficit (hm³)", row=1, col=2)

                    fig.update_layout(height=500, template='plotly_white', showlegend=True)

                    st.plotly_chart(fig, use_container_width=True)

                    # Residuals plot
                    st.markdown("**Residual Analysis:**")

                    residuals = y_test - y_test_pred

                    fig_residuals = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Residuals vs Predicted', 'Residuals Distribution')
                    )

                    # Residuals vs Predicted
                    fig_residuals.add_trace(
                        go.Scatter(x=y_test_pred, y=residuals, mode='markers',
                                   marker=dict(color='purple', opacity=0.6)),
                        row=1, col=1
                    )
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

                    # Residuals histogram
                    fig_residuals.add_trace(
                        go.Histogram(x=residuals, nbinsx=20, marker_color='purple', opacity=0.7),
                        row=1, col=2
                    )

                    fig_residuals.update_xaxes(title_text="Predicted Deficit (hm³)", row=1, col=1)
                    fig_residuals.update_yaxes(title_text="Residuals", row=1, col=1)
                    fig_residuals.update_xaxes(title_text="Residuals", row=1, col=2)
                    fig_residuals.update_yaxes(title_text="Frequency", row=1, col=2)

                    fig_residuals.update_layout(height=400, template='plotly_white', showlegend=False)

                    st.plotly_chart(fig_residuals, use_container_width=True)

                    # Export options
                    st.markdown("---")
                    st.markdown("#### Export Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Export predictions
                        predictions_df = pd.DataFrame({
                            'Observed': y_test,
                            'Predicted': y_test_pred,
                            'Residual': residuals,
                            'Abs_Error': np.abs(residuals)
                        })

                        csv = predictions_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv,
                            file_name="regression_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col2:
                        # Export model
                        model_data = {
                            'model': model,
                            'scaler': scaler,
                            'features': feature_cols,
                            'model_type': model_type
                        }
                        model_path = Path("regression_model.pkl")
                        joblib.dump(model_data, model_path)

                        with open(model_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Model (.pkl)",
                                data=f,
                                file_name="regression_model.pkl",
                                mime="application/octet-stream",
                                use_container_width=True
                            )

                    # Recommendations
                    st.markdown("---")
                    st.markdown("#### 💡 Key Insights & Recommendations")

                    # Create summary box with key findings
                    st.markdown("**📋 What We Learned:**")

                    insights = []

                    # Performance insight
                    if test_metrics['R²'] > 0.7:
                        insights.append(f"✅ **Strong Model**: R² = {test_metrics['R²']:.2%} - Model is highly reliable for predicting drought severity")
                    elif test_metrics['R²'] > 0.5:
                        insights.append(f"⚠️ **Moderate Model**: R² = {test_metrics['R²']:.2%} - Model has reasonable accuracy but room for improvement")
                    else:
                        insights.append(f"❌ **Weak Model**: R² = {test_metrics['R²']:.2%} - Model needs significant improvements")

                    # Feature importance insight
                    if model_type == 'rf' and 'importance_df' in locals():
                        top_feature = importance_df.iloc[0]
                        insights.append(f"🎯 **Most Important Factor**: {top_feature['Feature']} ({top_feature['Importance_Pct']:.1f}% importance)")
                    elif model_type == 'linear' and 'coef_df' in locals():
                        top_coef = coef_df.iloc[0]
                        insights.append(f"🎯 **Strongest Predictor**: {top_coef['Feature']} (coefficient: {top_coef['Coefficient']:.3f})")

                    # Correlation insight
                    if 'deficit_corr' in locals():
                        strongest_corr = deficit_corr.abs().idxmax()
                        corr_value = deficit_corr[strongest_corr]
                        insights.append(f"🔗 **Strongest Correlation**: {strongest_corr} (r = {corr_value:.3f}) has the strongest relationship with drought severity")

                    # Multicollinearity insight
                    if 'high_corr_pairs' in locals():
                        if len(high_corr_pairs) > 0:
                            insights.append(f"⚠️ **Feature Redundancy**: Found {len(high_corr_pairs)} highly correlated feature pairs - consider simplifying")
                        else:
                            insights.append("✅ **Independent Features**: All features provide unique information")

                    # Sample size insight
                    if len(features_df) < 50:
                        insights.append(f"⚠️ **Limited Data**: Only {len(features_df)} events - collect more data for better reliability")
                    else:
                        insights.append(f"✅ **Adequate Data**: {len(features_df)} events provide good training foundation")

                    for insight in insights:
                        st.markdown(f"- {insight}")

                    st.markdown("---")
                    st.markdown("**🚀 Next Steps:**")

                    recommendations = []

                    if test_metrics['R²'] > 0.7:
                        recommendations.append("✅ Model is ready for operational use")
                        recommendations.append("✅ Consider deploying for early drought severity estimation")
                        recommendations.append("📊 Monitor performance on new drought events and retrain if accuracy drops")
                    elif test_metrics['R²'] > 0.5:
                        recommendations.append("⚠️ Model works but needs improvement. Try:")
                        recommendations.append("  - Switch model type (Linear ↔ Random Forest)")
                        recommendations.append("  - Adjust lookback period (try 60-90 days)")
                        recommendations.append("  - Add climate variables (precipitation, temperature)")
                        if len(high_corr_pairs) > 0:
                            recommendations.append("  - Remove redundant features to reduce multicollinearity")
                    else:
                        recommendations.append("❌ Model needs significant improvements:")
                        recommendations.append("  - Try Random Forest if using Linear (or vice versa)")
                        recommendations.append("  - Increase lookback period to capture more pre-event patterns")
                        recommendations.append("  - Check data quality - ensure features are calculated correctly")
                        recommendations.append("  - Consider different feature engineering approach")
                        recommendations.append("  - Collect more training data if possible")

                    if model_type == 'rf' and 'importance_df' in locals():
                        low_importance = importance_df[importance_df['Importance_Pct'] < 5]
                        if len(low_importance) > 0:
                            recommendations.append(f"💡 Consider removing {len(low_importance)} low-importance features (<5%) to simplify the model")

                    for rec in recommendations:
                        st.markdown(f"{rec}")

                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
                    st.exception(e)

# ==================== TAB 3: METHODOLOGY ====================
with tab3:
    st.markdown("### 📚 Methodology & Help")

    with st.expander("📖 ARIMA/SARIMA Overview", expanded=True):
        st.markdown("""
        ### What is ARIMA?

        **ARIMA** (AutoRegressive Integrated Moving Average) is a popular time series forecasting method that combines:

        - **AR (AutoRegressive)**: Uses past values to predict future values
        - **I (Integrated)**: Differences the data to make it stationary
        - **MA (Moving Average)**: Uses past forecast errors

        ### ARIMA Parameters (p, d, q)

        - **p**: Number of autoregressive terms (lag order)
        - **d**: Degree of differencing (to make data stationary)
        - **q**: Number of moving average terms

        ### SARIMA Extension

        **SARIMA** adds seasonal components (P, D, Q, s):
        - **P, D, Q**: Seasonal equivalents of p, d, q
        - **s**: Seasonal period (e.g., 12 for monthly data, 365 for daily with yearly seasonality)

        ### When to Use ARIMA

        ✅ **Good for:**
        - Univariate time series
        - Data with trends
        - Short-term forecasts
        - Stationary or trend-stationary data

        ❌ **Not ideal for:**
        - Long-term forecasts
        - Highly irregular data
        - Multiple interacting variables
        - Structural breaks in data

        ### Parameter Selection Tips

        1. **Check stationarity** first (ADF test)
        2. If non-stationary, start with d=1
        3. Use ACF/PACF plots to identify p and q
        4. Start simple: (1,1,1) is often a good baseline
        5. Use information criteria (AIC, BIC) to compare models
        """)

    with st.expander("🎯 Regression Modeling for Drought Events", expanded=True):
        st.markdown("""
        ### Event Deficit Prediction

        Regression models predict the **water deficit** (hm³) of a drought event based on:

        ### Features Used

        1. **Event Characteristics**
           - Duration (days)

        2. **Pre-Event Flow Conditions**
           - Mean flow before event (lookback period)
           - Minimum flow before event
           - Standard deviation of flow
           - Flow trend (increasing/decreasing)

        3. **Temporal Features**
           - Month (seasonality)
           - Year (long-term trends)

        ### Model Types

        #### Linear Regression
        - Assumes linear relationships
        - Interpretable coefficients
        - Fast training
        - Good baseline

        #### Random Forest
        - Captures non-linear relationships
        - Handles feature interactions
        - More robust to outliers
        - Feature importance ranking

        ### Best Practices

        1. **Feature Engineering**: Create meaningful features from raw data
        2. **Scaling**: Standardize features for better performance
        3. **Train/Test Split**: Use time-based or random split (20-30% test)
        4. **Cross-Validation**: Consider k-fold CV for small datasets
        5. **Feature Selection**: Remove redundant or low-importance features

        ### Interpreting Results

        - **R² > 0.7**: Strong model
        - **0.5 < R² < 0.7**: Moderate model
        - **R² < 0.5**: Weak model, needs improvement

        - **RMSE**: Average prediction error (same units as target)
        - **MAE**: Mean absolute error (robust to outliers)
        - **MAPE**: Percentage error (scale-independent)
        """)

    with st.expander("⚙️ Data Preprocessing Tips", expanded=True):
        st.markdown("""
        ### Handling Missing Data

        1. **Forward Fill (ffill)**
           - Uses last known value
           - Good for slowly changing variables
           - May introduce bias in gaps

        2. **Linear Interpolation**
           - Fills gaps linearly
           - Assumes smooth transitions
           - Good for short gaps

        3. **Drop Missing**
           - Removes rows with missing data
           - Use when gaps are small (<5%)

        ### Stationarity

        Time series must be **stationary** for ARIMA:
        - Constant mean over time
        - Constant variance
        - No seasonality

        **Making data stationary:**
        - Differencing (d parameter)
        - Log transformation
        - Seasonal differencing

        ### Train/Test Split

        For **time series**:
        - Use chronological split
        - Train on past, test on future
        - Never shuffle data

        For **regression** (events):
        - Random split is OK
        - Or split by year
        - Ensure test set is representative
        """)

    with st.expander("📊 Model Evaluation Metrics", expanded=True):
        st.markdown("""
        ### Regression Metrics

        #### RMSE (Root Mean Squared Error)
        - Square root of average squared errors
        - Penalizes large errors
        - Same units as target variable
        - Lower is better

        #### MAE (Mean Absolute Error)
        - Average of absolute errors
        - Less sensitive to outliers than RMSE
        - Same units as target
        - Lower is better

        #### R² (R-squared)
        - Proportion of variance explained
        - Range: 0 to 1 (higher is better)
        - Can be negative for very poor models

        #### MAPE (Mean Absolute Percentage Error)
        - Average percentage error
        - Scale-independent
        - Interpretable (e.g., 10% error)
        - Be careful with values near zero

        ### Time Series Metrics

        Same as regression, plus:

        - **Persistence Baseline**: Tomorrow = today
        - **Confidence Intervals**: Uncertainty bounds
        - **Forecast Horizon**: How many steps ahead

        ### Residual Analysis

        Check residuals for:
        - **Zero mean**: Unbiased predictions
        - **No patterns**: Model captures all signal
        - **Constant variance**: Homoscedasticity
        - **Normal distribution**: For confidence intervals
        """)

    with st.expander("🚀 Usage Examples", expanded=True):
        st.markdown("""
        ### Example 1: ARIMA Forecasting

        **Scenario**: Predict next week's daily flow

        1. Select station with good historical data
        2. Choose training period (at least 1 year)
        3. Check stationarity (ADF test)
        4. Start with ARIMA(1,1,1)
        5. Evaluate on test set
        6. Compare to persistence baseline
        7. Export forecast for operational use

        ### Example 2: Drought Deficit Prediction

        **Scenario**: Predict water deficit of upcoming drought

        1. Create features from historical droughts
        2. Use 30-day lookback for flow conditions
        3. Train Linear Regression or Random Forest
        4. Check R² and residuals
        5. Identify important features
        6. Use model to estimate deficit when new drought starts

        ### Example 3: Seasonal Forecasting

        **Scenario**: Account for strong seasonality

        1. Detect seasonality in data (visual inspection)
        2. Enable SARIMA
        3. Set seasonal period (s=12 for monthly, s=365 for daily/yearly)
        4. Start with SARIMA(1,1,1)(1,1,1,s)
        5. Compare to non-seasonal ARIMA

        ### Example 4: Feature Engineering

        **Scenario**: Improve deficit predictions

        1. Calculate additional features:
           - Antecedent precipitation
           - Temperature anomalies
           - Soil moisture index
           - Previous year's drought severity
        2. Test feature importance
        3. Remove low-importance features
        4. Retrain and compare performance
        """)

    with st.expander("💡 Troubleshooting", expanded=True):
        st.markdown("""
        ### Common Issues & Solutions

        #### "Insufficient data" error
        - **Cause**: Too few observations
        - **Solution**: Extend date range or choose different station

        #### "Model does not converge"
        - **Cause**: Bad parameter choice or non-stationary data
        - **Solution**: Check stationarity, try simpler model (reduce p, q)

        #### "High test error, low train error"
        - **Cause**: Overfitting
        - **Solution**: Simplify model, add more training data, use regularization

        #### "Predictions are constant"
        - **Cause**: Model learned only the mean
        - **Solution**: Check feature variation, add more informative features

        #### "Negative predictions"
        - **Cause**: Linear model extrapolating beyond training range
        - **Solution**: Post-process to clip at zero, or use log-transformed target

        #### "Residuals show patterns"
        - **Cause**: Model missing important signal
        - **Solution**: Add features, try non-linear model, check for seasonality

        ### Performance Tips

        - Start simple, then increase complexity
        - Always compare to baseline
        - Use domain knowledge for feature engineering
        - Check data quality before modeling
        - Validate on multiple time periods
        - Consider ensemble methods for robustness
        """)

    st.markdown("---")
    st.markdown("### 📚 Additional Resources")

    st.markdown("""
    #### Recommended Reading

    - **ARIMA**: Hyndman & Athanasopoulos - "Forecasting: Principles and Practice"
    - **Time Series in Python**: statsmodels documentation
    - **Drought Forecasting**: Literature on hydrological forecasting methods

    #### Python Libraries Used

    - `statsmodels`: ARIMA/SARIMA implementation
    - `scikit-learn`: Regression models and preprocessing
    - `pandas`: Data manipulation
    - `numpy`: Numerical operations
    - `plotly`: Interactive visualizations
    - `joblib`: Model serialization

    #### Extending This Tool

    Future enhancements could include:
    - Prophet for trend+seasonality decomposition
    - LSTM neural networks for complex patterns
    - Multivariate models (VAR, VARMAX)
    - Exogenous variables (weather, climate indices)
    - Automated hyperparameter tuning
    - Online learning for model updates
    """)

# Footer
st.markdown("---")

col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown(f"""
        <div class="footer-author">
            <h3>{AUTHOR_INFO['name']}</h3>
            <p><strong>{AUTHOR_INFO['occupation']}</strong></p>
            <p>{AUTHOR_INFO['institution']}</p>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">{AUTHOR_INFO['institution_full']}</p>
        </div>
    """, unsafe_allow_html=True)

with col_right:
    try:
        logo_path = BASE_DIR / "data" / "png" / "logo-ipe22.png"
        if logo_path.exists():
            st.image(str(logo_path), width=200)
    except Exception:
        pass

st.markdown("""
<div class="footer">
    <p>🤖 ML Prediction & Forecasting | Advanced machine learning for drought analysis</p>
</div>
""", unsafe_allow_html=True)
