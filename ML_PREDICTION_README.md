# 🤖 ML Prediction & Forecasting Module

## Overview

The ML Prediction module (`9_🤖_ML_Prediction.py`) provides advanced machine learning capabilities for:
1. **Time Series Forecasting**: ARIMA/SARIMA models for daily streamflow prediction
2. **Event Prediction**: Regression models for drought event deficit prediction

---

## Features

### 1. ARIMA/SARIMA Time Series Forecasting

**Purpose**: Predict future daily streamflow values based on historical patterns.

**Key Capabilities**:
- Single-station analysis
- Configurable ARIMA parameters (p, d, q)
- Optional seasonal components (P, D, Q, s)
- Multi-step forecasting (1-30 days ahead)
- Stationarity testing (Augmented Dickey-Fuller)
- Baseline comparison (persistence model)
- Confidence interval visualization
- Model export (.pkl format)

**Workflow**:
1. Select station and date range
2. Configure ARIMA parameters or use defaults
3. Train model on historical data
4. Evaluate on test set
5. Generate future forecasts
6. Export predictions and model

**Metrics Provided**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

### 2. Drought Event Deficit Prediction

**Purpose**: Predict water deficit (hm³) of drought events based on event characteristics and pre-event flow conditions.

**Features Engineered**:
- Event duration (days)
- Mean flow before event (lookback period: 7-90 days)
- Minimum flow before event
- Standard deviation of flow before event
- Flow trend before event (increasing/decreasing)
- Temporal features (month, year)

**Model Options**:
- **Linear Regression**: Interpretable, fast, good baseline
- **Random Forest**: Captures non-linear relationships, feature importance

**Workflow**:
1. Configure lookback period for feature creation
2. Select features to include
3. Choose model type and parameters
4. Train on historical drought events
5. Evaluate on test set
6. Analyze feature importance
7. Export model and predictions

**Visualizations**:
- Observed vs Predicted scatter plot with 1:1 line
- Residual plots (vs predicted, distribution)
- Feature importance (for Random Forest)
- Model coefficients (for Linear Regression)

---

## Code Structure

### Main Functions

#### Data Preparation
```python
prepare_streamflow_data(df, station, start_date, end_date, fill_method)
```
- Prepares time series data for ARIMA
- Handles missing values (forward fill, interpolation, or drop)
- Creates complete date range

```python
create_event_features(drought_df, streamflow_df, lookback_days)
```
- Engineers features for event prediction
- Calculates flow statistics before each event
- Adds temporal features

#### ARIMA Modeling
```python
check_stationarity(timeseries)
```
- Performs Augmented Dickey-Fuller test
- Returns stationarity status and statistics

```python
train_arima_model(data, order, seasonal_order)
```
- Trains ARIMA or SARIMA model
- Returns fitted model object

```python
forecast_arima(model, steps)
```
- Generates forecast with confidence intervals
- Returns DataFrame with predictions and bounds

#### Regression Modeling
```python
train_regression_model(X_train, y_train, model_type, **kwargs)
```
- Trains Linear Regression or Random Forest
- Performs feature scaling
- Returns model and scaler

```python
calculate_metrics(y_true, y_pred)
```
- Computes RMSE, MAE, R², MAPE
- Returns metrics dictionary

#### Utilities
```python
create_baseline_forecast(data, steps)
```
- Creates persistence baseline (tomorrow = today)
- Used for model comparison

---

## User Interface

### Tab 1: Time Series Forecasting (ARIMA)

**Sidebar Controls**:
- Station selector
- Training period (date range)
- Test set size (days)
- Missing value handling method
- ARIMA parameters (p, d, q)
- Seasonal ARIMA toggle and parameters (P, D, Q, s)
- Forecast horizon (1-30 days)

**Main Content**:
- Data preparation summary
- Stationarity test results
- Model training and summary
- Performance metrics comparison (train/test/baseline)
- Observed vs Predicted plot with confidence intervals
- Residual analysis
- Future forecast visualization
- Export options (CSV predictions, PKL model)
- Recommendations and interpretations

### Tab 2: Event Prediction (Regression)

**Sidebar Controls**:
- Lookback period (days before event)
- Model type (Linear/Random Forest)
- Number of trees (for Random Forest)
- Test set size (%)
- Feature selection checkboxes

**Main Content**:
- Feature engineering summary
- Feature statistics table
- Model training confirmation
- Coefficients (Linear) or Feature Importance (RF)
- Performance metrics
- Observed vs Predicted scatter plots (train/test)
- Residual analysis (vs predicted, distribution)
- Export options (CSV predictions, PKL model bundle)
- Recommendations

### Tab 3: Methodology & Help

**Educational Content**:
- ARIMA/SARIMA overview
- Parameter selection tips
- Regression modeling explanation
- Data preprocessing best practices
- Model evaluation metrics definitions
- Usage examples
- Troubleshooting guide
- Additional resources

---

## Dependencies

All required packages are in `requirements.txt`:

```python
# Core
pandas>=2.1.0
numpy>=1.24.0

# Machine Learning
scikit-learn>=1.3.0
statsmodels>=0.14.0
joblib>=1.3.0

# Visualization
plotly>=5.15.0
streamlit>=1.26.0

# Statistical
scipy>=1.11.0
```

---

## Model Export

### ARIMA Model Export
- **Format**: `.pkl` (pickle via joblib)
- **Contents**: Fitted statsmodels ARIMA/SARIMA object
- **Loading**:
```python
import joblib
model = joblib.load('arima_model_station_123.pkl')
forecast = model.forecast(steps=7)
```

### Regression Model Export
- **Format**: `.pkl` (pickle via joblib)
- **Contents**: Dictionary with:
  - `model`: Trained sklearn model
  - `scaler`: StandardScaler for features
  - `features`: List of feature names (order matters!)
  - `model_type`: 'linear' or 'rf'
- **Loading**:
```python
import joblib
model_data = joblib.load('regression_model.pkl')
model = model_data['model']
scaler = model_data['scaler']

# For new prediction
X_new_scaled = scaler.transform(X_new)
prediction = model.predict(X_new_scaled)
```

---

## Best Practices

### Time Series Forecasting

1. **Always check stationarity** before fitting ARIMA
2. **Start simple**: Begin with ARIMA(1,1,1)
3. **Use AIC/BIC**: Lower is better for model selection
4. **Compare to baseline**: Persistence model provides context
5. **Validate chronologically**: Train on past, test on future
6. **Short-term only**: ARIMA is best for 1-7 day forecasts

### Event Prediction

1. **Feature engineering**: Domain knowledge is crucial
2. **Avoid data leakage**: Only use pre-event information
3. **Check for overfitting**: Compare train vs test R²
4. **Minimum sample size**: At least 20-30 events
5. **Feature selection**: Remove low-importance features
6. **Interpretability**: Start with Linear Regression

### General

1. **Document everything**: Save model parameters and performance
2. **Version control**: Track model versions and retraining
3. **Monitor performance**: Retrain periodically with new data
4. **Communicate uncertainty**: Always show confidence intervals
5. **Domain validation**: Check if predictions make physical sense

---

## Troubleshooting

### Common Issues

**Issue**: "Insufficient data for station X"
- **Solution**: Select different station or extend date range

**Issue**: "Model does not converge"
- **Solution**:
  - Check data quality (outliers, missing values)
  - Reduce ARIMA parameters (simpler model)
  - Try differencing (increase d)

**Issue**: "High test error, low train error"
- **Solution**: Model is overfitting
  - Simplify model (fewer features, lower RF trees)
  - Get more training data
  - Use regularization

**Issue**: "Predictions are unrealistic"
- **Solution**:
  - Check feature scaling
  - Verify no data leakage
  - Post-process predictions (e.g., clip negatives to zero)

---

## Future Enhancements

Potential additions:
- **Prophet**: Facebook's forecasting tool for trend+seasonality
- **LSTM**: Neural networks for complex temporal patterns
- **XGBoost**: Gradient boosting for event prediction
- **Multivariate models**: VAR, VARMAX with exogenous variables
- **Feature automation**: Auto-generate lag features
- **Hyperparameter tuning**: Grid search or Bayesian optimization
- **Cross-validation**: Time series CV for robustness
- **Ensemble methods**: Combine multiple models
- **Online learning**: Update models with new data
- **Uncertainty quantification**: Conformal prediction, quantile regression

---

## References

### ARIMA/Time Series
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.)
- Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time series analysis: forecasting and control*

### Drought Forecasting
- Mishra, A. K., & Singh, V. P. (2011). Drought modeling–A review. *Journal of hydrology*, 403(1-2), 157-175.
- Hao, Z., et al. (2018). A statistical method for categorical drought prediction based on NLDAS-2. *Journal of Applied Meteorology and Climatology*, 57(6), 1049-1061.

### Python Documentation
- statsmodels: https://www.statsmodels.org/
- scikit-learn: https://scikit-learn.org/
- plotly: https://plotly.com/python/

---

## Contact

For questions or issues with the ML module:
- **Author**: Alex Crespillo López
- **Institution**: IPE-CSIC
- **Project**: Drought Events Analysis Dashboard

---

**Note**: This module is designed for research and operational drought forecasting. Always validate predictions with domain expertise and consider physical constraints of the hydrological system.
