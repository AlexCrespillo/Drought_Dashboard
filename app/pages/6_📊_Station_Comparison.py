"""
Station Comparison Page - Multi-station Analysis and Comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from scipy import stats

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
        <h1>📊 Station Comparison & Multi-Station Analysis</h1>
        <p>Compare streamflow patterns, statistics, and anomalies across multiple stations</p>
    </div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
with st.spinner("Loading streamflow data..."):
    streamflow_df = load_streamflow_data()
    drought_df = load_drought_events()

if streamflow_df.empty:
    st.error("No streamflow data available")
    st.stop()

# ==================== SIDEBAR: STATION SELECTION ====================
st.sidebar.header("🔍 Station Selection")

# Get list of available stations
available_stations = sorted(streamflow_df['indroea'].unique())

# Multi-select widget for stations (2-6 stations)
selected_stations = st.sidebar.multiselect(
    "Select 2-6 stations to compare",
    options=available_stations,
    default=available_stations[:3] if len(available_stations) >= 3 else available_stations,
    max_selections=6,
    help="Choose between 2 and 6 stations for comparison"
)

# Validate selection
if len(selected_stations) < 2:
    st.warning("⚠️ Please select at least 2 stations to compare")
    st.stop()

if len(selected_stations) > 6:
    st.warning("⚠️ Please select no more than 6 stations")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Display Options")

# Normalize option
normalize = st.sidebar.checkbox(
    "Normalize series",
    value=False,
    help="Normalize data to 0-1 scale for visual comparison"
)

# Date range filter
if 'fecha' in streamflow_df.columns:
    streamflow_df['fecha'] = pd.to_datetime(streamflow_df['fecha'])
    date_min = streamflow_df['fecha'].min()
    date_max = streamflow_df['fecha'].max()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )

    if len(date_range) == 2:
        date_start, date_end = date_range
        streamflow_df = streamflow_df[
            (streamflow_df['fecha'] >= pd.to_datetime(date_start)) &
            (streamflow_df['fecha'] <= pd.to_datetime(date_end))
        ]

# ==================== PREPARE DATA ====================
# Filter data for selected stations
comparison_data = streamflow_df[streamflow_df['indroea'].isin(selected_stations)].copy()

# ==================== MAIN CONTENT ====================
st.markdown(f"### Comparing {len(selected_stations)} Stations")

# Display selected station info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Stations Selected", len(selected_stations))
with col2:
    st.metric("Date Range", f"{len(comparison_data['fecha'].unique())} days")
with col3:
    st.metric("Total Records", f"{len(comparison_data):,}")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Time Series Comparison",
    "📊 Summary Statistics",
    "📉 Anomaly Analysis",
    "📋 Detailed Comparison"
])

# ==================== TAB 1: TIME SERIES COMPARISON ====================
with tab1:
    st.markdown("### 📈 Streamflow Time Series")

    # Create interactive time series plot
    fig = go.Figure()

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    for idx, station in enumerate(selected_stations):
        station_data = comparison_data[comparison_data['indroea'] == station].sort_values('fecha')

        if not station_data.empty:
            y_values = station_data['caudal'].values

            # Normalize if requested
            if normalize:
                y_min = y_values.min()
                y_max = y_values.max()
                if y_max > y_min:
                    y_values = (y_values - y_min) / (y_max - y_min)

            # Get station name
            station_name = station_data['lugar'].iloc[0] if 'lugar' in station_data.columns else f"Station {station}"

            fig.add_trace(go.Scatter(
                x=station_data['fecha'],
                y=y_values,
                name=f"{station} - {station_name}",
                mode='lines',
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Flow: %{y:.2f}<extra></extra>'
            ))

    fig.update_layout(
        title="Streamflow Time Series Comparison" + (" (Normalized)" if normalize else ""),
        xaxis_title="Date",
        yaxis_title="Normalized Flow (0-1)" if normalize else "Streamflow (m³/s)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="timeseries_comparison")

    st.info("💡 **Tip:** Use the normalize checkbox in the sidebar to compare stations on the same relative scale")

# ==================== TAB 2: SUMMARY STATISTICS ====================
with tab2:
    st.markdown("### 📊 Summary Statistics by Station")

    # Calculate statistics for each station
    stats_list = []

    for station in selected_stations:
        station_data = comparison_data[comparison_data['indroea'] == station]

        if not station_data.empty and 'caudal' in station_data.columns:
            flows = station_data['caudal'].dropna()

            if len(flows) > 0:
                # Get station name
                station_name = station_data['lugar'].iloc[0] if 'lugar' in station_data.columns else "Unknown"

                # Calculate statistics
                stats_dict = {
                    'Station ID': station,
                    'Station Name': station_name,
                    'Count': len(flows),
                    'Mean (m³/s)': flows.mean(),
                    'Median (m³/s)': flows.median(),
                    'Std Dev (m³/s)': flows.std(),
                    'Q95 (m³/s)': flows.quantile(0.95),
                    'Q5 (m³/s)': flows.quantile(0.05),
                    'CV (%)': (flows.std() / flows.mean() * 100) if flows.mean() > 0 else 0,
                    'Min (m³/s)': flows.min(),
                    'Max (m³/s)': flows.max()
                }

                stats_list.append(stats_dict)

    if stats_list:
        stats_df = pd.DataFrame(stats_list)

        # Display metrics in columns
        st.markdown("#### Key Statistics")

        # Create comparison table
        st.dataframe(
            stats_df.style.format({
                'Mean (m³/s)': '{:.2f}',
                'Median (m³/s)': '{:.2f}',
                'Std Dev (m³/s)': '{:.2f}',
                'Q95 (m³/s)': '{:.2f}',
                'Q5 (m³/s)': '{:.2f}',
                'CV (%)': '{:.2f}',
                'Min (m³/s)': '{:.2f}',
                'Max (m³/s)': '{:.2f}'
            }).background_gradient(subset=['Mean (m³/s)', 'Q95 (m³/s)', 'Q5 (m³/s)'], cmap='Blues'),
            use_container_width=True,
            height=300
        )

        st.markdown("---")

        # Visual comparison of key statistics
        st.markdown("#### Visual Comparison")

        col1, col2 = st.columns(2)

        with col1:
            # Mean comparison
            fig_mean = go.Figure()
            fig_mean.add_trace(go.Bar(
                x=stats_df['Station ID'],
                y=stats_df['Mean (m³/s)'],
                marker_color=colors[:len(stats_df)],
                text=stats_df['Mean (m³/s)'].round(2),
                textposition='auto',
            ))
            fig_mean.update_layout(
                title="Mean Streamflow Comparison",
                xaxis_title="Station ID",
                yaxis_title="Mean Flow (m³/s)",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_mean, use_container_width=True, key="mean_comparison")

        with col2:
            # CV comparison
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                x=stats_df['Station ID'],
                y=stats_df['CV (%)'],
                marker_color=colors[:len(stats_df)],
                text=stats_df['CV (%)'].round(1),
                textposition='auto',
            ))
            fig_cv.update_layout(
                title="Coefficient of Variation Comparison",
                xaxis_title="Station ID",
                yaxis_title="CV (%)",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_cv, use_container_width=True, key="cv_comparison")

        # Percentile comparison
        st.markdown("#### Percentile Comparison (Q5 vs Q95)")

        fig_percentiles = go.Figure()

        fig_percentiles.add_trace(go.Bar(
            name='Q5 (Low Flow)',
            x=stats_df['Station ID'],
            y=stats_df['Q5 (m³/s)'],
            marker_color='#e74c3c'
        ))

        fig_percentiles.add_trace(go.Bar(
            name='Q95 (High Flow)',
            x=stats_df['Station ID'],
            y=stats_df['Q95 (m³/s)'],
            marker_color='#3498db'
        ))

        fig_percentiles.update_layout(
            title="Q5 and Q95 Percentiles by Station",
            xaxis_title="Station ID",
            yaxis_title="Flow (m³/s)",
            barmode='group',
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig_percentiles, use_container_width=True, key="percentiles_comparison")

        # Download button
        st.markdown("---")
        csv = stats_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Statistics (CSV)",
            data=csv,
            file_name="station_comparison_statistics.csv",
            mime="text/csv"
        )
    else:
        st.warning("No statistics could be calculated for the selected stations")

# ==================== TAB 3: ANOMALY ANALYSIS ====================
with tab3:
    st.markdown("### 📉 Synchronized Anomaly Analysis")

    st.info("💡 **Anomalies** are calculated as deviations from each station's mean flow (z-scores). "
            "This allows comparison of relative variations across stations with different flow magnitudes.")

    # Calculate anomalies (z-scores) for each station
    fig_anomaly = go.Figure()

    for idx, station in enumerate(selected_stations):
        station_data = comparison_data[comparison_data['indroea'] == station].sort_values('fecha')

        if not station_data.empty and len(station_data) > 1:
            flows = station_data['caudal'].values

            # Calculate z-scores (standardized anomalies)
            mean_flow = np.mean(flows)
            std_flow = np.std(flows)

            if std_flow > 0:
                z_scores = (flows - mean_flow) / std_flow

                station_name = station_data['lugar'].iloc[0] if 'lugar' in station_data.columns else f"Station {station}"

                fig_anomaly.add_trace(go.Scatter(
                    x=station_data['fecha'],
                    y=z_scores,
                    name=f"{station} - {station_name}",
                    mode='lines',
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Z-score: %{y:.2f}<extra></extra>'
                ))

    # Add reference line at z=0
    fig_anomaly.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig_anomaly.update_layout(
        title="Streamflow Anomalies (Z-scores)",
        xaxis_title="Date",
        yaxis_title="Standardized Anomaly (z-score)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    st.plotly_chart(fig_anomaly, use_container_width=True, key="anomaly_comparison")

    st.markdown("---")

    # Correlation analysis
    st.markdown("#### Cross-Station Correlation Analysis")

    # Create correlation matrix
    correlation_data = {}
    for station in selected_stations:
        station_flows = comparison_data[comparison_data['indroea'] == station].set_index('fecha')['caudal']
        correlation_data[station] = station_flows

    corr_df = pd.DataFrame(correlation_data)

    if not corr_df.empty:
        correlation_matrix = corr_df.corr()

        # Create heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))

        fig_corr.update_layout(
            title="Streamflow Correlation Matrix",
            template='plotly_white',
            height=400,
            xaxis_title="Station ID",
            yaxis_title="Station ID"
        )

        st.plotly_chart(fig_corr, use_container_width=True, key="correlation_matrix")

        st.markdown("""
        **Interpretation:**
        - Values close to **1**: Strong positive correlation (stations behave similarly)
        - Values close to **-1**: Strong negative correlation (opposite behavior)
        - Values close to **0**: Weak or no correlation
        """)

# ==================== TAB 4: DETAILED COMPARISON ====================
with tab4:
    st.markdown("### 📋 Detailed Data Comparison")

    # Create side-by-side comparison
    st.markdown("#### Station Information")

    for station in selected_stations:
        station_data = comparison_data[comparison_data['indroea'] == station]

        if not station_data.empty:
            station_name = station_data['lugar'].iloc[0] if 'lugar' in station_data.columns else "Unknown"

            with st.expander(f"📍 Station {station} - {station_name}", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Records", len(station_data))
                    if 'xutm' in station_data.columns:
                        st.metric("X (UTM)", f"{station_data['xutm'].iloc[0]:.0f}")

                with col2:
                    st.metric("Mean Flow", f"{station_data['caudal'].mean():.2f} m³/s")
                    if 'yutm' in station_data.columns:
                        st.metric("Y (UTM)", f"{station_data['yutm'].iloc[0]:.0f}")

                with col3:
                    # Get drought count for this station
                    station_droughts = drought_df[drought_df['indroea'] == station] if not drought_df.empty else pd.DataFrame()
                    st.metric("Drought Events", len(station_droughts))
                    st.metric("Max Flow", f"{station_data['caudal'].max():.2f} m³/s")

                # Show sample data
                st.markdown("**Recent Data Sample:**")
                sample_data = station_data.sort_values('fecha', ascending=False).head(10)[['fecha', 'caudal']]
                st.dataframe(sample_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Export combined data
    st.markdown("#### 💾 Export Comparison Data")

    export_data = comparison_data[['fecha', 'indroea', 'caudal', 'lugar']].sort_values(['indroea', 'fecha'])

    csv_export = export_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download All Selected Station Data (CSV)",
        data=csv_export,
        file_name=f"station_comparison_{len(selected_stations)}_stations.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(f"""
    <div class="footer-author">
        <h3>{AUTHOR_INFO['name']}</h3>
        <p><strong>{AUTHOR_INFO['occupation']}</strong></p>
        <p>{AUTHOR_INFO['institution']}</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>📊 Station Comparison | Multi-station streamflow analysis and statistical comparison</p>
</div>
""", unsafe_allow_html=True)
