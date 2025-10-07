"""
Data Overview Page - Dataset Exploration and Quality Metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

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
st.markdown(f"""
    <div class="main-header">
        <h1>📊 Data Overview & Exploration</h1>
        <p>Comprehensive analysis of drought events and streamflow patterns in the Ebro River Basin</p>
    </div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
with st.spinner("Loading datasets..."):
    drought_df = load_drought_events()
    streamflow_df = load_streamflow_data()

# ==================== SIDEBAR FILTERS ====================
st.sidebar.header("🔍 Filters")

# Year range filter
if not drought_df.empty:
    year_min = int(drought_df['year'].min())
    year_max = int(drought_df['year'].max())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max)
    )

# Station filter
stations = get_station_list(drought_df)
selected_stations = st.sidebar.multiselect(
    "Select Stations (leave empty for all)",
    options=stations,
    default=[]
)

# Apply filters
if selected_stations:
    drought_df = filter_by_stations(drought_df, selected_stations)
    streamflow_df = filter_by_stations(streamflow_df, selected_stations)

if not drought_df.empty:
    drought_df = drought_df[(drought_df['year'] >= year_range[0]) &
                            (drought_df['year'] <= year_range[1])]

# ==================== HERO SECTION WITH KEY METRICS ====================
st.markdown("## 🎯 Dataset at a Glance")

if not drought_df.empty:
    drought_stats = calculate_drought_statistics(drought_df)

    # Key metrics in attractive cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3 style="margin:0; font-size: 2.5em;">💧</h3>
                <h2 style="margin:10px 0; font-size: 2em;">{drought_stats['total_events']:,}</h2>
                <p style="margin:0;">Drought Events</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3 style="margin:0; font-size: 2.5em;">📍</h3>
                <h2 style="margin:10px 0; font-size: 2em;">{drought_stats['total_stations']}</h2>
                <p style="margin:0;">Stations</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                        padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3 style="margin:0; font-size: 2.5em;">⏱️</h3>
                <h2 style="margin:10px 0; font-size: 2em;">{drought_stats['avg_duration']:.0f}</h2>
                <p style="margin:0;">Avg Duration (days)</p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                        padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3 style="margin:0; font-size: 2.5em;">💦</h3>
                <h2 style="margin:10px 0; font-size: 2em;">{drought_stats['total_deficit']:.0f}</h2>
                <p style="margin:0;">Total Deficit (hm³)</p>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
                        padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3 style="margin:0; font-size: 2.5em;">📅</h3>
                <h2 style="margin:10px 0; font-size: 2em;">{drought_stats['years_covered']}</h2>
                <p style="margin:0;">Years Covered</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ==================== VISUAL OVERVIEW ====================
st.markdown("## 📈 Visual Dataset Overview")

if not drought_df.empty and not streamflow_df.empty:

    # Create two columns for side-by-side visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌊 Drought Events Timeline")

        # Annual drought events
        annual_events = drought_df.groupby('year').size().reset_index(name='count')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=annual_events['year'],
            y=annual_events['count'],
            mode='lines+markers',
            name='Events',
            line=dict(color='#d62728', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.2)'
        ))

        fig.update_layout(
            title="Annual Drought Events Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Events",
            template="plotly_white",
            hovermode='x unified',
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="drought_events_timeline")

        st.info(f"📊 **Insight:** On average, **{annual_events['count'].mean():.1f} drought events** occur per year, "
                f"with a maximum of **{annual_events['count'].max()} events** in {annual_events.loc[annual_events['count'].idxmax(), 'year']:.0f}.")

    with col2:
        st.markdown("### 💧 Water Deficit Evolution")

        # Annual water deficit
        annual_deficit = drought_df.groupby('year')['deficit_hm3'].sum().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=annual_deficit['year'],
            y=annual_deficit['deficit_hm3'],
            name='Water Deficit',
            marker=dict(
                color=annual_deficit['deficit_hm3'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="hm³")
            )
        ))

        fig.update_layout(
            title="Annual Total Water Deficit",
            xaxis_title="Year",
            yaxis_title="Water Deficit (hm³)",
            template="plotly_white",
            hovermode='x unified',
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="water_deficit_evolution")

        st.info(f"💦 **Insight:** The basin experienced a cumulative water deficit of **{annual_deficit['deficit_hm3'].sum():.0f} hm³** "
                f"during the study period.")

st.markdown("---")

# ==================== DISTRIBUTION ANALYSIS ====================
st.markdown("## 📊 Drought Characteristics Distribution")

if not drought_df.empty:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ⏱️ Duration Distribution")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=drought_df['duracion'],
            nbinsx=50,
            marker=dict(color='#4facfe', line=dict(color='white', width=1)),
            name='Duration'
        ))

        fig.update_layout(
            xaxis_title="Duration (days)",
            yaxis_title="Frequency",
            template="plotly_white",
            showlegend=False,
            height=300
        )

        st.plotly_chart(fig, use_container_width=True, key="duration_distribution")

        st.caption(f"**Median:** {drought_df['duracion'].median():.0f} days | "
                   f"**Max:** {drought_df['duracion'].max():.0f} days")

    with col2:
        st.markdown("### 💧 Deficit Distribution")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=drought_df['deficit_hm3'],
            nbinsx=50,
            marker=dict(color='#fa709a', line=dict(color='white', width=1)),
            name='Deficit'
        ))

        fig.update_layout(
            xaxis_title="Water Deficit (hm³)",
            yaxis_title="Frequency",
            template="plotly_white",
            showlegend=False,
            height=300
        )

        st.plotly_chart(fig, use_container_width=True, key="deficit_distribution")

        st.caption(f"**Median:** {drought_df['deficit_hm3'].median():.2f} hm³ | "
                   f"**Max:** {drought_df['deficit_hm3'].max():.2f} hm³")

    with col3:
        st.markdown("### 📅 Seasonal Pattern")

        if 'season' in drought_df.columns:
            seasonal = drought_df.groupby('season').size().reset_index(name='count')
            season_order = ['Winter', 'Spring', 'Summer', 'Fall']
            seasonal['season'] = pd.Categorical(seasonal['season'], categories=season_order, ordered=True)
            seasonal = seasonal.sort_values('season')

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=seasonal['season'],
                y=seasonal['count'],
                marker=dict(
                    color=['#667eea', '#2ca02c', '#ff7f0e', '#d62728'],
                    line=dict(color='white', width=1)
                )
            ))

            fig.update_layout(
                xaxis_title="Season",
                yaxis_title="Number of Events",
                template="plotly_white",
                showlegend=False,
                height=300
            )

            st.plotly_chart(fig, use_container_width=True, key="seasonal_pattern")

            max_season = seasonal.loc[seasonal['count'].idxmax(), 'season']
            st.caption(f"**Peak season:** {max_season} with {seasonal['count'].max()} events")

st.markdown("---")

# ==================== STREAMFLOW OVERVIEW ====================
if not streamflow_df.empty:
    st.markdown("## 🌊 Streamflow Data Overview")

    flow_stats = calculate_streamflow_statistics(streamflow_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Records", f"{flow_stats['total_records']:,}")
    with col2:
        st.metric("📍 Stations", flow_stats['total_stations'])
    with col3:
        st.metric("💧 Avg Flow", f"{flow_stats['avg_streamflow']:.2f} m³/s")
    with col4:
        st.metric("📈 Max Flow", f"{flow_stats['max_streamflow']:.2f} m³/s")

    st.markdown("---")

    # Streamflow statistics by station
    st.markdown("### 🏆 Top Stations by Average Streamflow")

    station_flow = streamflow_df.groupby('indroea')['caudal'].agg(['mean', 'median', 'max']).reset_index()
    station_flow.columns = ['Station', 'Mean Flow', 'Median Flow', 'Max Flow']
    station_flow = station_flow.sort_values('Mean Flow', ascending=False).head(10)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=station_flow['Station'].astype(str),
        y=station_flow['Mean Flow'],
        marker=dict(
            color=station_flow['Mean Flow'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="m³/s")
        ),
        text=station_flow['Mean Flow'].round(2),
        textposition='outside'
    ))

    fig.update_layout(
        title="Top 10 Stations by Average Streamflow",
        xaxis_title="Station ID",
        yaxis_title="Average Flow (m³/s)",
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True, key="top_stations_flow")

st.markdown("---")

# ==================== TABS FOR DETAILED EXPLORATION ====================
st.markdown("## 🔍 Detailed Data Exploration")

tab1, tab2, tab3 = st.tabs(["📋 Data Quality", "📊 Statistical Summary", "💾 Export Data"])

# ==================== TAB 1: DATA QUALITY ====================
with tab1:
    st.markdown("### 📈 Data Quality Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌊 Drought Events Quality")

        if not drought_df.empty:
            missing_counts = drought_df.isnull().sum()
            missing_pct = (missing_counts / len(drought_df) * 100).round(2)

            quality_df = pd.DataFrame({
                'Column': missing_counts.index,
                'Missing': missing_counts.values,
                'Percentage': missing_pct.values
            })

            st.dataframe(quality_df, use_container_width=True, hide_index=True)

            completeness = 100 - missing_pct.mean()

            # Completeness gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=completeness,
                title={'text': "Data Completeness"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen" if completeness > 90 else "orange"},
                       'steps': [
                           {'range': [0, 70], 'color': "lightgray"},
                           {'range': [70, 90], 'color': "lightyellow"},
                           {'range': [90, 100], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 95}}
            ))

            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True, key="drought_completeness_gauge")

    with col2:
        st.markdown("#### 💧 Streamflow Data Quality")

        if not streamflow_df.empty:
            missing_counts = streamflow_df.isnull().sum()
            missing_pct = (missing_counts / len(streamflow_df) * 100).round(2)

            quality_df = pd.DataFrame({
                'Column': missing_counts.index,
                'Missing': missing_counts.values,
                'Percentage': missing_pct.values
            })

            st.dataframe(quality_df, use_container_width=True, hide_index=True)

            completeness = 100 - missing_pct.mean()

            # Completeness gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=completeness,
                title={'text': "Data Completeness"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen" if completeness > 90 else "orange"},
                       'steps': [
                           {'range': [0, 70], 'color': "lightgray"},
                           {'range': [70, 90], 'color': "lightyellow"},
                           {'range': [90, 100], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 95}}
            ))

            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True, key="streamflow_completeness_gauge")

# ==================== TAB 2: STATISTICAL SUMMARY ====================
with tab2:
    st.markdown("### 📊 Complete Statistical Summary")

    if not drought_df.empty:
        st.markdown("#### 🌊 Drought Events Statistics")

        stats_df = drought_df[['duracion', 'deficit_hm3']].describe().T
        stats_df['median'] = drought_df[['duracion', 'deficit_hm3']].median()
        stats_df = stats_df[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']]

        st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")

    if not streamflow_df.empty:
        st.markdown("#### 💧 Streamflow Statistics")

        flow_stats_df = streamflow_df[['caudal']].describe().T

        st.dataframe(flow_stats_df, use_container_width=True)

# ==================== TAB 3: EXPORT DATA ====================
with tab3:
    st.markdown("### 💾 Export Filtered Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌊 Drought Events")
        if not drought_df.empty:
            st.info(f"📊 {len(drought_df):,} events ready for export")

            csv = drought_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Drought Events CSV",
                data=csv,
                file_name=f"drought_events_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv",
                key="drought_export"
            )

    with col2:
        st.markdown("#### 💧 Streamflow Data")
        if not streamflow_df.empty:
            st.info(f"📊 {len(streamflow_df):,} records ready for export")

            csv = streamflow_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Streamflow CSV",
                data=csv,
                file_name=f"streamflow_data_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv",
                key="streamflow_export"
            )

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
    <p>📊 Data Overview & Exploration | Interactive dashboard for drought and streamflow analysis</p>
</div>
""", unsafe_allow_html=True)
