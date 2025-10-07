"""
Spatial Analysis Page - Geographic Distribution and Basin Characteristics
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import plotly.express as px
import sys
from pathlib import Path

try:
    import pyproj
except ImportError:
    pyproj = None

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
        <h1>🗺️ Spatial Analysis & Geographic Distribution</h1>
        <p>Explore the geographic distribution of stations and drought patterns across the Ebro Basin</p>
    </div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
with st.spinner("Loading spatial and event data..."):
    drought_df = load_drought_events()
    streamflow_df = load_streamflow_data()
    spatial_data = load_spatial_data()

# ==================== SIDEBAR OPTIONS ====================
st.sidebar.header("🗺️ Map Settings")

# Map type selector
map_type = st.sidebar.selectbox(
    "Select Map Layer",
    ["Stations Overview", "Drought Intensity", "Event Frequency"]
)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

# Year filter for drought data
if not drought_df.empty:
    year_min = int(drought_df['year'].min())
    year_max = int(drought_df['year'].max())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max)
    )

    # Filter drought data
    drought_df_filtered = drought_df[(drought_df['year'] >= year_range[0]) &
                                     (drought_df['year'] <= year_range[1])]
else:
    drought_df_filtered = drought_df

# ==================== MAIN CONTENT ====================

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🖼️ Static Overview Map",
    "📊 Spatial Statistics",
    "🌍 Basin Overview"
])

# ==================== TAB 1: STATIC OVERVIEW MAP ====================
with tab1:
    # Calculate station metrics from drought data
    if not drought_df_filtered.empty:
        station_metrics = drought_df_filtered.groupby('indroea').agg({
            'event_id': 'count',
            'duracion': 'mean',
            'deficit_hm3': 'sum'
        }).reset_index()
        station_metrics.columns = ['indroea', 'num_events', 'avg_duration', 'total_deficit']
    else:
        station_metrics = pd.DataFrame()

with tab1:
    st.markdown("### 🖼️ Static Basin Overview Map")

    st.info("📍 **Complete view** of the Ebro River Basin with all spatial elements (basin, rivers, stations, reservoirs)")

    # Create static matplotlib map
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Plot basin boundary
    if spatial_data.get('basin') is not None:
        basin_gdf = spatial_data['basin']
        basin_gdf.boundary.plot(ax=ax, color='blue', linewidth=2, label='Basin Boundary')
        basin_gdf.plot(ax=ax, color='lightblue', alpha=0.2)

    # Plot rivers
    if spatial_data.get('rivers') is not None:
        rivers_gdf = spatial_data['rivers']
        rivers_gdf.plot(ax=ax, color='#4682B4', linewidth=0.5, alpha=0.6, label='River Network')

    # Plot reservoirs
    if spatial_data.get('reservoirs') is not None:
        reservoirs_gdf = spatial_data['reservoirs']
        reservoirs_gdf.plot(ax=ax, color='darkblue', marker='s', markersize=50, alpha=0.7, label='Reservoirs')

    # Plot stations
    if spatial_data.get('stations') is not None:
        stations_gdf = spatial_data['stations'].copy()

        # Merge with metrics if available
        if not station_metrics.empty and 'indroea' in stations_gdf.columns:
            stations_gdf = stations_gdf.merge(station_metrics, on='indroea', how='left')
            stations_gdf['num_events'] = stations_gdf['num_events'].fillna(0)

            # Color by number of events
            stations_gdf.plot(
                ax=ax,
                column='num_events',
                cmap='Reds',
                markersize=40,
                alpha=0.8,
                legend=True,
                legend_kwds={'label': 'Number of Drought Events', 'shrink': 0.5}
            )
        else:
            stations_gdf.plot(ax=ax, color='red', markersize=30, alpha=0.7, label='Gauging Stations')

    ax.set_title('Ebro River Basin - Spatial Overview', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### 📊 Basin Elements Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        basin_count = len(spatial_data.get('basin', []))
        st.metric("Basin Polygons", basin_count if basin_count > 0 else "N/A")

    with col2:
        river_count = len(spatial_data.get('rivers', []))
        st.metric("River Segments", river_count if river_count > 0 else "N/A")

    with col3:
        station_count = len(spatial_data.get('stations', []))
        st.metric("Gauging Stations", station_count if station_count > 0 else "N/A")

    with col4:
        reservoir_count = len(spatial_data.get('reservoirs', []))
        st.metric("Reservoirs", reservoir_count if reservoir_count > 0 else "N/A")

# ==================== TAB 2: SPATIAL STATISTICS ====================
with tab2:
    st.markdown("### 📊 Spatial Distribution Statistics")

    if not drought_df_filtered.empty:
        # Calculate spatial statistics
        spatial_stats = drought_df_filtered.groupby('indroea').agg({
            'event_id': 'count',
            'duracion': ['mean', 'max', 'std'],
            'deficit_hm3': ['mean', 'max', 'sum']
        }).reset_index()

        spatial_stats.columns = ['indroea', 'num_events', 'avg_duration', 'max_duration',
                                'std_duration', 'avg_deficit', 'max_deficit', 'total_deficit']

        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Stations", len(spatial_stats))
        with col2:
            st.metric("Avg Events/Station", f"{spatial_stats['num_events'].mean():.1f}")
        with col3:
            st.metric("Max Events (1 Station)", int(spatial_stats['num_events'].max()))
        with col4:
            st.metric("Total Events", int(spatial_stats['num_events'].sum()))

        st.markdown("---")

        # Top stations by different metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏆 Top 10 Stations by Event Count")
            top_events = spatial_stats.nlargest(10, 'num_events')[['indroea', 'num_events']]
            fig = create_bar_chart(
                top_events,
                'indroea', 'num_events',
                title="Top 10 Stations by Number of Events",
                orientation='h'
            )
            st.plotly_chart(fig, use_container_width=True, key="spatial_top_events")

        with col2:
            st.markdown("#### 💧 Top 10 Stations by Total Deficit")
            top_deficit = spatial_stats.nlargest(10, 'total_deficit')[['indroea', 'total_deficit']]
            fig = create_bar_chart(
                top_deficit,
                'indroea', 'total_deficit',
                title="Top 10 Stations by Total Water Deficit",
                orientation='h'
            )
            st.plotly_chart(fig, use_container_width=True, key="spatial_top_deficit")

        # Detailed table
        st.markdown("---")
        st.markdown("#### 📋 Detailed Station Statistics")

        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ['num_events', 'avg_duration', 'max_duration', 'total_deficit'],
            format_func=lambda x: x.replace('_', ' ').title()
        )

        sorted_stats = spatial_stats.sort_values(sort_by, ascending=False)
        st.dataframe(sorted_stats, use_container_width=True, height=400)

        # Download button
        csv = sorted_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Station Statistics",
            data=csv,
            file_name="station_spatial_statistics.csv",
            mime="text/csv"
        )

    else:
        st.warning("No drought data available for spatial analysis")

# ==================== TAB 3: BASIN OVERVIEW ====================
with tab3:
    st.markdown("### 🌍 Ebro River Basin Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Basin Characteristics")

        st.markdown(f"""
        <div class="info-box">
            <h4>Geographic Information</h4>
            <p><strong>Basin Name:</strong> {BASIN_INFO['name']}</p>
            <p><strong>Main River:</strong> {BASIN_INFO['main_river']}</p>
            <p><strong>Countries:</strong> {', '.join(BASIN_INFO['countries'])}</p>
            <p><strong>Basin Area:</strong> {BASIN_INFO['area_km2']:,} km²</p>
            <p><strong>River Length:</strong> {BASIN_INFO['length_km']} km</p>
            <p><strong>Number of Gauging Stations:</strong> {BASIN_INFO['num_stations']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Elevation information if DEM is available
        if spatial_data.get('dem_data') is not None:
            dem_data = spatial_data['dem_data']
            st.markdown("#### 🏔️ Elevation Profile")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Min Elevation", f"{np.nanmin(dem_data):.0f} m")
            with col_b:
                st.metric("Max Elevation", f"{np.nanmax(dem_data):.0f} m")
            with col_c:
                st.metric("Mean Elevation", f"{np.nanmean(dem_data):.0f} m")

    with col2:
        st.markdown("#### 📊 Data Coverage")

        if not drought_df_filtered.empty:
            # Temporal coverage
            years_covered = drought_df_filtered['year'].nunique()
            st.metric("Years of Data", years_covered)

            # Spatial coverage
            stations_with_data = drought_df_filtered['indroea'].nunique()
            st.metric("Stations with Events", stations_with_data)

            # Total events
            total_events = len(drought_df_filtered)
            st.metric("Total Events Recorded", f"{total_events:,}")

    # River network information
    if spatial_data.get('rivers') is not None:
        st.markdown("---")
        st.markdown("#### 🌊 River Network")

        rivers_gdf = spatial_data['rivers']
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total River Segments", len(rivers_gdf))
        with col2:
            if rivers_gdf.crs:
                total_length = rivers_gdf.to_crs('EPSG:3857').length.sum() / 1000
                st.metric("Total Network Length", f"{total_length:.0f} km")
        with col3:
            st.metric("Data Source", "CHE Spatial Database")

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
    <p>🗺️ Spatial Analysis | Geographic patterns and distribution of drought events</p>
</div>
""", unsafe_allow_html=True)
