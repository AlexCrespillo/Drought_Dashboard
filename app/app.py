"""
Drought Events Analysis Dashboard - Main Application
Ebro River Basin - Comprehensive Hydrological Assessment
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from config import *

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(**PAGE_CONFIG)

# ==================== CUSTOM CSS ====================
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==================== MAIN APPLICATION ====================

def main():
    """Main application function"""

    # Header
    st.markdown(f"""
        <div class="main-header">
            <h1>{APP_ICON} {APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                Temporal Coverage: {START_YEAR}-{END_YEAR} |
                Interactive EDA Dashboard
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Basin Image
    try:
        image_path = BASE_DIR / "data" / "png" / "Ebro.jpg"
        if image_path.exists():
            st.image(
                str(image_path),
                caption="Ebro River Basin - Study Area Overview",
                use_container_width=True
            )
        else:
            st.info("📸 Basin overview image not found")
    except Exception as e:
        st.warning(f"Could not load basin image: {str(e)}")

    st.markdown("---")

    # Welcome message
    st.markdown("""
    ## Welcome to the Drought Analysis Dashboard

    This interactive application provides comprehensive analysis of drought events and streamflow patterns
    in the Ebro River Basin. The dashboard is organized into multiple analytical modules:
    """)

    # Navigation guide
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📊 Available Analyses

        - **📊 Data Overview**: Dataset exploration and quality metrics
        - **🗺️ Spatial Analysis**: Geographic distribution and basin characteristics
        - **📈 Temporal Analysis**: Long-term trends and seasonal patterns
        """)

    with col2:
        st.markdown("""
        ### 🔬 Advanced Features

        - **🌊 Drought Analysis**: Event characteristics and severity metrics
        - **🔬 Station Clustering**: Hydrological regime classification
        - **🚀 Coming Soon**: ML forecasting and network analysis
        """)

    st.markdown("---")

    # Quick stats section
    st.markdown("### 📊 Dataset Quick Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Temporal Coverage</div>
                <div class="metric-value">{END_YEAR - START_YEAR}</div>
                <div class="metric-label">years</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Gauging Stations</div>
                <div class="metric-value">{BASIN_INFO['num_stations']}</div>
                <div class="metric-label">stations</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Basin Area</div>
                <div class="metric-value">{BASIN_INFO['area_km2']:,}</div>
                <div class="metric-label">km²</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Main River Length</div>
                <div class="metric-value">{BASIN_INFO['length_km']}</div>
                <div class="metric-label">km</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Study area information
    st.markdown("### 🗺️ Study Area: Ebro River Basin")

    st.markdown(f"""
    <div class="info-box">
        <h4 style="margin-top: 0;">Basin Characteristics</h4>
        <p><strong>Name:</strong> {BASIN_INFO['name']}</p>
        <p><strong>Main River:</strong> {BASIN_INFO['main_river']}</p>
        <p><strong>Countries:</strong> {', '.join(BASIN_INFO['countries'])}</p>
        <p><strong>Basin Area:</strong> {BASIN_INFO['area_km2']:,} km²</p>
        <p><strong>River Length:</strong> {BASIN_INFO['length_km']} km</p>
    </div>
    """, unsafe_allow_html=True)

    # Methodology
    with st.expander("🔍 Drought Detection Methodology", expanded=False):
        st.markdown(f"""
        #### Threshold-Based Detection System

        Drought events have been identified using an **advanced climatological threshold-based detection system**
        that analyzes daily streamflow time series:

        **Parameters:**
        - **Threshold Percentile:** P{DROUGHT_PERCENTILE_THRESHOLD} (5th percentile)
        - **Minimum Duration:** {MIN_DROUGHT_DURATION} days
        - **Event Pooling Window:** {POOLING_WINDOW} days

        **Process:**
        1. Calculate climatological percentiles for each day of year
        2. Identify periods below P{DROUGHT_PERCENTILE_THRESHOLD} threshold
        3. Apply intelligent event pooling for gaps ≤ {POOLING_WINDOW} days
        4. Filter events with duration ≥ {MIN_DROUGHT_DURATION} days
        5. Calculate volumetric water deficit (hm³)

        **References:**
        """)
        for ref in REFERENCES:
            st.markdown(f"- {ref}")

    # Navigation instructions
    st.markdown("---")
    st.info("👈 **Use the sidebar** to navigate between different analysis pages and explore the dashboard features.")

    # Author Information
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
        except Exception as e:
            st.warning(f"Logo not found: {str(e)}")

    # Footer
    st.markdown(f"""
        <div class="footer">
            <p><strong>Drought Events Analysis Dashboard</strong> | Version 1.0</p>
            <p>Data Period: {START_YEAR}-{END_YEAR} | Last Updated: {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
