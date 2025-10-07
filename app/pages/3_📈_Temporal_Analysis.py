"""
Temporal Analysis Page - Long-term Trends and Seasonal Patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
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
        <h1>📈 Temporal Analysis & Trends</h1>
        <p>Explore long-term trends, decadal variability, and seasonal patterns in drought events</p>
    </div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
with st.spinner("Loading drought events data..."):
    drought_df = load_drought_events()

# ==================== SIDEBAR OPTIONS ====================
st.sidebar.header("⚙️ Analysis Settings")

# Temporal aggregation
temporal_agg = st.sidebar.selectbox(
    "Temporal Aggregation",
    ["Annual", "Seasonal", "Monthly", "Decadal"]
)

# Variable to analyze
variable_choice = st.sidebar.selectbox(
    "Variable to Analyze",
    ["Event Count", "Duration", "Deficit", "All Variables"]
)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

# Station filter
if not drought_df.empty:
    stations = get_station_list(drought_df)
    selected_stations = st.sidebar.multiselect(
        "Select Stations (leave empty for all)",
        options=stations,
        default=[]
    )

    if selected_stations:
        drought_df = filter_by_stations(drought_df, selected_stations)

    # Year range
    year_min = int(drought_df['year'].min())
    year_max = int(drought_df['year'].max())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max)
    )

    drought_df = drought_df[(drought_df['year'] >= year_range[0]) &
                            (drought_df['year'] <= year_range[1])]

# ==================== MAIN CONTENT ====================

tabs = st.tabs([
    "📊 Annual Trends",
    "📅 Seasonal Patterns",
    "📆 Monthly Analysis",
    "🔢 Decadal Comparison",
    "📈 Statistical Tests"
])

# ==================== TAB 1: ANNUAL TRENDS ====================
with tabs[0]:
    st.markdown("### 📊 Annual Drought Trends")

    if not drought_df.empty:
        # Annual aggregation
        annual_data = drought_df.groupby('year').agg({
            'event_id': 'count',
            'duracion': ['mean', 'max', 'sum'],
            'deficit_hm3': ['mean', 'max', 'sum']
        }).reset_index()

        annual_data.columns = ['year', 'num_events', 'avg_duration', 'max_duration',
                              'total_duration', 'avg_deficit', 'max_deficit', 'total_deficit']

        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_events_per_year = annual_data['num_events'].mean()
            st.metric("Avg Events/Year", f"{avg_events_per_year:.1f}")

        with col2:
            max_events_year = annual_data.loc[annual_data['num_events'].idxmax()]
            st.metric("Max Events in a Year", f"{int(max_events_year['num_events'])} ({int(max_events_year['year'])})")

        with col3:
            trend_events = stats.linregress(annual_data['year'], annual_data['num_events'])
            st.metric("Event Trend", f"{trend_events.slope:+.2f} events/year",
                     delta=f"p-value: {trend_events.pvalue:.3f}")

        with col4:
            total_years = len(annual_data)
            st.metric("Years Analyzed", total_years)

        st.markdown("---")

        # Time series plots
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Number of Events per Year")
            fig = create_trend_plot(annual_data, 'year', 'num_events',
                                   title="Annual Drought Event Count with Trend")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Average Duration per Year")
            fig = create_trend_plot(annual_data, 'year', 'avg_duration',
                                   title="Annual Average Duration with Trend")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Deficit trends
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Total Water Deficit per Year")
            fig = create_trend_plot(annual_data, 'year', 'total_deficit',
                                   title="Annual Total Water Deficit with Trend")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Maximum Duration per Year")
            fig = create_trend_plot(annual_data, 'year', 'max_duration',
                                   title="Annual Maximum Duration with Trend")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Detailed annual data table
        st.markdown("---")
        st.markdown("#### 📋 Annual Statistics Table")

        display_data = annual_data.copy()
        display_data = display_data.sort_values('year', ascending=False)

        st.dataframe(display_data, use_container_width=True, height=400)

        # Download button
        csv = display_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Annual Data",
            data=csv,
            file_name="annual_drought_statistics.csv",
            mime="text/csv"
        )

    else:
        st.warning("No drought data available for analysis")

# ==================== TAB 2: SEASONAL PATTERNS ====================
with tabs[1]:
    st.markdown("### 📅 Seasonal Drought Patterns")

    if not drought_df.empty and 'season' in drought_df.columns:
        # Seasonal aggregation
        seasonal_data = drought_df.groupby('season').agg({
            'event_id': 'count',
            'duracion': ['mean', 'max', 'std'],
            'deficit_hm3': ['mean', 'max', 'sum']
        }).reset_index()

        seasonal_data.columns = ['season', 'num_events', 'avg_duration', 'max_duration',
                                'std_duration', 'avg_deficit', 'max_deficit', 'total_deficit']

        # Reorder seasons
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_data['season'] = pd.Categorical(seasonal_data['season'],
                                                 categories=season_order, ordered=True)
        seasonal_data = seasonal_data.sort_values('season')

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        for col, season in zip([col1, col2, col3, col4], season_order):
            if season in seasonal_data['season'].values:
                season_row = seasonal_data[seasonal_data['season'] == season].iloc[0]
                with col:
                    st.markdown(f"#### {season}")
                    st.metric("Events", int(season_row['num_events']))
                    st.metric("Avg Duration", f"{season_row['avg_duration']:.1f} days")

        st.markdown("---")

        # Seasonal visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Events by Season")
            fig = create_bar_chart(seasonal_data, 'season', 'num_events',
                                 title="Number of Events by Season")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Average Duration by Season")
            fig = create_bar_chart(seasonal_data, 'season', 'avg_duration',
                                 title="Average Duration by Season")
            st.plotly_chart(fig, use_container_width=True)

        # Box plots for distributions
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Duration Distribution by Season")
            fig = create_seasonal_plot(drought_df, 'duracion',
                                      title="Drought Duration by Season")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Deficit Distribution by Season")
            fig = create_seasonal_plot(drought_df, 'deficit_hm3',
                                      title="Water Deficit by Season")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Statistical table
        st.markdown("---")
        st.markdown("#### 📊 Seasonal Statistics Summary")
        st.dataframe(seasonal_data, use_container_width=True, hide_index=True)

    else:
        st.warning("Season information not available in the dataset")

# ==================== TAB 3: MONTHLY ANALYSIS ====================
with tabs[2]:
    st.markdown("### 📆 Monthly Drought Analysis")

    if not drought_df.empty and 'month' in drought_df.columns:
        # Monthly aggregation
        monthly_data = drought_df.groupby('month').agg({
            'event_id': 'count',
            'duracion': 'mean',
            'deficit_hm3': 'mean'
        }).reset_index()

        monthly_data.columns = ['month', 'num_events', 'avg_duration', 'avg_deficit']

        # Add month names
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        monthly_data['month_name'] = monthly_data['month'].map(month_names)

        # Monthly pattern visualization
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Events by Month")
            fig = create_bar_chart(monthly_data, 'month_name', 'num_events',
                                 title="Number of Events by Month")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Average Duration by Month")
            fig = create_bar_chart(monthly_data, 'month_name', 'avg_duration',
                                 title="Average Duration by Month")
            st.plotly_chart(fig, use_container_width=True)

        # Year-month heatmap
        st.markdown("---")
        st.markdown("#### 🔥 Year-Month Heatmap")

        heatmap_metric = st.selectbox(
            "Select metric for heatmap",
            ["Number of Events", "Average Duration", "Total Deficit"]
        )

        if heatmap_metric == "Number of Events":
            fig = create_monthly_heatmap(drought_df, 'event_id',
                                        title="Number of Events by Year and Month")
        elif heatmap_metric == "Average Duration":
            fig = create_monthly_heatmap(drought_df, 'duracion',
                                        title="Average Duration by Year and Month")
        else:
            fig = create_monthly_heatmap(drought_df, 'deficit_hm3',
                                        title="Average Deficit by Year and Month")

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Month information not available in the dataset")

# ==================== TAB 4: DECADAL COMPARISON ====================
with tabs[3]:
    st.markdown("### 🔢 Decadal Drought Comparison")

    if not drought_df.empty:
        # Create decade column
        drought_df['decade'] = (drought_df['year'] // 10) * 10

        # Decadal aggregation
        decadal_data = drought_df.groupby('decade').agg({
            'event_id': 'count',
            'duracion': ['mean', 'max'],
            'deficit_hm3': ['mean', 'sum']
        }).reset_index()

        decadal_data.columns = ['decade', 'num_events', 'avg_duration',
                               'max_duration', 'avg_deficit', 'total_deficit']

        # Format decade labels
        decadal_data['decade_label'] = decadal_data['decade'].astype(str) + 's'

        # Display metrics
        st.markdown("#### 📊 Events per Decade")

        cols = st.columns(len(decadal_data))
        for col, (idx, row) in zip(cols, decadal_data.iterrows()):
            with col:
                st.metric(row['decade_label'], int(row['num_events']))

        st.markdown("---")

        # Comparative visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Events by Decade")
            fig = create_bar_chart(decadal_data, 'decade_label', 'num_events',
                                 title="Number of Events by Decade")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Average Duration by Decade")
            fig = create_bar_chart(decadal_data, 'decade_label', 'avg_duration',
                                 title="Average Duration by Decade")
            st.plotly_chart(fig, use_container_width=True)

        # Box plot comparison
        st.markdown("---")
        st.markdown("#### 📊 Distribution Comparison Across Decades")

        col1, col2 = st.columns(2)

        with col1:
            fig = create_box_plot(drought_df, 'decade', 'duracion',
                                title="Duration Distribution by Decade")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = create_box_plot(drought_df, 'decade', 'deficit_hm3',
                                title="Deficit Distribution by Decade")
            st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.markdown("---")
        st.markdown("#### 📋 Decadal Statistics Summary")
        st.dataframe(decadal_data, use_container_width=True, hide_index=True)

    else:
        st.warning("No drought data available for decadal analysis")

# ==================== TAB 5: STATISTICAL TESTS ====================
with tabs[4]:
    st.markdown("### 📈 Statistical Trend Tests")

    if not drought_df.empty:
        st.markdown("""
        Statistical tests to assess the significance of observed trends in drought characteristics.
        """)

        # Annual data for trend tests
        annual_data = drought_df.groupby('year').agg({
            'event_id': 'count',
            'duracion': 'mean',
            'deficit_hm3': 'sum'
        }).reset_index()

        annual_data.columns = ['year', 'num_events', 'avg_duration', 'total_deficit']

        # Perform trend tests
        st.markdown("#### 📊 Linear Regression Trend Analysis")

        results_data = []

        for variable in ['num_events', 'avg_duration', 'total_deficit']:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                annual_data['year'], annual_data[variable]
            )

            results_data.append({
                'Variable': variable.replace('_', ' ').title(),
                'Slope': f"{slope:.4f}",
                'R²': f"{r_value**2:.4f}",
                'P-value': f"{p_value:.4f}",
                'Significance': '✓ Significant' if p_value < 0.05 else '✗ Not Significant'
            })

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Mann-Kendall test (simple version)
        st.markdown("#### 📈 Mann-Kendall Trend Test")

        st.info("""
        The Mann-Kendall test is a non-parametric test for detecting monotonic trends.
        A positive tau indicates an upward trend, while a negative tau indicates a downward trend.
        """)

        # Simple Mann-Kendall implementation
        def mann_kendall_test(data):
            n = len(data)
            s = 0
            for i in range(n-1):
                for j in range(i+1, n):
                    s += np.sign(data[j] - data[i])

            # Calculate variance
            var_s = n * (n - 1) * (2 * n + 5) / 18

            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0

            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            tau = s / (n * (n - 1) / 2)

            return tau, p_value

        mk_results = []
        for variable in ['num_events', 'avg_duration', 'total_deficit']:
            tau, p_value = mann_kendall_test(annual_data[variable].values)
            mk_results.append({
                'Variable': variable.replace('_', ' ').title(),
                'Tau': f"{tau:.4f}",
                'P-value': f"{p_value:.4f}",
                'Trend': 'Upward' if tau > 0 else 'Downward',
                'Significance': '✓ Significant' if p_value < 0.05 else '✗ Not Significant'
            })

        mk_df = pd.DataFrame(mk_results)
        st.dataframe(mk_df, use_container_width=True, hide_index=True)

        # Interpretation
        st.markdown("---")
        st.markdown("#### 💡 Interpretation Guidelines")
        st.markdown("""
        - **P-value < 0.05**: Statistically significant trend at 95% confidence level
        - **P-value < 0.01**: Highly significant trend at 99% confidence level
        - **R² > 0.5**: Strong linear relationship
        - **Tau**: Measure of correlation strength (-1 to +1)
        """)

    else:
        st.warning("No data available for statistical tests")

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
    <p>📈 Temporal Analysis | Trends and patterns in drought events over time</p>
</div>
""", unsafe_allow_html=True)
