"""
Drought Analysis Page - Event Characteristics and Severity Metrics
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
        <h1>🌊 Drought Event Analysis</h1>
        <p>Comprehensive analysis of drought characteristics, duration, intensity, and severity metrics</p>
    </div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
with st.spinner("Loading drought events data..."):
    drought_df = load_drought_events()

# ==================== SIDEBAR OPTIONS ====================
st.sidebar.header("⚙️ Analysis Settings")

# Analysis type
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Duration Analysis", "Deficit Analysis", "Duration-Deficit Relationship", "Severity Classification"]
)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

# Filters
if not drought_df.empty:
    # Year range
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

    # Duration filter
    duration_min = st.sidebar.number_input(
        "Minimum Duration (days)",
        min_value=0,
        max_value=int(drought_df['duracion'].max()),
        value=0
    )

    # Apply filters
    drought_df = drought_df[(drought_df['year'] >= year_range[0]) &
                            (drought_df['year'] <= year_range[1])]

    if selected_stations:
        drought_df = filter_by_stations(drought_df, selected_stations)

    if duration_min > 0:
        drought_df = drought_df[drought_df['duracion'] >= duration_min]

# ==================== MAIN CONTENT ====================

tabs = st.tabs([
    "📊 Event Characteristics",
    "⏱️ Duration Analysis",
    "💧 Deficit Analysis",
    "🔗 Correlations",
    "📈 Severity Classification"
])

# ==================== TAB 1: EVENT CHARACTERISTICS ====================
with tabs[0]:
    st.markdown("### 📊 Drought Event Characteristics Overview")

    if not drought_df.empty:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Events", f"{len(drought_df):,}")
        with col2:
            st.metric("Avg Duration", f"{drought_df['duracion'].mean():.1f} days")
        with col3:
            st.metric("Avg Deficit", f"{drought_df['deficit_hm3'].mean():.2f} hm³")
        with col4:
            st.metric("Stations Affected", drought_df['indroea'].nunique())

        st.markdown("---")

        # Detailed statistics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Duration Statistics")

            duration_stats = {
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3', 'IQR'],
                'Value (days)': [
                    f"{drought_df['duracion'].mean():.2f}",
                    f"{drought_df['duracion'].median():.2f}",
                    f"{drought_df['duracion'].std():.2f}",
                    f"{drought_df['duracion'].min():.0f}",
                    f"{drought_df['duracion'].max():.0f}",
                    f"{drought_df['duracion'].quantile(0.25):.2f}",
                    f"{drought_df['duracion'].quantile(0.75):.2f}",
                    f"{drought_df['duracion'].quantile(0.75) - drought_df['duracion'].quantile(0.25):.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(duration_stats), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### 💧 Deficit Statistics")

            deficit_stats = {
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3', 'Total'],
                'Value (hm³)': [
                    f"{drought_df['deficit_hm3'].mean():.2f}",
                    f"{drought_df['deficit_hm3'].median():.2f}",
                    f"{drought_df['deficit_hm3'].std():.2f}",
                    f"{drought_df['deficit_hm3'].min():.2f}",
                    f"{drought_df['deficit_hm3'].max():.2f}",
                    f"{drought_df['deficit_hm3'].quantile(0.25):.2f}",
                    f"{drought_df['deficit_hm3'].quantile(0.75):.2f}",
                    f"{drought_df['deficit_hm3'].sum():.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(deficit_stats), use_container_width=True, hide_index=True)

        st.markdown("---")

        # Distribution plots
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Duration Distribution")
            fig = create_histogram(drought_df, 'duracion',
                                 title="Drought Duration Distribution", bins=50)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Deficit Distribution")
            fig = create_histogram(drought_df, 'deficit_hm3',
                                 title="Water Deficit Distribution", bins=50)
            st.plotly_chart(fig, use_container_width=True)

        # Top events
        st.markdown("---")
        st.markdown("### 🏆 Top Drought Events")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top 10 Longest Droughts")
            top_duration = drought_df.nlargest(10, 'duracion')[
                ['indroea', 'inicio', 'fin', 'duracion', 'deficit_hm3']
            ]
            st.dataframe(top_duration, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Top 10 Highest Deficits")
            top_deficit = drought_df.nlargest(10, 'deficit_hm3')[
                ['indroea', 'inicio', 'fin', 'duracion', 'deficit_hm3']
            ]
            st.dataframe(top_deficit, use_container_width=True, hide_index=True)

    else:
        st.warning("No drought data available with current filters")

# ==================== TAB 2: DURATION ANALYSIS ====================
with tabs[1]:
    st.markdown("### ⏱️ Drought Duration Analysis")

    if not drought_df.empty:
        # Duration categories
        st.markdown("#### 📊 Duration Categories")

        # Define categories
        drought_df['duration_category'] = pd.cut(
            drought_df['duracion'],
            bins=[0, 10, 30, 60, 180, float('inf')],
            labels=['Very Short (0-10d)', 'Short (10-30d)', 'Medium (30-60d)',
                   'Long (60-180d)', 'Very Long (>180d)']
        )

        category_counts = drought_df['duration_category'].value_counts().sort_index()

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = create_bar_chart(
                pd.DataFrame({'category': category_counts.index, 'count': category_counts.values}),
                'category', 'count',
                title="Events by Duration Category"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Category Percentages")
            for cat, count in category_counts.items():
                pct = (count / len(drought_df)) * 100
                st.metric(str(cat), f"{pct:.1f}%")

        st.markdown("---")

        # Duration over time
        st.markdown("#### 📈 Duration Trends Over Time")

        annual_duration = drought_df.groupby('year').agg({
            'duracion': ['mean', 'max', 'median']
        }).reset_index()
        annual_duration.columns = ['year', 'mean_duration', 'max_duration', 'median_duration']

        fig = create_time_series_plot(
            annual_duration.melt(id_vars='year', var_name='metric', value_name='duration'),
            'year', 'duration',
            title="Annual Duration Metrics Over Time",
            color='metric'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Duration by season
        if 'season' in drought_df.columns:
            st.markdown("---")
            st.markdown("#### 📅 Duration by Season")

            fig = create_box_plot(drought_df, 'season', 'duracion',
                                title="Duration Distribution by Season")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No data available for duration analysis")

# ==================== TAB 3: DEFICIT ANALYSIS ====================
with tabs[2]:
    st.markdown("### 💧 Water Deficit Analysis")

    if not drought_df.empty:
        # Deficit categories
        st.markdown("#### 📊 Deficit Severity Categories")

        # Define categories
        drought_df['deficit_category'] = pd.cut(
            drought_df['deficit_hm3'],
            bins=[0, 1, 5, 20, 100, float('inf')],
            labels=['Minor (<1 hm³)', 'Moderate (1-5 hm³)', 'Severe (5-20 hm³)',
                   'Very Severe (20-100 hm³)', 'Extreme (>100 hm³)']
        )

        category_counts = drought_df['deficit_category'].value_counts().sort_index()

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = create_bar_chart(
                pd.DataFrame({'category': category_counts.index, 'count': category_counts.values}),
                'category', 'count',
                title="Events by Deficit Severity"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Category Percentages")
            for cat, count in category_counts.items():
                pct = (count / len(drought_df)) * 100
                st.metric(str(cat), f"{pct:.1f}%")

        st.markdown("---")

        # Deficit over time
        st.markdown("#### 📈 Deficit Trends Over Time")

        annual_deficit = drought_df.groupby('year').agg({
            'deficit_hm3': ['mean', 'sum', 'max']
        }).reset_index()
        annual_deficit.columns = ['year', 'mean_deficit', 'total_deficit', 'max_deficit']

        fig = create_time_series_plot(
            annual_deficit.melt(id_vars='year', var_name='metric', value_name='deficit'),
            'year', 'deficit',
            title="Annual Deficit Metrics Over Time",
            color='metric'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative deficit
        st.markdown("---")
        st.markdown("#### 📊 Cumulative Water Deficit")

        annual_cumulative = drought_df.groupby('year')['deficit_hm3'].sum().cumsum().reset_index()
        annual_cumulative.columns = ['year', 'cumulative_deficit']

        fig = create_time_series_plot(
            annual_cumulative, 'year', 'cumulative_deficit',
            title="Cumulative Water Deficit Over Time",
            x_label="Year", y_label="Cumulative Deficit (hm³)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"**Total cumulative water deficit:** {drought_df['deficit_hm3'].sum():.2f} hm³")

    else:
        st.warning("No data available for deficit analysis")

# ==================== TAB 4: CORRELATIONS ====================
with tabs[3]:
    st.markdown("### 🔗 Duration-Deficit Relationships")

    if not drought_df.empty:
        # Scatter plot
        st.markdown("#### 📊 Duration vs Deficit Scatter Plot")

        fig = create_scatter_plot(
            drought_df, 'duracion', 'deficit_hm3',
            title="Relationship between Duration and Water Deficit",
            trendline=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlation analysis
        st.markdown("---")
        st.markdown("#### 📈 Correlation Analysis")

        col1, col2, col3 = st.columns(3)

        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(drought_df['duracion'], drought_df['deficit_hm3'])
        with col1:
            st.metric("Pearson Correlation", f"{pearson_corr:.3f}")
            st.caption(f"p-value: {pearson_p:.4f}")

        # Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(drought_df['duracion'], drought_df['deficit_hm3'])
        with col2:
            st.metric("Spearman Correlation", f"{spearman_corr:.3f}")
            st.caption(f"p-value: {spearman_p:.4f}")

        # R-squared
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            drought_df['duracion'], drought_df['deficit_hm3']
        )
        with col3:
            st.metric("R² (Linear Fit)", f"{r_value**2:.3f}")
            st.caption(f"Slope: {slope:.4f}")

        # Interpretation
        st.markdown("---")
        st.markdown("#### 💡 Interpretation")

        if pearson_corr > 0.7:
            strength = "strong positive"
        elif pearson_corr > 0.4:
            strength = "moderate positive"
        elif pearson_corr > 0.2:
            strength = "weak positive"
        else:
            strength = "very weak or no"

        st.info(f"""
        The analysis shows a **{strength} correlation** between drought duration and water deficit.
        This suggests that {'longer droughts tend to have higher water deficits' if pearson_corr > 0.4 else 'the relationship is complex and may depend on other factors'}.
        """)

        # Log-scale analysis
        st.markdown("---")
        st.markdown("#### 📊 Log-Scale Analysis")

        drought_df_log = drought_df.copy()
        drought_df_log['log_duracion'] = np.log10(drought_df_log['duracion'] + 1)
        drought_df_log['log_deficit'] = np.log10(drought_df_log['deficit_hm3'] + 1)

        fig = create_scatter_plot(
            drought_df_log, 'log_duracion', 'log_deficit',
            title="Duration vs Deficit (Log Scale)",
            trendline=True
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No data available for correlation analysis")

# ==================== TAB 5: SEVERITY CLASSIFICATION ====================
with tabs[4]:
    st.markdown("### 📈 Drought Severity Classification")

    if not drought_df.empty:
        st.markdown("""
        Classify drought events based on multiple characteristics to create a composite severity index.
        """)

        # Create severity score
        drought_df['duration_percentile'] = drought_df['duracion'].rank(pct=True)
        drought_df['deficit_percentile'] = drought_df['deficit_hm3'].rank(pct=True)
        drought_df['severity_score'] = (drought_df['duration_percentile'] + drought_df['deficit_percentile']) / 2

        # Classify severity
        drought_df['severity_class'] = pd.cut(
            drought_df['severity_score'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low', 'Moderate', 'High', 'Extreme']
        )

        # Display classification results
        severity_counts = drought_df['severity_class'].value_counts()

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = create_bar_chart(
                pd.DataFrame({'severity': severity_counts.index, 'count': severity_counts.values}),
                'severity', 'count',
                title="Events by Severity Classification"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Classification Distribution")
            for sev, count in severity_counts.items():
                pct = (count / len(drought_df)) * 100
                st.metric(str(sev), f"{pct:.1f}%")

        st.markdown("---")

        # Severity over time
        st.markdown("#### 📈 Severity Trends Over Time")

        annual_severity = drought_df.groupby(['year', 'severity_class']).size().unstack(fill_value=0)
        annual_severity_pct = annual_severity.div(annual_severity.sum(axis=1), axis=0) * 100

        fig = create_time_series_plot(
            annual_severity_pct.reset_index().melt(id_vars='year', var_name='severity', value_name='percentage'),
            'year', 'percentage',
            title="Severity Distribution Over Time (%)",
            color='severity'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top extreme events
        st.markdown("---")
        st.markdown("#### 🚨 Top 20 Most Severe Events")

        top_severe = drought_df.nlargest(20, 'severity_score')[
            ['indroea', 'inicio', 'fin', 'duracion', 'deficit_hm3', 'severity_score', 'severity_class']
        ]
        st.dataframe(top_severe, use_container_width=True, hide_index=True)

        # Download severity data
        csv = drought_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Events with Severity Classification",
            data=csv,
            file_name="drought_events_with_severity.csv",
            mime="text/csv"
        )

    else:
        st.warning("No data available for severity classification")

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
    <p>🌊 Drought Analysis | Comprehensive characterization of drought patterns</p>
</div>
""", unsafe_allow_html=True)
