"""
Drought Events Visualization Page - Interactive Bubble Chart Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
st.markdown("""
    <div class="main-header">
        <h1>🌊 Drought Events Visualization</h1>
        <p>Interactive bubble chart analysis of drought event characteristics and severity</p>
    </div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
with st.spinner("Loading drought events and streamflow data..."):
    drought_df = load_drought_events()
    streamflow_df = load_streamflow_data()

if drought_df.empty:
    st.error("No drought events data available")
    st.stop()

# ==================== SIDEBAR: CONFIGURATION ====================
st.sidebar.header("🔍 Station Selection")

# Get list of available stations
available_stations = sorted(drought_df['indroea'].unique())

# Multi-select widget for stations
selected_stations = st.sidebar.multiselect(
    "Select stations to visualize",
    options=available_stations,
    default=available_stations[:5] if len(available_stations) >= 5 else available_stations,
    help="Choose one or more stations to display drought events"
)

if not selected_stations:
    st.warning("⚠️ Please select at least one station to visualize drought events")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Visualization Settings")

# Year filter
if 'year' in drought_df.columns:
    year_min = int(drought_df['year'].min())
    year_max = int(drought_df['year'].max())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max)
    )
else:
    year_range = None

# Bubble size scale
bubble_scale = st.sidebar.slider(
    "Bubble Size Scale",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
    help="Adjust the size of bubbles for better visibility"
)

# Color scheme
color_scheme = st.sidebar.selectbox(
    "Color Scheme",
    options=["Viridis", "Plasma", "Inferno", "Turbo", "RdYlGn_r", "RdBu_r"],
    index=0,
    help="Color scale for drought intensity"
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Intensity Metric")

st.sidebar.info("""
**Drought Intensity** is calculated as a composite score (0-100) combining:
- **Duration** (30%): Longer events = higher intensity
- **Deficit** (40%): Larger water deficit = higher intensity
- **Relative Impact** (30%): Deficit relative to station's mean flow
""")

# Allow custom weights
with st.sidebar.expander("⚙️ Customize Intensity Weights", expanded=False):
    weight_duration = st.slider("Duration Weight", 0.0, 1.0, 0.3, 0.1)
    weight_deficit = st.slider("Deficit Weight", 0.0, 1.0, 0.4, 0.1)
    weight_relative = st.slider("Relative Impact Weight", 0.0, 1.0, 0.3, 0.1)

    # Normalize weights to sum to 1
    total_weight = weight_duration + weight_deficit + weight_relative
    if total_weight > 0:
        weight_duration = weight_duration / total_weight
        weight_deficit = weight_deficit / total_weight
        weight_relative = weight_relative / total_weight

# ==================== PREPARE DATA ====================
# Filter drought events by selected stations
filtered_drought = drought_df[drought_df['indroea'].isin(selected_stations)].copy()

# Filter by year range
if year_range:
    filtered_drought = filtered_drought[
        (filtered_drought['year'] >= year_range[0]) &
        (filtered_drought['year'] <= year_range[1])
    ]

if filtered_drought.empty:
    st.warning("No drought events found for the selected stations and time period")
    st.stop()

# Calculate mean flow for each station (for relative impact)
station_mean_flow = {}
for station in selected_stations:
    station_data = streamflow_df[streamflow_df['indroea'] == station]
    if not station_data.empty and 'caudal' in station_data.columns:
        station_mean_flow[station] = station_data['caudal'].mean()
    else:
        station_mean_flow[station] = 1.0  # Default if no data

# Add mean flow to drought data
filtered_drought['mean_flow'] = filtered_drought['indroea'].map(station_mean_flow)

# Calculate event midpoint for x-axis
if 'fecha_inicio' in filtered_drought.columns and 'duracion' in filtered_drought.columns:
    filtered_drought['fecha_inicio'] = pd.to_datetime(filtered_drought['fecha_inicio'])
    filtered_drought['event_midpoint'] = filtered_drought['fecha_inicio'] + pd.to_timedelta(filtered_drought['duracion'] / 2, unit='D')
else:
    # Use year as proxy if no exact dates
    filtered_drought['event_midpoint'] = pd.to_datetime(filtered_drought['year'].astype(str) + '-06-15')

# Calculate drought intensity metric
# Normalize each component to 0-1 scale
duration_normalized = (filtered_drought['duracion'] - filtered_drought['duracion'].min()) / (filtered_drought['duracion'].max() - filtered_drought['duracion'].min() + 1e-10)
deficit_normalized = (filtered_drought['deficit_hm3'] - filtered_drought['deficit_hm3'].min()) / (filtered_drought['deficit_hm3'].max() - filtered_drought['deficit_hm3'].min() + 1e-10)

# Relative deficit (deficit per unit of mean flow)
filtered_drought['relative_deficit'] = filtered_drought['deficit_hm3'] / (filtered_drought['mean_flow'] + 1e-10)
relative_normalized = (filtered_drought['relative_deficit'] - filtered_drought['relative_deficit'].min()) / (filtered_drought['relative_deficit'].max() - filtered_drought['relative_deficit'].min() + 1e-10)

# Composite intensity score (0-100)
filtered_drought['intensity'] = (
    weight_duration * duration_normalized +
    weight_deficit * deficit_normalized +
    weight_relative * relative_normalized
) * 100

# Get station names
station_names = {}
for station in selected_stations:
    station_data = streamflow_df[streamflow_df['indroea'] == station]
    if not station_data.empty and 'lugar' in station_data.columns:
        station_names[station] = station_data['lugar'].iloc[0]
    else:
        station_names[station] = f"Station {station}"

filtered_drought['station_name'] = filtered_drought['indroea'].map(station_names)

# ==================== MAIN CONTENT ====================

# Display summary metrics
st.markdown(f"### Analyzing {len(filtered_drought)} Drought Events from {len(selected_stations)} Station(s)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Events", len(filtered_drought))

with col2:
    st.metric("Avg Duration", f"{filtered_drought['duracion'].mean():.1f} days")

with col3:
    st.metric("Total Deficit", f"{filtered_drought['deficit_hm3'].sum():.2f} hm³")

with col4:
    st.metric("Max Intensity", f"{filtered_drought['intensity'].max():.1f}")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 3D Visualization",
    "📊 Intensity Distribution",
    "📋 Event Details"
])

# ==================== TAB 1: 3D SCATTER PLOT ====================
with tab1:
    st.markdown("### 🎯 Interactive 3D Drought Events Visualization")

    st.info("""
    💡 **3D Chart Guide:**
    - **X-axis:** Time (event date)
    - **Y-axis:** Water deficit (hm³)
    - **Z-axis:** Drought intensity score (0-100)
    - **Marker size:** Event duration (days)
    - **Marker color:** Each station has a unique color
    - **Interaction:** Rotate, zoom, and pan the 3D view | Hover for details
    """)

    # Create 3D scatter plot
    fig = go.Figure()

    # Define color palette for stations
    station_colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold

    # Add trace for each station
    for idx, station in enumerate(selected_stations):
        station_events = filtered_drought[filtered_drought['indroea'] == station]

        if not station_events.empty:
            # Create hover text
            hover_text = []
            for _, row in station_events.iterrows():
                text = (
                    f"<b>Station:</b> {row['station_name']} ({row['indroea']})<br>"
                    f"<b>Date:</b> {row['event_midpoint'].strftime('%Y-%m-%d')}<br>"
                    f"<b>Duration:</b> {row['duracion']:.0f} days<br>"
                    f"<b>Deficit:</b> {row['deficit_hm3']:.2f} hm³<br>"
                    f"<b>Intensity:</b> {row['intensity']:.1f}<br>"
                    f"<b>Mean Flow:</b> {row['mean_flow']:.2f} m³/s"
                )
                hover_text.append(text)

            # Calculate scaled marker sizes
            base_sizes = np.sqrt(station_events['duracion']) * bubble_scale * 1.5
            capped_sizes = np.minimum(base_sizes, 30)

            # Assign a unique color for each station
            station_color = station_colors[idx % len(station_colors)]

            fig.add_trace(go.Scatter3d(
                x=station_events['event_midpoint'],
                y=station_events['deficit_hm3'],
                z=station_events['intensity'],
                mode='markers',
                name=f"{station} - {station_names[station]}",
                marker=dict(
                    size=capped_sizes,
                    color=station_color,
                    line=dict(width=0.5, color='white'),
                    opacity=0.8
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))

    fig.update_layout(
        title={
            'text': "3D Drought Events: Time, Deficit, and Intensity",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis=dict(
                title='Event Date',
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"
            ),
            yaxis=dict(
                title='Water Deficit (hm³)',
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"
            ),
            zaxis=dict(
                title='Intensity Score (0-100)',
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=700,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)"
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig, use_container_width=True, key="3d_scatter_chart")

    # Add explanation
    st.markdown("---")
    st.markdown("#### 📖 3D Visualization Interpretation Guide")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **X-Axis (Time):**
        - Temporal distribution of drought events
        - Observe patterns and clustering over time
        - Events spread across the analysis period
        """)

    with col2:
        st.markdown("""
        **Y-Axis (Water Deficit):**
        - Volume of water shortage (hm³)
        - Higher values = more severe water deficit
        - Represents total deficit during the event
        """)

    with col3:
        st.markdown("""
        **Z-Axis (Intensity Score):**
        - Composite drought severity (0-100)
        - Higher position = more intense drought
        - Combines duration, deficit, and relative impact
        """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Marker Size:**
        - Represents event duration (days)
        - Larger markers = longer drought events
        - Scaled for optimal visibility
        """)

    with col2:
        st.markdown("""
        **Marker Color:**
        - Each station has a unique color
        - Use legend to identify stations
        - Helps track station-specific patterns
        """)

    st.markdown("""
    **💡 Interaction Tips:**
    - **Rotate:** Click and drag to rotate the 3D view
    - **Zoom:** Use scroll wheel or pinch to zoom
    - **Pan:** Hold shift and drag to pan
    - **Reset:** Double-click to reset the view
    - **Hover:** Move cursor over markers for detailed information
    """)

# ==================== TAB 2: INTENSITY DISTRIBUTION ====================
with tab2:
    st.markdown("### 📊 Drought Intensity Distribution Analysis")

    # Intensity histogram
    st.markdown("#### Intensity Score Distribution")

    fig_hist = go.Figure()

    fig_hist.add_trace(go.Histogram(
        x=filtered_drought['intensity'],
        nbinsx=20,
        marker_color='#0066cc',
        opacity=0.7,
        name='All Events'
    ))

    fig_hist.update_layout(
        title="Distribution of Drought Intensity Scores",
        xaxis_title="Intensity Score (0-100)",
        yaxis_title="Number of Events",
        template='plotly_white',
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig_hist, use_container_width=True, key="intensity_hist")

    st.markdown("---")

    # Intensity by station
    st.markdown("#### Intensity Comparison by Station")

    fig_box = go.Figure()

    for station in selected_stations:
        station_events = filtered_drought[filtered_drought['indroea'] == station]

        fig_box.add_trace(go.Box(
            y=station_events['intensity'],
            name=f"{station}",
            marker_color='#0066cc',
            boxmean='sd'  # Show mean and std dev
        ))

    fig_box.update_layout(
        title="Drought Intensity Distribution by Station",
        xaxis_title="Station",
        yaxis_title="Intensity Score",
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig_box, use_container_width=True, key="intensity_box")

    st.markdown("---")

    # Component breakdown
    st.markdown("#### Intensity Components Breakdown")

    # Calculate average components
    component_data = pd.DataFrame({
        'Component': ['Duration', 'Deficit', 'Relative Impact'],
        'Weight': [weight_duration * 100, weight_deficit * 100, weight_relative * 100],
        'Avg Value': [
            filtered_drought['duracion'].mean(),
            filtered_drought['deficit_hm3'].mean(),
            filtered_drought['relative_deficit'].mean()
        ]
    })

    col1, col2 = st.columns(2)

    with col1:
        # Weight distribution
        fig_weights = go.Figure(data=[go.Pie(
            labels=component_data['Component'],
            values=component_data['Weight'],
            marker=dict(colors=['#e74c3c', '#3498db', '#2ecc71']),
            hole=0.3
        )])

        fig_weights.update_layout(
            title="Current Intensity Weights",
            template='plotly_white',
            height=350
        )

        st.plotly_chart(fig_weights, use_container_width=True, key="weights_pie")

    with col2:
        # Average values
        st.markdown("**Average Component Values:**")
        st.dataframe(
            component_data.style.format({
                'Weight': '{:.1f}%',
                'Avg Value': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        **Customize weights** in the sidebar to emphasize different aspects of drought severity.
        """)

# ==================== TAB 3: EVENT DETAILS ====================
with tab3:
    st.markdown("### 📋 Detailed Event Information")

    # Sort options
    sort_by = st.selectbox(
        "Sort events by",
        options=['intensity', 'deficit_hm3', 'duracion', 'event_midpoint'],
        format_func=lambda x: {
            'intensity': 'Intensity Score',
            'deficit_hm3': 'Water Deficit',
            'duracion': 'Duration',
            'event_midpoint': 'Date'
        }[x]
    )

    sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
    ascending = (sort_order == "Ascending")

    # Display table
    display_df = filtered_drought[
        ['indroea', 'station_name', 'event_midpoint', 'duracion',
         'deficit_hm3', 'intensity', 'mean_flow']
    ].sort_values(sort_by, ascending=ascending).copy()

    # Rename columns for display
    display_df = display_df.rename(columns={
        'indroea': 'Station ID',
        'station_name': 'Station Name',
        'event_midpoint': 'Event Date',
        'duracion': 'Duration (days)',
        'deficit_hm3': 'Deficit (hm³)',
        'intensity': 'Intensity Score',
        'mean_flow': 'Station Mean Flow (m³/s)'
    })

    # Style the dataframe
    st.dataframe(
        display_df.style.format({
            'Duration (days)': '{:.0f}',
            'Deficit (hm³)': '{:.2f}',
            'Intensity Score': '{:.1f}',
            'Station Mean Flow (m³/s)': '{:.2f}'
        }).background_gradient(subset=['Intensity Score'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=400
    )

    # Summary statistics by station
    st.markdown("---")
    st.markdown("#### Summary Statistics by Station")

    summary_stats = filtered_drought.groupby('indroea').agg({
        'event_id': 'count',
        'duracion': ['mean', 'max'],
        'deficit_hm3': ['mean', 'sum'],
        'intensity': ['mean', 'max']
    }).reset_index()

    summary_stats.columns = ['Station ID', 'Event Count', 'Avg Duration', 'Max Duration',
                             'Avg Deficit', 'Total Deficit', 'Avg Intensity', 'Max Intensity']

    # Add station names
    summary_stats['Station Name'] = summary_stats['Station ID'].map(station_names)

    # Reorder columns
    summary_stats = summary_stats[['Station ID', 'Station Name', 'Event Count', 'Avg Duration',
                                   'Max Duration', 'Avg Deficit', 'Total Deficit', 'Avg Intensity', 'Max Intensity']]

    st.dataframe(
        summary_stats.style.format({
            'Avg Duration': '{:.1f}',
            'Max Duration': '{:.0f}',
            'Avg Deficit': '{:.2f}',
            'Total Deficit': '{:.2f}',
            'Avg Intensity': '{:.1f}',
            'Max Intensity': '{:.1f}'
        }).background_gradient(subset=['Avg Intensity'], cmap='RdYlGn_r'),
        use_container_width=True
    )

    # Download button
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        csv_events = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Event Details (CSV)",
            data=csv_events,
            file_name=f"drought_events_detailed_{year_range[0]}-{year_range[1]}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        csv_summary = summary_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Summary Statistics (CSV)",
            data=csv_summary,
            file_name=f"drought_events_summary_{year_range[0]}-{year_range[1]}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==================== METHODOLOGY ====================
st.markdown("---")

with st.expander("📖 Drought Intensity Methodology", expanded=False):
    st.markdown("""
    ### Drought Intensity Calculation

    The **Drought Intensity Score** is a composite metric (0-100) that quantifies drought severity by combining three key factors:

    #### 1. Duration Component (Default: 30% weight)
    - Measures the temporal extent of the drought event
    - Normalized across all events: `(duration - min_duration) / (max_duration - min_duration)`
    - Longer droughts have higher duration scores

    #### 2. Deficit Component (Default: 40% weight)
    - Quantifies the total water shortage during the event
    - Measured in cubic hectometers (hm³)
    - Normalized across all events: `(deficit - min_deficit) / (max_deficit - min_deficit)`
    - Larger deficits indicate more severe water shortage

    #### 3. Relative Impact Component (Default: 30% weight)
    - Contextualizes deficit relative to the station's typical flow
    - Calculated as: `deficit / mean_station_flow`
    - Normalized across all events
    - Accounts for the fact that the same absolute deficit is more severe at a low-flow station

    #### Final Formula:
    ```
    Intensity = (w₁ × Duration_norm + w₂ × Deficit_norm + w₃ × Relative_norm) × 100
    ```

    Where:
    - `w₁, w₂, w₃` are the weights (default: 0.3, 0.4, 0.3)
    - Weights are normalized to sum to 1.0
    - Result is scaled to 0-100 for interpretability

    #### Customization:
    You can adjust the weights in the sidebar under "Customize Intensity Weights" to emphasize different aspects:
    - **High duration weight:** Emphasizes long-term drought stress
    - **High deficit weight:** Emphasizes water shortage volume
    - **High relative impact weight:** Emphasizes drought severity relative to station capacity

    #### Interpretation:
    - **0-30:** Low intensity (minor drought event)
    - **30-60:** Moderate intensity (significant drought)
    - **60-100:** High intensity (severe drought with major impacts)
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
    <p>🌊 Drought Events Visualization | Interactive bubble chart analysis of drought characteristics</p>
</div>
""", unsafe_allow_html=True)
