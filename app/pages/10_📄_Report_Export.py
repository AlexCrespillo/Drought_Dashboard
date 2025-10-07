"""
Report Export Page - Generate PDF/HTML Reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
from datetime import datetime
import base64

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
        <h1>📄 Report Export & Documentation</h1>
        <p>Generate comprehensive HTML reports of your analysis</p>
    </div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
with st.spinner("Loading data..."):
    drought_df = load_drought_events()
    streamflow_df = load_streamflow_data()

# ==================== SIDEBAR: REPORT CONFIGURATION ====================
st.sidebar.header("⚙️ Report Configuration")

report_title = st.sidebar.text_input(
    "Report Title",
    value="Drought Analysis Report - Ebro River Basin"
)

# Station selection for report
if not streamflow_df.empty:
    available_stations = sorted(streamflow_df['indroea'].unique())
    selected_stations_report = st.sidebar.multiselect(
        "Select Stations for Report",
        options=available_stations,
        default=available_stations[:5] if len(available_stations) >= 5 else available_stations,
        help="Select stations to include in the report"
    )
else:
    selected_stations_report = []

st.sidebar.markdown("---")
st.sidebar.header("📊 Include Sections")

include_summary = st.sidebar.checkbox("Executive Summary", value=True)
include_data_overview = st.sidebar.checkbox("Data Overview", value=True)
include_spatial = st.sidebar.checkbox("Spatial Analysis", value=True)
include_temporal = st.sidebar.checkbox("Temporal Analysis", value=True)
include_drought_events = st.sidebar.checkbox("Drought Events Analysis", value=True)
include_station_clustering = st.sidebar.checkbox("Station Clustering", value=True)
include_station_comparison = st.sidebar.checkbox("Station Comparison", value=True)
include_drought_viz = st.sidebar.checkbox("Drought Events Visualization", value=True)
include_ml_prediction = st.sidebar.checkbox("ML Prediction & Forecasting", value=True)

st.sidebar.markdown("---")

# Date range for report
if 'fecha' in streamflow_df.columns and not streamflow_df.empty:
    streamflow_df['fecha'] = pd.to_datetime(streamflow_df['fecha'])
    report_date_range = st.sidebar.date_input(
        "Report Date Range",
        value=(streamflow_df['fecha'].min(), streamflow_df['fecha'].max())
    )
else:
    report_date_range = None

# ==================== MAIN CONTENT ====================

st.markdown("### 📄 Report Generator")

st.info("Configure your report using the sidebar options, then click the button below to generate an HTML report.")

# Preview report sections
st.markdown("---")
st.markdown("### Report Preview")

tab1, tab2 = st.tabs(["📋 Report Configuration", "🔍 Content Preview"])

with tab1:
    st.markdown("#### Selected Report Sections:")

    sections = []
    if include_summary:
        sections.append("✅ Executive Summary")
    if include_data_overview:
        sections.append("✅ Data Overview")
    if include_spatial:
        sections.append("✅ Spatial Analysis")
    if include_temporal:
        sections.append("✅ Temporal Analysis")
    if include_drought_events:
        sections.append("✅ Drought Events Analysis")
    if include_station_clustering:
        sections.append("✅ Station Clustering")
    if include_station_comparison:
        sections.append("✅ Station Comparison")
    if include_drought_viz:
        sections.append("✅ Drought Events Visualization")
    if include_ml_prediction:
        sections.append("✅ ML Prediction & Forecasting")

    if sections:
        for section in sections:
            st.markdown(f"- {section}")
    else:
        st.warning("⚠️ No sections selected. Please enable at least one section in the sidebar.")

    st.markdown("---")
    st.markdown("#### Report Parameters:")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Stations", len(selected_stations_report))
        st.metric("Title", report_title[:30] + "..." if len(report_title) > 30 else report_title)

    with col2:
        if report_date_range and len(report_date_range) == 2:
            st.metric("Date Range", f"{(report_date_range[1] - report_date_range[0]).days} days")
        st.metric("Sections", len(sections))

with tab2:
    st.markdown("#### Content Preview")

    if not selected_stations_report:
        st.warning("⚠️ Please select at least one station to generate a report")
    else:
        # Filter data based on selections
        filtered_streamflow = streamflow_df[streamflow_df['indroea'].isin(selected_stations_report)].copy()
        filtered_drought = drought_df[drought_df['indroea'].isin(selected_stations_report)].copy() if not drought_df.empty else pd.DataFrame()

        if report_date_range and len(report_date_range) == 2:
            date_start, date_end = report_date_range
            filtered_streamflow = filtered_streamflow[
                (filtered_streamflow['fecha'] >= pd.to_datetime(date_start)) &
                (filtered_streamflow['fecha'] <= pd.to_datetime(date_end))
            ]

        # Show preview statistics
        st.markdown("**Key Statistics for Report:**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(filtered_streamflow):,}")

        with col2:
            if not filtered_drought.empty:
                st.metric("Drought Events", len(filtered_drought))
            else:
                st.metric("Drought Events", "N/A")

        with col3:
            if not filtered_streamflow.empty:
                st.metric("Mean Flow", f"{filtered_streamflow['caudal'].mean():.2f} m³/s")
            else:
                st.metric("Mean Flow", "N/A")

        with col4:
            if not filtered_streamflow.empty:
                st.metric("Date Span", f"{filtered_streamflow['fecha'].nunique()} days")
            else:
                st.metric("Date Span", "N/A")

st.markdown("---")

# ==================== GENERATE REPORT ====================

def generate_html_report(
    title,
    stations,
    streamflow_data,
    drought_data,
    include_sections
):
    """Generate HTML report"""

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load and encode IPE logo
    logo_base64 = ""
    try:
        logo_path = BASE_DIR / "data" / "png" / "logo-ipe22.png"
        if logo_path.exists():
            with open(logo_path, 'rb') as f:
                logo_base64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        pass  # Continue without logo if it fails

    # Start HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f8f9fa;
            }}
            .header {{
                background: linear-gradient(135deg, #0066cc 0%, #00a3e0 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .header-content {{
                flex: 1;
            }}
            .header-logo {{
                flex-shrink: 0;
                margin-left: 20px;
            }}
            .header-logo img {{
                max-height: 100px;
                max-width: 200px;
                object-fit: contain;
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 2.5em;
            }}
            .header p {{
                margin: 5px 0;
                opacity: 0.9;
            }}
            .section {{
                background: white;
                padding: 25px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #0066cc;
                border-bottom: 2px solid #0066cc;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .section h3 {{
                color: #004d99;
                margin-top: 20px;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #0066cc;
                text-align: center;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #0066cc;
                margin: 10px 0;
            }}
            .metric-label {{
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background: #0066cc;
                color: white;
                font-weight: bold;
            }}
            tr:hover {{
                background: #f8f9fa;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                margin-top: 40px;
                border-top: 2px solid #e9ecef;
                color: #666;
            }}
            .author-info {{
                background: linear-gradient(135deg, #0066cc 0%, #00a3e0 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                text-align: center;
            }}
            .station-list {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 15px 0;
            }}
            .station-tag {{
                background: #0066cc;
                color: white;
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-content">
                <h1>📊 {title}</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Stations Analyzed:</strong> {len(stations)}</p>
                <p><strong>Author:</strong> {AUTHOR_INFO['name']} - {AUTHOR_INFO['occupation']}, {AUTHOR_INFO['institution']}</p>
            </div>
    """

    # Add logo if available
    if logo_base64:
        html += f"""
            <div class="header-logo">
                <img src="data:image/png;base64,{logo_base64}" alt="IPE-CSIC Logo">
            </div>
        """

    html += """
        </div>
    """

    # Executive Summary
    if include_sections['summary']:
        html += """
        <div class="section">
            <h2>📋 Executive Summary</h2>
        """

        # Calculate summary statistics
        total_records = len(streamflow_data)
        mean_flow = streamflow_data['caudal'].mean() if not streamflow_data.empty else 0
        drought_events = len(drought_data) if not drought_data.empty else 0

        html += f"""
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Total Stations</div>
                    <div class="metric-value">{len(stations)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Data Records</div>
                    <div class="metric-value">{total_records:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Flow</div>
                    <div class="metric-value">{mean_flow:.2f}</div>
                    <div class="metric-label">m³/s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Drought Events</div>
                    <div class="metric-value">{drought_events}</div>
                </div>
            </div>

            <h3>Selected Stations</h3>
            <div class="station-list">
        """

        for station in stations:
            html += f'<span class="station-tag">Station {station}</span>'

        html += """
            </div>
        </div>
        """

    # Data Overview
    if include_sections['data_overview'] and not streamflow_data.empty:
        html += """
        <div class="section">
            <h2>📊 Data Overview</h2>
            <h3>Dataset Characteristics</h3>
        """

        # Calculate data completeness
        total_days = (streamflow_data['fecha'].max() - streamflow_data['fecha'].min()).days if 'fecha' in streamflow_data.columns else 0
        records_per_station = len(streamflow_data) / len(stations) if stations else 0

        html += f"""
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Temporal Span</div>
                    <div class="metric-value">{total_days:,}</div>
                    <div class="metric-label">days</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Records per Station</div>
                    <div class="metric-value">{records_per_station:,.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Flow</div>
                    <div class="metric-value">{streamflow_data['caudal'].mean():.2f}</div>
                    <div class="metric-label">m³/s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Flow Variability (CV)</div>
                    <div class="metric-value">{(streamflow_data['caudal'].std() / streamflow_data['caudal'].mean()):.2f}</div>
                </div>
            </div>
        """

        # Flow statistics table
        flow_stats = streamflow_data['caudal'].describe()
        html += """
            <h3>Flow Statistics Summary</h3>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Value (m³/s)</th>
                    </tr>
                </thead>
                <tbody>
        """

        for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            html += f"""
                <tr>
                    <td>{stat.upper()}</td>
                    <td>{flow_stats[stat]:.2f}</td>
                </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

    # Temporal Analysis
    if include_sections['temporal'] and not streamflow_data.empty:
        html += """
        <div class="section">
            <h2>📈 Temporal Analysis</h2>
        """

        # Yearly summary
        if 'year' in streamflow_data.columns:
            yearly_stats = streamflow_data.groupby('year')['caudal'].agg(['mean', 'min', 'max', 'count']).reset_index()

            html += """
                <h3>Annual Statistics</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Year</th>
                            <th>Mean Flow (m³/s)</th>
                            <th>Min Flow (m³/s)</th>
                            <th>Max Flow (m³/s)</th>
                            <th>Records</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for _, row in yearly_stats.head(10).iterrows():
                html += f"""
                    <tr>
                        <td>{int(row['year'])}</td>
                        <td>{row['mean']:.2f}</td>
                        <td>{row['min']:.2f}</td>
                        <td>{row['max']:.2f}</td>
                        <td>{int(row['count'])}</td>
                    </tr>
                """

            html += """
                    </tbody>
                </table>
            """

        html += "</div>"

    # Station Clustering Analysis
    if include_sections['station_clustering'] and not drought_data.empty:
        html += """
        <div class="section">
            <h2>🔬 Station Clustering Analysis</h2>
            <p>Hydrological regime classification using machine learning techniques (K-means clustering).</p>
            <h3>Cluster Methodology</h3>
            <p>Stations are grouped based on drought characteristics including:</p>
            <ul>
                <li>Average drought duration</li>
                <li>Maximum drought duration</li>
                <li>Total water deficit</li>
                <li>Event frequency</li>
                <li>Mean streamflow characteristics</li>
            </ul>
        """

        # Calculate basic clustering metrics by station
        station_metrics = drought_data.groupby('indroea').agg({
            'duracion': ['mean', 'max', 'count'],
            'deficit_hm3': 'sum'
        }).reset_index()

        station_metrics.columns = ['Station', 'Avg Duration', 'Max Duration', 'Event Count', 'Total Deficit']

        html += """
            <h3>Station Drought Metrics</h3>
            <table>
                <thead>
                    <tr>
                        <th>Station</th>
                        <th>Avg Duration (days)</th>
                        <th>Max Duration (days)</th>
                        <th>Event Count</th>
                        <th>Total Deficit (hm³)</th>
                    </tr>
                </thead>
                <tbody>
        """

        for _, row in station_metrics.iterrows():
            html += f"""
                <tr>
                    <td>{int(row['Station'])}</td>
                    <td>{row['Avg Duration']:.1f}</td>
                    <td>{int(row['Max Duration'])}</td>
                    <td>{int(row['Event Count'])}</td>
                    <td>{row['Total Deficit']:.2f}</td>
                </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

    # Station Comparison
    if include_sections['station_comparison'] and len(stations) >= 2:
        html += """
        <div class="section">
            <h2>📊 Station Comparison</h2>
            <p>Comparative analysis of selected stations across key hydrological indicators.</p>
        """

        # Compare flow statistics across stations
        comparison_data = []
        for station in stations:
            station_flow = streamflow_data[streamflow_data['indroea'] == station]['caudal']
            if not station_flow.empty:
                comparison_data.append({
                    'Station': station,
                    'Mean': station_flow.mean(),
                    'Median': station_flow.median(),
                    'Std': station_flow.std(),
                    'CV': station_flow.std() / station_flow.mean() if station_flow.mean() > 0 else 0
                })

        if comparison_data:
            html += """
                <h3>Flow Comparison</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Station</th>
                            <th>Mean Flow (m³/s)</th>
                            <th>Median Flow (m³/s)</th>
                            <th>Std Dev (m³/s)</th>
                            <th>Coefficient of Variation</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for row in comparison_data:
                html += f"""
                    <tr>
                        <td>{int(row['Station'])}</td>
                        <td>{row['Mean']:.2f}</td>
                        <td>{row['Median']:.2f}</td>
                        <td>{row['Std']:.2f}</td>
                        <td>{row['CV']:.3f}</td>
                    </tr>
                """

            html += """
                    </tbody>
                </table>
            """

        html += "</div>"

    # Station Details
    if include_sections.get('station_details', False):
        html += """
        <div class="section">
            <h2>📍 Station Details</h2>
        """

        for station in stations:
            station_data = streamflow_data[streamflow_data['indroea'] == station]

            if not station_data.empty:
                station_name = station_data['lugar'].iloc[0] if 'lugar' in station_data.columns else "Unknown"
                mean_flow = station_data['caudal'].mean()
                std_flow = station_data['caudal'].std()
                q95 = station_data['caudal'].quantile(0.95)
                q5 = station_data['caudal'].quantile(0.05)

                html += f"""
                <h3>Station {station} - {station_name}</h3>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">Mean Flow</div>
                        <div class="metric-value">{mean_flow:.2f}</div>
                        <div class="metric-label">m³/s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Std Dev</div>
                        <div class="metric-value">{std_flow:.2f}</div>
                        <div class="metric-label">m³/s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Q95</div>
                        <div class="metric-value">{q95:.2f}</div>
                        <div class="metric-label">m³/s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Q5</div>
                        <div class="metric-value">{q5:.2f}</div>
                        <div class="metric-label">m³/s</div>
                    </div>
                </div>
                """

        html += "</div>"

    # Drought Events
    if include_sections['drought_events'] and not drought_data.empty:
        html += """
        <div class="section">
            <h2>🌊 Drought Events Summary</h2>
        """

        # Summary by station
        drought_summary = drought_data.groupby('indroea').agg({
            'event_id': 'count',
            'duracion': 'mean',
            'deficit_hm3': 'sum'
        }).reset_index()

        html += """
            <table>
                <thead>
                    <tr>
                        <th>Station</th>
                        <th>Event Count</th>
                        <th>Avg Duration (days)</th>
                        <th>Total Deficit (hm³)</th>
                    </tr>
                </thead>
                <tbody>
        """

        for _, row in drought_summary.iterrows():
            html += f"""
                <tr>
                    <td>{int(row['indroea'])}</td>
                    <td>{int(row['event_id'])}</td>
                    <td>{row['duracion']:.1f}</td>
                    <td>{row['deficit_hm3']:.2f}</td>
                </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

    # Drought Events Visualization
    if include_sections['drought_viz'] and not drought_data.empty:
        html += """
        <div class="section">
            <h2>🌊 Drought Events Visualization Summary</h2>
            <p>Comprehensive visualization analysis of drought event characteristics.</p>
            <h3>Event Intensity Distribution</h3>
        """

        # Calculate intensity metrics
        if 'duracion' in drought_data.columns and 'deficit_hm3' in drought_data.columns:
            # Calculate composite intensity (simplified version for report)
            drought_data_copy = drought_data.copy()
            duration_norm = (drought_data_copy['duracion'] - drought_data_copy['duracion'].min()) / (drought_data_copy['duracion'].max() - drought_data_copy['duracion'].min() + 1e-10)
            deficit_norm = (drought_data_copy['deficit_hm3'] - drought_data_copy['deficit_hm3'].min()) / (drought_data_copy['deficit_hm3'].max() - drought_data_copy['deficit_hm3'].min() + 1e-10)
            drought_data_copy['intensity'] = ((duration_norm * 0.5 + deficit_norm * 0.5) * 100)

            # Get intensity statistics
            intensity_stats = drought_data_copy['intensity'].describe()

            html += f"""
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">Mean Intensity</div>
                        <div class="metric-value">{intensity_stats['mean']:.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Median Intensity</div>
                        <div class="metric-value">{intensity_stats['50%']:.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Intensity</div>
                        <div class="metric-value">{intensity_stats['max']:.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">High Intensity Events</div>
                        <div class="metric-value">{(drought_data_copy['intensity'] > 60).sum()}</div>
                    </div>
                </div>

                <h3>Intensity Classification</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Intensity Range</th>
                            <th>Classification</th>
                            <th>Event Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            # Classify by intensity
            low = (drought_data_copy['intensity'] < 30).sum()
            moderate = ((drought_data_copy['intensity'] >= 30) & (drought_data_copy['intensity'] < 60)).sum()
            high = (drought_data_copy['intensity'] >= 60).sum()
            total = len(drought_data_copy)

            html += f"""
                        <tr>
                            <td>0-30</td>
                            <td>Low Intensity</td>
                            <td>{low}</td>
                            <td>{(low/total*100):.1f}%</td>
                        </tr>
                        <tr>
                            <td>30-60</td>
                            <td>Moderate Intensity</td>
                            <td>{moderate}</td>
                            <td>{(moderate/total*100):.1f}%</td>
                        </tr>
                        <tr>
                            <td>60-100</td>
                            <td>High Intensity</td>
                            <td>{high}</td>
                            <td>{(high/total*100):.1f}%</td>
                        </tr>
                    </tbody>
                </table>
            """

        html += "</div>"

    # ML Prediction & Forecasting
    if include_sections.get('ml_prediction', False) and not drought_data.empty:
        html += """
        <div class="section">
            <h2>🤖 ML Prediction & Forecasting</h2>
            <p>Machine learning models for streamflow forecasting and drought event prediction.</p>

            <h3>Overview</h3>
            <p>This section presents advanced machine learning capabilities for predicting future streamflow patterns
            and estimating drought event severity based on pre-event conditions.</p>

            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
                <h4 style="margin-top: 0; color: #0066cc;">📈 ARIMA/SARIMA Time Series Forecasting</h4>
                <p><strong>Purpose:</strong> Predict future daily streamflow values based on historical patterns</p>
                <p><strong>Model Type:</strong> AutoRegressive Integrated Moving Average (ARIMA)</p>
                <p><strong>Use Cases:</strong></p>
                <ul>
                    <li>Short-term flow predictions (1-30 days ahead)</li>
                    <li>Early warning for low-flow conditions</li>
                    <li>Water resource management planning</li>
                    <li>Operational decision support</li>
                </ul>
                <p><strong>Key Features:</strong></p>
                <ul>
                    <li><strong>Trend Detection:</strong> Identifies increasing or decreasing flow patterns</li>
                    <li><strong>Seasonality:</strong> Captures repeating annual patterns (SARIMA)</li>
                    <li><strong>Confidence Intervals:</strong> Provides uncertainty bounds for predictions</li>
                    <li><strong>Baseline Comparison:</strong> Validated against persistence model</li>
                </ul>
            </div>

            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #00a3e0; margin: 20px 0;">
                <h4 style="margin-top: 0; color: #00a3e0;">🎯 Drought Event Prediction (Regression)</h4>
                <p><strong>Purpose:</strong> Predict water deficit (hm³) of drought events based on pre-event conditions</p>
                <p><strong>Model Types:</strong> Linear Regression & Random Forest</p>
                <p><strong>Features Used:</strong></p>
                <ul>
                    <li><strong>Event Duration:</strong> Length of drought event (days)</li>
                    <li><strong>Pre-Event Flow:</strong> Mean, minimum, and variability of flow before drought</li>
                    <li><strong>Flow Trend:</strong> Whether flow was increasing or decreasing before event</li>
                    <li><strong>Temporal Factors:</strong> Month and year (seasonality and long-term trends)</li>
                </ul>
                <p><strong>Applications:</strong></p>
                <ul>
                    <li>Early estimation of drought severity when event begins</li>
                    <li>Resource allocation planning based on expected water deficit</li>
                    <li>Trigger points for drought management actions</li>
                    <li>Risk assessment for water supply systems</li>
                </ul>
            </div>

            <h3>Model Performance Summary</h3>
            <p>The machine learning models have been trained and validated on historical data from the Ebro River Basin:</p>
        """

        # Calculate some basic ML-relevant metrics
        event_count = len(drought_data)
        avg_deficit = drought_data['deficit_hm3'].mean() if 'deficit_hm3' in drought_data.columns else 0
        avg_duration = drought_data['duracion'].mean() if 'duracion' in drought_data.columns else 0

        html += f"""
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Training Events</div>
                    <div class="metric-value">{event_count}</div>
                    <div class="metric-label">drought events</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Event Duration</div>
                    <div class="metric-value">{avg_duration:.0f}</div>
                    <div class="metric-label">days</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Water Deficit</div>
                    <div class="metric-value">{avg_deficit:.2f}</div>
                    <div class="metric-label">hm³</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Model Types</div>
                    <div class="metric-value">2</div>
                    <div class="metric-label">ARIMA + Regression</div>
                </div>
            </div>

            <h3>Key Insights from ML Analysis</h3>
            <ul>
                <li><strong>Predictability:</strong> Streamflow patterns show temporal autocorrelation, making short-term forecasting feasible with ARIMA models</li>
                <li><strong>Pre-Event Conditions:</strong> Flow statistics 30 days before a drought event are strong indicators of eventual drought severity</li>
                <li><strong>Seasonal Influence:</strong> Month of drought onset significantly affects water deficit patterns</li>
                <li><strong>Model Comparison:</strong> Random Forest typically outperforms Linear Regression for complex non-linear relationships</li>
                <li><strong>Operational Value:</strong> Models provide actionable early warnings for water resource managers</li>
            </ul>

            <h3>Recommended Usage</h3>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Best For</th>
                        <th>Forecast Horizon</th>
                        <th>Update Frequency</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>ARIMA Forecasting</td>
                        <td>Daily flow predictions</td>
                        <td>1-7 days (optimal)<br>up to 30 days (longer uncertainty)</td>
                        <td>Daily</td>
                    </tr>
                    <tr>
                        <td>Regression (Deficit)</td>
                        <td>Drought severity estimation</td>
                        <td>At drought onset</td>
                        <td>When new drought detected</td>
                    </tr>
                </tbody>
            </table>

            <h3>Model Limitations & Considerations</h3>
            <ul>
                <li><strong>ARIMA:</strong> Best for short-term forecasts; accuracy decreases with longer horizons</li>
                <li><strong>ARIMA:</strong> Assumes patterns continue; cannot predict sudden changes (dam operations, extreme rainfall)</li>
                <li><strong>Regression:</strong> Based on historical relationships; may not capture unprecedented drought conditions</li>
                <li><strong>Regression:</strong> Requires pre-event flow data; cannot predict droughts without 30-day lookback period</li>
                <li><strong>General:</strong> Models should be retrained periodically as new data becomes available</li>
                <li><strong>General:</strong> Predictions should be combined with expert judgment for critical decisions</li>
            </ul>

            <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ff7f0e; margin: 20px 0;">
                <h4 style="margin-top: 0; color: #856404;">💡 Implementation Notes</h4>
                <p><strong>For operational deployment:</strong></p>
                <ul>
                    <li>Validate models on most recent data before operational use</li>
                    <li>Establish confidence thresholds for triggering management actions</li>
                    <li>Monitor model performance and retrain when accuracy degrades</li>
                    <li>Consider ensemble approaches combining multiple models</li>
                    <li>Document model assumptions and limitations for stakeholders</li>
                </ul>
            </div>
        </div>
        """

    # Spatial Statistics
    if include_sections['spatial']:
        html += """
        <div class="section">
            <h2>🗺️ Spatial Statistics</h2>
            <p>Geographic distribution analysis of selected stations across the Ebro River Basin.</p>
        """

        # Station coordinates if available
        if 'xutm' in streamflow_data.columns and 'yutm' in streamflow_data.columns:
            station_coords = streamflow_data[['indroea', 'xutm', 'yutm', 'lugar']].drop_duplicates('indroea')

            html += """
                <table>
                    <thead>
                        <tr>
                            <th>Station</th>
                            <th>Location</th>
                            <th>X (UTM)</th>
                            <th>Y (UTM)</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for _, row in station_coords.iterrows():
                html += f"""
                    <tr>
                        <td>{int(row['indroea'])}</td>
                        <td>{row['lugar']}</td>
                        <td>{row['xutm']:.0f}</td>
                        <td>{row['yutm']:.0f}</td>
                    </tr>
                """

            html += """
                    </tbody>
                </table>
            """

        html += "</div>"

    # Footer
    html += f"""
        <div class="author-info">
    """

    # Add logo in author section if available
    if logo_base64:
        html += f"""
            <div style="display: flex; align-items: center; justify-content: center; gap: 30px; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <h3 style="margin: 0;">{AUTHOR_INFO['name']}</h3>
                    <p style="margin: 5px 0;"><strong>{AUTHOR_INFO['occupation']}</strong></p>
                    <p style="margin: 5px 0;">{AUTHOR_INFO['institution']}</p>
                    <p style="margin: 5px 0; font-size: 0.9em;">{AUTHOR_INFO['institution_full']}</p>
                </div>
                <div>
                    <img src="data:image/png;base64,{logo_base64}" alt="IPE-CSIC Logo" style="max-height: 80px; max-width: 150px; object-fit: contain;">
                </div>
            </div>
        """
    else:
        html += f"""
            <h3>{AUTHOR_INFO['name']}</h3>
            <p><strong>{AUTHOR_INFO['occupation']}</strong></p>
            <p>{AUTHOR_INFO['institution']}</p>
            <p>{AUTHOR_INFO['institution_full']}</p>
        """

    html += f"""
        </div>

        <div class="footer">
            <p><strong>Drought Events Analysis Dashboard</strong></p>
            <p>Report generated on {timestamp}</p>
            <p>Data Period: {START_YEAR}-{END_YEAR}</p>
        </div>
    </body>
    </html>
    """

    return html


# Generate Report Button
st.markdown("### 🚀 Generate Report")

if not selected_stations_report:
    st.warning("⚠️ Please select at least one station in the sidebar to generate a report")
else:
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("📄 Generate HTML Report", type="primary", use_container_width=True):
            with st.spinner("Generating report..."):
                # Filter data
                filtered_streamflow = streamflow_df[streamflow_df['indroea'].isin(selected_stations_report)].copy()
                filtered_drought = drought_df[drought_df['indroea'].isin(selected_stations_report)].copy() if not drought_df.empty else pd.DataFrame()

                if report_date_range and len(report_date_range) == 2:
                    date_start, date_end = report_date_range
                    filtered_streamflow = filtered_streamflow[
                        (filtered_streamflow['fecha'] >= pd.to_datetime(date_start)) &
                        (filtered_streamflow['fecha'] <= pd.to_datetime(date_end))
                    ]

                # Add year column for temporal analysis
                if 'fecha' in filtered_streamflow.columns:
                    filtered_streamflow['year'] = pd.to_datetime(filtered_streamflow['fecha']).dt.year

                # Generate report
                include_sections_dict = {
                    'summary': include_summary,
                    'data_overview': include_data_overview,
                    'spatial': include_spatial,
                    'temporal': include_temporal,
                    'drought_events': include_drought_events,
                    'station_clustering': include_station_clustering,
                    'station_comparison': include_station_comparison,
                    'drought_viz': include_drought_viz,
                    'ml_prediction': include_ml_prediction
                }

                html_report = generate_html_report(
                    title=report_title,
                    stations=selected_stations_report,
                    streamflow_data=filtered_streamflow,
                    drought_data=filtered_drought,
                    include_sections=include_sections_dict
                )

                # Provide download
                st.success("✅ Report generated successfully!")

                # Download button
                timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Download HTML Report",
                    data=html_report,
                    file_name=f"drought_analysis_report_{timestamp_file}.html",
                    mime="text/html",
                    use_container_width=True
                )

                st.info("💡 The HTML report can be opened in any web browser and shared with colleagues.")

st.markdown("---")

# Additional export options
st.markdown("### 📊 Additional Export Options")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Export Raw Data")

    if st.button("📥 Export Streamflow Data (CSV)", use_container_width=True):
        if not selected_stations_report:
            st.warning("Please select stations first")
        else:
            export_data = streamflow_df[streamflow_df['indroea'].isin(selected_stations_report)]
            csv = export_data.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Streamflow CSV",
                data=csv,
                file_name=f"streamflow_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

with col2:
    st.markdown("#### Export Drought Events")

    if st.button("📥 Export Drought Events (CSV)", use_container_width=True):
        if not selected_stations_report:
            st.warning("Please select stations first")
        elif drought_df.empty:
            st.warning("No drought data available")
        else:
            export_data = drought_df[drought_df['indroea'].isin(selected_stations_report)]
            csv = export_data.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Drought Events CSV",
                data=csv,
                file_name=f"drought_events_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
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
    <p>📄 Report Export | Generate comprehensive analysis reports in HTML format</p>
</div>
""", unsafe_allow_html=True)
