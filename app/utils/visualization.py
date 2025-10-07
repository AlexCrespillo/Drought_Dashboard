"""
Visualization utilities for Drought Dashboard
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
import folium
from streamlit_folium import folium_static
import sys
from pathlib import Path

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import *

# Set default style
plt.style.use(PLOT_STYLE)
sns.set_palette(COLOR_PALETTE)


def create_time_series_plot(df, x_col, y_col, title="Time Series",
                            x_label="Date", y_label="Value", color=None):
    """
    Create interactive time series plot using Plotly

    Args:
        df: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        color: Column for color grouping

    Returns:
        plotly.graph_objects.Figure
    """
    if color:
        fig = px.line(df, x=x_col, y=y_col, color=color, title=title)
    else:
        fig = px.line(df, x=x_col, y=y_col, title=title)

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_histogram(df, column, title="Distribution", bins=30, color=DROUGHT_COLOR):
    """
    Create histogram with Plotly

    Args:
        df: DataFrame
        column: Column to plot
        title: Plot title
        bins: Number of bins
        color: Bar color

    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.histogram(df, x=column, nbins=bins, title=title)
    fig.update_traces(marker_color=color, marker_line_color='white', marker_line_width=1)
    fig.update_layout(
        xaxis_title=column.replace('_', ' ').title(),
        yaxis_title='Frequency',
        template='plotly_white',
        showlegend=False
    )

    return fig


def create_box_plot(df, x_col, y_col, title="Box Plot"):
    """
    Create box plot with Plotly

    Args:
        df: DataFrame
        x_col: Column for x-axis (categories)
        y_col: Column for y-axis (values)
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.box(df, x=x_col, y=y_col, title=title)
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        template='plotly_white'
    )

    return fig


def create_scatter_plot(df, x_col, y_col, title="Scatter Plot",
                       color=None, size=None, trendline=False):
    """
    Create scatter plot with Plotly

    Args:
        df: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Plot title
        color: Column for color grouping
        size: Column for marker size
        trendline: Whether to add trendline

    Returns:
        plotly.graph_objects.Figure
    """
    trendline_arg = "ols" if trendline else None

    fig = px.scatter(df, x=x_col, y=y_col, color=color, size=size,
                     title=title, trendline=trendline_arg)

    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        template='plotly_white'
    )

    return fig


def create_heatmap(data, x_labels=None, y_labels=None, title="Heatmap",
                   colormap='RdYlBu_r', annotate=True):
    """
    Create heatmap with Plotly

    Args:
        data: 2D array or DataFrame
        x_labels: X-axis labels
        y_labels: Y-axis labels
        title: Plot title
        colormap: Colorscale
        annotate: Whether to show values

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale=colormap,
        text=data if annotate else None,
        texttemplate='%{text:.2f}' if annotate else None
    ))

    fig.update_layout(
        title=title,
        template='plotly_white'
    )

    return fig


def create_bar_chart(df, x_col, y_col, title="Bar Chart", color=None, orientation='v'):
    """
    Create bar chart with Plotly

    Args:
        df: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Plot title
        color: Column for color grouping
        orientation: 'v' for vertical, 'h' for horizontal

    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.bar(df, x=x_col, y=y_col, color=color, title=title,
                 orientation=orientation)

    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        template='plotly_white'
    )

    return fig


def create_correlation_matrix(df, columns=None, title="Correlation Matrix"):
    """
    Create correlation matrix heatmap

    Args:
        df: DataFrame
        columns: Columns to include (if None, use all numeric)
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    if columns:
        corr_data = df[columns].corr()
    else:
        corr_data = df.select_dtypes(include=[np.number]).corr()

    fig = create_heatmap(
        corr_data.values,
        x_labels=corr_data.columns,
        y_labels=corr_data.columns,
        title=title,
        colormap='RdBu_r',
        annotate=True
    )

    return fig


def create_trend_plot(df, x_col, y_col, title="Trend Analysis"):
    """
    Create trend plot with linear regression

    Args:
        df: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    # Calculate trend
    x = df[x_col].values
    y = df[y_col].values

    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        line_y = slope * x_clean + intercept

        # Create figure
        fig = go.Figure()

        # Add scatter
        fig.add_trace(go.Scatter(
            x=x_clean, y=y_clean,
            mode='markers',
            name='Data',
            marker=dict(size=8, opacity=0.6)
        ))

        # Add trend line
        fig.add_trace(go.Scatter(
            x=x_clean, y=line_y,
            mode='lines',
            name=f'Trend (R²={r_value**2:.3f}, p={p_value:.3f})',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            template='plotly_white'
        )

        return fig
    else:
        st.warning("Not enough data points for trend analysis")
        return None


def create_seasonal_plot(df, value_col, title="Seasonal Pattern"):
    """
    Create seasonal box plot

    Args:
        df: DataFrame with 'season' column
        value_col: Column to plot
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    if 'season' not in df.columns:
        st.warning("DataFrame must have 'season' column")
        return None

    # Define season order
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    df_plot = df.copy()
    df_plot['season'] = pd.Categorical(df_plot['season'], categories=season_order, ordered=True)
    df_plot = df_plot.sort_values('season')

    fig = px.box(df_plot, x='season', y=value_col, title=title)
    fig.update_layout(
        xaxis_title='Season',
        yaxis_title=value_col.replace('_', ' ').title(),
        template='plotly_white'
    )

    return fig


def create_monthly_heatmap(df, value_col, title="Monthly Pattern", aggfunc='mean'):
    """
    Create year-month heatmap

    Args:
        df: DataFrame with 'year' and 'month' columns
        value_col: Column to aggregate
        title: Plot title
        aggfunc: Aggregation function ('mean', 'sum', 'count', 'size')

    Returns:
        plotly.graph_objects.Figure
    """
    if 'year' not in df.columns or 'month' not in df.columns:
        st.warning("DataFrame must have 'year' and 'month' columns")
        return None

    # Create pivot table
    # Use 'count' for event_id to count occurrences, not average
    if aggfunc == 'size' or value_col == 'event_id':
        # Use size to count rows per group
        pivot_data = df.groupby(['year', 'month']).size().unstack(fill_value=0)
    else:
        pivot_data = df.pivot_table(
            values=value_col,
            index='year',
            columns='month',
            aggfunc=aggfunc,
            fill_value=0
        )

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Only show months that exist in the data
    available_months = [i for i in range(1, 13) if i in pivot_data.columns]
    month_labels = [month_names[i-1] for i in available_months]

    fig = create_heatmap(
        pivot_data[available_months].values,
        x_labels=month_labels,
        y_labels=pivot_data.index,
        title=title,
        colormap='RdYlBu_r',
        annotate=False
    )

    return fig


def create_station_map(stations_gdf, value_col=None, title="Station Map"):
    """
    Create interactive folium map with stations

    Args:
        stations_gdf: GeoDataFrame with station locations
        value_col: Column for color mapping (optional)
        title: Map title

    Returns:
        folium.Map
    """
    if stations_gdf is None or stations_gdf.empty:
        st.warning("No station data available")
        return None

    # Convert to WGS84 if needed
    if stations_gdf.crs != 'EPSG:4326':
        stations_gdf = stations_gdf.to_crs('EPSG:4326')

    # Calculate center
    center_lat = stations_gdf.geometry.y.mean()
    center_lon = stations_gdf.geometry.x.mean()

    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=MAP_ZOOM, tiles=MAP_TILE)

    # Add markers
    for idx, row in stations_gdf.iterrows():
        popup_text = f"Station: {row.get('indroea', 'N/A')}"
        if value_col and value_col in row:
            popup_text += f"<br>{value_col}: {row[value_col]:.2f}"

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            popup=popup_text,
            color='blue',
            fill=True,
            fillColor='lightblue',
            fillOpacity=0.7
        ).add_to(m)

    return m


def create_cluster_plot(df, x_col, y_col, cluster_col='cluster', title="Cluster Analysis"):
    """
    Create scatter plot colored by cluster

    Args:
        df: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        cluster_col: Column with cluster labels
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.scatter(df, x=x_col, y=y_col, color=cluster_col,
                     title=title, color_discrete_sequence=px.colors.qualitative.Set1)

    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        template='plotly_white'
    )

    return fig


def display_metrics_row(metrics_dict):
    """
    Display metrics in a row of columns

    Args:
        metrics_dict: Dictionary with metric_name: (value, label, format)
    """
    cols = st.columns(len(metrics_dict))

    for col, (metric_name, (value, label, fmt)) in zip(cols, metrics_dict.items()):
        with col:
            if fmt == 'int':
                formatted_value = f"{int(value):,}"
            elif fmt == 'float':
                formatted_value = f"{float(value):.2f}"
            elif fmt == 'percent':
                formatted_value = f"{float(value):.1f}%"
            else:
                formatted_value = str(value)

            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
            """, unsafe_allow_html=True)


def create_comparison_plot(df1, df2, column, labels, title="Comparison"):
    """
    Create comparison plot for two datasets

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        column: Column to compare
        labels: Labels for datasets
        title: Plot title

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=df1[column], name=labels[0], opacity=0.7))
    fig.add_trace(go.Histogram(x=df2[column], name=labels[1], opacity=0.7))

    fig.update_layout(
        title=title,
        xaxis_title=column.replace('_', ' ').title(),
        yaxis_title='Frequency',
        barmode='overlay',
        template='plotly_white'
    )

    return fig
