"""
Data loading and preprocessing utilities for Drought Dashboard
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import streamlit as st
from pathlib import Path
import sys

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import *


@st.cache_data(ttl=CACHE_TTL)
def load_drought_events():
    """
    Load drought events dataset

    Returns:
        pd.DataFrame: Drought events data with columns:
            - indroea: station identifier
            - event_id: unique event ID
            - inicio: start date
            - fin: end date
            - duracion: duration in days
            - deficit_hm3: water deficit in hm³
    """
    try:
        df = pd.read_csv(DROUGHT_EVENTS_FILE, parse_dates=['inicio', 'fin'])

        # Add derived columns
        if 'year' not in df.columns:
            df['year'] = df['inicio'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['inicio'].dt.month
        if 'season' not in df.columns:
            df['season'] = df['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })

        return df
    except FileNotFoundError:
        st.error(f"Drought events file not found: {DROUGHT_EVENTS_FILE}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading drought events: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_streamflow_data():
    """
    Load daily streamflow dataset

    Returns:
        pd.DataFrame: Streamflow data with columns:
            - indroea: station identifier
            - fecha: date
            - caudal: streamflow (m³/s)
            - lugar: station name
            - xutm: UTM X coordinate
            - yutm: UTM Y coordinate
    """
    try:
        df = pd.read_csv(STREAMFLOW_FILE, parse_dates=['fecha'])

        # Add derived columns
        if 'year' not in df.columns:
            df['year'] = df['fecha'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['fecha'].dt.month
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = df['fecha'].dt.dayofyear

        return df
    except FileNotFoundError:
        st.error(f"Streamflow file not found: {STREAMFLOW_FILE}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading streamflow data: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_spatial_data():
    """
    Load all geospatial datasets

    Returns:
        dict: Dictionary containing:
            - basin: Basin boundary GeoDataFrame
            - rivers: River network GeoDataFrame
            - stations: Gauging stations GeoDataFrame
            - reservoirs: Reservoirs GeoDataFrame
            - dem_data: DEM array
            - dem_meta: DEM metadata
    """
    spatial_data = {}

    try:
        # Load basin boundary
        if BASIN_BOUNDARY.exists():
            spatial_data['basin'] = gpd.read_file(BASIN_BOUNDARY)
        else:
            st.warning(f"Basin boundary file not found: {BASIN_BOUNDARY}")
            spatial_data['basin'] = None

        # Load river network
        if RIVER_NETWORK.exists():
            spatial_data['rivers'] = gpd.read_file(RIVER_NETWORK)
        else:
            st.warning(f"River network file not found: {RIVER_NETWORK}")
            spatial_data['rivers'] = None

        # Load stations
        if STATIONS_FILE.exists():
            spatial_data['stations'] = gpd.read_file(STATIONS_FILE)
        else:
            st.warning(f"Stations file not found: {STATIONS_FILE}")
            spatial_data['stations'] = None

        # Load reservoirs
        if RESERVOIRS_FILE.exists():
            spatial_data['reservoirs'] = gpd.read_file(RESERVOIRS_FILE)
        else:
            st.warning(f"Reservoirs file not found: {RESERVOIRS_FILE}")
            spatial_data['reservoirs'] = None

        # Load DEM
        if DEM_FILE.exists():
            with rasterio.open(DEM_FILE) as src:
                spatial_data['dem_data'] = src.read(1)
                spatial_data['dem_meta'] = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'bounds': src.bounds,
                    'height': src.height,
                    'width': src.width
                }
        else:
            st.warning(f"DEM file not found: {DEM_FILE}")
            spatial_data['dem_data'] = None
            spatial_data['dem_meta'] = None

    except Exception as e:
        st.error(f"Error loading spatial data: {str(e)}")

    return spatial_data


def get_station_list(df):
    """
    Get list of unique stations from dataframe

    Args:
        df: DataFrame with 'indroea' column

    Returns:
        list: Sorted list of unique station IDs
    """
    if df is not None and 'indroea' in df.columns:
        return sorted(df['indroea'].unique())
    return []


def filter_by_date_range(df, date_column, start_date, end_date):
    """
    Filter dataframe by date range

    Args:
        df: DataFrame to filter
        date_column: Name of date column
        start_date: Start date
        end_date: End date

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df is None or df.empty:
        return df

    mask = (df[date_column] >= pd.Timestamp(start_date)) & \
           (df[date_column] <= pd.Timestamp(end_date))
    return df[mask]


def filter_by_stations(df, stations):
    """
    Filter dataframe by station list

    Args:
        df: DataFrame to filter
        stations: List of station IDs

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if df is None or df.empty or not stations:
        return df

    return df[df['indroea'].isin(stations)]


def calculate_drought_statistics(df):
    """
    Calculate summary statistics for drought events

    Args:
        df: Drought events dataframe

    Returns:
        dict: Dictionary of statistics
    """
    if df is None or df.empty:
        return {}

    stats = {
        'total_events': len(df),
        'total_stations': df['indroea'].nunique(),
        'avg_duration': df['duracion'].mean(),
        'max_duration': df['duracion'].max(),
        'avg_deficit': df['deficit_hm3'].mean(),
        'max_deficit': df['deficit_hm3'].max(),
        'total_deficit': df['deficit_hm3'].sum(),
        'events_per_station': len(df) / df['indroea'].nunique(),
        'years_covered': df['year'].nunique() if 'year' in df.columns else 0
    }

    return stats


def calculate_streamflow_statistics(df):
    """
    Calculate summary statistics for streamflow data

    Args:
        df: Streamflow dataframe

    Returns:
        dict: Dictionary of statistics
    """
    if df is None or df.empty:
        return {}

    stats = {
        'total_records': len(df),
        'total_stations': df['indroea'].nunique(),
        'date_range_start': df['fecha'].min(),
        'date_range_end': df['fecha'].max(),
        'avg_streamflow': df['caudal'].mean(),
        'median_streamflow': df['caudal'].median(),
        'max_streamflow': df['caudal'].max(),
        'min_streamflow': df['caudal'].min(),
        'std_streamflow': df['caudal'].std()
    }

    return stats


def get_station_metrics(drought_df, streamflow_df):
    """
    Calculate metrics for each station for clustering analysis

    Args:
        drought_df: Drought events dataframe
        streamflow_df: Streamflow dataframe

    Returns:
        pd.DataFrame: Station metrics
    """
    if drought_df is None or drought_df.empty:
        return pd.DataFrame()

    # Drought metrics by station
    drought_metrics = drought_df.groupby('indroea').agg({
        'duracion': ['mean', 'max', 'std', 'count'],
        'deficit_hm3': ['mean', 'max', 'sum'],
        'year': ['min', 'max']
    }).reset_index()

    drought_metrics.columns = ['indroea', 'avg_duration', 'max_duration',
                                'std_duration', 'num_events', 'avg_deficit',
                                'max_deficit', 'total_deficit', 'first_year', 'last_year']

    # Add streamflow metrics if available
    if streamflow_df is not None and not streamflow_df.empty:
        flow_metrics = streamflow_df.groupby('indroea')['caudal'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).reset_index()
        flow_metrics.columns = ['indroea', 'mean_flow', 'median_flow',
                                'std_flow', 'min_flow', 'max_flow']

        drought_metrics = drought_metrics.merge(flow_metrics, on='indroea', how='left')

    return drought_metrics


def prepare_temporal_analysis_data(df, temporal_aggregation='year'):
    """
    Prepare data for temporal analysis

    Args:
        df: Drought events dataframe
        temporal_aggregation: 'year', 'month', 'season'

    Returns:
        pd.DataFrame: Aggregated temporal data
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if temporal_aggregation == 'year':
        agg_df = df.groupby('year').agg({
            'event_id': 'count',
            'duracion': ['mean', 'max', 'sum'],
            'deficit_hm3': ['mean', 'max', 'sum']
        }).reset_index()

    elif temporal_aggregation == 'month':
        agg_df = df.groupby(['year', 'month']).agg({
            'event_id': 'count',
            'duracion': 'mean',
            'deficit_hm3': 'sum'
        }).reset_index()

    elif temporal_aggregation == 'season':
        agg_df = df.groupby(['year', 'season']).agg({
            'event_id': 'count',
            'duracion': 'mean',
            'deficit_hm3': 'sum'
        }).reset_index()

    return agg_df
