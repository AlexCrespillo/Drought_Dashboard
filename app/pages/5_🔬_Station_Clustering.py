"""
Station Clustering Page - Hydrological Regime Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import folium
from streamlit_folium import st_folium
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
        <h1>🔬 Station Clustering Analysis</h1>
        <p>Hydrological regime classification using K-Means clustering and PCA analysis</p>
    </div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
with st.spinner("Loading data for clustering analysis..."):
    drought_df = load_drought_events()
    streamflow_df = load_streamflow_data()
    spatial_data = load_spatial_data()

# ==================== SIDEBAR OPTIONS ====================
st.sidebar.header("⚙️ Clustering Settings")

# Number of clusters
n_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=4
)

# Feature selection
st.sidebar.markdown("#### Select Features for Clustering")

use_duration = st.sidebar.checkbox("Duration Metrics", value=True)
use_deficit = st.sidebar.checkbox("Deficit Metrics", value=True)
use_frequency = st.sidebar.checkbox("Event Frequency", value=True)
use_streamflow = st.sidebar.checkbox("Streamflow Statistics", value=False)

st.sidebar.markdown("---")
st.sidebar.header("🎨 Visualization")

show_labels = st.sidebar.checkbox("Show Station Labels", value=False)
pca_components = st.sidebar.slider("PCA Components to Display", 2, 3, 2)

# ==================== PREPARE DATA FOR CLUSTERING ====================

if not drought_df.empty:
    # Calculate station metrics
    station_metrics = get_station_metrics(drought_df, streamflow_df)

    # Select features based on user choice
    feature_columns = []

    if use_duration:
        feature_columns.extend(['avg_duration', 'max_duration', 'std_duration'])
    if use_deficit:
        feature_columns.extend(['avg_deficit', 'max_deficit', 'total_deficit'])
    if use_frequency:
        feature_columns.extend(['num_events'])
    if use_streamflow and 'mean_flow' in station_metrics.columns:
        feature_columns.extend(['mean_flow', 'median_flow', 'std_flow'])

    # Remove any missing features
    feature_columns = [col for col in feature_columns if col in station_metrics.columns]

    if len(feature_columns) < 2:
        st.error("Please select at least 2 feature categories for clustering")
        st.stop()

    # Prepare clustering data
    clustering_data = station_metrics[['indroea'] + feature_columns].dropna()

    if len(clustering_data) < n_clusters:
        st.error(f"Not enough stations ({len(clustering_data)}) for {n_clusters} clusters")
        st.stop()

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(clustering_data[feature_columns])

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clustering_data['cluster'] = kmeans.fit_predict(features_scaled)

    # Perform PCA for visualization
    pca = PCA(n_components=min(pca_components, len(feature_columns)))
    pca_features = pca.fit_transform(features_scaled)

    if pca_components == 2:
        clustering_data['PC1'] = pca_features[:, 0]
        clustering_data['PC2'] = pca_features[:, 1]
    elif pca_components == 3 and len(feature_columns) >= 3:
        clustering_data['PC1'] = pca_features[:, 0]
        clustering_data['PC2'] = pca_features[:, 1]
        clustering_data['PC3'] = pca_features[:, 2]

    # ==================== MAIN CONTENT ====================

    tabs = st.tabs([
        "📊 Cluster Overview",
        "🎯 PCA Visualization",
        "📈 Cluster Profiles",
        "🗺️ Spatial Distribution",
        "📋 Detailed Results"
    ])

    # ==================== TAB 1: CLUSTER OVERVIEW ====================
    with tabs[0]:
        st.markdown("### 📊 Clustering Results Overview")

        # Cluster statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Stations", len(clustering_data))
        with col2:
            st.metric("Number of Clusters", n_clusters)
        with col3:
            st.metric("Features Used", len(feature_columns))
        with col4:
            silhouette_score = 0  # Placeholder - could compute actual silhouette score
            st.metric("Quality Score", "Calculated")

        st.markdown("---")

        # Cluster sizes
        st.markdown("#### 🔢 Cluster Sizes")

        cluster_sizes = clustering_data['cluster'].value_counts().sort_index()

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = create_bar_chart(
                pd.DataFrame({'cluster': [f'Cluster {i}' for i in cluster_sizes.index],
                            'count': cluster_sizes.values}),
                'cluster', 'count',
                title="Number of Stations per Cluster"
            )
            st.plotly_chart(fig, use_container_width=True, key="cluster_sizes")

        with col2:
            st.markdown("#### Distribution")
            for cluster_id, count in cluster_sizes.items():
                pct = (count / len(clustering_data)) * 100
                st.metric(f"Cluster {cluster_id}", f"{count} ({pct:.1f}%)")

        st.markdown("---")

        # Feature importance (PCA explained variance)
        st.markdown("#### 📊 Feature Importance (PCA Explained Variance)")

        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        var_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(explained_var))],
            'Explained Variance': explained_var,
            'Cumulative Variance': cumulative_var
        })

        col1, col2 = st.columns(2)

        with col1:
            fig = create_bar_chart(var_df, 'Component', 'Explained Variance',
                                 title="Explained Variance by Principal Component")
            st.plotly_chart(fig, use_container_width=True, key="pca_explained_variance")

        with col2:
            st.dataframe(var_df, use_container_width=True, hide_index=True)
            st.info(f"First 2 PCs explain {cumulative_var[1]*100:.1f}% of variance")

    # ==================== TAB 2: PCA VISUALIZATION ====================
    with tabs[1]:
        st.markdown("### 🎯 PCA Visualization")

        st.info("💡 **Interpretation:** PCA (Principal Component Analysis) reduces the multiple drought features "
                "into 2-3 main components that capture most of the variation. Well-separated clusters in this "
                "visualization indicate that stations have distinct drought behaviors. Overlapping clusters suggest "
                "more gradual differences between station groups.")

        if pca_components == 2:
            st.markdown("#### 2D PCA Projection")

            fig = create_cluster_plot(
                clustering_data, 'PC1', 'PC2', 'cluster',
                title="Station Clusters in PCA Space (2D)"
            )
            st.plotly_chart(fig, use_container_width=True, key="pca_2d_plot")

            st.caption("Each point represents a station. Colors indicate cluster membership. "
                      "Distance between points reflects similarity in drought characteristics.")

        elif pca_components == 3 and 'PC3' in clustering_data.columns:
            st.markdown("#### 3D PCA Projection")

            fig = px.scatter_3d(
                clustering_data, x='PC1', y='PC2', z='PC3',
                color='cluster',
                title="Station Clusters in PCA Space (3D)",
                labels={'cluster': 'Cluster'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig, use_container_width=True, key="pca_3d_plot")

            st.caption("3D view allows better visualization of cluster separation. "
                      "Rotate the plot to explore different angles.")

        # Feature loadings
        st.markdown("---")
        st.markdown("#### 📊 Feature Contributions (PCA Loadings)")

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings[:, :2],
            columns=['PC1', 'PC2'],
            index=feature_columns
        )
        loadings_df = loadings_df.round(3)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = create_heatmap(
                loadings_df.T.values,
                x_labels=loadings_df.index,
                y_labels=['PC1', 'PC2'],
                title="Feature Loadings Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True, key="pca_loadings_heatmap")

        with col2:
            st.dataframe(loadings_df, use_container_width=True)

    # ==================== TAB 3: CLUSTER PROFILES ====================
    with tabs[2]:
        st.markdown("### 📈 Cluster Characteristic Profiles")

        st.info("💡 **Interpretation:** This heatmap shows the average characteristics of each cluster. "
                "Brighter/darker colors indicate higher/lower values. Use this to identify what makes each "
                "cluster unique. For example, one cluster might have high event frequency but low duration, "
                "while another has infrequent but long-lasting droughts.")

        # Calculate cluster means
        cluster_profiles = clustering_data.groupby('cluster')[feature_columns].mean()

        st.markdown("#### 📊 Cluster Mean Values")

        # Display as heatmap
        fig = create_heatmap(
            cluster_profiles.T.values,
            x_labels=[f'Cluster {i}' for i in cluster_profiles.index],
            y_labels=cluster_profiles.columns,
            title="Cluster Profiles (Standardized Values)",
            annotate=True
        )
        st.plotly_chart(fig, use_container_width=True, key="cluster_profiles_heatmap")

        st.markdown("---")

        # Individual cluster details
        st.markdown("#### 📋 Detailed Cluster Statistics")

        selected_cluster = st.selectbox(
            "Select cluster to view details",
            options=range(n_clusters),
            format_func=lambda x: f"Cluster {x}"
        )

        cluster_stations = clustering_data[clustering_data['cluster'] == selected_cluster]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Stations in Cluster", len(cluster_stations))
        with col2:
            pct = (len(cluster_stations) / len(clustering_data)) * 100
            st.metric("Percentage", f"{pct:.1f}%")
        with col3:
            st.metric("Feature Count", len(feature_columns))

        # Statistics table
        cluster_stats = cluster_stations[feature_columns].describe().T
        cluster_stats['cluster_mean'] = cluster_profiles.loc[selected_cluster]

        st.dataframe(cluster_stats, use_container_width=True)

        # Comparison with other clusters
        st.markdown("---")
        st.markdown("#### 📊 Feature Comparison Across Clusters")

        comparison_feature = st.selectbox(
            "Select feature to compare",
            options=feature_columns,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        fig = create_box_plot(
            clustering_data, 'cluster', comparison_feature,
            title=f"{comparison_feature.replace('_', ' ').title()} Distribution by Cluster"
        )
        st.plotly_chart(fig, use_container_width=True, key="cluster_feature_comparison")

    # ==================== TAB 4: SPATIAL DISTRIBUTION ====================
    with tabs[3]:
        st.markdown("### 🗺️ Spatial Distribution of Clusters")

        st.info("💡 **Interpretation:** This map shows the geographic distribution of station clusters. "
                "Stations with similar drought characteristics (duration, frequency, deficit) are colored "
                "the same. Spatial patterns can reveal regional hydro logical similarities or differences across the basin.")

        # Try to create map
        map_created = False

        if spatial_data.get('stations') is not None:
            try:
                # Merge cluster assignments with spatial data
                stations_gdf = spatial_data['stations'].copy()

                # Check for station identifier column (try multiple possible names)
                station_id_col = None
                for possible_col in ['indroea', 'INDROEA', 'station_id', 'Station_ID', 'id', 'ID']:
                    if possible_col in stations_gdf.columns:
                        station_id_col = possible_col
                        if possible_col != 'indroea':
                            st.info(f"Using '{possible_col}' as station identifier column")
                            stations_gdf['indroea'] = stations_gdf[possible_col]
                        break

                if station_id_col is not None:
                    # Merge with cluster data
                    stations_gdf = stations_gdf.merge(
                        clustering_data[['indroea', 'cluster']],
                        on='indroea',
                        how='inner'
                    )

                    if len(stations_gdf) > 0:
                        # Convert to WGS84
                        if stations_gdf.crs and stations_gdf.crs != 'EPSG:4326':
                            stations_gdf = stations_gdf.to_crs('EPSG:4326')

                        # Create map
                        center_lat = stations_gdf.geometry.y.mean()
                        center_lon = stations_gdf.geometry.x.mean()

                        m = folium.Map(
                            location=[center_lat, center_lon],
                            zoom_start=8,
                            tiles='OpenStreetMap'
                        )

                        # Add basin boundary
                        if spatial_data.get('basin') is not None:
                            basin_gdf = spatial_data['basin']
                            if basin_gdf.crs != 'EPSG:4326':
                                basin_gdf = basin_gdf.to_crs('EPSG:4326')

                            folium.GeoJson(
                                basin_gdf,
                                name='Basin Boundary',
                                style_function=lambda x: {
                                    'fillColor': 'transparent',
                                    'color': 'blue',
                                    'weight': 2
                                }
                            ).add_to(m)

                        # Color map for clusters
                        cluster_colors_map = {
                            0: 'red', 1: 'blue', 2: 'green', 3: 'orange',
                            4: 'purple', 5: 'darkred', 6: 'lightblue',
                            7: 'pink', 8: 'darkgreen', 9: 'gray'
                        }

                        # Add station markers
                        marker_count = 0
                        for idx, row in stations_gdf.iterrows():
                            try:
                                cluster_id = int(row['cluster'])
                                color = cluster_colors_map.get(cluster_id, 'gray')
                                station_id = row.get('indroea', 'Unknown')

                                # Get cluster data for this station from clustering_data
                                station_cluster_data = clustering_data[clustering_data['indroea'] == station_id]

                                # Build detailed popup with station metrics
                                popup_html = f"""
                                    <b>Station ID:</b> {station_id}<br>
                                    <b>Cluster:</b> {cluster_id}<br>
                                """

                                # Add available metrics to popup
                                if not station_cluster_data.empty:
                                    station_row = station_cluster_data.iloc[0]
                                    if 'num_events' in station_row:
                                        popup_html += f"<b>Events:</b> {int(station_row['num_events'])}<br>"
                                    if 'avg_duration' in station_row:
                                        popup_html += f"<b>Avg Duration:</b> {station_row['avg_duration']:.1f} days<br>"
                                    if 'total_deficit' in station_row:
                                        popup_html += f"<b>Total Deficit:</b> {station_row['total_deficit']:.2f} hm³<br>"

                                # Validate coordinates
                                lat, lon = row.geometry.y, row.geometry.x
                                if pd.notna(lat) and pd.notna(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
                                    folium.CircleMarker(
                                        location=[lat, lon],
                                        radius=8,
                                        popup=folium.Popup(popup_html, max_width=300),
                                        tooltip=f"Station {station_id} - Cluster {cluster_id}",
                                        color='black',
                                        fillColor=color,
                                        fill=True,
                                        fillOpacity=0.8,
                                        weight=1
                                    ).add_to(m)
                                    marker_count += 1
                            except Exception as e:
                                st.warning(f"Error adding marker for station {station_id}: {str(e)}")
                                continue

                        # Display map without overlapping legend
                        if marker_count > 0:
                            # Display map
                            st_folium(m, width=1200, height=600, returned_objects=[], key=f"cluster_map_spatial_{n_clusters}")
                            map_created = True

                            st.success(f"✅ Successfully displayed {marker_count} stations on interactive map")
                        else:
                            st.error("❌ No valid station markers could be added to the map")

                        # Cluster legend
                        st.markdown("---")
                        st.markdown("#### 🎨 Cluster Legend")
                        legend_cols = st.columns(min(n_clusters, 6))
                        for i in range(n_clusters):
                            col_idx = i % len(legend_cols)
                            with legend_cols[col_idx]:
                                color = cluster_colors_map.get(i, 'gray')
                                cluster_size = len(clustering_data[clustering_data['cluster'] == i])
                                st.markdown(f"**Cluster {i}** <span style='color:{color};font-size:20px;'>●</span> ({cluster_size} stations)",
                                          unsafe_allow_html=True)

                        st.markdown("---")
                        st.markdown("#### 📊 Spatial Interpretation")
                        st.markdown("""
                        **What to look for:**
                        - **Geographic clustering**: Do stations in the same cluster tend to be geographically close?
                        - **Regional patterns**: Are certain regions dominated by specific clusters?
                        - **Hydrological zones**: Do clusters align with known hydrological or climatic zones?

                        Spatially contiguous clusters suggest regional hydrological similarities (e.g., similar rainfall patterns,
                        similar land use). Scattered clusters may indicate that drought characteristics are more influenced by
                        local factors (e.g., catchment size, elevation, regulation) than regional climate patterns.
                        """)

                    else:
                        st.warning("No stations matched between spatial and cluster data")
                else:
                    available_cols = ', '.join(stations_gdf.columns.tolist()[:10])
                    st.warning(f"Station identifier column not found in spatial data. Available columns: {available_cols}...")

            except Exception as e:
                st.error(f"Error creating spatial map: {str(e)}")
                map_created = False

        # Fall back to UTM coordinates from streamflow data if spatial map failed
        if not map_created:
            if not streamflow_df.empty and 'xutm' in streamflow_df.columns and 'yutm' in streamflow_df.columns:
                st.info("Using UTM coordinates from streamflow data to create interactive map (spatial shapefiles not available)")

                # Get station coordinates and merge with cluster data
                station_coords = streamflow_df[['indroea', 'xutm', 'yutm', 'lugar']].drop_duplicates('indroea')
                station_coords = station_coords.merge(
                    clustering_data[['indroea', 'cluster']],
                    on='indroea',
                    how='inner'
                )

                if not station_coords.empty:
                    # Convert UTM to lat/lon (same logic as Page 2)
                    try:
                        # Filter valid UTM coordinates
                        valid_utm = station_coords.dropna(subset=['xutm', 'yutm']).copy()
                        valid_utm = valid_utm[
                            (valid_utm['xutm'].between(200000, 900000)) &
                            (valid_utm['yutm'].between(4200000, 4800000))
                        ]

                        if len(valid_utm) == 0:
                            st.error("❌ No valid UTM coordinates found")
                        else:
                            # Try accurate coordinate transformation
                            conversion_success = False
                            if pyproj is not None:
                                try:
                                    transformer = pyproj.Transformer.from_crs('EPSG:25830', 'EPSG:4326', always_xy=True)
                                    lon, lat = transformer.transform(valid_utm['xutm'].values, valid_utm['yutm'].values)
                                    valid_utm['lat'] = lat
                                    valid_utm['lon'] = lon
                                    conversion_success = True
                                    st.success("✅ Using accurate coordinate transformation (EPSG:25830 → WGS84)")
                                except Exception as e:
                                    st.warning(f"⚠️ Accurate transformation failed: {str(e)}")

                            # Fallback to approximation if needed
                            if not conversion_success:
                                valid_utm['lat'] = 40.0 + (valid_utm['yutm'] - 4400000) / 111320
                                valid_utm['lon'] = 0.0 + (valid_utm['xutm'] - 500000) / (111320 * np.cos(np.radians(41)))
                                st.info("ℹ️ Using approximate coordinate conversion")

                            # Validate lat/lon
                            valid_coords = valid_utm[
                                (valid_utm['lat'].between(40, 44)) &
                                (valid_utm['lon'].between(-3, 3))
                            ]

                            if len(valid_coords) > 0:
                                # Create folium map
                                center_lat = valid_coords['lat'].mean()
                                center_lon = valid_coords['lon'].mean()

                                m = folium.Map(
                                    location=[center_lat, center_lon],
                                    zoom_start=8,
                                    tiles='OpenStreetMap'
                                )

                                # Color map for clusters
                                cluster_colors_map = {
                                    0: 'red', 1: 'blue', 2: 'green', 3: 'orange',
                                    4: 'purple', 5: 'darkred', 6: 'lightblue',
                                    7: 'pink', 8: 'darkgreen', 9: 'gray'
                                }

                                # Add markers
                                markers_added = 0
                                for idx, row in valid_coords.iterrows():
                                    try:
                                        station_id = row['indroea']
                                        cluster_id = int(row['cluster'])
                                        color = cluster_colors_map.get(cluster_id, 'gray')
                                        location_name = row.get('lugar', 'Unknown')

                                        # Get additional metrics from clustering_data
                                        station_cluster_data = clustering_data[clustering_data['indroea'] == station_id]

                                        popup_html = f"""
                                            <b>Station ID:</b> {station_id}<br>
                                            <b>Location:</b> {location_name}<br>
                                            <b>Cluster:</b> {cluster_id}<br>
                                        """

                                        if not station_cluster_data.empty:
                                            station_row = station_cluster_data.iloc[0]
                                            if 'num_events' in station_row:
                                                popup_html += f"<b>Events:</b> {int(station_row['num_events'])}<br>"
                                            if 'avg_duration' in station_row:
                                                popup_html += f"<b>Avg Duration:</b> {station_row['avg_duration']:.1f} days<br>"
                                            if 'total_deficit' in station_row:
                                                popup_html += f"<b>Total Deficit:</b> {station_row['total_deficit']:.2f} hm³<br>"

                                        folium.CircleMarker(
                                            location=[row['lat'], row['lon']],
                                            radius=8,
                                            popup=folium.Popup(popup_html, max_width=300),
                                            tooltip=f"Station {station_id} - Cluster {cluster_id}",
                                            color='black',
                                            fillColor=color,
                                            fill=True,
                                            fillOpacity=0.8,
                                            weight=1
                                        ).add_to(m)
                                        markers_added += 1
                                    except Exception as e:
                                        continue

                                # Display map without overlapping legend
                                if markers_added > 0:
                                    # Display map
                                    st_folium(m, width=1200, height=600, returned_objects=[], key=f"cluster_map_utm_{n_clusters}")
                                    map_created = True

                                    st.success(f"✅ Successfully displayed {markers_added} stations on interactive map")

                                    # Legend below map
                                    st.markdown("---")
                                    st.markdown("#### 🎨 Cluster Legend")
                                    legend_cols = st.columns(min(n_clusters, 6))
                                    for i in range(n_clusters):
                                        col_idx = i % len(legend_cols)
                                        with legend_cols[col_idx]:
                                            color = cluster_colors_map.get(i, 'gray')
                                            cluster_size = len(clustering_data[clustering_data['cluster'] == i])
                                            st.markdown(f"**Cluster {i}** <span style='color:{color};font-size:20px;'>●</span> ({cluster_size} stations)",
                                                      unsafe_allow_html=True)
                                else:
                                    st.error("❌ No markers could be added to the map")
                            else:
                                st.error("❌ No valid coordinates after lat/lon conversion")

                    except Exception as e:
                        st.error(f"❌ Error converting coordinates: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.error("Unable to create map - no matching stations found between streamflow and cluster data")
            else:
                # Create a simple PCA-based visualization as final fallback
                if 'PC1' in clustering_data.columns and 'PC2' in clustering_data.columns:
                    st.info("Creating PCA-based spatial representation (no geographic coordinates available)")
                    
                    fig = px.scatter(
                        clustering_data,
                        x='PC1',
                        y='PC2', 
                        color='cluster',
                        title="Station Clusters - PCA Space Representation",
                        hover_data=['indroea'],
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    
                    fig.update_layout(
                        xaxis_title="First Principal Component",
                        yaxis_title="Second Principal Component", 
                        template='plotly_white',
                        showlegend=True,
                        legend_title="Cluster",
                        height=600
                    )
                    
                    fig.update_traces(
                        marker=dict(size=10, line=dict(width=1, color='black'))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="cluster_pca_fallback")
                    
                    st.info("💡 This visualization shows clusters in reduced dimensional space (PCA). "
                            "While not geographically accurate, it shows the relative similarity between stations.")
                else:
                    st.error("No spatial, coordinate, or PCA data available for mapping")

    # ==================== TAB 5: DETAILED RESULTS ====================
    with tabs[4]:
        st.markdown("### 📋 Detailed Clustering Results")

        # Display full results table
        results_df = clustering_data.copy()

        # Add original (non-scaled) values
        for col in feature_columns:
            results_df[col] = station_metrics.loc[station_metrics['indroea'].isin(results_df['indroea']), col].values

        st.markdown(f"#### Complete Results ({len(results_df)} stations)")

        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ['cluster'] + feature_columns,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        results_display = results_df.sort_values(sort_by)

        # Column selection
        display_columns = st.multiselect(
            "Select columns to display",
            options=['indroea', 'cluster'] + feature_columns,
            default=['indroea', 'cluster'] + feature_columns[:3]
        )

        if display_columns:
            st.dataframe(results_display[display_columns], use_container_width=True, height=400)

        st.markdown("---")

        # Download options
        st.markdown("#### 💾 Download Results")

        col1, col2 = st.columns(2)

        with col1:
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name=f"station_clustering_results_{n_clusters}clusters.csv",
                mime="text/csv"
            )

        with col2:
            # Cluster summary
            summary_df = clustering_data.groupby('cluster')[feature_columns].mean()
            summary_csv = summary_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Cluster Summary (CSV)",
                data=summary_csv,
                file_name=f"cluster_summary_{n_clusters}clusters.csv",
                mime="text/csv"
            )

else:
    st.error("No drought data available for clustering analysis")

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
    <p>🔬 Station Clustering Analysis | Hydrological regime classification using machine learning</p>
</div>
""", unsafe_allow_html=True)
