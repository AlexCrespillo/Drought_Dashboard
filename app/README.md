# 🌊 Drought Events Analysis Dashboard

Interactive web application for comprehensive analysis of drought events and streamflow patterns in the Ebro River Basin.

## 📋 Overview

This Streamlit application provides an interactive interface to explore and analyze drought events based on extensive EDA performed on daily streamflow data. The dashboard includes multiple analytical modules covering temporal trends, spatial patterns, drought characteristics, and station clustering.

## 🚀 Quick Start

### Installation

1. **Navigate to the app directory:**
   ```bash
   cd app
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard:**
   Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
app/
├── app.py                          # Main application entry point
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── pages/                          # Multi-page application modules
│   ├── 1_📊_Data_Overview.py      # Dataset exploration and quality metrics
│   ├── 2_🗺️_Spatial_Analysis.py   # Geographic distribution and mapping
│   ├── 3_📈_Temporal_Analysis.py   # Temporal trends and patterns
│   ├── 4_🌊_Drought_Analysis.py    # Drought characteristics analysis
│   └── 5_🔬_Station_Clustering.py  # K-Means clustering and PCA
│
└── utils/                          # Utility modules
    ├── __init__.py
    ├── data_loader.py              # Data loading and preprocessing
    └── visualization.py            # Plotting and visualization functions
```

## 📊 Features

### 1. **Data Overview** 📊
- Comprehensive dataset statistics
- Data quality assessment
- Missing value analysis
- Interactive data filtering
- Export capabilities

### 2. **Spatial Analysis** 🗺️
- Interactive maps with basin boundaries
- Station distribution visualization
- River network overlay
- Reservoir locations
- Spatial statistics by station

### 3. **Temporal Analysis** 📈
- Annual drought trends
- Seasonal pattern analysis
- Monthly heatmaps
- Decadal comparisons
- Statistical trend tests (Linear regression, Mann-Kendall)

### 4. **Drought Analysis** 🌊
- Event duration analysis
- Water deficit quantification
- Duration-deficit correlations
- Severity classification system
- Top extreme events identification

### 5. **Station Clustering** 🔬
- K-Means clustering (2-10 clusters)
- PCA visualization (2D/3D)
- Cluster profiling and comparison
- Spatial distribution of clusters
- Feature importance analysis

## ⚙️ Configuration

### Data Paths
Edit `config.py` to update data file paths:

```python
# Data file paths
DROUGHT_EVENTS_FILE = DATA_DIR / "sequias_completo.csv"
STREAMFLOW_FILE = DATA_DIR / "caudales_completo.csv"

# Spatial data paths
BASIN_BOUNDARY = SPATIAL_DIR / "Limite_Cuenca_Ebro.shp"
RIVER_NETWORK = SPATIAL_DIR / "Redlinealgeo.shp"
STATIONS_FILE = SPATIAL_DIR / "Estaciones_bueno.shp"
```

### Analysis Parameters
Customize drought detection parameters in `config.py`:

```python
# Drought detection parameters
DROUGHT_PERCENTILE_THRESHOLD = 5  # P5
MIN_DROUGHT_DURATION = 5  # days
POOLING_WINDOW = 10  # days
```

### Visualization Settings
Adjust plot styling and colors in `config.py`:

```python
# Color schemes
DROUGHT_COLOR = "#d62728"
NORMAL_COLOR = "#2ca02c"
WARNING_COLOR = "#ff7f0e"

# Plot styling
DEFAULT_FIGSIZE = (14, 8)
FONT_SIZE = 11
```

## 🎯 Usage Guide

### Sidebar Navigation
- Use the sidebar to navigate between different analysis pages
- Each page has specific filters and settings
- Filters apply dynamically to visualizations

### Interactive Filters
Common filters across pages:
- **Year Range**: Filter data by temporal period
- **Station Selection**: Focus on specific stations
- **Duration/Deficit Thresholds**: Filter drought events by severity

### Data Export
Most pages include export functionality:
- Download filtered datasets as CSV
- Export analysis results and statistics
- Save clustering results with assignments

### Visualization Options
- All plots are interactive (Plotly-based)
- Zoom, pan, and hover for details
- Download plots as PNG images
- Toggle data series on/off

## 📦 Dependencies

Core libraries:
- **streamlit** >= 1.28.0 - Web application framework
- **pandas** >= 2.0.0 - Data manipulation
- **numpy** >= 1.24.0 - Numerical computing
- **matplotlib** >= 3.7.0 - Static plotting
- **seaborn** >= 0.12.0 - Statistical visualization
- **plotly** >= 5.14.0 - Interactive plotting
- **geopandas** >= 0.13.0 - Geospatial data
- **folium** >= 0.14.0 - Interactive maps
- **scikit-learn** >= 1.3.0 - Machine learning
- **scipy** >= 1.10.0 - Scientific computing

See `requirements.txt` for complete list.

## 🔧 Customization

### Adding New Pages
1. Create new Python file in `pages/` directory
2. Name format: `N_Icon_Page_Name.py` (e.g., `6_🚀_Forecasting.py`)
3. Streamlit automatically detects new pages
4. Use existing pages as templates

### Adding New Visualizations
Add custom visualization functions to `utils/visualization.py`:

```python
def create_custom_plot(df, x_col, y_col, **kwargs):
    """
    Custom visualization function
    """
    fig = px.scatter(df, x=x_col, y=y_col)
    # Customize as needed
    return fig
```

### Adding New Data Processing
Add utility functions to `utils/data_loader.py`:

```python
@st.cache_data(ttl=CACHE_TTL)
def load_custom_dataset():
    """
    Load custom dataset with caching
    """
    df = pd.read_csv("path/to/file.csv")
    return df
```

## 🐛 Troubleshooting

### Common Issues

**Issue: Data files not found**
```
Solution: Check paths in config.py and ensure data files exist
```

**Issue: Spatial data not loading**
```
Solution: Verify shapefile components (.shp, .shx, .dbf, .prj) are present
```

**Issue: Memory errors with large datasets**
```
Solution: Use filters to reduce data size or increase CACHE_TTL
```

**Issue: Plots not displaying**
```
Solution: Check browser console for errors, clear Streamlit cache
```

### Clear Cache
If experiencing data issues, clear Streamlit cache:
```bash
streamlit cache clear
```

## 🚀 Future Enhancements

Planned features (as mentioned in the EDA):
- 🤖 **Machine Learning Models**: Drought prediction using ML algorithms
- 📊 **Forecasting Module**: Time series forecasting for drought events
- 🔗 **Network Analysis**: Event propagation and connectivity analysis
- 📈 **Real-time Monitoring**: Integration with live data sources
- 📱 **Mobile Responsive**: Enhanced mobile device support

## 📚 Data Requirements

### Expected Data Format

**Drought Events CSV:**
```
indroea,event_id,inicio,fin,duracion,deficit_hm3
9001,1,1980-05-12,1980-06-20,39,15.2
...
```

**Streamflow Data CSV:**
```
indroea,fecha,caudal,lugar,xutm,yutm
9001,1980-01-01,25.3,Station Name,450000,4600000
...
```

## 📖 References

Drought detection methodology based on:
- Van Loon (2015) - Hydrological drought explained
- Tallaksen & Van Lanen (2004) - Hydrological Drought Processes
- Fleig et al. (2006) - Global evaluation of streamflow drought characteristics
- Smakhtin (2001) - Low flow hydrology review

## 📧 Support

For issues, questions, or contributions:
- Create an issue in the project repository
- Contact the development team
- Refer to Streamlit documentation: https://docs.streamlit.io

## 📄 License

This project is part of the Drought Events Analysis research initiative for the Ebro River Basin.

---

**Version:** 1.0
**Last Updated:** 2025-10-06
**Status:** Production Ready
