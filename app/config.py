"""
Configuration file for Drought Dashboard Application
"""

import os
from pathlib import Path

# ==================== PATH CONFIGURATION ====================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SPATIAL_DIR = DATA_DIR / "espacial"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Data file paths (relative to DATA_DIR)
DROUGHT_EVENTS_FILE = DATA_DIR / "csv" / "eventos_sequia_p5.csv"
STREAMFLOW_FILE = DATA_DIR / "csv" / "caudales_diarios.csv"

# Spatial data paths (relative to SPATIAL_DIR)
BASIN_BOUNDARY = SPATIAL_DIR / "Limite_Cuenca_Ebro.shp"
RIVER_NETWORK = SPATIAL_DIR / "Redlinealgeo.shp"
STATIONS_FILE = SPATIAL_DIR / "Estaciones_bueno.shp"
RESERVOIRS_FILE = SPATIAL_DIR / "embalses.shp"
DEM_FILE = SPATIAL_DIR / "Dem_bueno_fill.tif"

# ==================== APP CONFIGURATION ====================
APP_TITLE = "Drought Events Analysis Dashboard"
APP_SUBTITLE = "Ebro River Basin - Comprehensive Hydrological Assessment"
APP_ICON = "🌊"

# Page configuration
PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ==================== ANALYSIS PARAMETERS ====================
# Temporal coverage
START_YEAR = 1974
END_YEAR = 2021

# Drought detection parameters
DROUGHT_PERCENTILE_THRESHOLD = 5  # P5
MIN_DROUGHT_DURATION = 5  # days
POOLING_WINDOW = 10  # days

# ==================== VISUALIZATION SETTINGS ====================
# Enhanced color schemes
COLOR_PALETTE = "husl"
PRIMARY_COLOR = "#0066cc"
SECONDARY_COLOR = "#00a3e0"
DROUGHT_COLOR = "#d62728"
NORMAL_COLOR = "#2ca02c"
WARNING_COLOR = "#ff7f0e"
ACCENT_COLOR = "#9467bd"

# Plot styling
PLOT_STYLE = "seaborn-v0_8-whitegrid"
DEFAULT_FIGSIZE = (14, 8)
FONT_SIZE = 11

# Cluster colors for station analysis
CLUSTER_COLORS = {
    0: "#e74c3c",  # Red
    1: "#3498db",  # Blue
    2: "#2ecc71",  # Green
    3: "#f39c12",  # Orange
    4: "#9b59b6",  # Purple
    5: "#e67e22",  # Dark Orange
    6: "#1abc9c",  # Turquoise
    7: "#e91e63",  # Pink
    8: "#34495e",  # Dark Gray
    9: "#95a5a6",  # Light Gray
}

# ==================== DATA DISPLAY SETTINGS ====================
MAX_ROWS_DISPLAY = 20
DATE_FORMAT = "%Y-%m-%d"
FLOAT_FORMAT = "{:.2f}"

# ==================== CACHING SETTINGS ====================
CACHE_TTL = 3600  # 1 hour in seconds

# ==================== MAP SETTINGS ====================
MAP_CENTER = [41.5, -0.5]  # Approximate center of Ebro Basin
MAP_ZOOM = 7
MAP_TILE = "OpenStreetMap"

# ==================== FEATURE FLAGS ====================
ENABLE_ML_FEATURES = True
ENABLE_FORECASTING = True
ENABLE_NETWORK_ANALYSIS = True

# ==================== METADATA ====================
BASIN_INFO = {
    "name": "Ebro River Basin",
    "area_km2": 85362,
    "countries": ["Spain", "France", "Andorra"],
    "main_river": "Ebro River",
    "length_km": 930,
    "num_stations": 104
}

# Author information
AUTHOR_INFO = {
    "name": "Alex Crespillo López",
    "occupation": "Predoctoral Researcher",
    "institution": "IPE-CSIC",
    "institution_full": "Instituto Pirenaico de Ecología - Consejo Superior de Investigaciones Científicas"
}

REFERENCES = [
    "Van Loon (2015) - Hydrological drought explained",
    "Tallaksen & Van Lanen (2004) - Hydrological Drought Processes",
    "Fleig et al. (2006) - Global evaluation of streamflow drought characteristics",
    "Smakhtin (2001) - Low flow hydrology review"
]

# ==================== CUSTOM CSS STYLING ====================
CUSTOM_CSS = """
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0066cc 0%, #00a3e0 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .info-box h4 {
        color: #0066cc;
        margin-top: 0;
        font-size: 1.3rem;
        font-weight: 600;
    }

    .info-box p {
        margin: 0.5rem 0;
        color: #495057;
        line-height: 1.6;
    }

    /* Warning boxes */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff7f0e;
        margin: 1rem 0;
    }

    /* Success boxes */
    .success-box {
        background: linear-gradient(135deg, #d1f2eb 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2ca02c;
        margin: 1rem 0;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 6px;
        padding: 0 1.5rem;
        font-weight: 500;
        border: 1px solid #e9ecef;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0066cc 0%, #00a3e0 100%);
        color: white !important;
        border: none;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0066cc 0%, #004d99 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }

    section[data-testid="stSidebar"] label {
        color: white !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #0066cc 0%, #00a3e0 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #0052a3 0%, #0082b3 100%);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 2px solid #e9ecef;
        color: #6c757d;
    }

    .footer-author {
        background: linear-gradient(135deg, #0066cc 0%, #00a3e0 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .footer-author h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
    }

    .footer-author p {
        margin: 0.25rem 0;
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
"""
