# 🌊 Ebro River Basin Drought Analysis Dashboard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> **Comprehensive Hydrological Drought Analysis for the Ebro River Basin (1974-2021)**
> 
> Advanced statistical analysis, machine learning classification, and interactive visualization of drought patterns across Spain's largest river basin.

## 📊 Project Overview

This repository contains a comprehensive drought analysis dashboard focusing on **hydrological drought events** in the **Ebro River Basin**, Spain's largest river system. The analysis covers **48 years of data (1974-2021)** and represents one of the most extensive drought characterization studies conducted on a European river system.

### 🎯 Key Features

- **📈 Temporal Analysis**: Long-term trend detection using Mann-Kendall tests and change-point analysis
- **🔬 Statistical Validation**: Advanced non-parametric statistical tests with seasonal decomposition
- **🤖 Machine Learning**: K-means clustering for hydrological station classification
- **🗺️ Geospatial Visualization**: Interactive maps and spatial analysis of drought patterns
- **📊 Interactive Dashboards**: Publication-ready visualizations with Plotly and Matplotlib
- **🌍 Seasonal Patterns**: Comprehensive seasonal drought characterization

## 📁 Repository Structure

```
Drought_Dashboard/
├── 📓 notebooks/
│   ├── eda_bueno.ipynb          # Main comprehensive analysis notebook
├── 📊 data/
│   ├── csv/                     # Tabular drought and flow data
│   │   ├── caudales_diarios.csv # Daily streamflow records
│   │   └── eventos_sequia_p5.csv # Drought events dataset
│   └── espacial/                # Geospatial datasets
│       ├── Limite_Cuenca_Ebro.shp    # Basin boundary
│       ├── Estaciones_Bueno.shp       # Gauging stations
│       ├── Redlinealgeo.shp          # River network
│       ├── embalses.shp              # Reservoirs
│       └── dem_bueno_fill.tif # Digital elevation model
├── 📸 img/                      # Documentation images
└── 📖 README.md                # This documentation
```

## 🔬 Scientific Findings

### 🏆 Major Discoveries

| **Discovery** | **Finding** | **Significance** |
|:--------------|:------------|:-----------------|
| 🕒 **Temporal Stability** | No significant long-term trends in drought characteristics | Provides stable baseline for water resource planning |
| 🌸 **Seasonal Patterns** | Distinct seasonal drought signatures (p < 0.0001) | Enables season-specific management strategies |
| 🎯 **Station Classification** | 4 distinct hydrological clusters identified | Optimizes monitoring network and resource allocation |
| 📊 **Data Completeness** | 100% station overlap, 99.2% data coverage | Ensures robust statistical validation |

### 📈 Key Statistics

- **📅 Analysis Period**: 48 years (1974-2021)
- **🌊 Drought Events**: 3,302 events analyzed
- **🏭 Monitoring Stations**: 104 gauging stations
- **📊 Daily Observations**: 1,807,520 streamflow records
- **🎯 Spatial Coverage**: Entire Ebro River Basin (85,000 km²)

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.8+** 
- **Jupyter Notebook/Lab**
- **Git** (for cloning)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/AlexCrespilloLopez/Drought_Dashboard.git
cd Drought_Dashboard
```

2. **Create virtual environment**
```bash
python -m venv venv
source dd/bin/activate  # On Windows: dd\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook notebooks/eda_bueno.ipynb
```

### 📦 Key Dependencies

- **Core Analysis**: `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Statistical Tests**: `pymannkendall`, `ruptures`, `statsmodels`
- **Geospatial**: `geopandas`, `rasterio`, `folium`, `contextily`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `matplotlib-venn`
- **Environment**: `jupyter`, `ipywidgets`

## 🚀 Usage Guide

### 📓 Main Analysis Notebook (`eda_bueno.ipynb`)

The comprehensive analysis is organized into logical sections:

#### **🔍 Section 0: Data Preprocessing & Quality Assessment**
- Data loading and validation
- Quality metrics and completeness analysis
- Temporal coverage assessment

#### **⏰ Section 1: Temporal Analysis of Droughts**
- Annual drought frequency evolution
- Decadal variability assessment
- Seasonal pattern analysis
- Statistical trend validation

#### **🎯 Section 2: Comprehensive Station Analysis**
- K-means clustering of gauging stations
- Spatial coherence validation
- Hydrological typology development

#### **🏆 Section 3: Synthesis of Scientific Discoveries**
- Integrated findings and implications
- Management recommendations
- Future research directions

### 🎨 Visualization Examples

The notebook generates publication-ready visualizations including:

- **📊 Interactive Time Series**: Annual drought evolution with trend analysis
- **🗺️ Spatial Maps**: Geospatial distribution of drought patterns and station clusters
- **📈 Statistical Plots**: Box plots, violin plots, and correlation matrices
- **🔄 Seasonal Analysis**: Circular plots and seasonal decomposition
- **🤖 Machine Learning**: Cluster validation and feature importance

## 📚 Methodology

### 🔬 Statistical Methods

- **📈 Trend Detection**: Mann-Kendall tests for monotonic trends
- **🔍 Change Point Analysis**: Pettitt tests and ruptures algorithm
- **🌍 Seasonal Analysis**: Kruskal-Wallis tests for seasonal differences
- **📊 Non-parametric Statistics**: Robust methods for hydrological data

### 🤖 Machine Learning Approach

- **🎯 Clustering Algorithm**: K-means with silhouette optimization
- **📊 Feature Engineering**: Standardized hydrological metrics
- **✅ Validation**: Spatial coherence and statistical validation
- **🗺️ Interpretability**: Geographic and hydrological interpretation

### 🗺️ Geospatial Analysis

- **📍 Spatial Data Processing**: Basin boundaries, river networks, monitoring stations
- **🌍 Coordinate Systems**: Consistent CRS handling and reprojection
- **📊 Spatial Statistics**: Geographic coherence analysis
- **🎨 Cartographic Visualization**: Professional maps with multiple layers

## 🎯 Applications & Impact

### 💧 Water Resources Management

- **🚨 Early Warning Systems**: Season-specific drought alert protocols
- **🎯 Network Optimization**: Evidence-based monitoring station deployment
- **🌊 Adaptive Allocation**: Dynamic water allocation based on drought clusters
- **📋 Policy Development**: Scientific foundation for water management policies

### 🔬 Research Contributions

- **📚 Hydrological Science**: First comprehensive 48-year (1974-2021) drought analysis for Ebro Basin
- **🤖 Applied ML**: Novel application of unsupervised learning to hydrological classification
- **📊 Methodology**: Integrated framework combining statistical and ML approaches
- **🌍 International Reference**: Benchmark for Mediterranean drought studies

## 📊 Data Sources

### 🌊 Hydrological Data
- **Source**: Spanish Hydrological Information System (SAIH-Ebro)
- **Type**: Daily streamflow records and drought event catalogs
- **Quality**: Comprehensive quality control and validation
- **Coverage**: 104 gauging stations across the basin

### 🗺️ Geospatial Data
- **Basin Boundary**: Official Ebro River Basin delimitation
- **River Network**: Detailed hydrographic network
- **Topography**: Digital Elevation Model (100m resolution)
- **Infrastructure**: Reservoir and monitoring station locations

## 🤝 Contributing

We welcome contributions to improve the drought analysis dashboard! Please see our contribution guidelines:

### 🔧 Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### 📝 Issues & Suggestions
- Report bugs via GitHub Issues
- Suggest enhancements or new features
- Share applications in other river basins

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact & Citation

### 👥 Authors
- **Research Team**: Hydrological Drought Analysis Group
- **Institution**: Instituto Pirenaico de Ecología - CSIC
- **Contact**: alex@ipe.csic.es

### 📖 Citation
If you use this code or methodology in your research, please cite:

```bibtex
@software{ebro_drought_analysis,
  title={Comprehensive Drought Analysis Dashboard for the Ebro River Basin},
  author={Alex Crespillo López},
  year={2024},
  url={https://github.com/AlexCrespilloLopez/Drought_Dashboard},
  note={Advanced Statistical Analysis and Machine Learning for Hydrological Drought Characterization}
}
```

## 🔗 Related Publications

- [Manuscript in preparation]: "Temporal Stability and Seasonal Patterns of Drought in the Ebro River Basin: A 48-Year Statistical Analysis"
- [Conference paper]: "Machine Learning Classification of Hydrological Drought Behavior in Mediterranean River Systems"

## 🌟 Acknowledgments

- **Data Providers**: Spanish Hydrological Information System (SAIH)
- **Open Source Community**: Contributors to pandas, scikit-learn, geopandas, and other libraries
- **Research Community**: International drought research network

---

<div align="center">

**🌊 Understanding Drought Through Data Science 📊**

*Advancing hydrological science through comprehensive analysis and innovative methodologies*

</div>
