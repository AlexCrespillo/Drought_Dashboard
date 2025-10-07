# 📊 Data Access Guide

## Overview

Due to GitHub's file size limitations (100MB maximum), the large streamflow dataset is not included in this repository. This document provides instructions for obtaining the required data files.

## Required Data Files

### ✅ Included in Repository

- **`eventos_sequia_p5.csv`** (176 KB) - Drought events dataset
  - Location: `data/csv/eventos_sequia_p5.csv`
  - Contains: Drought event characteristics (duration, deficit, dates)

- **Geospatial data** - Shapefiles and DEM
  - Location: `data/espacial/`
  - All spatial data files are included

### ⬇️ Download Required

- **`caudales_diarios.csv`** (131 MB) - Daily streamflow records
  - **Required for**: Full analysis and Streamlit app
  - **Size**: 131 MB (exceeds GitHub limit)
  - **Contains**: Daily streamflow data (1974-2021) for 104 stations

- **`eventos_sequia_p5_diario.csv`** (131 MB) - Daily drought events (if needed)
  - **Note**: This file may be in your local repository history
  - **Size**: 131 MB (exceeds GitHub limit)
  - **Contains**: Daily drought event records

## Download Instructions

### Option 1: Direct Download (Recommended)

**Contact the author to obtain the data file:**

📧 **Alex Crespillo López**
- Email: acrespillo@ipe.csic.es
- Institution: Instituto Pirenaico de Ecología - CSIC

Request the `caudales_diarios.csv` file and place it in:
```
Drought_Dashboard/data/csv/caudales_diarios.csv
```

### Option 2: External Data Repository

The data may also be available from:

- **SAIH-Ebro**: Spanish Hydrological Information System
  - Website: [SAIH Ebro](https://www.saihebro.com/)
  - Data type: Daily streamflow records
  - Stations: 104 gauging stations
  - Period: 1974-2021

- **Zenodo/Figshare**: (Link will be added when data is published)

### Option 3: Generate from Source

If you have access to SAIH-Ebro data, you can regenerate the dataset:

1. Download raw streamflow data from SAIH-Ebro
2. Process according to methodology in the notebooks
3. Apply drought detection algorithm (P5 threshold, 5-day minimum duration)

## Data Structure

### Expected Format for `caudales_diarios.csv`

```csv
indroea,fecha,caudal,lugar,xutm,yutm
9001,1974-01-01,25.3,Station Name,450000,4600000
9001,1974-01-02,24.8,Station Name,450000,4600000
...
```

**Columns:**
- `indroea` (int): Station identifier code
- `fecha` (date): Date (YYYY-MM-DD format)
- `caudal` (float): Streamflow in m³/s
- `lugar` (str): Station name/location
- `xutm` (int): UTM X coordinate (ETRS89 Zone 30N)
- `yutm` (int): UTM Y coordinate (ETRS89 Zone 30N)

**Expected size:**
- ~1,807,520 records (104 stations × 48 years × 365 days)
- File size: ~131 MB

## Verification

After downloading, verify the file:

```bash
# Check file exists
ls -lh data/csv/caudales_diarios.csv

# Expected output: ~131M

# Quick validation (Python)
python -c "
import pandas as pd
df = pd.read_csv('data/csv/caudales_diarios.csv', parse_dates=['fecha'])
print(f'Records: {len(df):,}')
print(f'Stations: {df.indroea.nunique()}')
print(f'Date range: {df.fecha.min()} to {df.fecha.max()}')
"
```

**Expected output:**
```
Records: ~1,807,520
Stations: 104
Date range: 1974-... to 2021-...
```

## Troubleshooting

### File Not Found Error

If you see an error like:
```
FileNotFoundError: data/csv/caudales_diarios.csv not found
```

**Solution:**
1. Verify the file is in the correct location: `data/csv/caudales_diarios.csv`
2. Check filename spelling matches exactly (case-sensitive)
3. Ensure the file is not empty (size should be ~131 MB)

### Loading Errors

If pandas cannot read the file:
```python
# Try specifying encoding
df = pd.read_csv('data/csv/caudales_diarios.csv',
                 encoding='utf-8',
                 parse_dates=['fecha'])
```

## Alternative: Sample Data

For testing purposes, you can create a small sample dataset:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample data (10 stations, 1 year)
dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
stations = range(9001, 9011)

data = []
for station in stations:
    for date in dates:
        data.append({
            'indroea': station,
            'fecha': date,
            'caudal': np.random.lognormal(3, 1),  # Realistic streamflow distribution
            'lugar': f'Station_{station}',
            'xutm': 450000 + station,
            'yutm': 4600000 + station
        })

df = pd.DataFrame(data)
df.to_csv('data/csv/caudales_diarios.csv', index=False)
print(f"Sample dataset created: {len(df)} records")
```

**Note**: Sample data is for testing only and will not produce meaningful scientific results.

## Data Citation

If you use this data in your research, please cite:

```bibtex
@dataset{ebro_streamflow_data,
  title={Daily Streamflow Records - Ebro River Basin (1974-2021)},
  author={Spanish Hydrological Information System (SAIH-Ebro)},
  year={2021},
  publisher={Confederación Hidrográfica del Ebro},
  note={104 gauging stations, daily observations}
}
```

## Contact & Support

For questions about data access:

- **Technical Issues**: Open an issue on GitHub
- **Data Requests**: Contact acrespillo@ipe.csic.es
- **SAIH-Ebro**: Visit [www.saihebro.com](https://www.saihebro.com/)

---

**Last Updated**: 2024-10-07
**Author**: Alex Crespillo López
**Institution**: IPE-CSIC
