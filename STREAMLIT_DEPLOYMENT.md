# 🚀 Streamlit Deployment Guide - Large Files Solution

**Date**: 2024-10-07
**Author**: Alex Crespillo López
**Institution**: IPE-CSIC

---

## 📊 The Challenge

Your Streamlit app requires `caudales_diarios.csv` (131 MB), which exceeds GitHub's 100MB limit but is needed for the dashboard to function.

---

## ✅ Solution 1: Git LFS (Recommended)

**Best for**: Files between 100MB - 2GB
**Streamlit Cloud**: Fully supported
**Cost**: Free (GitHub LFS: 1GB storage + 1GB bandwidth/month free)

### Setup Git LFS

```bash
# 1. Install Git LFS (if not already installed)
# Windows: Download from https://git-lfs.github.com/
# Or use: git lfs install

# 2. Navigate to your repository
cd /c/Users/39401439R/Desktop/Drought_Dashboard

# 3. Initialize Git LFS
git lfs install

# 4. Track the large CSV file
git lfs track "data/csv/caudales_diarios.csv"

# 5. Verify .gitattributes was created
cat .gitattributes
# Should show: data/csv/caudales_diarios.csv filter=lfs diff=lfs merge=lfs -text

# 6. Add the large file
git add data/csv/caudales_diarios.csv
git add .gitattributes

# 7. Commit
git commit -m "Add large data file via Git LFS

- Enable Git LFS for caudales_diarios.csv (131MB)
- Streamlit Cloud will automatically download this file on deployment
- File is now properly tracked for deployment

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 8. Push to GitHub
git push origin main
```

### Deploy to Streamlit Cloud

1. Go to: https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub account
4. Select: `AlexCrespillo/Drought_Dashboard`
5. Main file path: `app/app.py`
6. Click "Deploy"

**Streamlit Cloud will automatically download Git LFS files during deployment!**

---

## ✅ Solution 2: External Data Hosting + Download on Startup

**Best for**: Very large files (>2GB) or frequently updated data
**Streamlit Cloud**: Supported
**Cost**: Free (using various hosting services)

### Step 1: Host Your Data File

Choose one of these free options:

#### Option A: Google Drive
1. Upload `caudales_diarios.csv` to Google Drive
2. Right-click → Get link → Set to "Anyone with the link"
3. Copy the file ID from URL: `https://drive.google.com/file/d/FILE_ID_HERE/view`
4. Use this download URL format: `https://drive.google.com/uc?id=FILE_ID_HERE&export=download`

#### Option B: Dropbox
1. Upload file to Dropbox
2. Get share link
3. Change `www.dropbox.com` to `dl.dropboxusercontent.com` in URL
4. Change `?dl=0` to `?dl=1`

#### Option C: Zenodo (Recommended for Research Data)
1. Go to: https://zenodo.org/
2. Click "Upload" → Create new upload
3. Add your CSV file
4. Fill metadata (title, description, author)
5. Publish → Get DOI and download URL

#### Option D: GitHub Release (Alternative)
1. Create a release on GitHub
2. Attach the large file to the release
3. Use the release asset URL

### Step 2: Modify Your Streamlit App to Download Data

Create a helper function in `app/utils/data_loader.py`:

```python
import os
import requests
import streamlit as st
from pathlib import Path

def download_large_file(url, destination):
    """
    Download large file from external source if not already present.
    Shows progress bar in Streamlit.
    """
    dest_path = Path(destination)

    # Check if file already exists
    if dest_path.exists():
        st.info(f"✅ Data file found: {destination}")
        return str(dest_path)

    # Download file
    st.warning(f"⬇️ Downloading large data file... (this may take a minute)")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Progress bar
        progress_bar = st.progress(0)
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(downloaded / total_size)

        progress_bar.empty()
        st.success(f"✅ Downloaded: {destination}")
        return str(dest_path)

    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        raise

# Configuration
DATA_URL = "YOUR_EXTERNAL_URL_HERE"  # Replace with actual URL
LOCAL_PATH = "data/csv/caudales_diarios.csv"

def load_streamflow_data():
    """Load streamflow data, downloading if necessary"""
    # Download if needed
    download_large_file(DATA_URL, LOCAL_PATH)

    # Load data
    import pandas as pd
    df = pd.read_csv(LOCAL_PATH, parse_dates=['fecha'])
    return df
```

### Step 3: Use Streamlit Secrets for URL

Create `.streamlit/secrets.toml` locally (don't commit this):

```toml
[data]
caudales_url = "https://your-external-host.com/caudales_diarios.csv"
```

Update your code:

```python
import streamlit as st

# Access secret URL
DATA_URL = st.secrets["data"]["caudales_url"]
```

Add the secret to Streamlit Cloud:
1. Go to your app settings on Streamlit Cloud
2. Click "Secrets"
3. Paste the same TOML content
4. Save

### Step 4: Update Requirements

Add to `app/requirements.txt`:

```txt
requests>=2.31.0
```

---

## ✅ Solution 3: Data Sampling (For Demo Purposes)

**Best for**: Public demos where full data isn't critical
**Streamlit Cloud**: Supported
**Cost**: Free

Create a smaller sample dataset:

```python
import pandas as pd
import numpy as np

# Load full dataset locally
df = pd.read_csv('data/csv/caudales_diarios.csv', parse_dates=['fecha'])

# Sample strategies:

# Option A: Random sample (10% of data)
sample = df.sample(frac=0.1, random_state=42)

# Option B: Recent years only (e.g., last 10 years)
recent = df[df['fecha'] >= '2011-01-01']

# Option C: Selected stations (e.g., 10 most important stations)
important_stations = [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010]
sample = df[df['indroea'].isin(important_stations)]

# Option D: Monthly averages instead of daily (dramatically reduces size)
monthly = df.groupby(['indroea', pd.Grouper(key='fecha', freq='M')]).agg({
    'caudal': ['mean', 'min', 'max', 'std'],
    'lugar': 'first',
    'xutm': 'first',
    'yutm': 'first'
}).reset_index()

# Save sample
sample.to_csv('data/csv/caudales_diarios_sample.csv', index=False)
print(f"Original size: {len(df):,} rows")
print(f"Sample size: {len(sample):,} rows")
print(f"Reduction: {len(sample)/len(df)*100:.1f}%")
```

Then update your app to use the sample file, and add a note:

```python
st.info("📊 Demo version: Showing sample of data. Contact for full dataset access.")
```

---

## ✅ Solution 4: Database Connection

**Best for**: Professional deployments with multiple users
**Streamlit Cloud**: Supported
**Cost**: Free tier available (PostgreSQL on Supabase, Railway, etc.)

1. **Upload data to cloud database** (PostgreSQL, MySQL, etc.)
2. **Use Streamlit's database connectors**:

```python
import streamlit as st

# Connect to database
conn = st.connection("postgresql", type="sql")

# Query data
df = conn.query("SELECT * FROM caudales WHERE fecha >= '2020-01-01'")
```

3. **Store credentials in Streamlit Secrets**

---

## 📊 Comparison Table

| Solution | File Size Limit | Setup Difficulty | Performance | Cost |
|----------|----------------|------------------|-------------|------|
| **Git LFS** | 2GB | ⭐ Easy | ⭐⭐⭐ Fast | Free (1GB) |
| **External Hosting** | Unlimited | ⭐⭐ Medium | ⭐⭐ Medium | Free |
| **Data Sampling** | N/A | ⭐ Easy | ⭐⭐⭐ Fast | Free |
| **Database** | Unlimited | ⭐⭐⭐ Hard | ⭐⭐⭐ Fast | Free tier |

---

## 🎯 Recommended Approach for Your Project

For your 131MB CSV file, I recommend **Git LFS (Solution 1)** because:

✅ **Simple**: Just a few git commands
✅ **Native**: Works seamlessly with Streamlit Cloud
✅ **Version controlled**: Data changes are tracked
✅ **No code changes**: App works the same locally and on cloud
✅ **Free**: Within GitHub LFS free tier (1GB storage, 1GB bandwidth/month)

### Quick Start Commands

```bash
# Install Git LFS
git lfs install

# Track large file
git lfs track "data/csv/caudales_diarios.csv"

# Add and commit
git add .gitattributes data/csv/caudales_diarios.csv
git commit -m "Add large data file via Git LFS"
git push origin main

# Deploy on Streamlit Cloud
# Go to: https://share.streamlit.io/
# Select your repo and deploy!
```

---

## 🔧 Troubleshooting

### Git LFS: "Rate limit exceeded"
**Solution**: GitHub LFS has bandwidth limits. For high-traffic apps, consider external hosting.

### Streamlit Cloud: "Out of memory"
**Solution**:
- Optimize data loading with `@st.cache_data`
- Load only necessary columns: `pd.read_csv(file, usecols=['fecha', 'caudal'])`
- Filter data on load: `pd.read_csv(file, parse_dates=['fecha']).query("fecha >= '2020-01-01'")`

### External download: "Connection timeout"
**Solution**:
- Use more reliable hosting (Zenodo recommended for research data)
- Implement retry logic with exponential backoff
- Consider compressing file (gzip)

---

## 📝 Update DATA.md After Deployment

Once deployed, update your `DATA.md` to include:

```markdown
## 🌐 Live Demo

Try the interactive dashboard: [Streamlit App URL]

**Note**: The demo uses [Git LFS / sample data / full dataset] for demonstration purposes.
```

---

## 📞 Need Help?

**Alex Crespillo López**
Email: acrespillo@ipe.csic.es
GitHub: @AlexCrespilloLopez

---

## 📚 Additional Resources

- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-community-cloud
- Git LFS Tutorial: https://git-lfs.github.com/
- Streamlit Secrets: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
- Zenodo Guide: https://help.zenodo.org/

---

**Status**: Ready to deploy with any of the solutions above ✅
**Recommended**: Git LFS for your 131MB file 🎯
