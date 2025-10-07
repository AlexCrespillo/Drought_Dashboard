# Contributing to Drought Analysis Dashboard

Thank you for your interest in contributing to the Drought Analysis Dashboard! We welcome contributions from the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## 🤝 Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to:
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive criticism
- Show empathy towards other community members

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of hydrology or data science (helpful but not required)

### Project Structure

```
Drought_Dashboard/
├── app/                    # Streamlit application
│   ├── pages/             # Dashboard pages
│   └── utils/             # Utility modules
├── data/                  # Data files
├── notebooks/             # Jupyter notebooks
└── requirements.txt       # Dependencies
```

## 💡 How to Contribute

We appreciate contributions in several forms:

### 1. Bug Reports
- Search existing issues first
- Include system information
- Provide reproducible examples
- Describe expected vs. actual behavior

### 2. Feature Requests
- Explain the use case
- Describe the proposed solution
- Consider alternative approaches

### 3. Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation enhancements
- Test coverage

### 4. Documentation
- Fix typos or clarify text
- Add examples
- Improve API documentation
- Translate documentation

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR-USERNAME/Drought_Dashboard.git
cd Drought_Dashboard
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Unix/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# For development (optional)
pip install pytest black flake8
```

### 4. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://peps.python.org/pep-0008/) with some additions:

#### Import Organization
```python
# Standard library
import os
from datetime import datetime
from pathlib import Path

# Third-party packages
import pandas as pd
import numpy as np
import streamlit as st

# Local modules
from config import PAGE_CONFIG
from utils.data_loader import load_drought_events
```

#### Docstrings (Google Style)
```python
def calculate_metrics(data: pd.DataFrame, threshold: float = 0.05) -> dict:
    """
    Calculate drought metrics from streamflow data.

    Args:
        data (pd.DataFrame): Streamflow data with date index
        threshold (float): Percentile threshold for drought detection.
            Defaults to 0.05 (P5).

    Returns:
        dict: Dictionary containing:
            - num_events (int): Number of drought events
            - avg_duration (float): Average event duration in days
            - total_deficit (float): Total water deficit in hm³

    Raises:
        ValueError: If data is empty or threshold is invalid

    Example:
        >>> df = load_streamflow_data()
        >>> metrics = calculate_metrics(df, threshold=0.05)
        >>> print(metrics['num_events'])
        125
    """
    # Implementation
    pass
```

#### Type Hints
```python
from typing import Optional, List, Dict, Tuple

def process_station(
    station_id: int,
    start_date: str,
    end_date: str,
    include_metadata: bool = True
) -> Optional[pd.DataFrame]:
    """Process data for a single station."""
    pass
```

#### Naming Conventions
- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Code Formatting

We recommend using `black` for automatic formatting:

```bash
black app/ --line-length 100
```

### Linting

Check code quality with `flake8`:

```bash
flake8 app/ --max-line-length=100 --ignore=E203,W503
```

## 🔍 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

### Writing Tests

```python
# tests/test_data_loader.py
import pytest
from app.utils.data_loader import load_drought_events

def test_load_drought_events():
    """Test drought events loading."""
    df = load_drought_events()
    assert not df.empty
    assert 'indroea' in df.columns
    assert 'duracion' in df.columns

def test_invalid_station():
    """Test behavior with invalid station ID."""
    with pytest.raises(ValueError):
        process_station(-1, '2020-01-01', '2020-12-31')
```

## 📤 Submitting Changes

### 1. Commit Your Changes

Write clear, descriptive commit messages:

```bash
# Good commit messages:
git commit -m "Add support for seasonal drought analysis"
git commit -m "Fix: Correct percentile calculation in drought detection"
git commit -m "Docs: Update README installation instructions"

# Prefix types:
# - feat: New feature
# - fix: Bug fix
# - docs: Documentation changes
# - style: Code formatting (no logic change)
# - refactor: Code restructuring
# - test: Adding or updating tests
# - chore: Maintenance tasks
```

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Open Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe tests performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed the code
- [ ] Commented complex code sections
- [ ] Updated documentation
- [ ] Added tests (if applicable)
- [ ] All tests pass
```

### 4. Code Review Process

- Maintainers will review your PR
- Address feedback by pushing new commits
- Once approved, your PR will be merged
- Your contribution will be credited

## 🐛 Reporting Issues

### Bug Reports

Use the issue template and include:

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

**Expected Behavior**
What you expected to happen

**Screenshots**
If applicable, add screenshots

**Environment:**
 - OS: [e.g., Windows 10, Ubuntu 20.04]
 - Python version: [e.g., 3.9.7]
 - Browser (if app-related): [e.g., Chrome 96]

**Additional Context**
Any other relevant information
```

### Feature Requests

```markdown
**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
What other approaches did you consider?

**Additional Context**
Mockups, examples, or references
```

## 📚 Additional Resources

- [Project README](README.md)
- [ML Prediction Documentation](ML_PREDICTION_README.md)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [GeoPandas Documentation](https://geopandas.org/)

## 👥 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: acrespillo@ipe.csic.es (for direct contact)

### Recognition

Contributors will be:
- Listed in project documentation
- Credited in release notes
- Acknowledged in academic publications (for significant contributions)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to advancing hydrological drought analysis!** 🌊📊

*Last updated: 2024-10-07*
