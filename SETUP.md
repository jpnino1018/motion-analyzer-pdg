# 🚀 Parkinson Foot Movement Analysis – Setup Guide

This project analyzes IMU sensor data from **foot tapping** and **stomping** exercises, extracting clinical features such as **magnitude, rhythm, fatigue, and asymmetry**.  
It includes a **Streamlit app** for interactive exploration and a **batch pipeline** for automated CSV export.

---

## Create and Activate a Virtual Environment

We recommend using a virtual environment to keep dependencies isolated.

### Using `venv` (built into Python)
```bash
python -m venv venv
```

# Activate venv (Windows)
```bash
venv\Scripts\activate
```
# Activate venv (Linux/Mac)
```bash
source venv/bin/activate
```
# Install venv requirements
```bash
pip install -r requirements.txt
```
# Run the app
```bash
streamlit run app.py
```