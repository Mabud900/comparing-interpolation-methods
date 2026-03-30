# 🔬 DataFill — Data Interpolation Lab

**Numerical Methods Project · 23SMH-341 · BE-CSE/IT (3rd Year) · Jan–June 2026**

A research-grade web app comparing **5 interpolation algorithms** for missing value recovery in datasets.

---

## 🚀 Features

| Feature | Details |
|---------|---------|
| **5 Methods** | Linear, Polynomial (deg 3), Cubic Spline, Lagrange, Newton Forward |
| **Visualization** | Interactive Plotly charts comparing all methods |
| **Error Analysis** | MAE & RMSE benchmarking via cross-validation simulation |
| **CSV Upload** | Works with any numeric CSV dataset |
| **Download** | Export cleaned data in one click |
| **Method Guide** | In-app research reference for each algorithm |

---

## ⚙️ Local Setup (5 minutes)

### 1. Clone / download this folder

```bash
# If using Git
git clone <your-repo-url>
cd interpolation_app
```

### 2. Install dependencies (all free, open-source)

```bash
pip install -r requirements.txt
```

> **No paid software needed.** Everything runs on Python + free libraries.

### 3. Run locally

```bash
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## ☁️ Deploy for Free (Streamlit Cloud)

1. Push this folder to a **GitHub repository** (free account)
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **"New app"** → select your repo → set `app.py` as the main file
4. Click **Deploy** — live in ~2 minutes at a public URL

That's it. Your app is live for anyone to use.

---

## 📂 Project Structure

```
interpolation_app/
├── app.py                          # Main Streamlit UI
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── utils/
│   └── interpolators.py            # All 5 interpolation engines + error analysis
└── sample_data/
    ├── sensor_data_with_missing.csv  # Demo dataset
    └── generate_sample.py           # Script to regenerate sample data
```

---

## 🧪 Interpolation Methods

### 1. Linear Interpolation
- Connects known points with straight lines
- Best for: slowly varying, near-linear data
- `pandas.Series.interpolate(method='linear')`

### 2. Polynomial Interpolation (degree 3)
- Fits a degree-3 polynomial through known values
- Best for: smooth, moderately curved data
- `pandas.Series.interpolate(method='polynomial', order=3)`

### 3. Cubic Spline (C²-continuous)
- Piecewise cubics with smooth joins at every knot
- Best for: general smooth data — most accurate in practice
- `scipy.interpolate.CubicSpline`

### 4. Lagrange Interpolation
- Classic polynomial through N points (uses local window of 8)
- Best for: educational, research; sparse data
- `scipy.interpolate.lagrange`

### 5. Newton's Forward Difference
- Divided-difference table evaluated at missing points
- Best for: uniformly spaced data
- Custom implementation using finite difference tables

---

## 📊 Error Analysis Methodology

The app uses a **simulation-based cross-validation**:

1. Take the known (non-null) values
2. Randomly hide 20% of them
3. Run each interpolation method on the masked series
4. Compare predicted vs actual values
5. Report **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error)

This tells you *which method fits your specific data best* before you commit to it.

---

## 🔬 Research Directions

- How does **polynomial degree** affect accuracy and Runge's phenomenon?
- Compare performance on **uniformly vs non-uniformly spaced** data
- How does the **percentage of missing values** affect each method's error?
- Study **error propagation** in Newton's divided differences
- Investigate **adaptive splines** for data with local sharp changes

---

## 📦 Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| Streamlit | ≥1.32 | Web UI framework |
| Pandas | ≥2.0 | Data manipulation |
| NumPy | ≥1.24 | Numerical computing |
| SciPy | ≥1.10 | Spline & Lagrange interpolation |
| Plotly | ≥5.18 | Interactive charts |
| Scikit-learn | ≥1.3 | MAE/RMSE metrics |

**All free and open-source. Zero cost to run or deploy.**
