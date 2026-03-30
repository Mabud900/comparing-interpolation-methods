import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import CubicSpline, lagrange
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# Individual interpolation methods
# ─────────────────────────────────────────────

def linear_interpolation(series: pd.Series) -> pd.Series:
    """pandas built-in linear interpolation (fast, good for near-linear trends)."""
    return series.interpolate(method="linear", limit_direction="both")


def polynomial_interpolation(series: pd.Series, order: int = 3) -> pd.Series:
    """Polynomial interpolation of given order."""
    return series.interpolate(method="polynomial", order=order, limit_direction="both")


def spline_interpolation(series: pd.Series, order: int = 3) -> pd.Series:
    """B-spline interpolation."""
    return series.interpolate(method="spline", order=order, limit_direction="both")


def cubic_spline_interpolation(series: pd.Series) -> pd.Series:
    """Scipy CubicSpline — C2-continuous, excellent for smooth data."""
    result = series.copy()
    known_idx = series.dropna().index
    known_vals = series.dropna().values
    if len(known_idx) < 3:
        return linear_interpolation(series)
    cs = CubicSpline(known_idx.astype(float), known_vals)
    missing_idx = series[series.isna()].index
    result.loc[missing_idx] = cs(missing_idx.astype(float))
    return result


def lagrange_interpolation(series: pd.Series) -> pd.Series:
    """
    Lagrange interpolation using scipy.interpolate.lagrange.
    NOTE: Uses only nearby known points (window=10) to avoid Runge's phenomenon.
    """
    result = series.copy()
    known_mask = series.notna()
    known_x = series[known_mask].index.astype(float).values
    known_y = series[known_mask].values
    missing_idx = series[series.isna()].index

    for idx in missing_idx:
        xi = float(idx)
        # Pick 8 nearest known points
        dists = np.abs(known_x - xi)
        nearest = np.argsort(dists)[:8]
        poly = lagrange(known_x[nearest], known_y[nearest])
        result.loc[idx] = float(poly(xi))
    return result


def newton_forward_interpolation(series: pd.Series) -> pd.Series:
    """
    Newton's Forward Difference interpolation.
    Works best when x-values are uniformly spaced.
    """
    result = series.copy()
    known = series.dropna()
    known_x = known.index.astype(float).values
    known_y = known.values
    n = len(known_x)
    if n < 2:
        return series.ffill().bfill()

    # Build divided difference table
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = known_y
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = (diff_table[i + 1][j - 1] - diff_table[i][j - 1]) / \
                                (known_x[i + j] - known_x[i])

    # Evaluate at missing points
    missing_idx = series[series.isna()].index
    for idx in missing_idx:
        xi = float(idx)
        # Find nearest base point (use the one just before xi)
        base_candidates = known_x[known_x <= xi]
        if len(base_candidates) == 0:
            base_i = 0
        else:
            base_i = np.searchsorted(known_x, xi) - 1
            base_i = max(0, min(base_i, n - 1))

        # Evaluate polynomial using Newton's forward formula
        val = diff_table[base_i][0]
        prod = 1.0
        for k in range(1, min(5, n - base_i)):  # Use up to 5 terms
            prod *= (xi - known_x[base_i + k - 1])
            val += diff_table[base_i][k] * prod

        result.loc[idx] = val
    return result


# ─────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────

METHODS = {
    "Linear": linear_interpolation,
    "Polynomial (deg 3)": lambda s: polynomial_interpolation(s, 3),
    "Cubic Spline": cubic_spline_interpolation,
    "Lagrange": lagrange_interpolation,
    "Newton Forward": newton_forward_interpolation,
}


def apply_all_methods(series: pd.Series) -> dict:
    """Return dict of {method_name: filled_series}."""
    results = {}
    for name, func in METHODS.items():
        try:
            results[name] = func(series.copy())
        except Exception as e:
            results[name] = series.copy().ffill().bfill()
    return results


# ─────────────────────────────────────────────
# Error Analysis
# ─────────────────────────────────────────────

def evaluate_methods(original_series: pd.Series, filled_results: dict) -> pd.DataFrame:
    """
    Simulate evaluation: hide 20% of known values, interpolate, compare.
    Returns a DataFrame with MAE, RMSE for each method.
    """
    known = original_series.dropna()
    if len(known) < 10:
        return pd.DataFrame()

    np.random.seed(0)
    test_idx = np.random.choice(known.index, size=max(3, len(known) // 5), replace=False)
    masked = original_series.copy()
    masked.loc[test_idx] = np.nan

    rows = []
    for name, func in METHODS.items():
        try:
            filled = func(masked.copy())
            y_true = known.loc[test_idx].values
            y_pred = filled.loc[test_idx].values
            mae  = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rows.append({"Method": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4)})
        except Exception:
            rows.append({"Method": name, "MAE": None, "RMSE": None})

    df_metrics = pd.DataFrame(rows).sort_values("RMSE")
    return df_metrics
