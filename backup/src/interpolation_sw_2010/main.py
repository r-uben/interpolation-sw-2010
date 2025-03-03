"""Main script for US GDP interpolation from quarterly to monthly values.

This script follows the methodology described by Stock & Watson (2010) to interpolate
US GDP from quarterly to monthly values. The detrending of series is done by cubic splines.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from interpolation_sw_2010.data_manager import transform_data, load_data
from interpolation_sw_2010.spline_detrending import (
    create_spline_qm2,
    create_spline,
    expand_data,
    detrend_data
)
from interpolation_sw_2010.regressors import prepare_regressors
from interpolation_sw_2010.kalman_optimization import optimize_parameters
from interpolation_sw_2010.kalman_filter import KalmanFilter


def main():
    """Main function to run the GDP interpolation process."""
    # Configuration
    VOA = [1, 4, 5, 0, 2, 4, 4, 7, 1]  # mapping from Mdata to X through transform
    VOA_X = [1, 2, 5, 2, 2, 4, 4, 5, 1]  # mapping from X to X_m
    NKNOTS_Y = [0, 5, 5, 5, 5, 4, 4, 5, 5]  # last entries are for GDP deflator
    NKNOTS_X = [1, 1, 4, 1, 1, 4, 4, 4, 5]

    # Load data
    data_path = Path("replication_materials/2020__jarocinski_karadi__2020")
    qdata, qtxt = load_data(data_path / "DISTRIBUTE_GDP_GDI_INPUT2.xlsx", "Quarterly")
    mdata, mtxt = load_data(data_path / "DISTRIBUTE_GDP_GDI_INPUT2.xlsx", "Monthly")

    # Transform data
    y, x, names = transform_data(qdata, mdata, VOA, qtxt, mtxt)
    months = x.shape[0]

    # Initialize arrays for detrended data
    y_m = []
    y_q = []
    y_ext = np.zeros((y.shape[0], y.shape[1]-2))

    # Compute detrended data using cubic splines
    for j in range(2, y.shape[1]):  # skip first two columns (Years and Months)
        y_q_j, y_m_j = create_spline_qm2(y[:, j], NKNOTS_Y[j-2])
        y_q.append(y_q_j)
        y_m.append(y_m_j)
        y_ext[:, j-2] = expand_data(y[:, j], months)

    y_q = np.column_stack(y_q)
    y_m = np.column_stack(y_m)

    # Special handling for fifth indicator (scaled by spline based on absolute values)
    w = np.abs(y[:, 6])
    y_q[:, 4], y_m[:, 4] = create_spline_qm2(w, NKNOTS_Y[4])

    # Process X data
    x_m = []
    index = 0
    for j in range(x.shape[1]):
        if j - sum(VOA_X[:index]) > 0:
            index += 1
            if VOA_X[index] == 0:
                index += 1
        x_m_j = create_spline(x[:, j], NKNOTS_X[index])
        x_m.append(x_m_j)

    x_m = np.column_stack(x_m)

    # Detrend variables
    y_reg, x_reg = detrend_data(y, x, y_m, y_q, x_m)
    y_reg = y_reg[:, 2:]  # Remove year and month columns

    # Prepare regressors and get starting values
    x_reg_dict, b_start = prepare_regressors(x_reg, y_ext, x)

    # Initialize results array
    results = []

    # Names for accessing dictionary elements
    name_vec = ['first', 'second', 'third', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    # Initialize Kalman Filter
    kf = KalmanFilter(y_reg=y_reg, x_reg=x_reg_dict, y_m=y_m, y_q=y_q)

    # Optimization and interpolation for each series
    for i in range(1, b_start.shape[1]):  # Start from second column
        print(f"Processing series {i}")
        element = name_vec[i]

        # Get starting values for current series
        b_st = b_start[:, i]
        b_st = b_st[~np.isnan(b_st).any(axis=1)]

        # Optimize parameters
        b_max = optimize_parameters(kf, b_st, i, element)

        # Repeat optimization for problematic cases
        if i in [5, 4, 2]:  # equivalent to MATLAB's i==6 || i==5 || i==3
            b_max = optimize_parameters(kf, b_max, i, element)

        # Interpolation and smoothing
        tmp = kf.interpolate_and_smooth(b_max, i, element)
        results.append(tmp)

    results = np.column_stack(results)
    
    # Add first indicator (PCE) since it was not interpolated
    results = np.column_stack([x[:, 0], results])

    # Calculate GDP
    gdp_nominal = np.sum(results[:, :-1], axis=1) - 2 * results[:, 6]  # subtract imports twice
    gdp_real = gdp_nominal / results[:, -1] * 100

    final_results = np.column_stack([gdp_nominal, gdp_real, results[:, -1]])

    # Export results
    output_path = Path("output")
    output_path.mkdir(exist_ok=True)

    # Save monthly indices
    pd.DataFrame(results).to_excel(
        output_path / "Interpolated_US_monthly_indices.xlsx",
        sheet_name="Monthly Indices",
        index=False
    )

    # Save GDP results
    pd.DataFrame(
        final_results,
        columns=["GDP_nominal", "GDP_real", "GDP_deflator"]
    ).to_excel(
        output_path / "Interpolated_US_gdp.xlsx",
        sheet_name="GDP",
        index=False
    )


if __name__ == "__main__":
    main() 