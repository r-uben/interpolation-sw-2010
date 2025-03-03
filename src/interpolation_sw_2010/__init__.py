"""
Stock and Watson 2010 Kalman Filter Method for Interpolating Quarterly US National Data into Monthly Data.

This package implements the methodology described in Stock and Watson (2010) for temporal disaggregation
of quarterly time series into monthly estimates using related monthly indicators and the Kalman filter.
"""

from interpolation_sw_2010.data_manager import DataManager
from interpolation_sw_2010.kalman_filter import KalmanFilter

__version__ = "0.1.0"
