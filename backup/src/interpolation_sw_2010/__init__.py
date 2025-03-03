"""
Stock-Watson (2010) Interpolation Package

This package provides tools for implementing the Stock-Watson (2010) method 
for interpolating quarterly GDP data to monthly frequency using related monthly indicators.

The method uses cubic spline detrending and Kalman filtering to produce monthly estimates
that are consistent with quarterly totals.
"""

__version__ = "0.1.0"

# Core functionality
from .sw2010_interpolator import interpolate_gdp_monthly

# Make the main entry point available
from .sw2010_interpolator import main

# Expose main components
from .data_manager import DataManager, TransformationResult
