"""
Stock-Watson (2010) interpolation method for GDP components.

This package implements the Stock-Watson (2010) methodology for interpolating
quarterly GDP components to monthly frequency using related monthly indicators.
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the main interpolator class
from .sw2010_interpolator import SW2010Interpolator

# Version information
__version__ = '0.1.0'

# Make the main entry point available
from .cli import main

# Expose main components
from .core.data_manager import DataManager
