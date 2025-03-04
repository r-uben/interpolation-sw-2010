"""
FRED (Federal Reserve Economic Data) package.
This module contains all functionality related to fetching and processing data from the Federal Reserve Economic Data API.
"""

from .services.data_fetcher import DataFetcher

__all__ = ['DataFetcher'] 