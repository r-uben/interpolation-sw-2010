"""
Census Bureau package.
This module contains all functionality related to fetching and processing data from the Census Bureau.
"""

from .services.data_fetcher import DataFetcher

__all__ = ['DataFetcher'] 