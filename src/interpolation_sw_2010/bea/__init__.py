"""
BEA (Bureau of Economic Analysis) package.
This module contains all functionality related to fetching and processing data from the Bureau of Economic Analysis API.
"""

from .client import BEAClient
from .services.data_fetcher import DataFetcher

__all__ = ['BEAClient', 'DataFetcher'] 