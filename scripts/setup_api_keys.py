#!/usr/bin/env python
"""
Script to set up API keys in AWS Secrets Manager.
"""
import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.interpolation_sw_2010.aws_manager.s3 import S3

def setup_api_keys(fred_key=None, bea_key=None, census_key=None):
    """Set up API keys in AWS Secrets Manager."""
    # FRED API Key
    if fred_key:
        print("Setting up FRED API key...")
        S3.store_secret('FRED_API_KEY', 'fred', fred_key, type='api')
    
    # BEA API Key
    if bea_key:
        print("Setting up BEA API key...")
        S3.store_secret('BEA_API_KEY', 'bea', bea_key, type='api')
    
    # Census Bureau API Key
    if census_key:
        print("Setting up Census Bureau API key...")
        S3.store_secret('CENSUS_API_KEY', 'census', census_key, type='api')

def main():
    parser = argparse.ArgumentParser(description="Set up API keys in AWS Secrets Manager")
    parser.add_argument('--fred-key', help='FRED API key')
    parser.add_argument('--bea-key', help='BEA API key')
    parser.add_argument('--census-key', help='Census Bureau API key')
    args = parser.parse_args()
    
    if not any([args.fred_key, args.bea_key, args.census_key]):
        print("Please provide at least one API key")
        parser.print_help()
        return 1
    
    setup_api_keys(args.fred_key, args.bea_key, args.census_key)
    return 0

if __name__ == '__main__':
    sys.exit(main()) 