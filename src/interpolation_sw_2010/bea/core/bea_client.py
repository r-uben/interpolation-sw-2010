"""
Core BEA client implementation.
"""
import logging
import json
import pandas as pd
import requests
from typing import Dict, Any, List, Optional
from interpolation_sw_2010.aws_manager.s3 import S3

logger = logging.getLogger(__name__)

class BEAClient:
    """Client for interacting with BEA API."""
    
    BASE_URL = "https://apps.bea.gov/api/data"
    
    # BEA Dataset codes
    DATASETS = {
        "nipa": "NIPA",               # National Income and Product Accounts
        "niunderlyingdetail": "NIUnderlyingDetail",  # National Income Underlying Detail
        "fixedassets": "FixedAssets", # Fixed Assets
        "iip": "IIP",                 # International Investment Position
        "gsp": "RegionalProduct",     # Gross State Product
        "regional": "Regional",       # Regional Data
        "international": "International",  # International Transactions
        "inputoutput": "InputOutput", # Input-Output Data
    }
    
    def __init__(self):
        """Initialize BEA client with API key from AWS Secrets Manager."""
        try:
            # Get API key and ensure it's a plain string
            api_key = S3.get_secret('BEA-API-KEY', key='api_key')
            if isinstance(api_key, dict):
                api_key = api_key.get('api_key')
            self.api_key = api_key
        except Exception as e:
            logger.error(f"Failed to initialize BEA client: {str(e)}")
            raise
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the BEA API."""
        # Add required parameters
        params.update({
            "UserID": self.api_key,
            "method": "GetData",
            "ResultFormat": "JSON"
        })
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"BEA API request failed: {str(e)}")
            raise
    
    def get_nipa_table(self, table_name: str, frequency: str = "Q", 
                       year: Optional[str] = None, quarter: Optional[str] = None) -> pd.DataFrame:
        """
        Get data from a NIPA table.
        
        Args:
            table_name: NIPA table name (e.g., "T20600", "T10101")
            frequency: Frequency of data (A=Annual, Q=Quarterly, M=Monthly)
            year: Year(s) of data (e.g., "2020" or "2018,2019,2020" or "X" for all)
            quarter: Quarter(s) for quarterly data (e.g., "Q1" or "Q1,Q2" or "X" for all)
        
        Returns:
            DataFrame with the requested data
        """
        params = {
            "TableName": table_name,
            "Frequency": frequency,
            "DatasetName": self.DATASETS["nipa"]
        }
        
        # Add year and quarter parameters if specified
        if year:
            params["Year"] = year
        if quarter and frequency == "Q":
            params["Quarter"] = quarter
        
        response_data = self._make_request(params)
        
        # Process the response data
        try:
            # Extract data from the nested JSON response
            data = response_data["BEAAPI"]["Results"]["Data"]
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Process the DataFrame
            # Convert time periods to datetime
            if frequency == "Q":
                # Handle quarterly data
                df["date"] = df.apply(lambda row: pd.to_datetime(f"{row['Year']}-{int(row['Quarter'][-1])*3-2}-01"), axis=1)
                # Set to end of quarter
                df["date"] = df["date"] + pd.offsets.QuarterEnd(0)
            elif frequency == "M":
                # Handle monthly data
                df["date"] = df.apply(lambda row: pd.to_datetime(f"{row['Year']}-{int(row['Month'])}-01"), axis=1)
                # Set to end of month
                df["date"] = df["date"] + pd.offsets.MonthEnd(0)
            elif frequency == "A":
                # Handle annual data
                df["date"] = df.apply(lambda row: pd.to_datetime(f"{row['Year']}-12-31"), axis=1)
                
            # Set date as index
            df = df.set_index("date")
            
            # Convert DataValue to float
            df["value"] = pd.to_numeric(df["DataValue"], errors="coerce")
            
            # Select relevant columns
            columns_to_keep = ["value", "SeriesName", "LineNumber", "LineDescription"]
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            # Sort by date
            df = df.sort_index()
            
            return df
            
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing BEA response: {str(e)}")
            if "BEAAPI" in response_data and "Error" in response_data["BEAAPI"]:
                logger.error(f"BEA API error: {response_data['BEAAPI']['Error']}")
            raise
    
    def get_international_data(self, table_name: str, frequency: str = "M",
                              year: Optional[str] = None, month: Optional[str] = None,
                              direction: Optional[str] = None) -> pd.DataFrame:
        """
        Get international trade data (exports/imports).
        
        Args:
            table_name: International accounts table (e.g., "ita" for International Transactions)
            frequency: Frequency of data (A=Annual, Q=Quarterly, M=Monthly)
            year: Year(s) of data (e.g., "2020" or "2018,2019,2020" or "X" for all)
            month: Month(s) for monthly data (e.g., "1" or "1,2,3" or "X" for all)
            direction: Trade direction ("exports" or "imports" or None for both)
        
        Returns:
            DataFrame with the requested data
        """
        params = {
            "TableName": table_name,
            "Frequency": frequency,
            "DatasetName": self.DATASETS["international"]
        }
        
        # Add optional parameters if specified
        if year:
            params["Year"] = year
        if month and frequency == "M":
            params["Month"] = month
        if direction:
            params["Direction"] = direction.capitalize()
        
        response_data = self._make_request(params)
        
        # Process the response data (similar to get_nipa_table)
        try:
            # Extract data from the nested JSON response
            data = response_data["BEAAPI"]["Results"]["Data"]
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Process dates based on frequency
            if frequency == "M":
                df["date"] = df.apply(lambda row: pd.to_datetime(f"{row['Year']}-{row['Month']}-01"), axis=1)
                df["date"] = df["date"] + pd.offsets.MonthEnd(0)
            elif frequency == "Q":
                df["date"] = df.apply(lambda row: pd.to_datetime(f"{row['Year']}-{int(row['Quarter'][-1])*3-2}-01"), axis=1)
                df["date"] = df["date"] + pd.offsets.QuarterEnd(0)
            elif frequency == "A":
                df["date"] = df.apply(lambda row: pd.to_datetime(f"{row['Year']}-12-31"), axis=1)
            
            # Set date as index
            df = df.set_index("date")
            
            # Convert DataValue to float
            df["value"] = pd.to_numeric(df["DataValue"], errors="coerce")
            
            # Select relevant columns
            columns_to_keep = ["value", "SeriesName", "LineNumber", "LineDescription"]
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            # Sort by date
            df = df.sort_index()
            
            return df
            
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing BEA response: {str(e)}")
            if "BEAAPI" in response_data and "Error" in response_data["BEAAPI"]:
                logger.error(f"BEA API error: {response_data['BEAAPI']['Error']}")
            raise
            
    def search_series(self, dataset: str, search_text: str) -> pd.DataFrame:
        """
        Search for series in a BEA dataset.
        
        Args:
            dataset: BEA dataset name (e.g., "NIPA", "International")
            search_text: Text to search for
            
        Returns:
            DataFrame with matching series
        """
        # Note: BEA API doesn't have a direct search endpoint
        # This is a simplified implementation to help find tables and series
        
        params = {
            "method": "GetParameterValues",
            "DatasetName": self.DATASETS.get(dataset.lower(), dataset),
            "ParameterName": "TableName",
            "UserID": self.api_key,
            "ResultFormat": "JSON"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract parameter values
            param_values = data["BEAAPI"]["Results"]["ParamValue"]
            
            # Convert to DataFrame
            df = pd.DataFrame(param_values)
            
            # Filter based on search text
            search_result = df[df["Description"].str.contains(search_text, case=False)]
            
            return search_result
            
        except Exception as e:
            logger.error(f"Error searching BEA series: {str(e)}")
            raise
    
    def get_underlying_detail_data(self, table_name: str, frequency: str = "M", 
                              year: Optional[str] = None) -> pd.DataFrame:
        """
        Get data from the NIUnderlyingDetail dataset.
        
        Args:
            table_name: Table name or ID (e.g., "T71200")
            frequency: Frequency of data (A=Annual, Q=Quarterly, M=Monthly)
            year: Year(s) of data (e.g., "2020" or "1967,1997" or "X" for all)
        
        Returns:
            DataFrame with the requested data
        """
        params = {
            "TableName": table_name,
            "Frequency": frequency,
            "DatasetName": self.DATASETS["niunderlyingdetail"]
        }
        
        # Add year parameter if specified
        if year:
            params["Year"] = year
        
        logger.info(f"Making BEA API request with params: {params}")
        response_data = self._make_request(params)
        
        # Process the response data
        try:
            # Log the full response for debugging
            logger.info(f"BEA API response keys: {response_data.keys()}")
            if "BEAAPI" in response_data:
                logger.info(f"BEAAPI keys: {response_data['BEAAPI'].keys()}")
                if "Results" in response_data["BEAAPI"]:
                    logger.info(f"Results keys: {response_data['BEAAPI']['Results'].keys()}")
                    if "Error" in response_data["BEAAPI"]["Results"]:
                        error_msg = response_data["BEAAPI"]["Results"]["Error"]
                        logger.error(f"BEA API error: {error_msg}")
                        raise ValueError(f"BEA API error: {error_msg}")
                if "Error" in response_data["BEAAPI"]:
                    logger.error(f"BEA API error: {response_data['BEAAPI']['Error']}")
                    raise ValueError(f"BEA API error: {response_data['BEAAPI']['Error']}")
            
            # Extract data from the nested JSON response
            data = response_data["BEAAPI"]["Results"]["Data"]
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Process the DataFrame
            # Convert time periods to datetime
            if frequency == "Q":
                # Handle quarterly data
                df["date"] = df.apply(lambda row: pd.to_datetime(f"{row['Year']}-{int(row['Quarter'][-1])*3-2}-01"), axis=1)
                # Set to end of quarter
                df["date"] = df["date"] + pd.offsets.QuarterEnd(0)
            elif frequency == "M":
                # Handle monthly data
                df["date"] = df.apply(lambda row: pd.to_datetime(f"{row['Year']}-{int(row['Month'])}-01"), axis=1)
                # Set to end of month
                df["date"] = df["date"] + pd.offsets.MonthEnd(0)
            elif frequency == "A":
                # Handle annual data
                df["date"] = df.apply(lambda row: pd.to_datetime(f"{row['Year']}-12-31"), axis=1)
                
            # Set date as index
            df = df.set_index("date")
            
            # Convert DataValue to float
            df["value"] = pd.to_numeric(df["DataValue"], errors="coerce")
            
            # Select relevant columns
            columns_to_keep = ["value", "SeriesName", "LineNumber", "LineDescription"]
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            # Sort by date
            df = df.sort_index()
            
            return df
            
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing BEA response: {str(e)}")
            if "BEAAPI" in response_data and "Results" in response_data["BEAAPI"] and "Error" in response_data["BEAAPI"]["Results"]:
                logger.error(f"BEA API error: {response_data['BEAAPI']['Results']['Error']}")
            elif "BEAAPI" in response_data and "Error" in response_data["BEAAPI"]:
                logger.error(f"BEA API error: {response_data['BEAAPI']['Error']}")
            # Print the full response for debugging
            logger.error(f"Full BEA API response: {json.dumps(response_data, indent=2)}")
            raise 