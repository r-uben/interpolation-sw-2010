import os
import pandas as pd
import requests
import json
import tempfile
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs
from pathlib import Path

from ..aws_manager.s3 import S3
from ..utils.logger import get_logger

logger = get_logger(__name__)

class BEAClient:
    """
    Client for interacting with the BEA API and website.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BEA client.
        
        Args:
            config (Dict[str, Any], optional): Configuration for the client
        """
        self.config = config or {}
        
        # Get API key from AWS Secrets Manager
        try:
            secret_data = S3.get_secret("BEA-API-KEY")
            if isinstance(secret_data, str):
                # If it's a JSON string, parse it
                secret_json = json.loads(secret_data)
                self.api_key = secret_json.get('api_key')
            elif isinstance(secret_data, dict):
                # If it's already a dictionary
                self.api_key = secret_data.get('api_key')
            else:
                self.api_key = None
                
            if not self.api_key:
                logger.warning("Could not retrieve BEA API key from secrets manager.")
                # Fallback to config or environment variable
                self.api_key = self.config.get('api_key') or os.environ.get('BEA_API_KEY')
        except Exception as e:
            logger.error(f"Error retrieving BEA API key from secrets manager: {str(e)}")
            # Fallback to config or environment variable
            self.api_key = self.config.get('api_key') or os.environ.get('BEA_API_KEY')
            
        if not self.api_key:
            logger.warning("No BEA API key provided. Some functionality may be limited.")
    
    def download_table(self, source: Dict[str, Any]) -> pd.DataFrame:
        """
        Download a table from BEA.
        
        Args:
            source (Dict[str, Any]): Source configuration with link to BEA table
            
        Returns:
            pd.DataFrame: DataFrame with the downloaded data
        """
        url = source.get('link')
        if not url:
            raise ValueError(f"No URL provided for source: {source.get('name')}")
        
        logger.info(f"Downloading BEA table from {url}")
        
        try:
            # First try API approach
            df = self._download_table_api(source)
            return df
        except Exception as e:
            logger.warning(f"API approach failed: {str(e)}. Trying web scraping approach...")
            try:
                # Fall back to web scraping approach
                df = self._download_table_scrape(source)
                return df
            except Exception as e2:
                logger.error(f"Both approaches failed. API error: {str(e)}. Web scraping error: {str(e2)}")
                raise ValueError(f"Failed to download BEA table: {source.get('name')}. Tried both API and web scraping approaches.")
    
    def _download_table_api(self, source: Dict[str, Any]) -> pd.DataFrame:
        """
        Download a table from BEA using the API.
        
        Args:
            source (Dict[str, Any]): Source configuration with link to BEA table
            
        Returns:
            pd.DataFrame: DataFrame with the downloaded data
        """
        # Extract parameters from URL
        url = source.get('link')
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # Get table name from source or URL
        table_name = source.get('table')
        if not table_name and 'nipa_table_list' in query_params:
            table_name = query_params['nipa_table_list'][0]
        
        if not table_name:
            raise ValueError(f"Could not determine table name for source: {source.get('name')}")
        
        # Format table name for API
        # For NIPA tables, the format is typically like "T10105" for table 1.1.5
        # Remove dots and add T prefix if not already there
        if not table_name.startswith('T'):
            formatted_table = f"T{table_name.replace('.', '')}"
        else:
            formatted_table = table_name.replace('.', '')
        
        # Get frequency from source
        frequency = source.get('frequency', 'A')
        
        # Try multiple table name formats
        table_formats = [
            formatted_table,                                # Primary format (e.g., "T115")
            f"T{table_name}",                              # With T prefix, keeping dots (e.g., "T1.1.5")
            table_name,                                    # As is (e.g., "1.1.5")
            table_name.replace('.', ''),                   # No dots (e.g., "115")
            f"{table_name.split('.')[0]}",                 # Just first number (e.g., "1")
            f"T{table_name.split('.')[0]}",                # T prefix + first number (e.g., "T1")
            f"{table_name.replace('.', '_')}",             # Replace dots with underscores (e.g., "1_1_5")
            f"T{table_name.replace('.', '_')}",            # T prefix + underscores (e.g., "T1_1_5")
            f"Table{table_name.replace('.', '')}",         # "Table" prefix (e.g., "Table115")
            f"Table_{table_name.replace('.', '')}",        # "Table_" prefix (e.g., "Table_115")
            f"NIPA{table_name.replace('.', '')}",          # "NIPA" prefix (e.g., "NIPA115")
            f"NIPA_{table_name.replace('.', '')}",         # "NIPA_" prefix (e.g., "NIPA_115")
        ]
        
        # Add additional formats for specific table patterns
        if len(table_name.split('.')) >= 3:
            # For tables like 1.1.5, also try formats like T1-1-5
            parts = table_name.split('.')
            table_formats.extend([
                f"T{'-'.join(parts)}",                     # T prefix + dash separator (e.g., "T1-1-5")
                f"{'-'.join(parts)}",                      # Just dash separator (e.g., "1-1-5")
                f"T{parts[0]}{parts[1]}{parts[2]}",        # T prefix + no separators (e.g., "T115")
            ])
        
        # Make API request with each format until one works
        last_error = None
        for table_format in table_formats:
            try:
                # Make API request
                api_url = "https://apps.bea.gov/api/data"
                api_params = {
                    "UserID": self.api_key,
                    "method": "GetData",
                    "DatasetName": "NIPA",
                    "TableName": table_format,
                    "Frequency": frequency,
                    "Year": "ALL",  # Use ALL instead of X
                    "ResultFormat": "JSON"
                }
                
                logger.info(f"Making API request with parameters: {api_params}")
                
                response = requests.get(api_url, params=api_params)
                response.raise_for_status()
                data = response.json()
                
                # Check for errors
                if "BEAAPI" in data:
                    if "Results" in data["BEAAPI"] and "Data" in data["BEAAPI"]["Results"]:
                        # Success! Process the response
                        result_data = data["BEAAPI"]["Results"]["Data"]
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(result_data)
                        
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
                        
                        logger.info(f"Successfully retrieved data using table format: {table_format}")
                        return df
                    elif "Error" in data["BEAAPI"]:
                        # API returned an error
                        error = data["BEAAPI"]["Error"]
                        logger.warning(f"API error with table format {table_format}: {error}")
                        last_error = error
                    elif "Error" in data["BEAAPI"].get("Results", {}):
                        # API returned an error in Results
                        error = data["BEAAPI"]["Results"]["Error"]
                        logger.warning(f"API error with table format {table_format}: {error}")
                        last_error = error
                    else:
                        # Unknown structure
                        logger.warning(f"Unknown API response structure with table format {table_format}: {data}")
                        last_error = f"Unknown API response structure: {data}"
                else:
                    # Unknown structure
                    logger.warning(f"Unknown API response structure with table format {table_format}: {data}")
                    last_error = f"Unknown API response structure: {data}"
            except Exception as e:
                logger.warning(f"Error with table format {table_format}: {str(e)}")
                last_error = str(e)
        
        # If we get here, all formats failed
        error_msg = f"No data found in API response after trying multiple table formats. Last error: {last_error}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    def _download_table_scrape(self, source: Dict[str, Any]) -> pd.DataFrame:
        """
        Download a table from BEA using web scraping.
        
        Args:
            source (Dict[str, Any]): Source configuration with link to BEA table
            
        Returns:
            pd.DataFrame: DataFrame with the downloaded data
        """
        import time
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, NoSuchElementException
        
        url = source.get('link')
        table_name = source.get('table')
        
        logger.info(f"Scraping BEA table data from: {url}")
        logger.info(f"Looking for table: {table_name}")
        
        # Set up headless Chrome browser
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")  # Set window size to ensure elements are visible
        
        # Create a temporary directory for downloads
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": str(temp_path),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False
        })
        
        driver = None
        try:
            # Initialize the driver
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            
            # Wait for the page to load
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
                )
            except TimeoutException:
                # If no table is found, try waiting for any content to load
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.ID, "MainContainer"))
                )
            
            # Log the page title and URL for debugging
            logger.info(f"Page title: {driver.title}")
            logger.info(f"Current URL: {driver.current_url}")
            
            # Take a screenshot for debugging
            screenshot_path = os.path.join(temp_dir, "bea_page.png")
            driver.save_screenshot(screenshot_path)
            logger.info(f"Saved screenshot to {screenshot_path}")
            
            # Check if we need to navigate to a specific section first
            if "step=2" in url:
                # We're at the table selection page, need to find and click on the specific table
                try:
                    # Look for the table in the list
                    table_link = None
                    
                    # Try different approaches to find the table
                    try:
                        # First try: Look for exact table number in text
                        table_link = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, f"//a[contains(text(), 'Table {table_name}')]"))
                        )
                    except TimeoutException:
                        try:
                            # Second try: Look for table in any element containing the table number
                            table_link = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.XPATH, f"//*[contains(text(), '{table_name}')]"))
                            )
                        except TimeoutException:
                            # Third try: Look for links with the table number in the URL
                            links = driver.find_elements(By.TAG_NAME, "a")
                            for link in links:
                                href = link.get_attribute("href")
                                if href and table_name in href:
                                    table_link = link
                                    break
                    
                    if table_link:
                        logger.info(f"Found table link: {table_link.text}")
                        table_link.click()
                        time.sleep(5)  # Wait for the table page to load
                    else:
                        logger.warning(f"Could not find link for table {table_name}")
                except Exception as e:
                    logger.warning(f"Error finding table link: {str(e)}")
            
            # Now we should be on the table page, look for the download button
            try:
                download_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Download')]"))
                )
                logger.info("Found download button, clicking...")
                download_button.click()
                time.sleep(2)  # Wait for download options to appear
            except Exception as e:
                logger.warning(f"Could not find download button: {str(e)}")
                # Try alternative selectors
                try:
                    download_button = driver.find_element(By.CSS_SELECTOR, "button.download-button")
                    download_button.click()
                    time.sleep(2)
                except:
                    try:
                        download_button = driver.find_element(By.XPATH, "//button[contains(@class, 'download')]")
                        download_button.click()
                        time.sleep(2)
                    except:
                        logger.warning("Could not find download button with any selector")
                        
                        # Try to scrape the table directly from the page
                        try:
                            logger.info("Attempting to scrape table directly from page")
                            tables = pd.read_html(driver.page_source)
                            if tables:
                                logger.info(f"Found {len(tables)} tables on the page, using the first one")
                                return tables[0]
                        except Exception as e:
                            logger.error(f"Failed to scrape table directly: {str(e)}")
            
            # Select CSV format
            try:
                csv_option = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'CSV')]"))
                )
                logger.info("Found CSV option, clicking...")
                csv_option.click()
                time.sleep(5)
            except Exception as e:
                logger.warning(f"Could not find CSV option: {str(e)}")
                # Try alternative selector
                try:
                    csv_option = driver.find_element(By.CSS_SELECTOR, "a[data-format='csv']")
                    csv_option.click()
                    time.sleep(5)
                except:
                    logger.warning("Could not find CSV option with alternative selector")
            
            # Wait for download to complete
            time.sleep(10)
            
            # Find the downloaded file
            csv_files = list(temp_path.glob("*.csv"))
            if not csv_files:
                logger.warning("No CSV files found in download directory")
                # Try to get the table directly from the page
                try:
                    logger.info("Attempting to scrape table directly from page")
                    tables = pd.read_html(driver.page_source)
                    if tables:
                        logger.info(f"Found {len(tables)} tables on the page, using the first one")
                        df = tables[0]
                    else:
                        raise ValueError("No tables found on the page")
                except Exception as e:
                    logger.error(f"Failed to scrape table directly: {str(e)}")
                    
                    # Last resort: try to extract data from any visible table on the page
                    try:
                        logger.info("Attempting to extract data from visible table elements")
                        table_elements = driver.find_elements(By.TAG_NAME, "table")
                        if table_elements:
                            logger.info(f"Found {len(table_elements)} table elements")
                            # Extract data from the first table
                            rows = table_elements[0].find_elements(By.TAG_NAME, "tr")
                            data = []
                            for row in rows:
                                cols = row.find_elements(By.TAG_NAME, "td")
                                if not cols:  # Try th if td is empty (header row)
                                    cols = row.find_elements(By.TAG_NAME, "th")
                                row_data = [col.text for col in cols]
                                if row_data:  # Only add non-empty rows
                                    data.append(row_data)
                            
                            # Convert to DataFrame
                            if data:
                                headers = data[0]
                                df = pd.DataFrame(data[1:], columns=headers)
                                logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
                            else:
                                raise ValueError("No data found in table elements")
                        else:
                            raise ValueError("No table elements found on the page")
                    except Exception as e:
                        logger.error(f"Failed to extract data from table elements: {str(e)}")
                        raise ValueError(f"Could not download or scrape table data: {str(e)}")
            else:
                # Read the CSV file
                latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"Reading downloaded file: {latest_file}")
                df = pd.read_csv(latest_file)
                logger.info(f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns")
            
            # Process the DataFrame
            # Try to identify date columns
            date_cols = [col for col in df.columns if any(year in col for year in [str(y) for y in range(1900, 2100)])]
            
            if date_cols:
                # Wide format (years as columns)
                logger.info(f"Found date columns: {date_cols}")
                
                # Melt the DataFrame to convert from wide to long format
                id_vars = [col for col in df.columns if col not in date_cols]
                df_melted = pd.melt(
                    df,
                    id_vars=id_vars,
                    value_vars=date_cols,
                    var_name='date',
                    value_name='value'
                )
                
                # Convert date strings to datetime
                df_melted['date'] = pd.to_datetime(df_melted['date'], errors='coerce')
                
                # Convert values to float
                df_melted['value'] = pd.to_numeric(df_melted['value'], errors='coerce')
                
                # Set date as index
                df_melted = df_melted.set_index('date')
                
                # Sort by date
                df_melted = df_melted.sort_index()
                
                return df_melted
            else:
                # Try to find date column
                for col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        if not df[col].isna().all():
                            logger.info(f"Found date column: {col}")
                            df = df.set_index(col)
                            break
                    except:
                        continue
                
                # If no date column found, use the first column as index
                if df.index.name is None:
                    logger.warning("No date column found, using first column as index")
                    df = df.set_index(df.columns[0])
                
                # Try to convert values to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            
        except Exception as e:
            logger.error(f"Error scraping BEA table: {str(e)}")
            raise
        finally:
            if driver:
                driver.quit() 