"""
Service for fetching data from BEA.
"""
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any, List
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ...aws_manager.bucket_manager import BucketManager
from ...utils.logger import get_logger

logger = get_logger(__name__)

class DataFetcher:
    """Service for fetching and managing BEA data."""
    
    # Mapping of Stock-Watson components to BEA data sources
    SW_DATA_SOURCES = {
        # PCE
        'pce': {
            'table': 'T20600',  # Table 2.6 Personal Income and Disposable Personal Income
            'frequency': 'M',
            'description': 'Personal Consumption Expenditures',
            'line_number': '1'  # Line number in Table 2.6
        },
        # Government
        'government': {
            'table': 'T20700',  # Table 2.7 Wages and Salaries by Industry and Government
            'frequency': 'M',
            'description': 'Government Consumption Expenditures and Gross Investment',
            'line_number': '23'  # Line number for government wages
        },
        # Exports
        'exports': {
            'dataset': 'international',
            'table': 'ita',  # International Transactions Accounts
            'frequency': 'M',
            'description': 'Exports of Goods and Services',
            'direction': 'exports'
        },
        # Imports
        'imports': {
            'dataset': 'international',
            'table': 'ita',  # International Transactions Accounts
            'frequency': 'M',
            'description': 'Imports of Goods and Services',
            'direction': 'imports'
        },
        # Employee Compensation (GDI)
        'employee_compensation': {
            'table': 'T20600',  # Table 2.6
            'frequency': 'M',
            'description': 'Compensation of Employees',
            'line_number': '2'
        },
        # Proprietors' Income (GDI)
        'proprietors_income': {
            'table': 'T20600',  # Table 2.6
            'frequency': 'M',
            'description': 'Proprietors\' Income',
            'line_number': '7'
        },
        # Rental Income (GDI)
        'rental_income': {
            'table': 'T20600',  # Table 2.6
            'frequency': 'M',
            'description': 'Rental Income with Capital Consumption Adjustment',
            'line_number': '10'
        },
        # Net Interest (GDI)
        'net_interest': {
            'table': 'T20600',  # Table 2.6
            'frequency': 'M',
            'description': 'Personal Interest Income',
            'line_number': '12'
        },
        # Corporate Profits (GDI) - Quarterly only, needs special handling
        'corporate_profits': {
            'table': 'T11200',  # Table 1.12
            'frequency': 'Q',
            'description': 'Corporate Profits with Inventory Valuation and Capital Consumption Adjustments',
            'monthly_method': 'distribute_equally'  # Distribute quarterly values equally over months
        },
        # Investment - Fixed Investment (from NIUnderlyingDetail)
        'inv_fixed': {
            'dataset': 'niunderlyingdetail',
            'table': '5.7.5AM1',  # Updated from '5.7.5A' to '5.7.5AM1'
            'frequency': 'M',
            'description': 'Private Fixed Investment',
            'line_number': '1'  # Line number for total private fixed investment
        },
        # Investment - Change in Inventories (from NIUnderlyingDetail)
        'inv_ch': {
            'dataset': 'niunderlyingdetail',
            'table': '5.7.5BM1',  # Updated from '5.7.5B' to '5.7.5BM1'
            'frequency': 'M',
            'description': 'Change in Private Inventories',
            'line_number': '1'  # Line number for total change in private inventories
        }
    }
    
    def __init__(self, client=None):
        """
        Initialize the data fetcher.
        
        Args:
            client: BEA client instance
        """
        # We'll import BEAClient here to avoid circular imports
        # If client is provided, use it; otherwise create a new one
        if client is None:
            from ..core.bea_client import BEAClient
            self.client = BEAClient()
        else:
            self.client = client
            
        self.bucket_manager = BucketManager(bucket_name="macroeconomic-data")
    
    def _save_to_s3(self, data: pd.DataFrame, component: str, metadata: dict):
        """Save data and metadata to S3."""
        # Create metadata
        full_metadata = {
            'component': component,
            'title': metadata.get('title', ''),
            'table': metadata.get('table', ''),
            'frequency': metadata.get('frequency', ''),
            'last_updated': datetime.now().isoformat(),
            'source': 'Bureau of Economic Analysis (BEA)',
            'observation_start': data.index.min().isoformat() if not data.empty else '',
            'observation_end': data.index.max().isoformat() if not data.empty else '',
            'notes': metadata.get('notes', '')
        }
        
        # Save data
        data_path = f"bea/{component}/data.csv"
        self.bucket_manager.upload_file(
            file_content=data.to_csv().encode(),
            file_path=data_path,
            metadata={'content_type': 'text/csv'}
        )
        
        # Save metadata
        metadata_path = f"bea/{component}/metadata.json"
        self.bucket_manager.upload_file(
            file_content=json.dumps(full_metadata, indent=2).encode(),
            file_path=metadata_path,
            metadata={'content_type': 'application/json'}
        )
    
    def _save_locally(self, data: pd.DataFrame, component: str, metadata: dict):
        """Save data and metadata locally."""
        try:
            # Create base directory
            base_dir = Path('data/bea') / component
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Save data
            data_path = base_dir / "data.csv"
            data.to_csv(data_path)
            
            # Save metadata
            metadata_path = base_dir / "metadata.txt"
            with metadata_path.open('w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                    
            logger.info(f"Successfully saved {component} data locally")
            
        except Exception as e:
            logger.error(f"Error saving {component} locally: {str(e)}")
            raise

    def get_component_data(self, component: str, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Get data for a specific Stock-Watson component.
        
        Args:
            component: One of the components defined in SW_DATA_SOURCES
            start_date: Start date for the data (optional)
            end_date: End date for the data (optional)
            
        Returns:
            DataFrame with the component data
        """
        # Check if the component exists
        if component not in self.SW_DATA_SOURCES:
            raise ValueError(f"Unknown component: {component}. Valid components are: {list(self.SW_DATA_SOURCES.keys())}")
        
        # Get component configuration
        config = self.SW_DATA_SOURCES[component]
        
        try:
            # Different handling based on the source dataset
            if 'dataset' in config and config['dataset'] == 'international':
                # Handle international data (exports, imports)
                df = self.client.get_international_data(
                    table_name=config['table'],
                    frequency=config['frequency'],
                    direction=config.get('direction')
                )
            elif 'dataset' in config and config['dataset'] == 'niunderlyingdetail':
                # Handle NIUnderlyingDetail data (investment components)
                df = self.client.get_underlying_detail_data(
                    table_name=config['table'],
                    frequency=config['frequency'],
                    year="1967,1997"  # Use the years from the hash parameters
                )
                
                # Filter by line number if specified
                if 'line_number' in config:
                    df = df[df['LineNumber'] == config['line_number']]
            elif config.get('frequency') == 'Q' and config.get('monthly_method') == 'distribute_equally':
                # Handle quarterly data that needs to be distributed to monthly
                # (e.g., corporate profits)
                quarterly_df = self.client.get_nipa_table(
                    table_name=config['table'],
                    frequency='Q'
                )
                
                # Filter by line number if specified
                if 'line_number' in config:
                    quarterly_df = quarterly_df[quarterly_df['LineNumber'] == config['line_number']]
                
                # Convert quarterly to monthly by distributing values equally
                df = self._distribute_quarterly_to_monthly(quarterly_df)
            else:
                # Standard NIPA table data
                df = self.client.get_nipa_table(
                    table_name=config['table'],
                    frequency=config['frequency']
                )
                
                # Filter by line number if specified
                if 'line_number' in config:
                    df = df[df['LineNumber'] == config['line_number']]
            
            # Apply date filtering if specified
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
                
            # Create metadata
            metadata = {
                'component': component,
                'title': config['description'],
                'table': config['table'],
                'frequency': config['frequency'],
                'last_updated': datetime.now().isoformat(),
                'source': 'Bureau of Economic Analysis (BEA)',
                'notes': f"Stock-Watson component: {component}"
            }
            
            # Save to S3
            self._save_to_s3(df, component, metadata)
            
            # Save locally
            self._save_locally(df, component, metadata)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for component '{component}': {str(e)}")
            raise
    
    def _distribute_quarterly_to_monthly(self, quarterly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Distribute quarterly values equally across months.
        This is used for corporate profits which are only reported quarterly.
        
        Args:
            quarterly_df: DataFrame with quarterly data
            
        Returns:
            DataFrame with monthly data
        """
        monthly_data = []
        
        for idx, row in quarterly_df.iterrows():
            # Get the quarter
            year = idx.year
            quarter = idx.quarter
            
            # Determine the months in this quarter
            if quarter == 1:
                months = [1, 2, 3]
            elif quarter == 2:
                months = [4, 5, 6]
            elif quarter == 3:
                months = [7, 8, 9]
            else:  # quarter == 4
                months = [10, 11, 12]
            
            # Create a row for each month with value = quarterly_value / 3
            for month in months:
                monthly_data.append({
                    'date': pd.to_datetime(f"{year}-{month}-01") + pd.offsets.MonthEnd(0),
                    'value': row['value'] / 3,  # Distribute equally
                    'SeriesName': row.get('SeriesName', ''),
                    'LineNumber': row.get('LineNumber', ''),
                    'LineDescription': row.get('LineDescription', '')
                })
        
        # Convert to DataFrame
        monthly_df = pd.DataFrame(monthly_data)
        monthly_df = monthly_df.set_index('date')
        
        # Sort by date
        monthly_df = monthly_df.sort_index()
        
        return monthly_df
    
    def get_all_components(self, start_date=None, end_date=None) -> Dict[str, pd.DataFrame]:
        """
        Get data for all Stock-Watson components.
        
        Args:
            start_date: Start date for the data (optional)
            end_date: End date for the data (optional)
            
        Returns:
            Dictionary mapping component names to DataFrames
        """
        components_data = {}
        
        for component in self.SW_DATA_SOURCES.keys():
            logger.info(f"Fetching {component} data...")
            try:
                df = self.get_component_data(component, start_date, end_date)
                components_data[component] = df
                logger.info(f"Successfully fetched {component} data")
            except Exception as e:
                logger.error(f"Error fetching {component} data: {str(e)}")
                # Continue with other components even if one fails
        
        return components_data 

    def scrape_bea_table_data(self, source: Dict[str, Any]) -> pd.DataFrame:
        """
        Scrape data directly from BEA's interactive tables interface.
        
        Args:
            source (Dict[str, Any]): Source configuration with link to BEA table
            
        Returns:
            pd.DataFrame: DataFrame with the scraped data
        """
        import time
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
        
        url = source.get('link')
        if not url:
            raise ValueError(f"No URL provided for source: {source.get('name')}")
        
        logger.info(f"Scraping BEA table data from: {url}")
        
        # Set up headless Chrome browser
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            # Initialize the driver with webdriver-manager
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get(url)
            
            # Wait for the page to load
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".section-header"))
            )
            
            # Log the page title and URL for debugging
            logger.info(f"Page title: {driver.title}")
            logger.info(f"Current URL: {driver.current_url}")
            
            # Find and click on the appropriate section (based on table ID)
            table_id = source.get('table', '')
            section_number = table_id.split('.')[0] if '.' in table_id else ''
            
            if section_number:
                logger.info(f"Looking for section {section_number}")
                section_selector = f"a[href='#section{section_number}']"
                try:
                    section_element = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, section_selector))
                    )
                    section_element.click()
                    time.sleep(2)  # Wait for section to expand
                except Exception as e:
                    logger.warning(f"Could not find section {section_number}: {str(e)}")
                    # Take a screenshot for debugging
                    driver.save_screenshot(f"section_{section_number}_not_found.png")
            
            # Find and click on the specific table
            logger.info(f"Looking for table {table_id}")
            # Use XPath for more flexibility in finding the table
            table_xpath = f"//a[contains(text(), '{table_id}')]"
            try:
                table_element = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, table_xpath))
                )
                table_element.click()
                time.sleep(5)  # Wait for table to load
            except Exception as e:
                logger.warning(f"Could not find table {table_id}: {str(e)}")
                # Take a screenshot for debugging
                driver.save_screenshot(f"table_{table_id}_not_found.png")
                # Try to find any table as a fallback
                all_tables = driver.find_elements(By.XPATH, "//a[contains(@class, 'table-link')]")
                if all_tables:
                    logger.info(f"Found {len(all_tables)} tables, clicking the first one")
                    all_tables[0].click()
                    time.sleep(5)
            
            # Click on the modify button to open options
            try:
                modify_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.modify-button"))
                )
                modify_button.click()
            except Exception as e:
                logger.warning(f"Could not find modify button: {str(e)}")
                # Take a screenshot for debugging
                driver.save_screenshot("modify_button_not_found.png")
                # Try alternative selector
                try:
                    modify_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Modify')]")
                    modify_button.click()
                except:
                    logger.warning("Could not find modify button with alternative selector")
            
            # Check "Select All Years" option
            try:
                select_all_years = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='checkbox'][name='selectAllYears']"))
                )
                select_all_years.click()
            except Exception as e:
                logger.warning(f"Could not find 'Select All Years' checkbox: {str(e)}")
                # Take a screenshot for debugging
                driver.save_screenshot("select_all_years_not_found.png")
                # Try alternative selector
                try:
                    select_all_years = driver.find_element(By.XPATH, "//label[contains(text(), 'Select All Years')]/input")
                    select_all_years.click()
                except:
                    logger.warning("Could not find 'Select All Years' with alternative selector")
            
            # Click refresh table
            try:
                refresh_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.refresh-table-button"))
                )
                refresh_button.click()
                time.sleep(5)  # Wait for table to refresh
            except Exception as e:
                logger.warning(f"Could not find refresh button: {str(e)}")
                # Take a screenshot for debugging
                driver.save_screenshot("refresh_button_not_found.png")
                # Try alternative selector
                try:
                    refresh_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Refresh')]")
                    refresh_button.click()
                    time.sleep(5)
                except:
                    logger.warning("Could not find refresh button with alternative selector")
            
            # Click download button
            try:
                download_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.download-button"))
                )
                download_button.click()
            except Exception as e:
                logger.warning(f"Could not find download button: {str(e)}")
                # Take a screenshot for debugging
                driver.save_screenshot("download_button_not_found.png")
                # Try alternative selector
                try:
                    download_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Download')]")
                    download_button.click()
                except:
                    logger.warning("Could not find download button with alternative selector")
            
            # Select CSV format
            try:
                csv_option = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a[data-format='csv']"))
                )
                csv_option.click()
            except Exception as e:
                logger.warning(f"Could not find CSV option: {str(e)}")
                # Take a screenshot for debugging
                driver.save_screenshot("csv_option_not_found.png")
                # Try alternative selector
                try:
                    csv_option = driver.find_element(By.XPATH, "//a[contains(text(), 'CSV')]")
                    csv_option.click()
                except:
                    logger.warning("Could not find CSV option with alternative selector")
            
            # Wait for download to complete (this is tricky in headless mode)
            time.sleep(10)
            
            # Get the downloaded file path (this would need to be adjusted based on your environment)
            import glob
            import os
            download_dir = os.path.expanduser("~/Downloads")
            downloaded_files = glob.glob(f"{download_dir}/*.csv")
            
            if not downloaded_files:
                logger.error("No CSV files found in downloads directory")
                raise ValueError("Download failed - no CSV files found")
                
            latest_file = max(downloaded_files, key=os.path.getctime)
            logger.info(f"Found downloaded file: {latest_file}")
            
            # Read the CSV file
            df = pd.read_csv(latest_file)
            logger.info(f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns")
            
            # Process the DataFrame
            # Convert to proper format with dates as index
            # This will need to be customized based on the specific table format
            
            # Example processing (adjust as needed):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            elif 'Year' in df.columns and 'Month' in df.columns:
                df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))
                df = df.set_index('Date')
            
            logger.info(f"Successfully scraped data from {source.get('name')}")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping BEA table: {str(e)}")
            raise
        finally:
            if 'driver' in locals():
                driver.quit() 

    def download_bea_table(self, source: Dict[str, Any]) -> pd.DataFrame:
        """
        Download data from BEA website.
        
        Args:
            source (Dict[str, Any]): Source configuration with link to BEA table
            
        Returns:
            pd.DataFrame: DataFrame with the downloaded data
        """
        try:
            # First try web scraping approach
            return self._download_bea_table_scrape(source)
        except Exception as e:
            logger.warning(f"Web scraping approach failed: {str(e)}. Trying direct API approach...")
            try:
                # Fall back to direct API approach
                return self.download_bea_table_direct(source)
            except Exception as e2:
                logger.error(f"Both approaches failed. Web scraping error: {str(e)}. Direct API error: {str(e2)}")
                raise ValueError(f"Failed to download BEA table: {source.get('name')}. Tried both web scraping and direct API approaches.")
                
    def _download_bea_table_scrape(self, source: Dict[str, Any]) -> pd.DataFrame:
        """
        Download data from BEA website using web scraping.
        
        Args:
            source (Dict[str, Any]): Source configuration with link to BEA table
            
        Returns:
            pd.DataFrame: DataFrame with the downloaded data
        """
        url = source.get('link')
        if not url:
            raise ValueError(f"No URL provided for source: {source.get('name')}")
        
        logger.info(f"Downloading BEA table from {url}")
        
        # Initialize Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Initialize WebDriver
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # Navigate to the URL
            driver.get(url)
            
            # Wait for the table to load
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.DataTable"))
            )
            
            # Get the table HTML
            table_html = driver.find_element(By.CSS_SELECTOR, "table.DataTable").get_attribute('outerHTML')
            
            # Parse the table with pandas
            tables = pd.read_html(table_html)
            if not tables:
                raise ValueError(f"No tables found at {url}")
            
            df = tables[0]
            
            # Process the DataFrame
            # First row is usually the header
            if df.iloc[0, 0] == 'Line':
                df.columns = df.iloc[0]
                df = df.iloc[1:]
            
            # Rename columns
            df = df.rename(columns={
                'Line': 'line',
                'Description': 'description',
            })
            
            # Convert date columns to values
            date_columns = [col for col in df.columns if col not in ['line', 'description']]
            
            # Melt the DataFrame to convert from wide to long format
            df_melted = pd.melt(
                df,
                id_vars=['line', 'description'],
                value_vars=date_columns,
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
            
            logger.info(f"Successfully downloaded data with {len(df_melted)} rows")
            return df_melted
            
        except Exception as e:
            logger.error(f"Error downloading BEA table: {str(e)}")
            raise
        
        finally:
            if 'driver' in locals():
                driver.quit()

    def download_bea_table_direct(self, source: Dict[str, Any]) -> pd.DataFrame:
        """
        Download data directly from BEA API using the table ID extracted from the URL.
        This is an alternative approach that doesn't require web scraping.
        
        Args:
            source (Dict[str, Any]): Source configuration with link to BEA table
            
        Returns:
            pd.DataFrame: DataFrame with the downloaded data
        """
        import base64
        import json
        import requests
        
        url = source.get('link')
        if not url:
            raise ValueError(f"No URL provided for source: {source.get('name')}")
        
        logger.info(f"Extracting parameters from BEA URL: {url}")
        
        # Extract hash part from URL
        hash_part = url.split('#')[-1] if '#' in url else None
        if not hash_part:
            raise ValueError(f"No hash parameters found in URL: {url}")
        
        # Try to decode the hash part (it's usually base64 encoded)
        try:
            # Try standard base64 decoding
            decoded = base64.b64decode(hash_part).decode('utf-8')
        except:
            try:
                # If that fails, try URL-safe base64 decoding
                decoded = base64.urlsafe_b64decode(hash_part).decode('utf-8')
            except:
                # If that fails too, use the hash part as is
                decoded = hash_part
        
        logger.info(f"Decoded hash parameters: {decoded}")
        
        # Try to parse as JSON
        try:
            params = json.loads(decoded)
            logger.info(f"Parsed parameters: {params}")
        except:
            logger.warning(f"Could not parse parameters as JSON: {decoded}")
            params = {}
        
        # Extract relevant parameters
        table_id = source.get('table')
        if not table_id:
            # Try to extract from params
            if 'data' in params:
                for item in params['data']:
                    if item[0] == 'NIPA_Table_List':
                        table_id = item[1]
                        break
        
        if not table_id:
            raise ValueError(f"Could not determine table ID for source: {source.get('name')}")
        
        logger.info(f"Using table ID: {table_id}")
        
        # Extract year range
        start_year = None
        end_year = None
        if 'data' in params:
            for item in params['data']:
                if item[0] == 'First_Year':
                    start_year = item[1]
                elif item[0] == 'Last_Year':
                    end_year = item[1]
        
        # Use default years if not found
        if not start_year:
            start_year = source.get('start_date', '1967').split('-')[0]
        if not end_year:
            end_year = source.get('end_date', '2025').split('-')[0]
            if end_year == '9999':
                end_year = '2025'  # Use a reasonable default
        
        logger.info(f"Year range: {start_year} to {end_year}")
        
        # Extract frequency
        frequency = 'M'  # Default to monthly
        if 'data' in params:
            for item in params['data']:
                if item[0] == 'Series':
                    frequency = item[1]
                    break
        
        logger.info(f"Using frequency: {frequency}")
        
        # Construct API URL
        api_url = "https://apps.bea.gov/api/data"
        api_params = {
            "UserID": self.client.api_key,
            "method": "GetData",
            "DatasetName": "NIUnderlyingDetail",
            "TableName": table_id,
            "Frequency": frequency,
            "Year": f"{start_year},{end_year}",
            "ResultFormat": "JSON"
        }
        
        logger.info(f"Making direct API request with params: {api_params}")
        
        # Make the request
        try:
            response = requests.get(api_url, params=api_params)
            response.raise_for_status()
            data = response.json()
            
            # Check for errors
            if "BEAAPI" in data and "Results" in data["BEAAPI"]:
                if "Error" in data["BEAAPI"]["Results"]:
                    error = data["BEAAPI"]["Results"]["Error"]
                    logger.error(f"BEA API error: {error}")
                    
                    # If invalid parameters, try with different table name format
                    if "@APIErrorCode" in error and error["@APIErrorCode"] == "1":
                        # Try removing the 'M' suffix if present
                        if table_id.endswith('M1'):
                            new_table_id = table_id[:-2]
                            logger.info(f"Retrying with modified table ID: {new_table_id}")
                            api_params["TableName"] = new_table_id
                            response = requests.get(api_url, params=api_params)
                            response.raise_for_status()
                            data = response.json()
                        
                        # If still error, try with just the numeric part
                        if "BEAAPI" in data and "Results" in data["BEAAPI"] and "Error" in data["BEAAPI"]["Results"]:
                            # Extract just the numeric part (e.g., "5.7.5" -> "575")
                            numeric_id = ''.join(c for c in table_id if c.isdigit() or c == '.')
                            numeric_id = numeric_id.replace('.', '')
                            logger.info(f"Retrying with numeric table ID: {numeric_id}")
                            api_params["TableName"] = numeric_id
                            response = requests.get(api_url, params=api_params)
                            response.raise_for_status()
                            data = response.json()
            
            # Process the response
            if "BEAAPI" in data and "Results" in data["BEAAPI"] and "Data" in data["BEAAPI"]["Results"]:
                # Extract data from the nested JSON response
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
                
                logger.info(f"Successfully downloaded data with {len(df)} rows")
                return df
            else:
                logger.error(f"No data found in API response: {data}")
                raise ValueError("No data found in API response")
                
        except Exception as e:
            logger.error(f"Error downloading BEA table: {str(e)}")
            raise 