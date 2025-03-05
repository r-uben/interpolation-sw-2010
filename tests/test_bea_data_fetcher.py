import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
import sys

# Remove the sys.path.insert line since we're using Poetry

from interpolation_sw_2010.bea.client import BEAClient
from interpolation_sw_2010.bea.services.data_fetcher import BEADataFetcher
from interpolation_sw_2010.data_fetcher import DataFetcher


class TestBEADataFetcher(unittest.TestCase):
    """
    Test cases for the BEA data fetcher.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock BEA client
        self.mock_client = MagicMock()
        self.mock_client.api_key = "test_api_key"
        
        # Create a BEA data fetcher with the mock client
        self.data_fetcher = BEADataFetcher(self.mock_client)
        
        # Sample source configuration
        self.sample_source = {
            'name': 'test_source',
            'type': 'bea',
            'description': 'Test Source',
            'link': 'https://apps.bea.gov/iTable/?reqid=19&step=2&isuri=1&categories=survey&nipa_table_list=1.1.5',
            'table': '1.1.5',
            'start_date': '2020-01-01',
            'end_date': '2022-12-31',
            'frequency': 'Q'
        }
    
    @patch('interpolation_sw_2010.bea.services.data_fetcher.webdriver.Chrome')
    def test_download_bea_table_scrape(self, mock_chrome):
        """
        Test the web scraping method for downloading BEA tables.
        """
        # Mock the WebDriver and its methods
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        
        # Mock the table HTML
        mock_table_html = """
        <table class="DataTable">
            <tr>
                <th>Line</th>
                <th>Description</th>
                <th>2020Q1</th>
                <th>2020Q2</th>
            </tr>
            <tr>
                <td>1</td>
                <td>Gross domestic product</td>
                <td>21481.4</td>
                <td>19520.1</td>
            </tr>
        </table>
        """
        
        # Mock the find_element method
        mock_element = MagicMock()
        mock_element.get_attribute.return_value = mock_table_html
        mock_driver.find_element.return_value = mock_element
        
        # Mock pandas.read_html to return a DataFrame
        sample_df = pd.DataFrame({
            'Line': ['1'],
            'Description': ['Gross domestic product'],
            '2020Q1': [21481.4],
            '2020Q2': [19520.1]
        })
        
        with patch('pandas.read_html', return_value=[sample_df]):
            # Call the method
            result = self.data_fetcher._download_bea_table_scrape(self.sample_source)
            
            # Check that the result is a DataFrame
            self.assertIsInstance(result, pd.DataFrame)
            
            # Check that the DataFrame has the expected structure
            self.assertIn('value', result.columns)
            self.assertIn('line', result.columns)
            self.assertIn('description', result.columns)
            
            # Check that the DataFrame has the expected values
            self.assertEqual(len(result), 2)  # Two quarters
    
    @patch('requests.get')
    def test_download_bea_table_direct(self, mock_get):
        """
        Test the direct API method for downloading BEA tables.
        """
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "BEAAPI": {
                "Results": {
                    "Data": [
                        {
                            "TableName": "T10105",
                            "SeriesCode": "A191RC",
                            "LineNumber": "1",
                            "LineDescription": "Gross domestic product",
                            "TimePeriod": "2020Q1",
                            "CL_UNIT": "Billions of dollars",
                            "UNIT_MULT": "6",
                            "DataValue": "21481.4",
                            "NoteRef": "None"
                        },
                        {
                            "TableName": "T10105",
                            "SeriesCode": "A191RC",
                            "LineNumber": "1",
                            "LineDescription": "Gross domestic product",
                            "TimePeriod": "2020Q2",
                            "CL_UNIT": "Billions of dollars",
                            "UNIT_MULT": "6",
                            "DataValue": "19520.1",
                            "NoteRef": "None"
                        }
                    ]
                }
            }
        }
        mock_get.return_value = mock_response
        
        # Call the method
        result = self.data_fetcher.download_bea_table_direct(self.sample_source)
        
        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that the DataFrame has the expected structure
        self.assertIn('value', result.columns)
        
        # Check that the DataFrame has the expected values
        self.assertEqual(len(result), 2)  # Two quarters
    
    @patch.object(BEADataFetcher, '_download_bea_table_scrape')
    @patch.object(BEADataFetcher, 'download_bea_table_direct')
    def test_download_bea_table_fallback(self, mock_direct, mock_scrape):
        """
        Test that the download_bea_table method falls back to the direct API method
        if web scraping fails.
        """
        # Mock the web scraping method to raise an exception
        mock_scrape.side_effect = Exception("Web scraping failed")
        
        # Mock the direct API method to return a DataFrame
        sample_df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=2, freq='Q'),
            'value': [21481.4, 19520.1],
            'SeriesName': ['Gross domestic product', 'Gross domestic product'],
            'LineNumber': ['1', '1'],
            'LineDescription': ['Gross domestic product', 'Gross domestic product']
        }).set_index('date')
        mock_direct.return_value = sample_df
        
        # Call the method
        result = self.data_fetcher.download_bea_table(self.sample_source)
        
        # Check that the web scraping method was called
        mock_scrape.assert_called_once_with(self.sample_source)
        
        # Check that the direct API method was called as a fallback
        mock_direct.assert_called_once_with(self.sample_source)
        
        # Check that the result is the DataFrame from the direct API method
        self.assertIs(result, sample_df)
    
    @patch.object(BEADataFetcher, '_download_bea_table_scrape')
    @patch.object(BEADataFetcher, 'download_bea_table_direct')
    def test_download_bea_table_both_fail(self, mock_direct, mock_scrape):
        """
        Test that the download_bea_table method raises an exception if both
        web scraping and direct API methods fail.
        """
        # Mock both methods to raise exceptions
        mock_scrape.side_effect = Exception("Web scraping failed")
        mock_direct.side_effect = Exception("Direct API failed")
        
        # Call the method and check that it raises an exception
        with self.assertRaises(ValueError):
            self.data_fetcher.download_bea_table(self.sample_source)
        
        # Check that both methods were called
        mock_scrape.assert_called_once_with(self.sample_source)
        mock_direct.assert_called_once_with(self.sample_source)


class TestDataFetcher(unittest.TestCase):
    """
    Test cases for the main data fetcher.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock configuration
        self.mock_config = {
            'bea': {
                'api_key': 'test_api_key'
            },
            'sources': [
                {
                    'name': 'test_source',
                    'type': 'bea',
                    'description': 'Test Source',
                    'link': 'https://apps.bea.gov/iTable/?reqid=19&step=2&isuri=1&categories=survey&nipa_table_list=1.1.5',
                    'table': '1.1.5',
                    'start_date': '2020-01-01',
                    'end_date': '2022-12-31',
                    'frequency': 'Q'
                }
            ]
        }
    
    @patch('interpolation_sw_2010.data_fetcher.DataFetcher._load_config')
    @patch('interpolation_sw_2010.data_fetcher.BEAClient')
    def test_fetch_data(self, mock_bea_client, mock_load_config):
        """
        Test the fetch_data method.
        """
        # Mock the configuration
        mock_load_config.return_value = self.mock_config
        
        # Mock the BEA client
        mock_client = MagicMock()
        mock_bea_client.return_value = mock_client
        
        # Mock the download_table method
        sample_df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=2, freq='Q'),
            'value': [21481.4, 19520.1],
            'line': ['1', '1'],
            'description': ['Gross domestic product', 'Gross domestic product']
        }).set_index('date')
        mock_client.download_table.return_value = sample_df
        
        # Create a data fetcher
        data_fetcher = DataFetcher()
        
        # Call the method
        result = data_fetcher.fetch_data('test_source')
        
        # Check that the BEA client was initialized with the correct configuration
        mock_bea_client.assert_called_once_with(self.mock_config.get('bea', {}))
        
        # Check that the download_table method was called with the correct source
        mock_client.download_table.assert_called_once()
        source_arg = mock_client.download_table.call_args[0][0]
        self.assertEqual(source_arg['name'], 'test_source')
        
        # Check that the result is the DataFrame from the BEA client
        self.assertIs(result, sample_df)
    
    @patch('interpolation_sw_2010.data_fetcher.DataFetcher._load_config')
    def test_list_sources(self, mock_load_config):
        """
        Test the list_sources method.
        """
        # Mock the configuration
        mock_load_config.return_value = self.mock_config
        
        # Create a data fetcher
        data_fetcher = DataFetcher()
        
        # Call the method
        result = data_fetcher.list_sources()
        
        # Check that the result is the list of source names
        self.assertEqual(result, ['test_source'])


if __name__ == '__main__':
    unittest.main() 