"""
Visualization module for the Stock-Watson interpolation procedure.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any
from .config import Config

logger = logging.getLogger(__name__)

class Visualization:
    """Class for generating visualizations of the economic data."""
    
    def __init__(self, output_dir: Path, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the visualization class.
        
        Args:
            output_dir: Directory to save visualizations
            config: Optional configuration dictionary. If not provided, uses default config.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get configuration
        self.config = config or Config().get('visualization', {})
        self._configure_matplotlib()
    
    def _configure_matplotlib(self):
        """Configure matplotlib based on settings."""
        plt.rcParams.update(self.config.get('matplotlib', {}))
        sns.set(style=self.config.get('seaborn_style', "whitegrid"))
    
    def create_correlation_heatmap(self, df: pd.DataFrame, title: str = 'Correlation Matrix') -> Path:
        """
        Create and save a correlation heatmap.
        
        Args:
            df: DataFrame to create correlation matrix from
            title: Title for the plot
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=self.config['figure_sizes']['correlation'])
        corr_matrix = df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                   linewidths=0.5, vmin=-1, vmax=1)
        plt.title(title)
        plt.tight_layout()
        
        output_path = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(output_path, dpi=self.config.get('dpi', 300))
        plt.close()
        
        logger.info(f"Saved correlation heatmap to {output_path}")
        return output_path
    
    def create_time_series_plot(self, df: pd.DataFrame, title: str = 'Time Series Plot') -> Path:
        """Create and save a time series plot."""
        plt.figure(figsize=self.config['figure_sizes']['time_series'])
        
        # If date is not the index, try to create a date index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            else:
                df = df.copy()
                df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='ME')
        
        # Plot first 5 variables
        df.iloc[:, :5].plot()
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.tight_layout()
        
        output_path = self.output_dir / 'time_series_plot.png'
        plt.savefig(output_path, dpi=self.config.get('dpi', 300))
        plt.close()
        
        logger.info(f"Saved time series plot to {output_path}")
        return output_path
    
    def create_distribution_plots(self, df: pd.DataFrame) -> Path:
        """Create and save distribution plots for each variable."""
        plt.figure(figsize=self.config['figure_sizes']['distribution'])
        
        for i, col in enumerate(df.columns[:min(9, len(df.columns))]):
            plt.subplot(3, 3, i+1)
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        
        output_path = self.output_dir / 'distribution_plots.png'
        plt.savefig(output_path, dpi=self.config.get('dpi', 300))
        plt.close()
        
        logger.info(f"Saved distribution plots to {output_path}")
        return output_path
    
    def create_boxplot(self, df: pd.DataFrame) -> Path:
        """Create and save a boxplot of all variables."""
        plt.figure(figsize=self.config['figure_sizes']['boxplot'])
        df_melt = df.melt(var_name='Variable', value_name='Value')
        sns.boxplot(x='Variable', y='Value', data=df_melt)
        plt.title('Boxplot of All Variables')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        output_path = self.output_dir / 'boxplot.png'
        plt.savefig(output_path, dpi=self.config.get('dpi', 300))
        plt.close()
        
        logger.info(f"Saved boxplot to {output_path}")
        return output_path
    
    def generate_all_plots(self, df: pd.DataFrame) -> Dict[str, Path]:
        """Generate all available plots for the data."""
        plots = {
            'correlation': self.create_correlation_heatmap(df),
            'time_series': self.create_time_series_plot(df),
            'distribution': self.create_distribution_plots(df),
            'boxplot': self.create_boxplot(df)
        }
        
        logger.info(f"Generated all plots in {self.output_dir}")
        return plots 