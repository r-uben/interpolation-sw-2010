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
        
        # Define units for each component
        self.component_units = {
            'PCE': 'Billions of Dollars',
            'I_NS': 'Billions of Dollars',
            'I_ES': 'Billions of Dollars',
            'I_RS': 'Billions of Dollars',
            'I_chPI': 'Billions of Dollars',
            'X': 'Billions of Dollars',
            'IM': 'Billions of Dollars',
            'G': 'Billions of Dollars',
            'PGDP': 'Index (2017=100)',
            'GDP_nominal': 'Billions of Dollars',
            'GDP_real': 'Billions of 2017 Dollars'
        }
    
    def _configure_matplotlib(self):
        """Configure matplotlib based on settings."""
        # Set up Latin Modern Roman font with larger text sizes
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Latin Modern Roman'],
            'mathtext.fontset': 'cm',
            'axes.titlesize': 18,
            'axes.labelsize': 15,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 15,
            'figure.constrained_layout.use': False,
            'figure.autolayout': False,
            # Configure default spine visibility
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True
        })
        sns.set_style("whitegrid", {
            'grid.linestyle': ':',
            'grid.alpha': 0.3,
            'axes.facecolor': 'white',
            'axes.grid': True,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': False,
            'axes.spines.top': False
        })
    
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
    
    def visualize_quarterly_series(self, quarterly_df: pd.DataFrame, 
                                 components: Optional[list] = None) -> Dict[str, Path]:
        """
        Visualize quarterly time series that are being interpolated.
        
        Creates individual plots for each quarterly series and a combined plot 
        with all selected components.
        
        Args:
            quarterly_df: DataFrame with quarterly data
            components: Optional list of components to plot. If None, plots all columns except Year/Quarter.
            
        Returns:
            Dictionary mapping component names to paths of saved plots
        """
        logger.info(f"Visualizing quarterly series. Columns: {quarterly_df.columns.tolist()}")
        
        # Create quarterly date index
        if 'Year' in quarterly_df.columns and 'Quarter' in quarterly_df.columns:
            quarterly_dates = []
            for _, row in quarterly_df.iterrows():
                # Map quarter to actual quarter (Q1, Q2, Q3, Q4)
                quarter_str = f"Q{int(row['Quarter'])}"
                # Create a period index which is better for quarterly data
                quarterly_dates.append(f"{int(row['Year'])}-{quarter_str}")
            
            quarterly_df = quarterly_df.copy()
            quarterly_df.index = pd.PeriodIndex(quarterly_dates, freq='Q')
            logger.info(f"Created quarterly date index. Shape: {quarterly_df.shape}")
        else:
            logger.warning(f"Could not create quarterly date index. Missing Year/Quarter columns.")
            return {}
        
        # Identify columns to plot
        exclude_cols = ['Year', 'Quarter']
        if components is None:
            components = [col for col in quarterly_df.columns if col not in exclude_cols]
        else:
            # Filter to only include columns that exist in the DataFrame
            components = [col for col in components if col in quarterly_df.columns]
        
        logger.info(f"Components to plot: {components}")
        
        # Create individual plots for each component
        plot_paths = {}
        for component in components:
            logger.info(f"Creating plot for {component}")
            plt.figure(figsize=self.config['figure_sizes'].get('time_series', (12, 8)))
            
            # Plot the quarterly series
            quarterly_df[component].plot(marker='o', linestyle='-', 
                                      label=f'{component}', 
                                      markersize=4)
            
            plt.title(f'Quarterly Series: {component}')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            output_path = self.output_dir / f'quarterly_{component}.png'
            logger.info(f"Saving plot to {output_path}")
            plt.savefig(output_path, dpi=self.config.get('dpi', 300))
            plt.close()
            
            plot_paths[component] = output_path
            logger.info(f"Saved plot for {component} to {output_path}")
        
        # Create a combined plot with all components (if there are multiple)
        if len(components) > 1:
            try:
                plt.figure(figsize=self.config['figure_sizes'].get('time_series', (14, 10)))
                
                # Plot each component
                for component in components:
                    # Normalize to first value to compare on same scale
                    series = quarterly_df[component]
                    normalized = series / series.iloc[0] * 100 if len(series) > 0 else series
                    normalized.plot(label=component)
                
                plt.title('Comparison of Quarterly Series (Normalized to 100)')
                plt.ylabel('Value (Normalized)')
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the plot
                output_path = self.output_dir / 'quarterly_all_components.png'
                plt.savefig(output_path, dpi=self.config.get('dpi', 300))
                plt.close()
                
                plot_paths['all_components'] = output_path
                logger.info(f"Saved combined plot to {output_path}")
                
                # Also create a non-normalized version
                plt.figure(figsize=self.config['figure_sizes'].get('time_series', (14, 10)))
                quarterly_df[components].plot()
                plt.title('Comparison of Quarterly Series (Original Values)')
                plt.ylabel('Value')
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the plot
                output_path = self.output_dir / 'quarterly_all_components_raw.png'
                plt.savefig(output_path, dpi=self.config.get('dpi', 300))
                plt.close()
                
                plot_paths['all_components_raw'] = output_path
                logger.info(f"Saved non-normalized combined plot to {output_path}")
            except Exception as e:
                logger.warning(f"Failed to create combined plot: {e}")
        
        return plot_paths
    
    def compare_quarterly_and_monthly(self, quarterly_df: pd.DataFrame, monthly_df: pd.DataFrame, 
                                    components: Optional[list] = None) -> Dict[str, Path]:
        """
        Compare original quarterly data with interpolated monthly data.
        
        Creates comparison plots for each variable, showing how the monthly interpolated
        data compares to the original quarterly data. The quarterly data points are plotted
        at the middle month of each quarter (Feb, May, Aug, Nov).
        
        Args:
            quarterly_df: DataFrame with quarterly data
            monthly_df: DataFrame with monthly interpolated data
            components: Optional list of components to plot. If None, plots all common columns.
            
        Returns:
            Dictionary mapping component names to paths of saved plots
        """
        logger.info(f"Starting comparison plots. Quarterly columns: {quarterly_df.columns.tolist()}")
        logger.info(f"Monthly columns: {monthly_df.columns.tolist()}")
        logger.info(f"Requested components: {components}")
        
        # Create quarterly date index
        if 'Year' in quarterly_df.columns and 'Quarter' in quarterly_df.columns:
            quarterly_dates = []
            for _, row in quarterly_df.iterrows():
                # Map quarter to middle month (Q1->Feb, Q2->May, Q3->Aug, Q4->Nov)
                month = int(row['Quarter']) * 3 - 1
                quarterly_dates.append(pd.Timestamp(int(row['Year']), month, 15))
            
            quarterly_df = quarterly_df.copy()
            quarterly_df['date'] = quarterly_dates
            quarterly_df = quarterly_df.set_index('date')
            logger.info(f"Created quarterly date index. Shape: {quarterly_df.shape}")
        else:
            logger.warning(f"Could not create quarterly date index. Missing Year/Quarter columns.")
        
        # Create monthly date index
        if 'Year' in monthly_df.columns and 'Month' in monthly_df.columns:
            monthly_dates = []
            for _, row in monthly_df.iterrows():
                monthly_dates.append(pd.Timestamp(int(row['Year']), int(row['Month']), 15))
            
            monthly_df = monthly_df.copy()
            monthly_df['date'] = monthly_dates
            monthly_df = monthly_df.set_index('date')
            logger.info(f"Created monthly date index. Shape: {monthly_df.shape}")
        else:
            logger.warning(f"Could not create monthly date index. Missing Year/Month columns.")
        
        # Identify common columns to plot
        exclude_cols = ['Year', 'Quarter', 'Month', 'date']
        if components is None:
            components = [col for col in quarterly_df.columns 
                         if col in monthly_df.columns and col not in exclude_cols]
        
        logger.info(f"Components to plot: {components}")
        
        # Plot comparison for each component
        plot_paths = {}
        for component in components:
            if component in quarterly_df.columns and component in monthly_df.columns:
                logger.info(f"Creating plot for {component}")
                
                # Create a figure with a main plot and an inset zoom
                fig, ax = plt.subplots(figsize=self.config['figure_sizes'].get('comparison', (14, 8)))
                
                # Set up the plot style with more vertical space
                plt.subplots_adjust(bottom=0.2, top=0.9)  # Increased bottom margin for footnote
                
                # First plot quarterly data as points WITHOUT connecting lines (in the background)
                ax.plot(quarterly_df.index, quarterly_df[component], 
                       marker='o', 
                       linestyle='', 
                       linewidth=0,
                       markersize=4,  # Increased from 6
                       label=f'Quarterly (Original)', 
                       color='#d62728',
                       markeredgecolor='black',
                       markeredgewidth=0.5,
                       alpha=0.7,
                       zorder=2)
                
                # Then plot monthly data as a prominent line (on top)
                ax.plot(monthly_df.index, monthly_df[component],
                       label=f'Monthly (Interpolated)', 
                       linewidth=3.0,  # Increased from 2.5
                       color='#1f77b4',
                       linestyle='-',
                       alpha=1.0,
                       zorder=1)
                
                # Add title and labels with Latin Modern Roman font
                ax.set_title(f'Stock-Watson (2010) Interpolation: {component}', 
                            fontsize=21, fontweight='bold', pad=20)  # Increased from 14
                ax.set_xlabel('')
                unit = self.component_units.get(component, 'Value')
                ax.set_ylabel(unit, fontsize=18, labelpad=10)  # Increased from 12
                
                # Customize grid and spines
                ax.grid(True, linestyle=':', alpha=0.3)
                # Set spine visibility and style
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_linewidth(0.5)
                ax.spines['bottom'].set_linewidth(0.5)
                
                # Customize legend with larger font
                ax.legend(loc='best', fontsize=15, framealpha=0.9, edgecolor='none')  # Increased from 10
                
                # Adjust tick parameters for better spacing
                ax.tick_params(axis='both', which='major', pad=8, labelsize=14)  # Increased labelsize
                ax.tick_params(axis='x', rotation=0)
                
                # Add a zoom inset showing recent years (if data extends past 2000)
                try:
                    # Check if we have data past 2000
                    recent_quarterly = quarterly_df[quarterly_df.index >= pd.Timestamp('2013-01-01')]
                    recent_monthly = monthly_df[monthly_df.index >= pd.Timestamp('2013-01-01')]
                    
                    if not recent_quarterly.empty and not recent_monthly.empty:
                        # Create an inset axes - moved to the left side and higher up
                        axins = ax.inset_axes([0.15, 0.4, 0.4, 0.3])
                        
                        # Plot the recent data in the inset
                        axins.plot(recent_quarterly.index, recent_quarterly[component], 
                                marker='o', markersize=5, linestyle='',  # Increased from 4
                                color='#d62728', alpha=0.7, linewidth=0)
                        axins.plot(recent_monthly.index, recent_monthly[component], 
                                linewidth=2.5, color='#1f77b4')  # Increased from 2.0
                        
                        # Set title and style for inset with larger font
                        axins.set_title('Zoom: 2000-Present', fontsize=14)  # Increased from 9
                        axins.tick_params(labelsize=12)  # Increased from 8
                        axins.grid(alpha=0.2)
                        
                        # Indicate the zoomed region with a box
                        ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5)
                        
                        # Also update the zoom inset spines if it exists
                        axins.spines['top'].set_visible(False)
                        axins.spines['right'].set_visible(False)
                        axins.spines['left'].set_visible(True)
                        axins.spines['bottom'].set_visible(True)
                        axins.spines['left'].set_linewidth(0.5)
                        axins.spines['bottom'].set_linewidth(0.5)
                except Exception as e:
                    logger.warning(f"Could not create zoom inset: {e}")
                
                # Add a footnote with source and author - moved below the x-axis
                plt.figtext(0.99, 0.02,  # Keep y position at 0.02 for below x-axis
                          'Source: Stock-Watson (2010) GDP Interpolation | Author: Ruben Fernandez-Fuertes', 
                          horizontalalignment='right', 
                          fontsize=12,  # Increased from 8
                          fontstyle='italic',
                          family='Latin Modern Roman')
                
                # Save the plot with higher DPI for better quality and more padding
                output_path = self.output_dir / f'comparison_{component}.png'
                logger.info(f"Saving plot to {output_path}")
                plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)  # Increased padding
                plt.close()
                
                plot_paths[component] = output_path
                logger.info(f"Saved comparison plot for {component} to {output_path}")
            else:
                logger.warning(f"Component {component} not found in both quarterly and monthly data.")
        
        return plot_paths
    
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