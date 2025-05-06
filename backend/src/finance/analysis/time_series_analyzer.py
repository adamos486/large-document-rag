"""
Financial Time Series Analyzer

This module provides tools for analyzing financial time series data, detecting trends,
seasonality, and generating forecasts. It handles the complexities of financial
time series analysis including handling of irregularly spaced data, fiscal periods,
and financial-specific patterns.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import math

import numpy as np
import pandas as pd
from scipy import stats
from seasonal import fit_seasons, adjust_seasons

from ..models.statement import (
    FinancialStatement,
    StatementType,
    TimePeriod,
    TimePeriodType
)

logger = logging.getLogger(__name__)


class TrendDirection(str, Enum):
    """Enum for trend directions."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class TimeSeriesMetric:
    """Represents a financial metric tracked over time."""
    
    name: str
    values: List[float]
    periods: List[TimePeriod]
    dates: List[Optional[date]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize dates list if not provided."""
        if not self.dates:
            # Try to use end_dates from periods if available
            self.dates = []
            for period in self.periods:
                if period.end_date:
                    self.dates.append(period.end_date)
                else:
                    # Approximate date if not available
                    if period.period_type == TimePeriodType.ANNUAL and period.fiscal_year:
                        # Assume December 31 for annual periods
                        self.dates.append(date(period.fiscal_year, 12, 31))
                    elif period.period_type == TimePeriodType.QUARTERLY and period.fiscal_year and period.fiscal_quarter:
                        # Approximate quarter end dates
                        month = (period.fiscal_quarter * 3)
                        self.dates.append(date(period.fiscal_year, month, 30))
                    else:
                        self.dates.append(None)
    
    def to_pandas(self) -> pd.Series:
        """Convert to pandas Series with dates as index."""
        # Filter out None dates and corresponding values
        valid_data = [(d, v) for d, v in zip(self.dates, self.values) if d is not None]
        if not valid_data:
            return pd.Series(dtype=float)
            
        dates, values = zip(*valid_data)
        return pd.Series(values, index=dates)


@dataclass
class TrendAnalysisResult:
    """Results of trend analysis on a financial metric."""
    
    metric_name: str
    direction: TrendDirection
    slope: float  # Linear regression slope
    r_squared: float  # Goodness of fit
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    growth_rate: Optional[float] = None  # CAGR if applicable
    seasonal_components: Optional[Dict[str, float]] = None
    is_statistically_significant: bool = False
    p_value: Optional[float] = None
    forecast_values: Optional[List[float]] = None
    forecast_periods: Optional[List[TimePeriod]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis results to a dictionary."""
        result = {
            "metric_name": self.metric_name,
            "direction": self.direction.value,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "is_statistically_significant": self.is_statistically_significant
        }
        
        # Add optional fields if present
        if self.growth_rate is not None:
            result["growth_rate"] = self.growth_rate
        if self.seasonal_components:
            result["seasonal_components"] = self.seasonal_components
        if self.p_value is not None:
            result["p_value"] = self.p_value
        if self.forecast_values:
            result["forecast_values"] = self.forecast_values
        
        return result


class FinancialTimeSeriesAnalyzer:
    """
    Analyzes financial metrics over time to identify trends and patterns.
    
    This class provides methods for analyzing time series of financial data,
    detecting trends, seasonality, and generating forecasts.
    """
    
    def __init__(self, min_data_points: int = 3):
        """
        Initialize the analyzer.
        
        Args:
            min_data_points: Minimum number of data points required for analysis
        """
        self.min_data_points = min_data_points
    
    def create_time_series(
        self, 
        statements: List[FinancialStatement], 
        metric_accessor: Union[str, Callable]
    ) -> TimeSeriesMetric:
        """
        Create a time series from a list of financial statements.
        
        Args:
            statements: List of financial statements sorted by time
            metric_accessor: String property name or callable to extract metric value
                
        Returns:
            TimeSeriesMetric object with values and periods
        """
        values = []
        periods = []
        
        for statement in statements:
            # Extract value using accessor
            if isinstance(metric_accessor, str):
                # Try to get the value as a property first
                if hasattr(statement, metric_accessor):
                    value = getattr(statement, metric_accessor)
                else:
                    # Try as a line item
                    line_item = statement.get_line_item(metric_accessor)
                    value = line_item.scaled_value if line_item else None
            else:
                # Use callable accessor
                value = metric_accessor(statement)
                
            if value is not None:
                values.append(value)
                periods.append(statement.metadata.period)
        
        # Use metric accessor as name if it's a string
        name = metric_accessor if isinstance(metric_accessor, str) else "custom_metric"
            
        return TimeSeriesMetric(
            name=name,
            values=values,
            periods=periods
        )
    
    def analyze_trend(self, time_series: TimeSeriesMetric) -> Optional[TrendAnalysisResult]:
        """
        Analyze the trend in a financial time series.
        
        Args:
            time_series: Financial metric time series
                
        Returns:
            TrendAnalysisResult with trend analysis or None if insufficient data
        """
        # Check if we have enough data points
        if len(time_series.values) < self.min_data_points:
            logger.warning(f"Insufficient data points for trend analysis of {time_series.name}")
            return None
        
        # Basic statistics
        values = np.array(time_series.values)
        mean = np.mean(values)
        std_dev = np.std(values)
        min_value = np.min(values)
        max_value = np.max(values)
        
        # Try to convert dates to numerical format for regression
        dates_numerical = []
        for i, d in enumerate(time_series.dates):
            if d is not None:
                # Convert date to numerical value (days since epoch)
                dates_numerical.append((d - date(1970, 1, 1)).days)
            else:
                # If date not available, use index as proxy
                dates_numerical.append(i)
        
        x = np.array(dates_numerical)
        y = values
        
        # Normalize x to avoid numerical issues
        x_mean = np.mean(x)
        x_std = np.std(x)
        x_norm = (x - x_mean) / x_std if x_std > 0 else x - x_mean
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_norm, y)
        
        # Determine trend direction
        if abs(slope) < 0.01 * mean:  # Slope is less than 1% of mean
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
            
        # Check for volatility
        if std_dev > 0.25 * mean:  # Standard deviation is more than 25% of mean
            direction = TrendDirection.VOLATILE
        
        # Calculate growth rate (CAGR) if possible
        growth_rate = None
        if len(values) >= 2 and values[0] > 0 and values[-1] > 0:
            n_periods = len(values) - 1
            growth_rate = (values[-1] / values[0]) ** (1 / n_periods) - 1
        
        # Create result
        result = TrendAnalysisResult(
            metric_name=time_series.name,
            direction=direction,
            slope=slope,
            r_squared=r_value ** 2,
            mean=mean,
            std_dev=std_dev,
            min_value=min_value,
            max_value=max_value,
            growth_rate=growth_rate,
            is_statistically_significant=p_value < 0.05,
            p_value=p_value
        )
        
        return result
    
    def detect_seasonality(self, time_series: TimeSeriesMetric) -> Dict[str, float]:
        """
        Detect seasonal patterns in a financial time series.
        
        Args:
            time_series: Financial metric time series
                
        Returns:
            Dictionary of seasonal components or empty dict if no seasonality detected
        """
        # Convert to pandas for easier time series analysis
        series = time_series.to_pandas()
        
        # Need at least 2 years of quarterly data or 4 years of annual data
        if len(series) < 8:
            return {}
        
        try:
            # Check if we have regular quarterly data
            is_quarterly = all(p.period_type == TimePeriodType.QUARTERLY for p in time_series.periods)
            
            if is_quarterly:
                # Try to extract quarterly seasonality
                periods = 4  # Quarterly data
                components = {}
                
                # Group by quarter
                quarters = {}
                for i, (period, value) in enumerate(zip(time_series.periods, time_series.values)):
                    if period.fiscal_quarter:
                        quarter = period.fiscal_quarter
                        if quarter not in quarters:
                            quarters[quarter] = []
                        quarters[quarter].append(value)
                
                # Calculate average for each quarter
                if len(quarters) == 4:
                    baseline = np.mean(time_series.values)
                    for quarter, values in quarters.items():
                        if values:
                            avg = np.mean(values)
                            components[f"Q{quarter}"] = avg / baseline - 1  # Seasonal factor
                    
                    return components
            
            # More sophisticated seasonality detection
            # Use seasonal decomposition if we have enough data
            if len(series) >= 12:
                # Ensure series is sorted and has no duplicates
                series = series.sort_index()
                series = series[~series.index.duplicated()]
                
                # Convert to regular time series if needed
                if not series.index.is_monotonic:
                    # Reindex to regular frequency
                    freq = 'Q' if is_quarterly else 'A'
                    series = series.asfreq(freq)
                
                # Fill missing values if any
                if series.isna().any():
                    series = series.interpolate()
                
                # Detect seasonality using seasonal package
                seasons, trend = fit_seasons(series.values)
                adjusted = adjust_seasons(series.values, seasons=seasons)
                
                # Convert seasons to components dictionary
                components = {}
                for i, factor in enumerate(seasons):
                    components[f"Season_{i+1}"] = factor - 1  # Convert to seasonal factor
                
                return components
                
        except Exception as e:
            logger.warning(f"Error detecting seasonality: {e}")
        
        return {}
    
    def detect_anomalies(self, time_series: TimeSeriesMetric, z_threshold: float = 2.0) -> List[int]:
        """
        Detect anomalies in a financial time series.
        
        Args:
            time_series: Financial metric time series
            z_threshold: Z-score threshold for anomaly detection
                
        Returns:
            List of indices of anomalous values
        """
        if len(time_series.values) < self.min_data_points:
            return []
        
        try:
            values = np.array(time_series.values)
            
            # Calculate Z-scores
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:  # Handle case where all values are the same
                return []
                
            z_scores = np.abs((values - mean) / std)
            
            # Find anomalies
            anomalies = np.where(z_scores > z_threshold)[0].tolist()
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Error detecting anomalies: {e}")
            return []
    
    def forecast_values(
        self, 
        time_series: TimeSeriesMetric, 
        periods: int = 4,
        method: str = "linear"
    ) -> Tuple[List[float], List[TimePeriod]]:
        """
        Forecast future values of a financial time series.
        
        Args:
            time_series: Financial metric time series
            periods: Number of periods to forecast
            method: Forecasting method ('linear', 'growth_rate', 'average')
                
        Returns:
            Tuple of (forecasted_values, forecasted_periods)
        """
        if len(time_series.values) < self.min_data_points:
            return [], []
        
        values = np.array(time_series.values)
        
        # Generate forecast periods
        forecast_periods = self._generate_forecast_periods(time_series.periods, periods)
        
        if method == "linear":
            # Linear regression forecast
            try:
                # Create numeric X values (use indices for simplicity)
                x = np.arange(len(values))
                
                # Fit linear regression
                slope, intercept, _, _, _ = stats.linregress(x, values)
                
                # Forecast future values
                future_x = np.arange(len(values), len(values) + periods)
                forecasted_values = intercept + slope * future_x
                
                return forecasted_values.tolist(), forecast_periods
                
            except Exception as e:
                logger.warning(f"Error in linear forecast: {e}")
                method = "average"  # Fall back to average method
        
        if method == "growth_rate":
            # Compound growth rate forecast
            try:
                if values[0] <= 0 or any(v <= 0 for v in values):
                    raise ValueError("Growth rate method requires positive values")
                
                # Calculate average growth rate
                growth_rates = [values[i] / values[i-1] - 1 for i in range(1, len(values))]
                avg_growth_rate = np.mean(growth_rates) + 1  # Convert back to multiplier
                
                # Apply growth rate for future periods
                base = values[-1]
                forecasted_values = [base * (avg_growth_rate ** (i+1)) for i in range(periods)]
                
                return forecasted_values, forecast_periods
                
            except Exception as e:
                logger.warning(f"Error in growth rate forecast: {e}")
                method = "average"  # Fall back to average method
        
        # Average method (fallback)
        avg_value = np.mean(values)
        forecasted_values = [avg_value] * periods
        
        return forecasted_values, forecast_periods
    
    def _generate_forecast_periods(
        self, 
        historical_periods: List[TimePeriod], 
        num_periods: int
    ) -> List[TimePeriod]:
        """
        Generate future time periods based on historical patterns.
        
        Args:
            historical_periods: List of historical time periods
            num_periods: Number of future periods to generate
                
        Returns:
            List of forecasted time periods
        """
        if not historical_periods:
            return []
        
        # Get the most recent period
        last_period = historical_periods[-1]
        
        forecast_periods = []
        
        if last_period.period_type == TimePeriodType.ANNUAL:
            # Generate future annual periods
            for i in range(1, num_periods + 1):
                if last_period.fiscal_year:
                    next_year = last_period.fiscal_year + i
                    forecast_periods.append(TimePeriod(
                        period_type=TimePeriodType.ANNUAL,
                        fiscal_year=next_year,
                        label=f"FY{next_year} (Forecast)"
                    ))
                else:
                    # If no fiscal year, just use generic labels
                    forecast_periods.append(TimePeriod(
                        period_type=TimePeriodType.ANNUAL,
                        label=f"Year +{i} (Forecast)"
                    ))
                    
        elif last_period.period_type == TimePeriodType.QUARTERLY:
            # Generate future quarterly periods
            for i in range(1, num_periods + 1):
                if last_period.fiscal_year and last_period.fiscal_quarter:
                    next_quarter = last_period.fiscal_quarter + i
                    next_year = last_period.fiscal_year
                    
                    # Adjust for quarter overflow
                    while next_quarter > 4:
                        next_quarter -= 4
                        next_year += 1
                    
                    forecast_periods.append(TimePeriod(
                        period_type=TimePeriodType.QUARTERLY,
                        fiscal_year=next_year,
                        fiscal_quarter=next_quarter,
                        label=f"Q{next_quarter} FY{next_year} (Forecast)"
                    ))
                else:
                    # If no fiscal quarter/year, just use generic labels
                    forecast_periods.append(TimePeriod(
                        period_type=TimePeriodType.QUARTERLY,
                        label=f"Quarter +{i} (Forecast)"
                    ))
        else:
            # For other period types, use generic labels
            for i in range(1, num_periods + 1):
                forecast_periods.append(TimePeriod(
                    period_type=last_period.period_type,
                    label=f"Period +{i} (Forecast)"
                ))
        
        return forecast_periods
    
    def analyze_multiple_metrics(
        self, 
        metrics: Dict[str, TimeSeriesMetric]
    ) -> Dict[str, Optional[TrendAnalysisResult]]:
        """
        Analyze multiple financial metrics.
        
        Args:
            metrics: Dictionary of metric name to TimeSeriesMetric
                
        Returns:
            Dictionary of metric name to TrendAnalysisResult
        """
        results = {}
        
        for name, metric in metrics.items():
            results[name] = self.analyze_trend(metric)
            
        return results
