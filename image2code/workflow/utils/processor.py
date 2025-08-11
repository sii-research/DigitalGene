"""
Parameter Processing Module.

This module provides functionality for processing, normalizing, and discretizing
parameters in vision-language model outputs with statistical analysis.
"""

import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np

class ParameterProcessor:
    """
    Processor for normalizing and discretizing model parameters.
    
    This class handles the collection of statistics from datasets and provides
    methods for converting between continuous parameter values and discrete tokens.
    """
    
    def __init__(self, num_bins: int = 1024, token_start_id: int = 2048):
        """
        Initialize the parameter processor.
        
        Args:
            num_bins: Number of bins for discretization
            token_start_id: Starting ID for token mapping
        """
        self.num_bins = num_bins
        self.token_start_id = token_start_id
        # Store statistics for each category-template-parameter-dimension
        self.stats = {}
        self.distributions = {}
        self.param_dimensions = {}
        
    def collect_statistics(self, dataset: List[Dict]) -> None:
        """
        Collect statistical information for each parameter dimension.
        
        Args:
            dataset: List of data items containing parameter information
        """
        for item in dataset:
            category = item['category']
            if category not in self.stats:
                self.stats[category] = {}
                self.distributions[category] = {}
                self.param_dimensions[category] = {}
                
            for template_info in item['conceptualization']:
                template = template_info['template']
                if template not in self.stats[category]:
                    self.stats[category][template] = {}
                    self.distributions[category][template] = {}
                    self.param_dimensions[category][template] = {}
                param_key = "parameters" if "parameters" in template_info else "parameter"
                for param_name, param_values in template_info[param_key].items():
                    if isinstance(param_values, list):
                        if param_name not in self.stats[category][template]:
                            self.stats[category][template][param_name] = {}
                            self.distributions[category][template][param_name] = {}
                            self.param_dimensions[category][template][param_name] = [len(param_values)]
                        if not len(param_values) in self.param_dimensions[category][template][param_name]:
                            self.param_dimensions[category][template][param_name].append(len(param_values))
                        for dim_idx in range(len(param_values)):
                            if dim_idx not in self.distributions[category][template][param_name]:
                                self.distributions[category][template][param_name][dim_idx] = []
                        for dim_idx, value in enumerate(param_values):
                            self.distributions[category][template][param_name][dim_idx].append(value)
                    elif isinstance(param_values, float) or isinstance(param_values, int):
                        if param_name not in self.stats[category][template]:
                            self.stats[category][template][param_name] = {}
                            self.distributions[category][template][param_name] = {}
                            self.param_dimensions[category][template][param_name] = [1]
                        if 0 not in self.distributions[category][template][param_name]:
                            self.distributions[category][template][param_name][0] = []
                        self.distributions[category][template][param_name][0].append(param_values)
                    else:
                        raise RuntimeError(f"Error type: {type(param_values)}")
        for category in self.distributions:
            for template in self.distributions[category]:
                for param_name in self.distributions[category][template]:
                    for dim_idx in range(max(self.param_dimensions[category][template][param_name])):
                        values_array = np.array(self.distributions[category][template][param_name][dim_idx])
                        # fix nan or other error
                        values_array = values_array[np.isfinite(values_array)]
                        if dim_idx not in self.stats[category][template][param_name]:
                            self.stats[category][template][param_name][dim_idx] = {}
                        self.stats[category][template][param_name][dim_idx] = {
                            'min': float(np.min(values_array)),
                            'max': float(np.max(values_array)),
                            'mean': float(np.mean(values_array)),
                            'std': float(np.std(values_array)),
                            'quantiles': np.percentile(values_array, np.linspace(0, 100, self.num_bins + 1)).tolist()
                        }
    
    def normalize_and_discretize(
        self, 
        value: float, 
        category: str, 
        template: str, 
        param_name: str, 
        dim_idx: str
    ) -> int:
        """
        Normalize and discretize a parameter value to token ID.
        
        Args:
            value: The parameter value to discretize
            category: Category name
            template: Template name
            param_name: Parameter name
            dim_idx: Dimension index as string
            
        Returns:
            Discretized token ID
        """
        try:
            stats = self.stats[category][template][param_name][int(dim_idx)]
            quantiles = stats['quantiles']
            # Use binary search to find appropriate bin
            bin_idx = np.searchsorted(quantiles, value) - 1
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
            # Convert to token ID
            token_id = self.token_start_id + int(bin_idx)
            return token_id
        except (KeyError, IndexError) as e:
            print(f"Error in normalize_and_discretize: {e}")
            return self.token_start_id
    
    def _process_item_parameter(
        self, 
        param_values: Union[List, float, int], 
        category: str, 
        template: str, 
        param_name: str
    ) -> Union[List[int], int]:
        """Process individual parameter values during item processing."""
        if isinstance(param_values, list):
            return [
                self.normalize_and_discretize(
                    value, category, template, param_name, str(dim_idx)
                )
                for dim_idx, value in enumerate(param_values)
            ]
        elif isinstance(param_values, (float, int)):
            return self.normalize_and_discretize(
                param_values, category, template, param_name, "0"
            )
        else:
            return param_values
    
    def process_item(self, item: Dict) -> Dict:
        """
        Process a single data item by discretizing its parameters.
        
        Args:
            item: Data item containing parameters to process
            
        Returns:
            Processed item with discretized parameters
        """
        processed = item.copy()
        for template_info in processed['conceptualization']:
            template = template_info['template']
            param_key = "parameters" if "parameters" in template_info else "parameter"
            processed_params = {}
            for param_name, param_values in template_info[param_key].items():
                processed_params[param_name] = self._process_item_parameter(
                    param_values, item['category'], template, param_name
                )
            template_info[param_key] = processed_params
        return processed
    
    def token_to_value(
        self, 
        token_id: int, 
        category: str, 
        template: str, 
        param_name: str, 
        dim_idx: int
    ) -> float:
        """
        Convert token ID back to original parameter value.
        
        Args:
            token_id: Token ID to convert
            category: Category name
            template: Template name
            param_name: Parameter name
            dim_idx: Dimension index
            
        Returns:
            Recovered parameter value
        """
        try:
            # Parse token ID to bin index
            bin_idx = (token_id - self.token_start_id) % self.num_bins
            # Get statistics for the dimension
            stats = self.stats[category][template][param_name][int(dim_idx)]
            quantiles = stats['quantiles']
            # Ensure bin_idx is within valid range
            bin_idx = max(0, min(bin_idx, len(quantiles) - 2))
            # Return the upper bound of the quantile bin
            return float(quantiles[bin_idx + 1])
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error in token_to_value: {e}")
            return 0.0
    
    def _recover_item_parameter(
        self, 
        param_values: Union[List, int, float], 
        category: str, 
        template: str, 
        param_name: str
    ) -> Union[List[float], float]:
        """Recover individual parameter values during item recovery."""
        if isinstance(param_values, list):
            return [
                self.token_to_value(
                    token, category, template, param_name, str(dim_idx)
                )
                for dim_idx, token in enumerate(param_values)
            ]
        elif isinstance(param_values, (int, float)):
            return self.token_to_value(
                param_values, category, template, param_name, "0"
            )
        else:
            return param_values
    
    def recover_item(self, item: Dict) -> Dict:
        """
        Recover a processed item by converting tokens back to values.
        
        Args:
            item: Processed item with discretized parameters
            
        Returns:
            Recovered item with continuous parameter values
        """
        recovered = item.copy()
        conceptualizations = []
        for template_info in recovered['conceptualization']:
            template = template_info['template']
            param_key = "parameters" if "parameters" in template_info else "parameter"
            recovered_params = {}
            for param_name, param_values in template_info[param_key].items():
                try:
                    recovered_params[param_name] = self._recover_item_parameter(
                        param_values, item['category'], template, param_name
                    )
                except Exception as e:
                    print(f"Error recovering parameter {param_name}: {e}")
                    continue
            if recovered_params:
                template_info[param_key] = recovered_params
                conceptualizations.append(template_info)
        recovered["conceptualization"] = conceptualizations
        return recovered

    def get_param_statistics(
        self, 
        category: str, 
        template: str, 
        param_name: str
    ) -> List[Dict]:
        """
        Get statistical information for each dimension of a parameter.
        
        Args:
            category: Category name
            template: Template name
            param_name: Parameter name
            
        Returns:
            List of statistics for each parameter dimension
        """
        try:
            dims = max(self.param_dimensions[category][template][param_name])
            dim_stats = []
            for dim_idx in range(dims):
                if dim_idx in self.stats[category][template][param_name]:
                    stats = self.stats[category][template][param_name][dim_idx]
                    dim_stats.append({
                        'dimension': dim_idx,
                        'min': stats['min'],
                        'max': stats['max'],
                        'mean': stats['mean'],
                        'std': stats['std']
                    })
            return dim_stats
        except (KeyError, ValueError) as e:
            print(f"Error getting parameter statistics: {e}")
            return []
