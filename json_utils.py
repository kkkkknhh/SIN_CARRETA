"""
JSON Utilities with NaN/Inf Handling
====================================

Comprehensive utilities for safe JSON serialization with automatic cleaning
of special float values (NaN, Infinity, -Infinity) that are not JSON-compliant.

Features:
- Recursive data structure cleaning
- Configurable replacement strategies for special float values  
- Thread-safe operations
- Consistent json.dump() parameters across all JSON writes
- Support for nested data structures (lists, dicts, tuples)
"""

import json
import math
from typing import Any, Dict, List, Union, Optional, TextIO
import numpy as np


def clean_data_for_json(data: Any, 
                       nan_replacement: Union[str, None] = None,
                       inf_replacement: Union[str, None] = "Infinity",
                       neg_inf_replacement: Union[str, None] = "-Infinity") -> Any:
    """
    Recursively clean data structure by replacing special float values.
    
    Args:
        data: Input data structure to clean
        nan_replacement: Replacement for NaN values (default: None)
        inf_replacement: Replacement for positive infinity (default: "Infinity") 
        neg_inf_replacement: Replacement for negative infinity (default: "-Infinity")
    
    Returns:
        Cleaned data structure safe for JSON serialization
    """
    if isinstance(data, dict):
        return {key: clean_data_for_json(value, nan_replacement, inf_replacement, neg_inf_replacement) 
                for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        cleaned_list = [clean_data_for_json(item, nan_replacement, inf_replacement, neg_inf_replacement) 
                       for item in data]
        return cleaned_list if isinstance(data, list) else type(data)(cleaned_list)
    elif isinstance(data, (float, np.floating)):
        if math.isnan(data):
            return nan_replacement
        elif math.isinf(data):
            return inf_replacement if data > 0 else neg_inf_replacement
        else:
            return float(data)  # Convert numpy floats to Python floats
    elif isinstance(data, np.integer):
        return int(data)  # Convert numpy integers to Python integers
    elif isinstance(data, np.ndarray):
        return clean_data_for_json(data.tolist(), nan_replacement, inf_replacement, neg_inf_replacement)
    else:
        return data


def safe_json_dump(data: Any, 
                   file_handle: TextIO,
                   nan_replacement: Union[str, None] = None,
                   inf_replacement: Union[str, None] = "Infinity", 
                   neg_inf_replacement: Union[str, None] = "-Infinity",
                   **kwargs) -> None:
    """
    Safe JSON dump with automatic data cleaning and consistent parameters.
    
    Args:
        data: Data to serialize
        file_handle: File handle to write to
        nan_replacement: Replacement for NaN values
        inf_replacement: Replacement for positive infinity
        neg_inf_replacement: Replacement for negative infinity
        **kwargs: Additional arguments passed to json.dump (will override defaults)
    """
    # Default parameters for consistent JSON formatting
    default_params = {
        'ensure_ascii': False,
        'indent': 2,
        'allow_nan': False
    }
    
    # Update with any user-provided parameters
    default_params.update(kwargs)
    
    # Clean the data
    cleaned_data = clean_data_for_json(data, nan_replacement, inf_replacement, neg_inf_replacement)
    
    # Write to file
    json.dump(cleaned_data, file_handle, **default_params)


def safe_json_dumps(data: Any,
                    nan_replacement: Union[str, None] = None,
                    inf_replacement: Union[str, None] = "Infinity",
                    neg_inf_replacement: Union[str, None] = "-Infinity", 
                    **kwargs) -> str:
    """
    Safe JSON dumps with automatic data cleaning and consistent parameters.
    
    Args:
        data: Data to serialize
        nan_replacement: Replacement for NaN values
        inf_replacement: Replacement for positive infinity
        neg_inf_replacement: Replacement for negative infinity
        **kwargs: Additional arguments passed to json.dumps (will override defaults)
        
    Returns:
        JSON string representation
    """
    # Default parameters for consistent JSON formatting
    default_params = {
        'ensure_ascii': False,
        'indent': 2,
        'allow_nan': False
    }
    
    # Update with any user-provided parameters
    default_params.update(kwargs)
    
    # Clean the data
    cleaned_data = clean_data_for_json(data, nan_replacement, inf_replacement, neg_inf_replacement)
    
    # Return JSON string
    return json.dumps(cleaned_data, **default_params)


def safe_json_write(data: Any, 
                    filepath: str,
                    nan_replacement: Union[str, None] = None,
                    inf_replacement: Union[str, None] = "Infinity",
                    neg_inf_replacement: Union[str, None] = "-Infinity",
                    encoding: str = "utf-8",
                    **kwargs) -> None:
    """
    Convenience function to write JSON data to a file with safety checks.
    
    Args:
        data: Data to serialize
        filepath: Path to the output file
        nan_replacement: Replacement for NaN values
        inf_replacement: Replacement for positive infinity
        neg_inf_replacement: Replacement for negative infinity
        encoding: File encoding (default: utf-8)
        **kwargs: Additional arguments passed to json.dump
    """
    with open(filepath, 'w', encoding=encoding) as f:
        safe_json_dump(data, f, nan_replacement, inf_replacement, neg_inf_replacement, **kwargs)