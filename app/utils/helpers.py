"""
Utility Helper Functions for Compliance System
"""
import re
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import uuid
from decimal import Decimal
import json


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix"""
    unique_id = str(uuid.uuid4())
    return f"{prefix}_{unique_id}" if prefix else unique_id


def calculate_percentage(value: float, total: float, precision: int = 4) -> float:
    """Calculate percentage with proper error handling"""
    if total == 0:
        return 0.0
    return round((value / total) * 100, precision)


def calculate_weight(market_value: float, total_market_value: float, precision: int = 6) -> float:
    """Calculate portfolio weight with proper precision"""
    if total_market_value == 0:
        return 0.0
    return round(market_value / total_market_value, precision)


def clean_symbol(symbol: str) -> str:
    """Clean and standardize security symbols"""
    if not symbol:
        return ""
    
    # Remove common prefixes/suffixes and standardize
    symbol = symbol.upper().strip()
    symbol = re.sub(r'[^A-Z0-9\-\.]', '', symbol)
    
    # Handle common variations
    if symbol.endswith('.'):
        symbol = symbol[:-1]
    
    return symbol


def parse_rating(rating: str) -> str:
    """Standardize credit rating format"""
    if not rating:
        return "NR"  # Not Rated
    
    rating = rating.upper().strip()
    
    # Map common variations
    rating_map = {
        'AAA': 'AAA',
        'AA+': 'AA+', 'AA': 'AA', 'AA-': 'AA-',
        'A+': 'A+', 'A': 'A', 'A-': 'A-',
        'BBB+': 'BBB+', 'BBB': 'BBB', 'BBB-': 'BBB-',
        'BB+': 'BB+', 'BB': 'BB', 'BB-': 'BB-',
        'B+': 'B+', 'B': 'B', 'B-': 'B-',
        'CCC+': 'CCC+', 'CCC': 'CCC', 'CCC-': 'CCC-',
        'CC': 'CC', 'C': 'C', 'D': 'D',
        'NR': 'NR', 'NOTRATED': 'NR', 'NOT RATED': 'NR'
    }
    
    return rating_map.get(rating, rating)


def rating_to_numeric(rating: str) -> int:
    """Convert credit rating to numeric score for comparison"""
    rating_scores = {
        'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
        'A+': 17, 'A': 16, 'A-': 15,
        'BBB+': 14, 'BBB': 13, 'BBB-': 12,
        'BB+': 11, 'BB': 10, 'BB-': 9,
        'B+': 8, 'B': 7, 'B-': 6,
        'CCC+': 5, 'CCC': 4, 'CCC-': 3,
        'CC': 2, 'C': 1, 'D': 0
    }
    
    return rating_scores.get(parse_rating(rating), 0)


def is_investment_grade(rating: str) -> bool:
    """Check if rating is investment grade (BBB- or higher)"""
    return rating_to_numeric(rating) >= 12


def format_currency(amount: float, currency: str = "USD", precision: int = 2) -> str:
    """Format currency amount with proper formatting"""
    if currency == "USD":
        return f"${amount:,.{precision}f}"
    else:
        return f"{amount:,.{precision}f} {currency}"


def format_percentage(value: float, precision: int = 2) -> str:
    """Format percentage with proper precision"""
    return f"{value:.{precision}f}%"


def format_basis_points(value: float) -> str:
    """Format basis points"""
    return f"{value:.0f} bps"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with default for zero division"""
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_hash(data: Union[str, Dict, List]) -> str:
    """Calculate SHA256 hash of data"""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode()).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def calculate_herfindahl_index(weights: List[float]) -> float:
    """Calculate Herfindahl-Hirschman Index for concentration"""
    if not weights:
        return 0.0
    
    return sum(w * w for w in weights if w > 0)


def safe_get(dictionary: Dict, key_path: str, default: Any = None) -> Any:
    """Safely get nested dictionary value using dot notation"""
    keys = key_path.split('.')
    current = dictionary
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default