"""
Date parsing utilities for PRADO9_EVO CLI
"""

from datetime import datetime
from typing import Tuple


def parse_date_args(args: list) -> Tuple[str, str]:
    """
    Parse date arguments from CLI in format: start MM DD YYYY end MM DD YYYY

    Args:
        args: List of command-line arguments

    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format

    Raises:
        ValueError: If date format is invalid
    """
    try:
        # Find 'start' keyword
        start_idx = args.index('start')
        # Extract MM DD YYYY
        start_month = args[start_idx + 1]
        start_day = args[start_idx + 2]
        start_year = args[start_idx + 3]

        # Find 'end' keyword
        end_idx = args.index('end')
        # Extract MM DD YYYY
        end_month = args[end_idx + 1]
        end_day = args[end_idx + 2]
        end_year = args[end_idx + 3]

        # Validate and format
        start_date = datetime(int(start_year), int(start_month), int(start_day))
        end_date = datetime(int(end_year), int(end_month), int(end_day))

        # Return in YYYY-MM-DD format
        return (
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

    except (ValueError, IndexError) as e:
        raise ValueError(
            "Invalid date format. Use: start MM DD YYYY end MM DD YYYY\n"
            "Example: start 01 01 2020 end 12 31 2023"
        ) from e


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate that start_date is before end_date

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        True if valid, raises ValueError otherwise
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    if start >= end:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")

    return True
