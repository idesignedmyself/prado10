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


def parse_backtest_args(args: list) -> dict:
    """
    Parse backtest arguments in cleaner positional format.

    Supports:
    - QQQ standard
    - QQQ standard start 01 01 2020 end 12 31 2023
    - QQQ combo start 01 01 2020 end 12 31 2023 wf 12 31 2025
    - QQQ walk-forward start 01 01 2023 end 12 31 2025
    - QQQ crisis
    - QQQ monte-carlo 10000
    - QQQ mc2 1000

    Args:
        args: List of command-line arguments

    Returns:
        Dictionary with mode, start_date, end_date, wf_date, iterations

    Raises:
        ValueError: If format is invalid
    """
    if len(args) < 2:
        raise ValueError("At least SYMBOL and MODE required")

    result = {
        'symbol': args[0].upper(),
        'mode': args[1].lower(),
        'start_date': None,
        'end_date': None,
        'wf_date': None,
        'iterations': None
    }

    # Handle modes with iteration counts
    if result['mode'] in ['monte-carlo', 'mc2']:
        if len(args) >= 3:
            try:
                result['iterations'] = int(args[2])
            except ValueError:
                raise ValueError(f"{result['mode']} requires iteration count")
        return result

    # Handle crisis (no dates needed)
    if result['mode'] == 'crisis':
        return result

    # Parse dates if present
    remaining_args = args[2:]

    try:
        if 'start' in remaining_args:
            start_idx = remaining_args.index('start')
            start_month = remaining_args[start_idx + 1]
            start_day = remaining_args[start_idx + 2]
            start_year = remaining_args[start_idx + 3]
            start_date = datetime(int(start_year), int(start_month), int(start_day))
            result['start_date'] = start_date.strftime('%Y-%m-%d')

        if 'end' in remaining_args:
            end_idx = remaining_args.index('end')
            end_month = remaining_args[end_idx + 1]
            end_day = remaining_args[end_idx + 2]
            end_year = remaining_args[end_idx + 3]
            end_date = datetime(int(end_year), int(end_month), int(end_day))
            result['end_date'] = end_date.strftime('%Y-%m-%d')

        if 'wf' in remaining_args:
            wf_idx = remaining_args.index('wf')
            wf_month = remaining_args[wf_idx + 1]
            wf_day = remaining_args[wf_idx + 2]
            wf_year = remaining_args[wf_idx + 3]
            wf_date = datetime(int(wf_year), int(wf_month), int(wf_day))
            result['wf_date'] = wf_date.strftime('%Y-%m-%d')

    except (ValueError, IndexError) as e:
        raise ValueError(
            "Invalid date format. Use: start MM DD YYYY end MM DD YYYY [wf MM DD YYYY]\n"
            "Example: prado backtest QQQ combo start 01 01 2020 end 12 31 2023 wf 12 31 2025"
        ) from e

    return result
