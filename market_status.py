import pandas_market_calendars as mcal
from datetime import datetime


def is_market_open_today():
    # Create a calendar
    nyse = mcal.get_calendar('NYSE')

    # Get NYSE exchange info for day
    market_times = nyse.schedule(start_date=datetime.now(), end_date=datetime.now())

    if market_times.empty:
        return False
    else:
        return True

