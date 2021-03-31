import time
import math
import numpy as np
import random
import pandas as pd

ticker = 'IVV'
# n-size of window in days that trendline is based on
n_days = 252

# start date of strategy
strategy_start = pd.Timestamp("2019-01-02")
# threshold for each increment (%)
threshold = .05
# Number of shares to buy for each threshold increment
alpha = 5
# Beta - sell point based on current price (number greater than equal to 1)
beta = 1.6
# Determine portfolio starting balance
portfolio_balance = 10000

# Get dataframe of dates/closing prices for IVV
df_ticker_all = pd.read_csv('stock_data/' + ticker + '.csv')
df_ticker_all.rename(columns={df_ticker_all.columns[0]: "id"}, inplace=True)
df_ticker_all['date'] = pd.to_datetime(df_ticker_all['date'])

# Create dataframe to track mean prices during backtest and how closing prices compare
df_backtest_data = pd.DataFrame(columns=["id", "date", "close", "high", "trend_price", "close_to_trend_ratio"])

# Create blotter to track orders placed
df_blotter = pd.DataFrame(columns=["trade_id", "date", "symb", "actn", "size", "price", "type"])

def get_date_id(df, date):
    try:
        return df.loc[df['date'] == date, 'id'].values[0]
    except IndexError:
        return -1


def calc_expected_value(m, b, x):
    y = m * x + b
    return y


def calc_trend_line(df_ticker_all, n_days, current_date):
    df_window = get_price_data_for_window(df_ticker_all, n_days, current_date)

    # Get date and close prices as x and y arrays
    date_column = df_window.loc[:, 'id']
    dates = np.array(date_column.values)
    close_column = df_window.loc[:, 'close']
    close_prices = np.array(close_column.values)

    # Calculate slope and intercept of line based on dates and close-prices
    m, b = np.polyfit(dates, close_prices, 1)
    return m, b


def get_price_data_for_window(df, n_days, end_date):
    # Create new dataframe that only goes up until the current_date
    mask = (df['date'] <= end_date)
    df_window = df.loc[mask]
    # Find start id for window and update df_window to start at that date
    end_id = df_window["id"].max()
    if end_id - n_days + 1 < 0:
        start_id = 0
    else:
        start_id = end_id - n_days + 1
    mask2 = (df_window['id'] >= start_id)
    df_window = df_window.loc[mask2]
    return df_window


def build_backtest_dataset(df_all, df_backtest, start_date):
    i = 0
    for index, row in df_all.iterrows():
        if (row['date'] < start_date):
            continue
        current_date = pd.Timestamp(row['date'])
        date_id = row['id']
        m, b = calc_trend_line(df_all, n_days, current_date)
        expected_price = calc_expected_value(m, b, date_id)
        close = row['close']
        ratio = close / expected_price
        high = row['high']
        df1 = pd.DataFrame({'id': date_id, 'date': current_date, 'close': close, 'high': high, 'trend_price': expected_price,
                            'close_to_trend_ratio': ratio}, index=[i])
        df_backtest = df_backtest.append(df1)
        i = i + 1
    return df_backtest


def find_fill_date(df, date, lmt_price):
    # Create new dataframe that starts at date
    mask = (df['date'] > date)
    df2 = df.loc[mask]
    for index, row in df2.iterrows():
        high = row['high']
        if high >= lmt_price:
            return row['date']
    return None


# Find fill dates if an order will be filled during strategy and log date
def log_fill_dates(df_blotter, df_backtest):
    fill_dates = []
    filled = []
    # Go through blotter to look at each order
    for index, row in df_blotter.iterrows():
        # For buys use same date
        date = row['date']
        price = row['price']
        if row['actn'] == "BUY":
            fill_dates.append(date)
        elif row['actn'] == "SELL":
            print(find_fill_date(df_backtest, date, price))
            fill_dates.append(find_fill_date(df_backtest, date, price))
        else:
            fill_dates.append(None)
    for i in fill_dates:
        if i is not None:
            filled.append(True)
        else:
            filled.append(False)
    df_blotter['fill_dates'] = fill_dates
    df_blotter['filled'] = filled
    return df_blotter


# Might be useless - logs high and date of high out in the future (does not have to be close price) - keeping in
# case it's needed
def log_future_highs(df):
    df_highs = df[['id', 'date', 'high']].copy()
    future_highs = []
    future_high_dates = []
    max = 0
    for i in range(df_highs.shape[0] - 1, -1, -1):
        row = df_highs.iloc[i]
        if row['high'] > max:
            max = row['high']
            future_max_date = row['date']
        future_highs.append(max)
        future_high_dates.append(future_max_date)
    future_highs.reverse()
    future_high_dates.reverse()
    df['future_high'] = future_highs
    df['future_high_date'] = future_high_dates
    return df


def decide_to_buy_or_not(ratio, threshold):
    if 1 - ratio > threshold:
        return True
    else:
        return False


def create_order(i, date, symbol, action, size, price):
    df1 = pd.DataFrame({"trade_id": i, "date": date, "symb": symbol, "actn": action, "size": size, "price": price,
                        "type": "LMT"}, index=[i])
    return df1


def trade_size(ratio):
    return math.floor((abs(1-ratio)) / threshold) * alpha


# Function to create blotter of trades
def build_testing_blotter(blotter, backtest):
    i = 0
    # Look at price every day from testing period
    for index, row in backtest.iterrows():
        # Decide if a buy order needs to happen
        if not decide_to_buy_or_not(row['close_to_trend_ratio'], threshold):
            continue
        else:
            date = row['date']
            size = trade_size(row['close_to_trend_ratio'])
            buy_price = row['close']
            sell_price = buy_price * beta
            new_buy_order = create_order(i, date, ticker, "BUY", size, buy_price)
            new_sell_order = create_order(i+1, date, ticker, "SELL", size, sell_price)
            blotter = blotter.append(new_buy_order)
            blotter = blotter.append(new_sell_order)
            i = i + 2
    return blotter


def calc_portfolio_gain(df):
    balance = 0
    for index, row in df.iterrows():
        value = row['size'] * row['price']
        if row['actn'] == "BUY" and row['filled']:
            balance = balance - value
        if row['actn'] == "SELL" and row['filled']:
            balance = balance + value
    return balance

# Check whether to buy or sell
def decide_to_buy_or_sell():
    choice = random.choice([1, 2])
    print(choice)
    return choice


df_backtest_data = build_backtest_dataset(df_ticker_all, df_backtest_data, strategy_start)
df_blotter = build_testing_blotter(df_blotter, df_backtest_data)
df_blotter = log_fill_dates(df_blotter, df_backtest_data)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df_blotter)

print(calc_portfolio_gain(df_blotter))
