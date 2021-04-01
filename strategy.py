from datetime import datetime as dt
import math
import numpy as np
import random
import pandas as pd

#ticker = 'IVV'
# n-size of window in days that trendline is based on
# n_days = 252

# start date of strategy
# strategy_start = pd.Timestamp("2019-01-02")
# threshold for each increment (%)
# threshold = .05
# Number of shares to buy for each threshold increment
# alpha = .03
# Beta - sell point based on current price (number greater than equal to 1)
# beta = 1.3

# Determine portfolio starting balance
# portfolio_balance = 100000

# Get dataframe of dates/closing prices for IVV
def get_stock_data_from_csv(ticker):
    df = pd.read_csv('stock_data/' + ticker + '.csv')
    df.rename(columns={df.columns[0]: "id"}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df

# # Create dataframe to track mean prices during backtest and how closing prices compare
# df_backtest_data = pd.DataFrame(columns=["id", "date", "close", "high", "trend_price", "close_to_trend_ratio"])

# # Create blotter to track orders placed
# df_blotter = pd.DataFrame(columns=["trade_id", "date", "symb", "actn", "size_factor", "size", "price", "type"])

# # Track portfolio balance in USD and IVV
# df_portfolio = pd.DataFrame({"date": strategy_start - pd.Timedelta(days=1),
#                              "USD": portfolio_balance, "IVV_shares": 0, "IVV_value": 0,
#                              "Total": portfolio_balance}, index=[0])


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


def build_backtest_dataset(df_all, start_date, n_days):
    # Create dataframe to track mean prices during backtest and how closing prices compare
    df_backtest = pd.DataFrame(columns=["id", "date", "close", "high", "trend_price", "close_to_trend_ratio"])
    i = 0
    for index, row in df_all.iterrows():
        if row['date'] < start_date:
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
            fill_dates.append(find_fill_date(df_backtest, date, price))
        else:
            fill_dates.append(None)
    for i in fill_dates:
        if i is not None:
            filled.append(True)
        else:
            filled.append(False)
    df_blotter['fill_date'] = fill_dates
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


# Determine if price has even broken one threshold
def decide_to_buy_or_not(ratio, threshold):
    if 1 - ratio > threshold:
        return True
    else:
        return False


# Create an order to insert into the blotter
def create_order(i, date, symbol, action, size_factor, price):
    df1 = pd.DataFrame({"trade_id": i, "date": date, "symb": symbol, "actn": action, "size_factor": size_factor,
                        "size": 0, "price": price, "type": "LMT"}, index=[i])
    return df1


# Update order size in blotter with number of shares (based on portfolio balance at time of order)
def update_trade_order_size(df, n_shares, trade_id, action):
    if action == 'BUY':
        # Update size of order for buy trade
        df.loc[(df['trade_id'] == trade_id), 'size'] = n_shares
        # Update size of order for sell trades, we know the id is always the next one
        df.loc[(df['trade_id'] == trade_id + 1), 'size'] = n_shares

    return df


# Determine how many threshold deviations from the mean the price is (integer)
def trade_size_factor(ratio, threshold):
    return math.floor((abs(1-ratio)) / threshold)


# Determine how many whole shares to purchase
def trade_size(sz_f, a, cash, close):
    return math.floor((sz_f * a * cash) / close)


# Function to create blotter of trades
def build_testing_blotter(backtest, ticker, threshold, beta):
    # Create blotter to track orders placed
    blotter = pd.DataFrame(columns=["trade_id", "date", "symb", "actn", "size_factor", "size", "price", "type"])
    i = 0
    # Look at price every day from testing period
    for index, row in backtest.iterrows():
        # Decide if a buy order needs to happen
        if not decide_to_buy_or_not(row['close_to_trend_ratio'], threshold):
            continue
        else:
            date = row['date']
            size_factor = trade_size_factor(row['close_to_trend_ratio'], threshold)
            buy_price = row['close']
            sell_price = buy_price * beta
            new_buy_order = create_order(i, date, ticker, "BUY", size_factor, buy_price)
            new_sell_order = create_order(i+1, date, ticker, "SELL", size_factor, sell_price)
            blotter = blotter.append(new_buy_order)
            blotter = blotter.append(new_sell_order)
            i = i + 2
    return blotter


# Order any dataframe by any date column
def order_by_date_column(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])
    df = df.sort_values(by=column_name)
    return df


def get_cash_balance(df):
    return df.iloc[-1:]['USD'].values[0]


# Get dataframe with any orders filled on a certain date
def get_orders_for_date(date, df):
    mask = (df['fill_date'] == date)
    df = df.loc[mask]
    return df


# Update portfolio for certain date and make any adjustments based on buys/sells
def update_portfolio(df, date, shares_bought, share_price, current_close_price):
    last_row = df.iloc[-1:].copy()
    last_date = pd.Timestamp(last_row['date'].values[0])
    last_row['USD'] = last_row['USD'] - (shares_bought * share_price)
    last_row['IVV_shares'] = last_row['IVV_shares'] + shares_bought
    last_row['IVV_value'] = last_row['IVV_shares'] * current_close_price
    last_row['Total'] = last_row['USD'] + last_row['IVV_value']
    if last_date != date:
        # Replace row with new date and append row
        last_row['date'] = date
        df = df.append(last_row)
    else:
        df = df[df.date != last_date]
        df = df.append(last_row)
    return df


# Track portfolio progress and fill in blotter
def track_portfolio_progress(df_blot, df_backtest, a, strategy_start, portfolio_balance):
    # Track portfolio balance in USD and IVV
    df_port = pd.DataFrame({"date": strategy_start - pd.Timedelta(days=1),
                                 "USD": portfolio_balance, "IVV_shares": 0, "IVV_value": 0,
                                 "Total": portfolio_balance}, index=[0])
    for index, row in df_backtest.iterrows():
        date = pd.Timestamp(row['date'])
        close = row['close']
        df_orders = get_orders_for_date(date, df_blot)
        if df_orders.empty:
            df_port = update_portfolio(df_port, date, 0, 0, close)
        else:
            for ix, row2 in df_orders.iterrows():
                # Find out USD balance of portfolio
                cash_balance = get_cash_balance(df_port)
                # Update the blotter with correct size for shares
                if row2['actn'] == "BUY":
                    n_shares = trade_size(row2['size_factor'], a, cash_balance, close)
                else:
                    n_shares = -row2['size']
                df_blot = update_trade_order_size(df_blot, n_shares, row2['trade_id'], row2['actn'])
                # Update portfolio
                df_port = update_portfolio(df_port, date, n_shares, row2['price'], close)
    return df_port


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


def run_strategy(ticker, window, start_date, thresh, alpha, b, port_bal):
    df_ticker_all = get_stock_data_from_csv(ticker)
    df_backtest_data = build_backtest_dataset(df_ticker_all, start_date, window)
    df_blotter = build_testing_blotter(df_backtest_data, ticker, thresh, b)
    df_blotter = log_fill_dates(df_blotter, df_backtest_data)
    df_blotter_by_fill_date = order_by_date_column(df_blotter, 'fill_date')
    df_portfolio = track_portfolio_progress(df_blotter_by_fill_date, df_backtest_data, alpha, start_date, port_bal)
    df_blotter = order_by_date_column(df_blotter_by_fill_date, 'date')
    return df_backtest_data, df_portfolio, df_blotter


df_backtest, df_portfolio, df_blotter = run_strategy('IVV', 252, pd.Timestamp("2019-01-02"), 0.05, .03, 1.3, 100000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df_blotter)
print(df_portfolio)
