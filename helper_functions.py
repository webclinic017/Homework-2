# Contains helper functions for your apps
import os
from ib_insync import *
import pandas as pd


# If the io following files are in the current directory, remove them!
# 1. 'currency_pair.txt'
# 2. 'currency_pair_history.csv'
# 3. 'trade_order.p'
def check_for_and_del_strategy_files():
    # Your code goes here.
    if not os.path.exists('strategy_files'):
        os.makedirs('strategy_files')
    file_list = ['strategy_files/backtest.csv', 'strategy_files/blotter.csv', 'strategy_files/portfolio.csv']
    for i in file_list:
        try:
            os.remove(i)
        except:
            print("Could not delete file: " + i + ". File does not exist")

    pass  # nothing gets returned by this function, so end it with 'pass'.


def get_historical_us_stock_data(ticker, port=7497, stock_data_client_id=12347):
    ib_strat = IB()
    # Connect your app to a running instance of IBG or TWS
    ib_strat.connect(host='127.0.0.1', port=port, clientId=stock_data_client_id)
    with open('stock_data/' + ticker + '.csv', 'w'):
        stock_contract = Stock(ticker, "smart", 'USD')
        bars = ib_strat.reqHistoricalData(stock_contract, endDateTime='', durationStr='50 Y', barSizeSetting='1 day',
                                               whatToShow='MIDPOINT', useRTH=True)
        bars = pd.DataFrame(bars)
        bars.to_csv('stock_data/' + ticker + '.csv')
    ib_strat.disconnect()


def get_stock_close_price_today(ib_connection, ticker):
    stock_contract = Stock(ticker, "smart", 'USD')
    bars = ib_connection.reqHistoricalData(stock_contract, endDateTime='', durationStr='1 D', barSizeSetting='1 day',
                                               whatToShow='MIDPOINT', useRTH=True)
    bars = pd.DataFrame(bars)
    return bars['close'].values[0]


def plain_stock_contract(ib_connect, stock_ticker):
    c = Stock(stock_ticker, 'SMART', 'USD')
    ib_connect.qualifyContracts(c)
    return c


def place_limit_order(port, orders_client_id, type, size, px):
    ib = IB()
    ib.connect(host='127.0.0.1', port=port, clientId=orders_client_id)
    contract = Stock('IVV', 'SMART', 'USD')
    ib.qualifyContracts(contract)

    #Find current price
    #px = ib.reqMktData(contract, 221).last
    #px = 391.12

    new_order = LimitOrder(type, size, px)
    ib.placeOrder(contract, new_order)

    ib.disconnect()


def place_market_order(port, orders_client_id, size):
    ib = IB()
    ib.connect(host='127.0.0.1', port=port, clientId=orders_client_id)
    contract = Stock('IVV', 'SMART', 'USD')
    ib.qualifyContracts(contract)

    new_order = MarketOrder("Buy", size)
    ib.placeOrder(contract, new_order)

    ib.disconnect()


def get_available_funds(ib, account):
    account_summary = ib.accountSummary(account)
    for l in account_summary:
        for i in l:
            if i == 'AvailableFunds':
                return float(l[2])


def get_order_fills(ib_connection):
    #stock_contract = Stock(ticker, "smart", 'USD')
    filled_orders = ib_connection.fills()
    print(filled_orders)

