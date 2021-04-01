# Contains helper functions for your apps!
from os import listdir, remove
from ib_insync import *
import pandas as pd


# If the io following files are in the current directory, remove them!
# 1. 'currency_pair.txt'
# 2. 'currency_pair_history.csv'
# 3. 'trade_order.p'
def check_for_and_del_io_files():
    # Your code goes here.
    file_list = []
    for i in file_list:
        try:
            remove(i)
            print("File deleted")
        except:
            print("Could not delete file: " + i + ". File does not exist")

    pass  # nothing gets returned by this function, so end it with 'pass'.


def get_historical_us_stock_data(ib_connection, ticker):
    with open('stock_data/' + ticker + '.csv', 'w'):
        stock_contract = Stock(ticker, "smart", 'USD')
        bars = ib_connection.reqHistoricalData(stock_contract, endDateTime='', durationStr='5 Y', barSizeSetting='1 day',
                                               whatToShow='MIDPOINT', useRTH=True)
        bars = pd.DataFrame(bars)
        bars.to_csv('stock_data/' + ticker + '.csv')


def place_order(port, orders_client_id):
    ib = IB()
    ib.connect(host='127.0.0.1', port=port, clientId=orders_client_id)
    contract = Stock('IVV', 'SMART', 'USD')
    ib.qualifyContracts(contract)

    #Find current price
    px = ib.reqMktData(contract, 221).last
    px = 391.12

    new_order = LimitOrder("Buy", 5, px)
    print(new_order)
    ib.placeOrder(contract, new_order)

    sleep(100)
    ib.disconnect()




def get_order_fills(ib_connection):
    #stock_contract = Stock(ticker, "smart", 'USD')
    filled_orders = ib_connection.fills()
    print(filled_orders)

