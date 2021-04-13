import pandas as pd
from datetime import datetime
from ib_insync import *
import market_status as ms
import strategy as strat
import os
from time import sleep
import helper_functions as hf
import ib_insync


def setup_portfolio(date, portfolio_balance):
    if not os.path.exists('live_trading_files'):
        os.makedirs('live_trading_files')
    if not os.path.exists('live_trading_files/live_portfolio.csv'):
        print("test")
        df_portfolio = pd.DataFrame({"date": date - pd.Timedelta(days=1),
                                "USD": portfolio_balance, "IVV_shares": 0, "IVV_value": 0,
                                "Total": portfolio_balance}, index=[0])
        df_portfolio.to_csv('live_trading_files/live_portfolio.csv', index=False)


def run_strategy_live(port=7497, order_client_id=2222, live_client_id=12347, account='DU3576436', date=datetime.today(),
                      ticker='IVV', n_days=252, threshold=0.001, alpha=0.06, beta=1.2):
    # Check if market is open. Prevents strategy from being run live on weekends or holidays
    if not ms.is_market_open_today():
        return
    # If portfolio has not been set up yet, create file to track
    setup_portfolio(date, portfolio_balance=100000)

    ib_live = IB()
    ib_live.connect(host='127.0.0.1', port=port, clientId=live_client_id)
    while not ib_live.isConnected():
        sleep(.01)

    available_funds = hf.get_available_funds(ib_live, account)

    # Determine if a buy trade should execute
    close = hf.get_stock_close_price_today(ib_live, ticker)
    expected = strat.calc_today_expected_price(ticker, n_days)
    print(expected)
    ratio = close / expected
    if not strat.decide_to_buy_or_not(ratio, threshold):
        return
    print(ratio)
    # Determine price and size of buy order
    c = hf.plain_stock_contract(ib_live, 'IVV')
    #px = ib_live.reqMktData(c, 221)
    sz_f = strat.trade_size_factor(ratio, threshold)
    n_shares = strat.trade_size(sz_f, alpha, available_funds, close)

    # Place buy and sell order if they should be executed
    hf.place_market_order(port, order_client_id, n_shares)
    hf.place_limit_order(port, order_client_id+1, 'SELL', n_shares, round(beta*close, 2))
    ib_live.disconnect()


run_strategy_live()
