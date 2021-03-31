#Trade Order Output
1. ID: Unique ID for a single trade
2. TimeStamp (i.e. 16MAR21)
3. Symbol (Ticker)
4. Action (BUY/SELL)
5. Size (number shares, i.e. 150)
6. Price ($347.50)
7. Type + Special Parameters (LMT)


#Strategy
1. Wait till market close
2. Draw trend line from historical data through volume weighted average for n days
3. Calculate difference in closing price from mean (% deviation) using trend line for mean
4. If deviation percent is less than threshold (-5%), then put out limit order to buy (% deviation) * (#shares alpha) at market close price (goes live when market opens)
    - Limit order is good till market close (GTC)
5. Check if limit order would fill based on high-low 
6. If order fills, put out sell order for the same number of shares at some price (beta, greater than one) reltive to price
7. 


#Parameters
1. n - size of window in day
2. start date of strategy
3. threshold
4. alpha
5. beta



TODO
1. Scrape data to get S&P500 data (done?)
2. Check if there is enough data for the window and the start of strategy date
3. Don't create dataframe twice when creating df_window
