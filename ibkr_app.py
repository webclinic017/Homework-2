from ib_insync import *
from os import listdir, remove
from time import sleep
import pickle
from strategy import *
from helper_functions import *

# Define your variables here ###########################################################################################
sampling_rate = 1 # How often, in seconds, to check for inputs from Dash?
# For TWS Paper account, default port is 7497
# For IBG Paper account, default port is 4002
port = 7497
# choose your master id. Mine is 10645. You can use whatever you want, just set it in API Settings within TWS or IBG.
master_client_id = 12346
# choose your dedicated id just for orders. I picked 1111.
orders_client_id = 1111
# account number: you'll need to fill in yourself. The below is one of my paper trader account numbers.
acc_number = 'DU3576436'
########################################################################################################################

# Run your helper function to clear out any io files left over from old runs
check_for_and_del_io_files()

# Create an IB app; i.e., an instance of the IB() class from the ib_insync package
ib = IB()
# Connect your app to a running instance of IBG or TWS
ib.connect(host='127.0.0.1', port=port, clientId=master_client_id)

# Make sure you're connected -- stay in this while loop until ib.isConnected() is True.
while not ib.isConnected():
    sleep(.01)

# If connected, script proceeds and prints a success message.
print('Connection Successful!')

#print(get_historical_data(ib, 'IVV'))

# Main while loop of the app. Stay in this loop until the app is stopped by the user.
while True:

    # Check whether to buy or sell shares of IVV
    decision = decide_to_buy_or_sell()
    # Use decision to place buy or sell order
    if (decision == 1):
        print('buy')
        place_order(port, orders_client_id)
    else:
        print('sell')


    sleep(2)
