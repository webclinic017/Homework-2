import time
import random


# Check whether to buy or sell
def decide_to_buy_or_sell():
    choice = random.choice([1, 2])
    print(choice)
    return choice
