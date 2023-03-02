import time
import MetaTrader5 as mt5

print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)

# establish connection to the MetaTrader 5 terminal
if not mt5.initialize(login=113808435, server="Exness-MT5Trial6",password="Tranthong98"):
    print("initialize() failed, error code =",mt5.last_error())
    quit()
