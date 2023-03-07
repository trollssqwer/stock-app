import time
import MetaTrader5 as mt5
import pandas as pd

print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)

# establish connection to the MetaTrader 5 terminal
if not mt5.initialize(login=113808435, server="Exness-MT5Trial6",password="Tranthong98"):
    print("initialize() failed, error code =",mt5.last_error())
    quit()

import re
import os
path = 'stock_data_M5/'
list_stock = []
for file_name in os.listdir(path):
  list_stock.append(re.search(r"(.+)\_.+" ,file_name).group(1))
list_stock_new = list(dict.fromkeys(list_stock))
list_stock_new
test_df = pd.DataFrame(columns=['ticker', 'point', 'spread' , 'deviation'])
for ticker in list_stock_new:
    print(ticker)
    point = mt5.symbol_info(ticker).point
    spread = mt5.symbol_info(ticker).spread + 1
    row = {'ticker': ticker,'point': point,'spread': spread,'deviation': 5}
    test_df = test_df.append(row, ignore_index=True)
test_df.to_csv('spread2.csv')
print('ok')