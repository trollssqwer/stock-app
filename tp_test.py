import pandas as pd

tp_data = pd.read_csv('/Users/tranthong/Desktop/stock-app/state-output.csv')
#print(tp_data[['state_close_estimate','state_max_r','state_MA_r','state_RSI_r','last_r']].iloc[0:50])
#print(tp_data[['state_close_estimate','state_max_r','state_MA_r','state_RSI_r','last_r']].describe())
print(tp_data[['state_close_estimate','state_max_r','state_MA_r','state_RSI_r','last_r']].sum())