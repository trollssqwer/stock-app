import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from collections import defaultdict
import warnings
import time
pd.options.mode.chained_assignment = None
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime
from datetime import timedelta, date
from scipy.stats import linregress
import MetaTrader5 as mt5
print('System v3 new 1')
import time

import schedule

# MinMax, CI, consolidate_value
def get_max_min(prices, smoothing, window_range):

  smooth_prices = prices['Close'].rolling(window=smoothing).mean().dropna()
  local_max = argrelextrema(smooth_prices.values, np.greater)[0]
  local_min = argrelextrema(smooth_prices.values, np.less)[0]
  price_local_max_dt = []
  for i in local_max:
      if (i>window_range) and (i<len(prices)-window_range):
          price_local_max_dt.append(prices.iloc[i-window_range:i+window_range]['Close'].idxmax())
  price_local_min_dt = []
  for i in local_min:
      if (i>window_range) and (i<len(prices)-window_range):
          price_local_min_dt.append(prices.iloc[i-window_range:i+window_range]['Close'].idxmin())  
  maxima = pd.DataFrame(prices.loc[price_local_max_dt])
  minima = pd.DataFrame(prices.loc[price_local_min_dt])
  max_min = pd.concat([maxima, minima]).sort_index()
  max_min = max_min[~max_min.day_num.duplicated()]
  a = pd.DataFrame()
  a.index = max_min.day_num
  a['Close'] = max_min.Close
  return a

def get_ci(data, lookback):
    high, low, close =  data['High'], data['Low'], data['Close']
    tr1 = pd.DataFrame(high - low).rename(columns = {0:'tr1'})
    tr2 = pd.DataFrame(abs(high - close.shift(1))).rename(columns = {0:'tr2'})
    tr3 = pd.DataFrame(abs(low - close.shift(1))).rename(columns = {0:'tr3'})
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').dropna().max(axis = 1)
    atr = tr.rolling(1).mean()
    highh = high.rolling(lookback).max()
    lowl = low.rolling(lookback).min()
    ci = 100 * np.log10((atr.rolling(lookback).sum()) / (highh - lowl)) / np.log10(lookback)
    new_df = pd.DataFrame()
    new_df['day_num'] = data.day_num
    new_df['ci']= ci.values
    new_df
    return new_df

def get_consolidate_value(ticker, ci_thresh = 50, ci_lookback = 20 , timeframe = mt5.TIMEFRAME_M5 , day1 = 30 , day2= 1):
    start = datetime.now() - timedelta(days=day1)
    end = datetime.now() - timedelta(days=day2)
    rates = mt5.copy_rates_range(ticker, timeframe , start, end)
    rates_frame = pd.DataFrame(rates)
    # convert time in seconds into the 'datetime' format
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame.columns=['index', 'Open', 'High', 'Low', 'Close', 'Volume', 'spread', 'real_volume']
    rates_frame['day_num'] = rates_frame.index.values
    ci = get_ci(rates_frame, ci_lookback)
    rates_frame = rates_frame.merge(ci, how = "inner")
    test_data = rates_frame.loc[rates_frame.ci >= ci_thresh]

    a =(test_data.High - test_data.Low)
    print(str(ticker) + ': ' + 'Consolidation thresh ' + str(a.mean() * 2))
    return a.mean() * 2
#get_consolidate_value('AAPL', '2022-07-01', '2022-08-20', '1h', 50)
def get_rsi(data_raw , rsi_thresh = 30):
  data_raw['close_diff'] = data_raw.Close.diff()
  change_up = data_raw.close_diff.copy()
  change_down = data_raw.close_diff.copy()
  change_up[change_up<0] = 0
  change_down[change_down>0] = 0

  avg_up = change_up.rolling(rsi_thresh).mean()
  avg_down = change_down.rolling(rsi_thresh).mean().abs()
  rsi = 100 * avg_up/ (avg_up + avg_down)
  data_raw['rsi'] = rsi
  return data_raw

def get_mt5_raw_data(ticker,timeframe,bar_nums):
    rates = mt5.copy_rates_from_pos(ticker,timeframe , 0, bar_nums)
    rates_frame = pd.DataFrame(rates)
    start = datetime.fromtimestamp(rates_frame['time'].iloc[0])
    # convert time in seconds into the datetime format
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    start = rates_frame['time'].iloc[0]
    print(str(ticker) + ': ' + 'check start from ' + str(start))
    rates_frame.columns=['index', 'Open', 'High', 'Low', 'Close', 'Volume', 'spread', 'real_volume']
    rates_frame['day_num'] = rates_frame.index.values
    return rates_frame, start

def get_mt5_raw_data_range(ticker, start, end ,timeframe):
    rates = mt5.copy_rates_range(ticker,timeframe , start, end)
    rates_frame = pd.DataFrame(rates)
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame.columns=['index', 'Open', 'High', 'Low', 'Close', 'Volume', 'spread', 'real_volume']
    rates_frame['day_num'] = rates_frame.index.values
    return rates_frame

# Imbalance class
class Imbalance:
  def __init__(self, data):
    self.data = data
  def get_gap(self, row):
    i = int(row.name)
    open1 = self.data.Close.iloc[i]
    open2 = self.data.Close.iloc[i - 1] if i != 0 else self.data.Open.iloc[i]
    return [open1,open2] if open2 - open1 >  0 else [open2,open1] 

  def get_next_range(self, row):
    i = int(row.name)+1 
    if i < len(self.data):
      next_high = self.data.High.iloc[i:].max()
      next_low = self.data.Low.iloc[i:].min() 
    else:
      next_low = 0
      next_high = 0
    return [next_low,next_high]

  def get_imbalance(self ,row):
    low = row.gap[0]
    high = row.gap[1]
    if row.next_range[0] <= low <= row.next_range[1] and high >= row.next_range[1]:
      return row.next_range[1],high
    elif row.next_range[0] <= high <= row.next_range[1] and low <= row.next_range[0]:
      return low,row.next_range[0]
    elif row.name == len(self.data) - 1:
      return low,high
    return None,None

# WyckOff class
class WyckOff:
  def __init__(self, data, consolidate_thresh = 2, spread_thresh = 2, ci_thresh = 50, \
               window = 100, tail_rate = 0.3 ,imb_rate = 0.3 ,break_rate = 0.5 , \
               ci_lookback = 14 , minmax_smoothing = 3, minmax_window = 14, min_boxsize = 20, MA_thresh = 50):
    self.data = data
    self.consolidate_thresh = consolidate_thresh
    self.spread_thresh = spread_thresh
    self.ci_thresh = ci_thresh
    self.ci_lookback = ci_lookback
    self.minmax_smoothing = minmax_smoothing
    self.minmax_window = minmax_window
    self.tail_rate = tail_rate
    self.imb_rate = imb_rate
    self.break_rate = break_rate
    self.window = window
    self.min_boxsize = min_boxsize
    self.MA_thresh = MA_thresh

  def detect_box2(self):
    def get_mean(row):
      x = int(row.name)
      return window_data.return_on_point.iloc[0:x].abs().mean()
    k = 0
    box_list = []
    data_check = self.data.loc[self.data.ci > self.ci_thresh]
    while k < len(data_check):
      i = data_check.iloc[k].name
      window_data = self.data.iloc[i:] if i > len(self.data) - self.window else self.data.iloc[i:i+ self.window]
      close = self.data.Close.iloc[i]
      window_data['return_on_point'] = window_data['Close'] - close
      window_data['mean_return'] = window_data.apply(get_mean, axis = 1)
      last_point = window_data.loc[(window_data.mean_return < self.consolidate_thresh) & \
                                    (window_data.return_on_point.abs() <  self.spread_thresh ) ]
      if last_point.empty:
        k = k + 1 
      #elif last_point.day_num.iloc[-1] == self.data.iloc[-2].day_num:
      elif last_point.day_num.iloc[-1] != 1:
        last_point = last_point.iloc[-1]
        x1 = self.data.iloc[i].day_num
        x2 = last_point.day_num
        open_box = self.data.Open.iloc[self.data.iloc[i].name:last_point.name]
        close_box = self.data.Close.iloc[self.data.iloc[i].name:last_point.name]
        y1 = min(open_box.min() , close_box.min())
        y2 = max(open_box.max() , close_box.max())
        #new_cons = (open_box - close_box).abs().mean()
        if x2 - x1 < self.min_boxsize:
          k = k + 1 
        else:
          box = [x1,x2,y1,y2]
          #print(box)
          #print('New cons : ' + str(new_cons))
          box_list.append(box)
          k = k + 1 
          #break
      else:
        k = k + 1
    return box_list


  def bos_imbalance(self, box_list):
    bos_box = []
    for i in range(len(box_list)):
      x1,x2,y1,y2 = box_list[i]
      break_thresh = self.consolidate_thresh * self.break_rate
      imb_thresh =  self.consolidate_thresh * self.imb_rate
      imb_tail_thresh = self.tail_rate
      bos = self.data.loc[(((self.data.imbalance2 - y2 > break_thresh) & (self.data.Open < self.data.Close))  | \
                           ((y1 - self.data.imbalance1 > break_thresh) & (self.data.Open > self.data.Close)) | \
                           ((y1 - self.data.imbalance1 > break_thresh) & (self.data.blank_gap > self.consolidate_thresh / 2)) | \
                           ((self.data.imbalance2 - y2 > break_thresh) & (self.data.blank_gap > self.consolidate_thresh / 2)) ) & \
                    (((self.data.imbalance2 - self.data.imbalance1) / (self.data.High - self.data.Low)).abs() > imb_tail_thresh) & \
                    (self.data.imbalance2 - self.data.imbalance1 > imb_thresh)  &  \
                    (self.data.day_num > x2) & (x2 - x1 > self.min_boxsize) ]
      if not bos.empty:
        bos = bos.iloc[0]
        new_x2 = bos.day_num - 1
        x1_index = self.data.loc[self.data.day_num == x1].iloc[0].name
        x2_index = self.data.loc[self.data.day_num == bos.day_num - 1].iloc[0].name

        open_box = self.data.iloc[x1_index:x2_index].Open.max()
        close_box = self.data.iloc[x1_index:x2_index].Close.min()
        new_y1 = min(open_box.min() , close_box.min())
        new_y2 = max(open_box.max() , close_box.max())
        bos = self.data.loc[(((self.data.imbalance2 - new_y2 > break_thresh) & (self.data.Open < self.data.Close))  | \
                           ((new_y1 - self.data.imbalance1 > break_thresh) & (self.data.Open > self.data.Close)) | \
                           ((new_y1 - self.data.imbalance1 > break_thresh) & (self.data.blank_gap > self.consolidate_thresh / 2)) | \
                           ((self.data.imbalance2 - new_y2 > break_thresh) & (self.data.blank_gap > self.consolidate_thresh / 2)) ) & \
                    (((self.data.imbalance2 - self.data.imbalance1) / (self.data.High - self.data.Low)).abs() > imb_tail_thresh) & \
                    (self.data.imbalance2 - self.data.imbalance1 > imb_thresh)  & \
                    (self.data.day_num > new_x2) & (new_x2 - x1 > self.min_boxsize) & (self.data.day_num < new_x2 + 3) ]
        #print(new_y1 , bos.imbalance1.iloc[0], break_thresh)

        if not bos.empty :
          if (abs(self.data.Close.iloc[x1_index] - self.data.Close.iloc[x2_index]) > self.spread_thresh):
            continue
            #print('Bos box but break to far!')
            #print([x1,new_x2,new_y1,new_y2])
          else:
            bos = bos.iloc[0]
            bos_gap = bos.imbalance2 - bos.imbalance1
            if(bos_gap > self.consolidate_thresh * 1.5 and bos_gap < self.consolidate_thresh * 3):
              if bos.Close > bos.Open:
                bos.imbalance1 = bos.imbalance1 + (bos.imbalance2 - bos.imbalance1) / 2
              elif bos.Close <= bos.Open:
                bos.imbalance2 = bos.imbalance2 - (bos.imbalance2 - bos.imbalance1) / 2          
            new_box =[ x1,new_x2,new_y1,new_y2 ]
            
            self.data.bos_imbalance1.iloc[bos.name] = bos.imbalance1
            self.data.bos_imbalance2.iloc[bos.name] = bos.imbalance2
            bos_box.append(new_box)
    return self.data,bos_box


  def convert_data(self):
    # data.reset_index(inplace = True)
    # data['day_num'] = data.index.values
    minmax = get_max_min(self.data, self.minmax_smoothing, self.minmax_window)
    self.data['minmax'] = None
    for i in minmax.index.values:
      self.data['minmax'].loc[self.data.day_num == i] = self.data['Close'].loc[self.data.day_num == i] 

    ci = get_ci(self.data, self.ci_lookback).dropna()
    self.data['ci'] = None
    for i in ci.day_num.to_numpy():
      self.data['ci'].loc[self.data.day_num == i] = ci.ci.loc[ci.day_num == i].values[0]
    imb = Imbalance(self.data)
    self.data['blank_gap'] = abs(self.data['Open'] - self.data['Close'].shift(1))
    self.data['gap'] = self.data.apply(imb.get_gap, axis = 1)
    self.data['MA'] =  self.data.Close.rolling(self.MA_thresh).mean()
    self.data['next_range'] = self.data.apply(imb.get_next_range, axis = 1)
    self.data['imbalance1'] = self.data.apply(lambda x: imb.get_imbalance(x)[0], axis = 1)
    self.data['imbalance2'] = self.data.apply(lambda x: imb.get_imbalance(x)[1], axis = 1)
    self.data['bos_imbalance1'] = None
    self.data['bos_imbalance2'] = None
    #data.drop(columns = ['gap', 'next_range'], inplace = True)
    
    box_list = self.detect_box2()
    if box_list:
      self.data, bos_box = self.bos_imbalance(box_list)
      if bos_box:
        return self.data, bos_box[0]
      else:
        return self.data, None
    else:
      return self.data, None
    # return box_list

# try new trend
class Trend:
  def __init__(self, data, bos , box , bot_thresh):
    self.data = data
    self.bos = bos
    self.box = box
    self.bot_thresh = bot_thresh

  def detect_trend(self):
    data1 = self.data
    reg = []
    reg2 = []
    while len(data1)>=3:
      reg = linregress( x=data1['day_num'], y=data1['High'])
      data1 = data1.loc[data1['High'] > reg[0] * data1['day_num'] + reg[1]]
      if(len(data1) >= 3):
        reg_1 = linregress( x=data1['day_num'],y=data1['High'])
        if reg_1 and reg_1 == reg:
          data1 =[]
    data1 = self.data
    while len(data1)>=3:
      reg2 = linregress( x=data1['day_num'], y=data1['Low'])
      data1 = data1.loc[data1['Low'] < reg2[0] * data1['day_num'] + reg2[1]]
      if(len(data1) >= 3):
        reg2_1 = linregress( x=data1['day_num'],y=data1['Low'])
        if reg2_1 and reg2_1 == reg2:
          data1 = []
    if(len(reg) > 0 and len(reg2) > 0 ):
        return reg[0],reg[1], reg2[0], reg2[1]
    else:
        print('Cant detect reg!!!')
        return 0,0,0,0


  def get_current_trend(self, reg_high_x, reg_low_x):
    if self.bos.Close > self.bos.Open:
      current_trend = 1  if reg_high_x > 0 else  2
    elif self.bos.Close < self.bos.Open:
      current_trend = 2 if reg_low_x < 0 else 1
    else:
      current_trend = 0
    return current_trend

  def get_current_trend2(self, reg_high_x, reg_low_x):
    if reg_high_x > 0 and reg_low_x > 0:
      current_trend = 1
    elif reg_high_x < 0 and  reg_low_x < 0:
      current_trend = 2
    else:
      current_trend = 0
    return current_trend

  def get_break_trend(self, current_trend, reg):
    reg_high_x, reg_high_y, reg_low_x, reg_low_y = reg[0], reg[1], reg[2], reg[3]
    # lower imbalance
    if(self.bos.imbalance1 < self.box[2]):
      low_trend = reg_low_x * self.bos.day_num + reg_low_y
      bot = low_trend - self.bos.imbalance1
      print('bot/bot_thresh: ' + str(self.bos.imbalance1) + '/'+ str(low_trend))
      # break of trend
      if(bot > self.bot_thresh):
        print('break of trend from ' + str(self.bos.imbalance1) + ' to ' + str(low_trend))
        if(current_trend == 1):
          print('change to bearist trend')
        elif(current_trend == 2):
          print('continue bearist trend')
        else:
          print('No trend but BOT')
        print('SELL IMBALANCE')
        return 2

      # not break trend and
      else:
        if(current_trend == 2):
          print('continue current bearish trend')
          print("SELL IMBALANCE")
          return 2
        elif(current_trend == 1):
          print('BOS lower of bullish trend not BOT ... do not thing!')
          return 0
        else:
          print('Not trend, not BOT ... do not thing')
          return 0
    elif(self.bos.imbalance2 > self.box[3]):
      high_trend = reg_high_x * self.bos.day_num + reg_high_y
      bot = self.bos.imbalance2 - high_trend 
      print('bot/bot_thresh: ' + str(self.bos.imbalance2) + '/'+ str(high_trend))
      if(bot > self.bot_thresh):
        print('break of trend from ' + str(self.bos.imbalance2) + ' to ' + str(high_trend))
        if(current_trend == 1):
          print('continue bullish trend')
        elif(current_trend == 2):
          print('change to bullish trend')
        else:
          print('No trend but BOT')
        print("BUY IMBALANCE")
        return 1
      else:
        if(current_trend == 1):
          print('continue current bullish trend')
          print('BUY IMBALANCE')
          return 1
        elif(current_trend == 2):
          print('BOS upper of bearish trend not BOT ... do not thing!')
          return 0
        else:
          print('Not trend, not BOT ... do not thing')
          return 0

class Order:
  def __init__(self,ticker, risk, consolidate_thresh, data_order_raw, deviation = 5):
    self.deviation = deviation
    self.ticker = ticker
    self.risk = risk
    self.lot = 0
    self.point = mt5.symbol_info(self.ticker).point
    self.spread = mt5.symbol_info(self.ticker).spread + 1
    self.profit_cut = []
    self.data_order_raw = data_order_raw
    self.order_list = []
    self.order_state = []
    self.order_count_win = 0
    self.order_count_loss = 0
    self.order_profit = 0
    self.order_count_cut = 0
    self.consolidate_thresh = consolidate_thresh
    self.posititon = []

  def detect_trend(self, data_check):
    data1 = data_check
    reg = []
    reg2 = []
    while len(data1)>=3:
      reg = linregress( x=data1['day_num'], y=data1['High'])
      data1 = data1.loc[data1['High'] > reg[0] * data1['day_num'] + reg[1]]
      if(len(data1) >= 3):
        reg_1 = linregress( x=data1['day_num'],y=data1['High'])
        if reg_1 and reg_1 == reg:
          data1 =[]
    data1 = data_check
    while len(data1)>=3:
      reg2 = linregress( x=data1['day_num'], y=data1['Low'])
      data1 = data1.loc[data1['Low'] < reg2[0] * data1['day_num'] + reg2[1]]
      if(len(data1) >= 3):
        reg2_1 = linregress( x=data1['day_num'],y=data1['Low'])
        if reg2_1 and reg2_1 == reg2:
          data1 = []
    if(len(reg) > 0 and len(reg2) > 0 ):
        return reg[0],reg[1], reg2[0], reg2[1]
    else:
        print('Cant detect reg!!!')
        return 0,0,0,0


  def detect_order_trend(self, data_check , order_check):
    reg_high_x, reg_high_y, reg_low_x, reg_low_y = self.detect_trend(data_check)
    if order_check == 1 and reg_low_x > 0:
      return reg_low_x ,reg_low_y
    elif order_check == 2 and reg_high_x < 0:
      return reg_high_x ,reg_high_y
    else:
      data_check_split = data_check.iloc[int(len(data_check) / 2):]
      reg_high_x, reg_high_y, reg_low_x, reg_low_y = self.detect_trend(data_check_split)
      if order_check == 2:
        return reg_high_x ,reg_high_y 
      else:
        return  reg_low_x ,reg_low_y

  def send_order(self , price, stoploss, mt5_order_type):
      if mt5_order_type == mt5.ORDER_TYPE_BUY_LIMIT:
          price = price + self.deviation * self.point
          stoploss = stoploss - self.spread * self.point
          total_point = int((price - stoploss) / self.point)
          takeprofit1 = price + total_point * 3 * self.point
          takeprofit2 = price + total_point * 10 * self.point
      else:
          price = price - self.deviation * self.point
          stoploss = stoploss + self.spread * self.point
          total_point = int((stoploss - price) / self.point)
          takeprofit1 = price - total_point * 3 * self.point
          takeprofit2 = price - total_point * 10 * self.point      
      self.lot = float("{:.2f}".format(self.risk /total_point))
      if(self.lot < 0.01):
        print('min order lot. Cant set order!')
        return None
      request1 = {
          "action": mt5.TRADE_ACTION_PENDING,
          "symbol": self.ticker,
          "volume": self.lot,
          "type": mt5_order_type,
          "price": price,
          "sl": stoploss,
          "deviation": self.deviation,
          "magic": 234000,
          "comment": "python script open",
          "type_time": mt5.ORDER_TIME_GTC,
          "type_filling": mt5.ORDER_FILLING_RETURN,
      }
      result1 = mt5.order_send(request1)
      if result1.retcode == mt5.TRADE_RETCODE_DONE:
          print(str(self.ticker) + ': ' + 'Send stop limit realtime request done!')
          posititon = [request1, result1]
          return posititon
      else:
        print(str(self.ticker) + ': ' + 'Send stop limit FAIL!')
        if(mt5_order_type == mt5.ORDER_TYPE_BUY_LIMIT):
          mt5_order_type = mt5.ORDER_TYPE_BUY
          price = mt5.symbol_info_tick(self.ticker).ask
          total_point = (price -stoploss) / self.point
        else:
          mt5_order_type = mt5.ORDER_TYPE_SELL
          price = mt5.symbol_info_tick(self.ticker).bid
          total_point = (stoploss - price) / self.point
        self.lot =  float("{:.2f}".format(self.risk /total_point))
        request1 = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.ticker,
            "volume": self.lot,
            "type": mt5_order_type,
            "price": price,
            "sl": stoploss,
            "deviation": self.deviation,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result1 = mt5.order_send(request1)
        if result1.retcode == mt5.TRADE_RETCODE_DONE:
          print(str(self.ticker) + ': ' + 'Send deal realtime request done!')
          posititon = [request1, result1]
          return posititon
        else:
          print(str(self.ticker) + ': ' + 'Send realtime request FAIL !')
          return None
  
  def update_sl(self):
    positions_list=mt5.positions_get()
    risk = 10
    min_thresh = 6
    cut_thresh = 0.61
    for pos in positions_list:
        pos = pos._asdict()
        if('profit' in pos.keys() and pos['profit'] / risk > min_thresh):
            new_sl = abs(pos['profit'] * cut_thresh * mt5.symbol_info(pos['symbol']).point / pos['volume'] - pos['price_open'])
            print(pos['symbol'] ,pos['volume'] ,pos['type'] ,pos['ticket'], pos['price_open'] ,new_sl)
            pair,volume,pos_type,ticket,p_open,SL = pos['symbol'] ,pos['volume'] ,pos['type'] ,pos['ticket'], pos['price_open'] ,new_sl
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pair,
                "volume": volume,
                "type": pos_type,
                "position": ticket,
                "price_open": p_open,
                "sl": SL,
                "deviation": 1,
                "magic": 234000,
                "comment": "python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
                "ENUM_ORDER_STATE": mt5.ORDER_FILLING_RETURN,
            }
            #// perform the check and display the result 'as is'
            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print("order_update_sl failed, retcode={}".format(result.comment))

  def close_trade(self, action, buy_request, result, deviation = 1):
      # create a close request
      symbol = buy_request['symbol']
      if action == 'buy':
          trade_type = mt5.ORDER_TYPE_BUY
          price = mt5.symbol_info_tick(symbol).ask
      elif action =='sell':
          trade_type = mt5.ORDER_TYPE_SELL
          price = mt5.symbol_info_tick(symbol).bid
      position_id=result.order
      lot = buy_request['volume']
      ea_magic_number = 234000
      close_request={
          "action": mt5.TRADE_ACTION_DEAL,
          "symbol": symbol,
          "volume": lot,
          "type": trade_type,
          "position": position_id,
          "price": price,
          "deviation": deviation,
          "magic": ea_magic_number,
          "comment": "python script close",
          "type_time": mt5.ORDER_TIME_GTC, # good till cancelled
          "type_filling": mt5.ORDER_FILLING_RETURN,
      }
      # send a close request
      result=mt5.order_send(close_request)

  def get_order2(self, row):
    for order in self.order_list:
      #{"bos_day" :self.bos['index'], "bos_day_num" :self.bos.day_num,"bos_imbalance1" : self.bos.bos_imbalance1, "bos_imbalance2" :self.bos.bos_imbalance2, "order_type" :order,"box_day_num" :self.box[0], "realtime" :1}
      order_day_num, imb1, imb2, order_type , trend_point_check  = order['bos_day_num'], order['bos_imbalance1'] , order['bos_imbalance2'] , order['order_type'] , order['box_day_num']
      imb_gap = (imb2 - imb1) * 0.6
      major_state = self.order_state[0] if len(self.order_state) > 0 else {}
      if(order_day_num < row.day_num - 200):
        print('remove old order at ' + str(order_day_num))
        self.order_list.remove(order)

      elif order_type == 1:
        if (major_state and major_state['order_type'] == 1):
          imb1 = imb1 - imb_gap * 2
          imb2 = imb2 - imb_gap 
          if('realtime' in order.keys()):
            pos = self.send_order(imb2, imb1, mt5.ORDER_TYPE_BUY_LIMIT)
            time.sleep(1)
            if pos:
              order['pos'] = pos
              del order['realtime']
            else:
              self.order_list.remove(order)

          if 'pos' in order.keys() and not mt5.positions_get(symbol= self.ticker):
            # order hes been set in realtime
            print('set buy order at : ' + str(row.day_num) +  ' stoploss at : ' + str(imb1) + ' open_order at: ' + str(imb2))
            self.order_list.remove(order)
            order_row = {"order_type": 1, "open_order": imb2, "order_day_num": row.day_num, "stop_loss": imb1 , 'order_start': order_day_num, 'box_start': trend_point_check ,"position":order['pos']}
            self.order_state.append(order_row)

        elif (not major_state):
          imb1 = imb1 - imb_gap
          imb2 = imb2 - imb_gap    
          if('realtime' in order.keys()):
            pos = self.send_order(imb2, imb1, mt5.ORDER_TYPE_BUY_LIMIT)
            time.sleep(1)
            if pos:
              order['pos'] = pos
              del order['realtime']
            else:
              self.order_list.remove(order)
  
          if 'pos' in order.keys() and not mt5.positions_get(symbol= self.ticker):
            # order hes been set in realtime
            print('set buy order at : ' + str(row.day_num) +  ' stoploss at : ' + str(imb1) + ' open_order at: ' + str(imb2))
            data_check = self.data_order_raw.loc[(self.data_order_raw.day_num >= trend_point_check) & (self.data_order_raw.day_num <= row.day_num)]
            reg_x, reg_y = self.detect_order_trend(data_check , order_type)
            if reg_x > 0:
              self.order_list.remove(order)
              order_row = {"order_type": 1, "open_order": imb2, "order_day_num": row.day_num, "stop_loss": imb1 , 'order_start': order_day_num, 'box_start': trend_point_check ,"position":order['pos']}
              self.order_state.append(order_row)
            else:
              print('Cant detect reg. Remove order !!!')
              self.order_list.remove(order)
              if mt5.orders_get(symbol= self.ticker):
                self.close_trade('sell', order['pos'][0], order['pos'][1])
        else:
          self.order_list.remove(order)
          print('No reverse order!!! Remove order')

      elif order_type == 2:
        if (major_state and major_state['order_type'] == 2):
          imb1 = imb1 + imb_gap 
          imb2 = imb2 + imb_gap * 2
          if('realtime' in order.keys()):
            pos = self.send_order(imb1, imb2, mt5.ORDER_TYPE_SELL_LIMIT)
            time.sleep(1)
            if pos:
              order['pos'] = pos
              del order['realtime']
            else:
              self.order_list.remove(order)
          if 'pos' in order.keys() and not mt5.positions_get(symbol= self.ticker):
            # order hes been set in realtime
            print('set sell order at :' + str(row.day_num) +  ' stoploss at : ' + str(imb2) + ' open_order at: ' + str(imb1))
            self.order_list.remove(order)
            order_row = {"order_type": 2, "open_order": imb1, "order_day_num": row.day_num, "stop_loss": imb2 , 'order_start': order_day_num, 'box_start': trend_point_check ,"position":order['pos']}
            self.order_state.append(order_row)
        elif (not major_state):
          imb1 = imb1 + imb_gap
          imb2 = imb2 + imb_gap
          if('realtime' in order.keys()):
            pos = self.send_order(imb1, imb2, mt5.ORDER_TYPE_SELL_LIMIT)
            time.sleep(1)
            if pos:
              order['pos'] = pos
              del order['realtime']
            else:
              self.order_list.remove(order)
          if 'pos' in order.keys() and not mt5.positions_get(symbol= self.ticker):
             # order hes been set in realtime
            print('set sell order at :' + str(row.day_num) +  ' stoploss at : ' + str(imb2) + ' open_order at: ' + str(imb1))
            data_check = self.data_order_raw.loc[(self.data_order_raw.day_num >= trend_point_check) & (self.data_order_raw.day_num <= row.day_num)]
            reg_x, reg_y = self.detect_order_trend(data_check , order_type)
            if reg_x < 0:
              print('set sell order at :' + str(row.day_num) +  ' stoploss at : ' + str(imb2) + ' open_order at: ' + str(imb1))
              self.order_list.remove(order)
              order_row = {"order_type": 2, "open_order": imb1, "order_day_num": row.day_num, "stop_loss": imb2 , 'order_start': order_day_num, 'box_start': trend_point_check ,"position":order['pos']}
              self.order_state.append(order_row)            
            else:
              self.order_list.remove(order)
              print('Cant detect reg. Remove order !!!')
              if mt5.orders_get(symbol= self.ticker):
                self.close_trade('buy', order['pos'][0], order['pos'][1])
        else:
          self.order_list.remove(order)
          print('No reverse order!!! Remove order')
      else:
        print('Out of MA baseline')
        self.order_list.remove(order)

  def check_order2(self, row):
    # print(len(self.data_order_raw) , row.day_num , self.data_order_raw.day_num.iloc[-2] )
    # print(self.data_order_raw.MA.loc[self.data_order_raw.day_num == row.day_num].iloc[0])
    for state in self.order_state:
      if state['order_type'] == 1 and row.day_num > state['order_day_num']:
        r = abs(state['stop_loss'] - state['open_order'])
        print('Current order profit of ' + str(state['order_day_num']) +': ' + str((row.Close - state['open_order']) / r ))
        if(row.Low < state['stop_loss']):
          print('Lose buy order of : ' + str(state['order_day_num']))
          self.order_state.remove(state)
          self.order_profit = self.order_profit - 1
          self.order_count_loss = self.order_count_loss + 1
        elif row.High < self.data_order_raw.MA.loc[self.data_order_raw.day_num == row.day_num].iloc[0] - self.consolidate_thresh * 2 and row.day_num > state['order_day_num'] + 10:
          profit = (- state['open_order'] + row.Close ) / r
          self.order_count_win = self.order_count_win + 1
          self.order_profit = self.order_profit + profit 
          self.order_state.remove(state)
          print('Close buy order of : ' + str(state['order_day_num'])+" PROFIT: " +str(profit))
          request, result = state['position'][0],state['position'][1]
          self.close_trade('sell', request, result)

      elif state['order_type'] == 2 and row.day_num > state['order_day_num']:
        r = abs(state['stop_loss'] - state['open_order'])
        print('Current order profit of ' + str(state['order_day_num']) +': ' + str((state['open_order'] - row.Close ) / r))
        if(row.High > state['stop_loss']):
          print('Lose sell order of : ' + str(state['order_day_num']))
          self.order_state.remove(state)
          self.order_profit = self.order_profit - 1
          self.order_count_loss = self.order_count_loss + 1
        elif row.Low > self.data_order_raw.MA.loc[self.data_order_raw.day_num == row.day_num].iloc[0] + self.consolidate_thresh * 2:
          profit = (state['open_order'] - row.Close ) / r
          self.order_count_win = self.order_count_win + 1
          self.order_profit = self.order_profit + profit 
          self.order_state.remove(state)
          print('Close sell order of : ' + str(state['order_day_num'])+" PROFIT: " +str(profit))
          request, result = state['position'][0],state['position'][1]
          self.close_trade('buy', request, result)

class Trade:
  def __init__(self, ticker, timeframe , bar_nums, risk , ci_lookback = 20):
    self.timeframe = timeframe
    self.consolidate_thresh = get_consolidate_value(ticker, timeframe = self.timeframe) 
    #print('consolidate : ' + str(consolidate_thresh)  + ' spread: ' + str(consolidate_thresh))
    self.ticker = ticker
    self.bot_thresh = self.consolidate_thresh / 2
    self.bar_nums = bar_nums
    self.data_raw , self.init_date = get_mt5_raw_data(self.ticker, self.timeframe ,self.bar_nums)
    self.data_raw = get_rsi(self.data_raw, rsi_thresh= 25)
    self.bos_list = []
    self.boxbos_list = []
    self.start_point = 0
    self.window = 1
    i = self.window
    self.data_trend_raw = self.data_raw
    self.current_trend = 0
    self.previous_trend = 0
    self.previous_trend_point = 0
    self.box = None
    self.bos = None
    self.risk = risk
    self.data_order_raw = self.data_raw.copy()
    self.data_order_raw['MA'] =  self.data_order_raw.Close.rolling(50).mean()

    self.trade_order = Order(self.ticker, self.risk, self.consolidate_thresh, self.data_order_raw)
    self.ci_lookback = ci_lookback
  def check_per_data(self, data_check):
    wo = WyckOff(data_check , ci_thresh = 40 , tail_rate = 0.6 ,imb_rate= 0.3, break_rate=0.3,\
                consolidate_thresh = self.consolidate_thresh  , ci_lookback = self.ci_lookback,\
                spread_thresh = self.consolidate_thresh, min_boxsize = 30)
    data, self.box = wo.convert_data()
    last_row = data.iloc[-1]
    self.trade_order.check_order2(last_row)
    self.trade_order.get_order2(last_row)
    self.trade_order.update_sl()
    if self.box:
      print(self.ticker,self.box)
    bos_data = data.loc[data.bos_imbalance1.notna()]  
    if not bos_data.empty:
      self.bos = bos_data.iloc[-1]
      print(self.ticker, self.bos.day_num,self.bos['index'], self.bos.bos_imbalance1, self.bos.bos_imbalance2 )
      if((self.bos.bos_imbalance2 - self.bos.bos_imbalance1) / self.consolidate_thresh < 3):
        self.bos_list.append([self.bos.day_num, self.bos.bos_imbalance1, self.bos.bos_imbalance2])
        self.boxbos_list.append(self.box)
        return self.bos
    else:
      return None

  def check_trend(self, check_trend_data , realtime = False):
    previous_trend_temp = 0
    trend = Trend(check_trend_data  , self.bos , self.box , self.bot_thresh)
    reg = trend.detect_trend()
    self.current_trend = trend.get_current_trend2(reg[0], reg[2])
    print('current_trend is ' + str(self.current_trend) + ' trend from ' + str(self.start_point) + ' to ' + str(self.bos.day_num - 1))
    if(self.previous_trend == self.current_trend):
      check_trend_data = self.data_trend_raw.iloc[self.previous_trend_point:self.bos.day_num - 1]
      trend = Trend(check_trend_data , self.bos , self.box , self.bot_thresh)
      reg = trend.detect_trend()
      if self.current_trend == 0:
        self.current_trend = trend.get_current_trend2(reg[0], reg[2])
      print('current_trend longer lookback is ' + str(self.current_trend) + ' trend from ' + str(self.previous_trend_point) + ' to ' + str(self.bos.day_num - 1))
      order = trend.get_break_trend(self.current_trend, reg)
    else:
      previous_trend_temp = self.previous_trend
      order = trend.get_break_trend(self.current_trend, reg)
      self.previous_trend_point = self.start_point
      self.previous_trend = self.current_trend
    if order:
      if(order > 0):
        if realtime:
          order_imbalance = {"bos_day" :self.bos['index'], "bos_day_num" :self.bos.day_num,"bos_imbalance1" : self.bos.bos_imbalance1, "bos_imbalance2" :self.bos.bos_imbalance2, "order_type" :order,"box_day_num" :self.box[0], "realtime" :1}
        else:
          order_imbalance = {"bos_day" :self.bos['index'], "bos_day_num" :self.bos.day_num,"bos_imbalance1" : self.bos.bos_imbalance1, "bos_imbalance2" :self.bos.bos_imbalance2, "order_type" :order,"box_day_num" :self.box[0]}
        self.trade_order.order_list.append(order_imbalance)
      else:
        self.previous_trend = previous_trend_temp

  def init_trader(self):
    i = 1
    while i < len(self.data_raw):
      print(str(self.ticker) + ': ' + str(self.start_point) + ' forward ' + str(i))
      data = self.data_raw.iloc[self.start_point:i]
      data.reset_index(inplace = True)
      bos = self.check_per_data(data)
      if bos is not None and not bos.empty:
        check_trend_data = self.data_trend_raw.iloc[self.start_point:bos.day_num - 1]
        self.check_trend(check_trend_data)
        print("-----------------")
        self.start_point = bos.day_num + 1 - self.ci_lookback
        i = self.start_point + self.window + self.ci_lookback if self.start_point + self.window < len(self.data_raw) else len(self.data_raw) - 1
        print(str(self.ticker) + ': ' + str(self.start_point) + ' to ' + str(i)) 
      else:
        if(len(data) > 200):
          self.start_point = self.start_point + 100 - self.ci_lookback
          i = self.start_point + 100 + self.ci_lookback if self.start_point + 100 < len(self.data_raw) else len(self.data_raw) - 1
          print(str(self.ticker) + ': ' + 'cut old price ' + str(self.start_point) + ' to ' + str(i)) 
        else:
          i = i + 1

  def run_trader(self):
    data_raw_realtime = get_mt5_raw_data_range(self.ticker, self.init_date, datetime.now(), timeframe= self.timeframe)
    data_raw_realtime = get_rsi(data_raw_realtime, rsi_thresh= 25)
    self.data_trend_raw = data_raw_realtime.copy()
    self.trade_order.data_order_raw = data_raw_realtime.copy()
    self.trade_order.data_order_raw['MA'] =  self.trade_order.data_order_raw.Close.rolling(50).mean()
    print(str(self.ticker) + ': ' + str(self.start_point) + ' forward realtime to ' + str(len(data_raw_realtime) - 1) + ' time: ' + str(data_raw_realtime['index'].iloc[-2]))
    data = data_raw_realtime.iloc[self.start_point:-1].copy()
    data.reset_index(inplace = True)
    bos = self.check_per_data(data)
    if bos is not None and not bos.empty:
      check_trend_data = self.data_trend_raw.iloc[self.start_point:bos.day_num - 1]
      self.check_trend(check_trend_data ,realtime = True)
      self.start_point = bos.day_num + 1 - self.ci_lookback
  def live_trading(self):
      self.init_trader()
      for i in range(len(self.trade_order.order_list)):
        self.trade_order.order_list[i].append(1) 
      self.trade_order.order_state = []
      print(self.trade_order.order_listd)
      print('go to live trading !')
      schedule.every().minute.at(':10').do(self.run_trader)
      while True:
        schedule.run_pending()

class Portfolio:
  def __init__(self,  portfolio ,bar_nums, timeframe , risk):
    self.portfolio = portfolio
    self.list_trade = []
    self.timeframe = timeframe
    self.bar_nums = bar_nums
    self.risk  = risk
  def portfolio_init(self):
    for symbol in self.portfolio:
      trade = Trade(symbol , self.timeframe , self.bar_nums , risk = self.risk)
      trade.init_trader()
      for i in range(len(trade.trade_order.order_list)):
        if mt5.orders_get(symbol):
          trade.trade_order.order_list[i].append(1) 
        else:
          print('order have been already set or no order with this symbol !')
      trade.trade_order.order_state = []
      self.list_trade.append(trade)
      print(trade.trade_order.order_list)
      print("-----------------")

  def run_portfolio(self):
    for trade in self.list_trade:
      trade.run_trader()
      
  def live_trading_portfolio(self):
    self.portfolio_init()
    print('go to live trading! Let make some money!')
    schedule.every().minute.at(':05').do(self.run_portfolio)
    while True:
      schedule.run_pending()

import time
import MetaTrader5 as mt5

print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)

# establish connection to the MetaTrader 5 terminal
if not mt5.initialize(login=114057554, server="Exness-MT5Trial6",password="Tranthong98"):
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# portfolio =['MMM',  'IBM' ,'XOM', 'LIN' , 'LMT', 'MCD' , 'CVS' , 'INTU', 'BMY', 'AAPL', 'AMZN', 'ABBV', 'TSLA', 'EA', 'F', 'NVDA']
# trader_list = Portfolio(portfolio, 200 ,mt5.TIMEFRAME_M5 , risk = 5)
# trader_list.live_trading_portfolio()

import os 
import re
# path = 'stock_data_M5/'
# list_stock = []
# for file_name in os.listdir(path):
#   list_stock.append(re.search(r"(.+)\_.+" ,file_name).group(1))
# list_stock_new = list(dict.fromkeys(list_stock))
portfolio = ['ADBE', 'ADP', 'AMGN', 'AVGO', 'BMY', 'CHTR',
       'CMCSA', 'COST', 'EA', 'GILD', 'HD', 'IBM', 'INTU', 'KO', 'LIN',
       'LMT', 'MA', 'MCD', 'MDLZ', 'MMM', 'MO', 'MRK', 'NKE', 'TMO', 'TSLA']
trader_list = Portfolio(portfolio, 150 ,mt5.TIMEFRAME_M5 , risk = 10)
trader_list.live_trading_portfolio()


# portfolio = ['JP225', 'US30' ,'DE30', 'STOXX50' ]
# trader_list = Portfolio(portfolio, 200 ,mt5.TIMEFRAME_M5 , risk = 10)
# trader_list.live_trading_portfolio()