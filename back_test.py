import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import csv
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime
from datetime import timedelta, date
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")
path = '/home/tranthong/stock_data_M5/stock_data_M5/'
#path = '/Users/tranthong/Desktop/stock-app/stock_data_M5/'
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

def get_consolidate_value(ticker, ci_thresh = 50, ci_lookback = 14 , day1 = 80 , day2= 50):
    start = datetime.now() - timedelta(days=day1)
    end = datetime.now() - timedelta(days=day2)
    stock_dir = path + ticker +'_2022.csv'
    rates =  pd.read_csv(stock_dir, index_col =[0])
    rates['index'] = rates['index'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    a = rates.loc[ (rates['index'] > start) & (rates['index'] < end )]
    a.reset_index(inplace = True ,drop = True)
    a['day_num'] = a.index
    ci = get_ci(a, ci_lookback)
    a = a.merge(ci, how = "inner")
    a = a.loc[a.ci >= ci_thresh]
    b =(a.High - a.Low)
    print(str(ticker) + ': ' + 'Consolidation thresh ' + str(b.mean() * 2))
    return b.mean() * 2

def get_mt5_raw_data_range(ticker, start, end):
    stock_dir = path + ticker +'_2022.csv'
    rates =  pd.read_csv(stock_dir, index_col =[0])
    rates['index'] = rates['index'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    a = rates.loc[ (rates['index'] > start) & (rates['index'] < end )]
    a.reset_index(inplace = True ,drop = True)
    a['day_num'] = a.index
    return a


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

class WyckOff:
  def __init__(self, data, consolidate_thresh = 2, spread_thresh = 2, ci_thresh = 50, \
               window = 100, tail_rate = 0.3 ,imb_rate = 0.3 ,break_rate = 0.5 , \
               ci_lookback = 14 , minmax_smoothing = 3, minmax_window = 14, min_boxsize = 20):
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
  def __init__(self, data_order_raw , consolidate_thresh):
    self.data_order_raw = data_order_raw
    self.order_list = []
    self.order_state = []
    self.order_reg = []
    self.order_count_win = 0
    self.order_count_loss = 0
    self.order_profit = 0
    self.order_count_cut = 0
    self.order_cut = []
    self.consolidate_thresh = consolidate_thresh
    self.order_state_portfolio = []


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
      data_check_split = data_check.iloc[int(len(data_check) * 0.6):]
      reg_high_x, reg_high_y, reg_low_x, reg_low_y = self.detect_trend(data_check_split)
      if order_check == 2:
        return reg_high_x ,reg_high_y
      else:
        return  reg_low_x ,reg_low_y

  def update_trend(self, temp_day_num = 0):
      list_box_start = []
      list_order_day_num = []
      order_check = 0
      for s in self.order_state:
        list_box_start.append(s['box_start'])
        list_order_day_num.append(s['order_day_num'])
        order_check = s['order_type']
      new_trend_x1 = min(list_box_start)
      new_trend_x2 = max(list_order_day_num)
      if(temp_day_num == 0):
        data_check = self.data_order_raw.loc[(self.data_order_raw.day_num >= new_trend_x1) & (self.data_order_raw.day_num <= new_trend_x2)]
      else:
        data_check = self.data_order_raw.loc[(self.data_order_raw.day_num >= new_trend_x1) & (self.data_order_raw.day_num <= temp_day_num)]
      reg_x, reg_y = self.detect_order_trend(data_check , order_check)
      temp_state = []
      for s in self.order_state:
        s['reg_x'] = reg_x
        s['reg_y'] = reg_y
        temp_state.append(s)
      self.order_state = temp_state

  def get_order2(self, row):
    for order in self.order_list:
      order_day_num, imb1, imb2, order_type , trend_point_check  = order[1], order[2] , order[3] , order[4] , order[5]
      imb_gap = (imb2 - imb1) * 0.6
      
      major_state = self.order_state[0] if len(self.order_state) > 0 else {}

      if(order_day_num < row.day_num - 200):
        self.order_list.remove(order)
      elif order_type == 1:
        if (major_state and major_state['order_type'] == 1):
          imb1 = imb1 - imb_gap * 2
          imb2 = imb2 - imb_gap 
          if(row.Low <= imb2):
            #set order 
            data_check = self.data_order_raw.loc[(self.data_order_raw.day_num >= trend_point_check) & (self.data_order_raw.day_num <= row.day_num)]
            reg_x, reg_y = self.detect_order_trend(data_check , order_type)
            #reg_x, reg_y = order_reg[2] ,  order_reg[3] 
            order_row = {"order_type": 1, "open_oder": imb2, "order_day_num": row.day_num, "stop_loss": imb1 , \
                        "take_porfit_1": imb2 + (imb2 - imb1) * 3 , "take_porfit_2": imb1 + (imb2 - imb1) * 10, \
                        "reg_x" : reg_x, "reg_y" : reg_y , 'order_start': order_day_num, 'box_start': trend_point_check , 'current_max_r' : 0}
            self.order_list.remove(order)
            self.order_state.append(order_row)
            self.update_trend()
        elif (not major_state):
          imb1 = imb1 - imb_gap
          imb2 = imb2 - imb_gap           
          if(row.Low <= imb2):
            #set order 
            print('set buy order at : ' + str(row.day_num) +  ' stoploss at : ' + str(imb1) + ' open_order at: ' + str(imb2))
            data_check = self.data_order_raw.loc[(self.data_order_raw.day_num >= trend_point_check) & (self.data_order_raw.day_num <= row.day_num)]
            reg_x, reg_y = self.detect_order_trend(data_check , order_type)
            #reg_x, reg_y = order_reg[2] ,  order_reg[3] 
            if reg_x > 0:
              order_row = {"order_type": 1, "open_oder": imb2, "order_day_num": row.day_num, "stop_loss": imb1 , \
                          "take_porfit_1": imb2 + (imb2 - imb1) * 3 , "take_porfit_2": imb1 + (imb2 - imb1) * 10, \
                          "reg_x" : reg_x, "reg_y" : reg_y , 'order_start': order_day_num, 'box_start': trend_point_check ,  'current_max_r' : 0}
              self.order_list.remove(order)
              self.order_state.append(order_row)
            else:
              print('no reverse order')
              self.order_list.remove(order)
        else:
          print('no reverse order')
          self.order_list.remove(order)
      elif order_type == 2:
        if (major_state and major_state['order_type'] == 2):
          imb1 = imb1 + imb_gap 
          imb2 = imb2 + imb_gap * 2
          if(row.High >= imb1):
            data_check = self.data_order_raw.loc[(self.data_order_raw.day_num >= trend_point_check) & (self.data_order_raw.day_num <= row.day_num)]
            reg_x, reg_y = self.detect_order_trend(data_check , order_type)
            #reg_x, reg_y = order_reg[0] ,  order_reg[1] 
            order_row = {"order_type": 2, "open_oder": imb1,"order_day_num": row.day_num, "stop_loss": imb2 ,\
                        "reg_x" : reg_x, "reg_y" : reg_y, 'order_start': order_day_num , 'box_start': trend_point_check, 'current_max_r' : 0}
            
            self.order_list.remove(order)
            self.order_state.append(order_row)
            self.update_trend()

        elif (not major_state):
          imb1 = imb1 + imb_gap
          imb2 = imb2 + imb_gap
          if(row.High >= imb1):
            #set order 

            data_check = self.data_order_raw.loc[(self.data_order_raw.day_num >= trend_point_check) & (self.data_order_raw.day_num <= row.day_num)]
            reg_x, reg_y = self.detect_order_trend(data_check , order_type)
            if reg_x < 0:
              #reg_x, reg_y = order_reg[0] ,  order_reg[1] 
              order_row = {"order_type": 2, "open_oder": imb1,"order_day_num": row.day_num, "stop_loss": imb2 ,\
                          "reg_x" : reg_x, "reg_y" : reg_y, 'order_start': order_day_num , 'box_start': trend_point_check , 'current_max_r' : 0}
              
              self.order_list.remove(order)
              self.order_state.append(order_row)
        else:
          print('no reverse order')
          self.order_list.remove(order)
      else:
        print('no reverse order')
        self.order_list.remove(order)


  def update_order_state(self, state, row):
    state_order = state['order_type']
    state_day_start = str(self.data_order_raw['index'].loc[self.data_order_raw.day_num == state['order_day_num']].iloc[0])
    state_day_end = str(self.data_order_raw['index'].loc[self.data_order_raw.day_num == row.day_num].iloc[0])
    state_open = state['open_oder']
    state_stop_loss = state['stop_loss']
    state_r = abs(state_open - state_stop_loss)
    data_test = self.data_order_raw.loc[(self.data_order_raw.day_num <= row.day_num) & (self.data_order_raw.day_num >= state['order_day_num'])]
    state_close_estimate = state['state_close_estimate'] if('state_close_estimate' in state.keys()) else None
    last_r = state['last_r'] 
    if state_order == 1:
      state_chain = ((data_test.Close - state_open) / state_r).values.tolist() if not data_test.empty else None
      state_max_r = (data_test.Close.max() - state_open) / state_r
      state_order_MA = data_test.Close.loc[data_test.High < data_test.MA - self.consolidate_thresh * 2]
      state_MA_r = None if state_order_MA.empty else (state_order_MA.iloc[0] - state_open) / state_r
      state_order_RSI = data_test.Close.loc[data_test.rsi > 80]
      state_RSI_r = None if state_order_RSI.empty else (state_order_RSI.iloc[0] - state_open) / state_r
      temp_row = {'state_ticker': row.ticker, 'state_order':state_order, 'state_day_start':state_day_start,  'state_day_end':state_day_end, 'state_open':state_open, 'state_stop_loss':state_stop_loss,\
                 'state_close_estimate':state_close_estimate, 'state_max_r':state_max_r, 'state_MA_r':state_MA_r, 'state_RSI_r':state_RSI_r, 'last_r': last_r, 'state_chain':state_chain}
    elif state_order == 2:
      state_chain = (-(data_test.Close - state_open) / state_r).values.tolist() if not data_test.empty else None
      state_max_r = (state_open - data_test.Close.min() )/ state_r
      state_order_MA = data_test.Close.loc[data_test.Low > data_test.MA + self.consolidate_thresh * 2]
      state_MA_r = None if state_order_MA.empty else (state_open - state_order_MA.iloc[0]) / state_r
      state_order_RSI = data_test.Close.loc[data_test.rsi < 20]
      state_RSI_r = None if state_order_RSI.empty else (state_open -  state_order_RSI.iloc[0]) / state_r
      temp_row = {'state_ticker': row.ticker, 'state_order':state_order, 'state_day_start':state_day_start,  'state_day_end':state_day_end, 'state_open':state_open, 'state_stop_loss':state_stop_loss,\
                 'state_close_estimate':state_close_estimate, 'state_max_r':state_max_r, 'state_MA_r':state_MA_r, 'state_RSI_r':state_RSI_r, 'last_r': last_r , 'state_chain':state_chain}
    return temp_row

  def update_max_order(self, row , cut_thresh = 0.66 , cut_min_profit = 12):
    state_temp = []
    for state in self.order_state:
      r = abs(state['stop_loss'] - state['open_oder'])
      if state['order_type'] == 1 and row.day_num > state['order_day_num']:
        profit = (row.Close - state['open_oder'] ) / r
        state['last_r'] = profit if profit > -1 else -1
        if state['current_max_r'] > cut_min_profit and profit < state['current_max_r'] * cut_thresh and 'state_close_estimate' not in state.keys():
          state['state_close_estimate'] =  state['current_max_r'] * cut_thresh
        elif profit >= state['current_max_r']:
          state['current_max_r'] = profit
      elif state['order_type'] == 2 and row.day_num > state['order_day_num']:
        profit = (state['open_oder'] - row.Close) / r
        state['last_r'] = profit if profit > -1 else -1
        if state['current_max_r'] > cut_min_profit and profit < state['current_max_r'] * cut_thresh and 'state_close_estimate' not in state.keys():
          state['state_close_estimate'] =  state['current_max_r'] * cut_thresh
        elif profit >= state['current_max_r']:
          state['current_max_r'] = profit
      state_temp.append(state)
    return state_temp           

  def check_order_portfolio(self, row):
    self.order_state = self.update_max_order(row)
    for state in self.order_state:
      if state['order_type'] == 1 and row.day_num > state['order_day_num']:

        r = abs(state['stop_loss'] - state['open_oder'])
        if(row.Low < state['stop_loss']):
          print('Lose buy order of : ' + str(state['order_day_num']))
          new_state = self.update_order_state(state, row)
          self.order_state_portfolio.append(new_state)
          self.order_state.remove(state)
          self.order_profit = self.order_profit - 1
          self.order_count_loss = self.order_count_loss + 1
        elif row.High < self.data_order_raw.MA.loc[self.data_order_raw.day_num == row.day_num].iloc[0] - self.consolidate_thresh * 2 and row.day_num > state['order_day_num'] + 10:
          profit = (- state['open_oder'] + row.Close ) / r
          self.order_count_win = self.order_count_win + 1
          new_state = self.update_order_state(state, row)
          self.order_state_portfolio.append(new_state)
          if (state in self.order_cut):
            self.order_profit = self.order_profit + profit / 2
          else:
            self.order_profit = self.order_profit + profit 
          self.order_state.remove(state)
          print('Close buy order of : ' + str(state['order_day_num'])+" PROFIT: " +str(profit))

        elif self.data_order_raw.rsi.loc[self.data_order_raw.day_num == row.day_num].iloc[0] > 80 or (- state['open_oder'] + row.Close ) / r > 10:
          if(state not in self.order_cut):
            profit = (- state['open_oder'] + row.Close ) / r
            self.order_count_cut = self.order_count_cut + 1
            self.order_profit = self.order_profit + profit / 2
            self.order_cut.append(state)
            print('Close OVER buy order of : ' + str(state['order_day_num'])+" PROFIT: " +str(profit))
            

      elif state['order_type'] == 2 and row.day_num > state['order_day_num']:
        r = abs(state['stop_loss'] - state['open_oder'])
        if(row.High > state['stop_loss']):
          print('Lose sell order of : ' + str(state['order_day_num']))
          new_state = self.update_order_state(state, row)
          self.order_state_portfolio.append(new_state)
          self.order_state.remove(state)
          self.order_profit = self.order_profit - 1
          self.order_count_loss = self.order_count_loss + 1
        elif row.Low > self.data_order_raw.MA.loc[self.data_order_raw.day_num == row.day_num].iloc[0] + self.consolidate_thresh * 2:
          profit = (state['open_oder'] - row.Close ) / r
          new_state = self.update_order_state(state, row)
          self.order_state_portfolio.append(new_state)
          print(new_state)
          self.order_count_win = self.order_count_win + 1
          if (state in self.order_cut):
            self.order_profit = self.order_profit + profit / 2
          else:
            self.order_profit = self.order_profit + profit 
          self.order_state.remove(state)
          print('Close sell order of : ' + str(state['order_day_num'])+" PROFIT: " +str(profit))

        elif self.data_order_raw.rsi.loc[self.data_order_raw.day_num == row.day_num].iloc[0] < 20 or (state['open_oder'] - row.Close ) / r > 10:
          if(state not in self.order_cut):
            profit = (state['open_oder'] - row.Close ) / r
            self.order_count_cut = self.order_count_cut + 1
            self.order_profit = self.order_profit + profit / 2
            self.order_cut.append(state)
            print('Close OVER buy order of : ' + str(state['order_day_num'])+" PROFIT: " +str(profit))


def write_state_output(csv_file, dict_data):
  csv_columns = ['state_ticker', 'state_order', 'state_day_start',  'state_day_end', 'state_open', 'state_stop_loss', 'state_close_estimate', 'state_max_r', 'state_MA_r', 'state_RSI_r', 'last_r' , 'state_chain']
  if not os.path.isfile(csv_file):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for row in dict_data:
                writer.writerow(row)
            csvfile.close()
    except IOError:
        print("I/O error")
  else:
    try:
        with open(csv_file, 'a') as csvfile:
            for row in dict_data:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writerow(row)
    except IOError:
        print("I/O error")


def stock_check(ticker, d1, d2, d3 , output_path):
  start = datetime.now() - timedelta(days=d2)
  end = datetime.now() - timedelta(days=d1)
  print(start)
  consolidate_thresh = get_consolidate_value(ticker, day1 = d3, day2 = d2) 
  print('consolidate : ' + str(consolidate_thresh)  + ' spread: ' + str(consolidate_thresh))
  bot_thresh = consolidate_thresh / 2
  data_raw = get_mt5_raw_data_range(ticker, start, end)
  data_raw = get_rsi(data_raw, rsi_thresh= 25)

  print('data len ' + str(len(data_raw)))
  bos_list = []
  boxbos_list = []
  start_point = 0
  window = 1
  i = window
  ci_lookback = 20
  data_trend_raw = data_raw.copy()
  data_order_raw = data_raw.copy()
  data_order_raw['MA'] =  data_order_raw.Close.rolling(50).mean()
  previous_trend = 0
  previous_trend_point = 0
  order_status = Order(data_order_raw, consolidate_thresh)
  previous_trend_temp = 0
  while i < len(data_raw):
    print(str(start_point) + ' forward ' + str(i))
    data  = data_raw.iloc[start_point:i]
    data.reset_index(inplace = True)
    wo = WyckOff(data , ci_thresh = 40 , tail_rate = 0.6 ,imb_rate= 0.3, break_rate=0.3,\
                consolidate_thresh = consolidate_thresh  , ci_lookback = ci_lookback,\
                spread_thresh = consolidate_thresh , min_boxsize = 30)
    data, box = wo.convert_data()
    last_row = data.iloc[-1]
    order_status.check_order_portfolio(last_row)
    order_status.get_order2(last_row)
    bos_data = data.loc[data.bos_imbalance1.notna()]  
    if not bos_data.empty:
      bos = bos_data.iloc[-1]
      print(bos.bos_imbalance1, bos.bos_imbalance2 , (bos.bos_imbalance2 - bos.bos_imbalance1) / consolidate_thresh )
      if((bos.bos_imbalance2 - bos.bos_imbalance1) / consolidate_thresh < 3):
        bos_list.append([bos.day_num, bos.bos_imbalance1, bos.bos_imbalance2 , bos.imbalance1, bos.imbalance2])
        boxbos_list.append(box)
        print(bos.day_num)
    
        check_trend_data = data_trend_raw.iloc[start_point:bos.day_num - 1]
        trend = Trend(check_trend_data  , bos , box , bot_thresh)
        reg = trend.detect_trend()
        current_trend = trend.get_current_trend2(reg[0], reg[2])
        if(previous_trend == current_trend):
          check_trend_data = data_trend_raw.iloc[previous_trend_point:bos.day_num - 1]
          trend = Trend(check_trend_data , bos , box , bot_thresh)
          reg = trend.detect_trend()
          if current_trend == 0:
            current_trend = trend.get_current_trend2(reg[0], reg[2])
          order = trend.get_break_trend(current_trend, reg)

        else:
          previous_trend_temp = previous_trend
          order = trend.get_break_trend(current_trend, reg)
          previous_trend_point = box[0]
          previous_trend = current_trend
        
        
        if(order > 0):
          previous_trend = order
          order_imbalance = [bos['index'],bos.day_num, bos.bos_imbalance1, bos.bos_imbalance2, order , box[0]]
          order_status.order_list.append(order_imbalance)
        else:
          previous_trend = previous_trend_temp

      start_point = bos.day_num + 1 - ci_lookback
      i = start_point + window + ci_lookback if start_point + window < len(data_raw) else len(data_raw) - 1
    else:
      if(len(data) > 200):
        start_point = start_point + 100 - ci_lookback
        i = start_point + 100 + ci_lookback if start_point + 100 < len(data_raw) else len(data_raw) - 1
      else:     
        i = i + 1 
  write_state_output(output_path , order_status.order_state_portfolio)


import os 
import re

# list_stock = []
# for file_name in os.listdir(path):
#   list_stock.append(re.search(r"(.+)\_.+" ,file_name).group(1))
# list_stock = list(dict.fromkeys(list_stock))
# d1 = 90
# d2 = 120
# d3 = 150

# output_path = '/home/tranthong/state_output_new/stateoutput' + str(d1)  + '.csv'
# df = pd.read_csv(output_path)
# if(not df.empty):
#   array1 = df.state_ticker.unique()
#   print(len(list_stock) , len(array1))
#   list_stock_new = list(np.setdiff1d(np.array(list_stock) , array1))
#   print('list stock '+ str(list_stock_new))
# else:
#   list_stock_new = list_stock
# for ticker in list_stock_new:
#   stock_check(ticker, d1, d2, d3 , output_path)
d1 = 90
d2 = 120
d3 = 150
path = '/Users/tranthong/Desktop/stock-app/stock_data_M5/'
stock_check('AMT', d1, d2, d3 , 'a')