import time
import MetaTrader5 as mt5

print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)

# establish connection to the MetaTrader 5 terminal
if not mt5.initialize(login=113808435, server="Exness-MT5Trial6",password="Tranthong98"):
    print("initialize() failed, error code =",mt5.last_error())
    quit()
import MetaTrader5 as mt5


ea_magic_number = 9986989 # if you want to give every bot a unique identifier

def get_info(symbol):
    '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5symbolinfo_py
    '''
    # get symbol properties
    info=mt5.symbol_info(symbol)
    return info

def open_trade(action, symbol, lot, sl_points, tp_points, deviation):
    '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
    '''
    # prepare the buy request structure
    symbol_info = get_info(symbol)

    if action == 'buy':
        trade_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif action =='sell':
        trade_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    point = mt5.symbol_info(symbol).point

    buy_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "price": price,
        "sl": price - sl_points * point,
        "tp": price + tp_points * point,
        "deviation": deviation,
        "magic": ea_magic_number,
        "comment": "sent by python",
        "type_time": mt5.ORDER_TIME_GTC, # good till cancelled
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    # send a trading request
    result = mt5.order_send(buy_request)        
    return result, buy_request 

def close_trade(action, buy_request, result, deviation):
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


# This is how I would execute the order
result, buy_request = open_trade('buy', 'USDJPY', 0.1, 50, 50, 10)
close_trade('sell', buy_request, result, 10)

orders = mt5.orders_get(symbol = 'IBM')
print(orders)
if orders:
    print('a')
    request1 = {
    "action": mt5.TRADE_ACTION_CLOSE_BY,
    "symbol": 'IBM',
    "volume": 0.01,
    "type": mt5.ORDER_TYPE_SELL,
    "position": orders.order,
    "comment": "python script open",
    "magic": 234000,
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result=mt5.order_send(request1)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("4. order_send failed, retcode={}".format(result.retcode))
        print("   result",result)
else:
    print('b')
# request1 = {
#     "action": mt5.TRADE_ACTION_DEAL,
#     "symbol": 'AUDNZD',
#     "volume": float("{:.2f}".format(0.0132)),
#     "type": mt5.ORDER_TYPE_SELL,
#     "price":  mt5.symbol_info_tick('AUDNZD').bid,
#     "sl": 1.06900,
#     "tp": 1.06000,
#     "deviation": 10,
#     "magic": 234000,
#     "comment": "python script open",
#     "type_time": mt5.ORDER_TIME_GTC,
#     "type_filling": mt5.ORDER_FILLING_RETURN,
# }
# print( mt5.symbol_info_tick('AUDNZD').bid)
# result1 = mt5.order_send(request1)
# if result1.retcode == mt5.TRADE_RETCODE_DONE:
#     print('Send deal realtime request done!')
    