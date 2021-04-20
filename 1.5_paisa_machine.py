########## auto login ############
import login
from login import *
import pandas as pd
from loguru import logger
logger.add("./logs/SUPERTREND_logs_{time}.log")
# auto_login()

################ NEW VERSION ######################################
#########################
# Update this part only #
#########################

one_dollar_margin = 450     # (USD-MIS=984/NRML=1967)(GBP-MIS=1948/NRML=3895)(EUR-MIS=1313/NRML=2625)(JPY-MIS=1333/NRML=2667)
per_order_quant = 20
initial_difference = 0.0000

order_type = 'MIS'           # MIS / NRML
symbol_ip = 'USDINR20OCTFUT' # USDINR20OCTFUT / GBPINR20OCTFUT / EURINR20OCTFUT / JPYINR20OCTFUT
inst_token = 690691          # 690691(USD) / 490755(GBP) / 278019(EUR) / 690435(JPY)

moving_pivot_on = False      # True / False

stoploss_on = False          # True / False

hl_range = 0             # (Default=0.00)(USD=0.23)(GBP=?)(EUR=?)(JPY=?)
days_ago_data = 1            # Today Date - day_ago_data
#############################################################################################################################

import datetime
interval='day'
todaydt=datetime.date.today()
hud_ago=todaydt-datetime.timedelta(days=10)
to_date=datetime.date.isoformat(todaydt)

from_date=datetime.date.isoformat(hud_ago)
data = kite.historical_data(inst_token, from_date, to_date, interval, continuous=False)

import pandas as pd
mydata = pd.DataFrame(data)
from datetime import date
today = date.today()

print('Today Date : '+str(today))
yest_date =  mydata[-days_ago_data:][['date']].to_numpy()[0][0]
last_close =  mydata[-days_ago_data:][['close']].to_numpy()[0][0]
last_high =  mydata[-days_ago_data:][['high']].to_numpy()[0][0]
last_low =  mydata[-days_ago_data:][['low']].to_numpy()[0][0]
hl_range = last_high - last_low

print('Yestreday Date : '+str(yest_date))
print('Last Close : '+ str(last_close))
print('Last High : '+ str(last_high))
print('Last Low : '+ str(last_low))
print('High Low Range : '+ str(round(hl_range,4)))

mymargin = round(kite.margins(segment = 'equity')['available']['live_balance'],4)
#mymargin = 185777.23

pivot_price = last_close

fullquantity = int(mymargin / one_dollar_margin)

num_of_steps = int(fullquantity / per_order_quant)

difference = round(hl_range / num_of_steps,4)

difference = round(0.0025 * round(difference /0.0025),4)

last_buy_price = round(pivot_price -  difference*num_of_steps,4)
last_sell_price = round(pivot_price +  difference*num_of_steps,4)

print('My Margin : '+str(mymargin))
print('Pivot Price : ' + str(pivot_price))
print('Full Quantity : '+str(fullquantity))
print('Number of Steps : '+str(num_of_steps))
print('Price Difference Between One Trade : ' + str(difference))
print('Quantity Difference Between One Trade : ' + str(per_order_quant))
print('Last Buy Pivot Price : '+str(last_buy_price))
print('Last Sell Pivot Price : '+str(last_sell_price))
print('Stoploss On ? : '+str(stoploss_on))
print('Moving Pivot On ? : '+str(moving_pivot_on))
base=0.0025
max_no_of_order = int(fullquantity / per_order_quant)
total_parts = 2*max_no_of_order

print()
print('pivot_price : '+str(pivot_price))
print()
upper_limit = 0.12
lower_limit = 0.12
def calc_pivots():
    global upper_limit, lower_limit
    for i in range(1,max_no_of_order+1):
        globals()['p'+str(i)+'b'] = pivot_price - (initial_difference + i* difference)
        globals()['p'+str(i)+'b'] = round(base * round(globals()['p'+str(i)+'b'] /base),4)
        print('p'+str(i)+'b : '+str(globals()['p'+str(i)+'b']))

    print()
    for i in range(1,max_no_of_order+1):
        globals()['p'+str(i)+'s'] = pivot_price + (initial_difference + i* difference)
        globals()['p'+str(i)+'s'] = round(base * round(globals()['p'+str(i)+'s'] /base),4)
        print('p'+str(i)+'s : '+str(globals()['p'+str(i)+'s']))

    print()
    for i in range(1,max_no_of_order+1):
        globals()['p'+str(i)+'q'] = per_order_quant*i
        print('p'+str(i)+'q : '+str(globals()['p'+str(i)+'q']))
        
    print()
    upper_limit = round(((globals()['p'+str(max_no_of_order)+'s']) + hl_range*0.236),4)
    lower_limit = round(((globals()['p'+str(max_no_of_order)+'b']) - hl_range*0.236),4)
    print('S Limit : '+str(upper_limit))
    print('B Limit : '+str(lower_limit))

calc_pivots()

#####################
# NORMAL RUN-2 of 3 #
#####################

myquantity = 0
def getquant():
    global myquantity
    global order_type
    global symbol_ip
    allpos = kite.positions()['net']
    for i in range(len(allpos)):
        if(allpos[i]['tradingsymbol']==symbol_ip and  allpos[i]['product']== order_type):
            myquantity = allpos[i]['quantity']
            print('My Quantity : ' + str(allpos[i]['quantity']))

getquant()

stopbuy = False
stopsell = False

def placeneworder(quantdiff):
    global stopbuy
    global stopsell
    global order_type
    global symbol_ip
    if(quantdiff>0 and stopbuy == False):
        try:
                order_id= kite.place_order(tradingsymbol=symbol_ip,
                            exchange=kite.EXCHANGE_CDS,
                            transaction_type=kite.TRANSACTION_TYPE_BUY,
                            quantity=abs(quantdiff),
                            order_type=kite.ORDER_TYPE_MARKET,
                            variety = kite.VARIETY_REGULAR,
                            product=kite.PRODUCT_MIS if order_type=='MIS' else kite.PRODUCT_NRML)
                getquant()
                stopsell = False
        except Exception as e:
            stopbuy = False
            print(e)
            getquant()
    if(quantdiff<0 and stopsell == False):
        try:
                order_id= kite.place_order(tradingsymbol=symbol_ip,
                            exchange=kite.EXCHANGE_CDS,
                            transaction_type=kite.TRANSACTION_TYPE_SELL,
                            quantity=abs(quantdiff),
                            order_type=kite.ORDER_TYPE_MARKET,
                            variety = kite.VARIETY_REGULAR,
                            product=kite.PRODUCT_MIS if order_type=='MIS' else kite.PRODUCT_NRML)
                getquant()
                stopbuy = False
        except Exception as e:
            stopsell = True
            print(e)
            getquant()


def ckqnt (orderprice,orderquant):
    global myquantity
    quantdiff = orderquant - myquantity
    placeneworder(quantdiff)
    return myquantity


last_price = pivot_price

def checkpivot(ltps):
    global last_price, upper_limit, lower_limit
    trade_quant = 0
    
    if((ltps > upper_limit or ltps < lower_limit) and stoploss_on ):
        ckqnt(pivot_price,0)
    
    elif ((ltps>=(pivot_price-0.0025)) and (ltps<=(pivot_price+0.0025))):
        ckqnt(pivot_price,0)
        print('in pivot range')
        
    elif(upper_limit>ltps>(globals()['p'+str(max_no_of_order)+'s'])):
        ckqnt(globals()['p'+str(max_no_of_order)+'s'],-globals()['p'+str(max_no_of_order)+'q'])
        
    elif(lower_limit<ltps<(globals()['p'+str(max_no_of_order)+'b'])):
        ckqnt(globals()['p'+str(max_no_of_order)+'b'],globals()['p'+str(max_no_of_order)+'q'])
        
    else:
        for i in range(1,int(total_parts/2)+1):

            if(ltps<pivot_price):
                trade_quant = globals()['p'+str(i)+'q']
            elif(ltps>pivot_price):
                trade_quant = -globals()['p'+str(i)+'q']

            if ((ltps == globals()['p'+str(i)+'b'] or ltps == globals()['p'+str(i)+'s']) and ltps < last_price):
                ckqnt(globals()['p'+str(i)+'b'],trade_quant)
                print('==========' +'p'+str(i)+'b @ ' +str(ltps) + '==============' )
                last_price=ltps

            if ((ltps == globals()['p'+str(i)+'b'] or ltps == globals()['p'+str(i)+'s']) and ltps > last_price):
                ckqnt(globals()['p'+str(i)+'s'],trade_quant)
                print('==========' +'p'+str(i)+'s @ ' +str(ltps) + '==============' )
                last_price=ltps
                
#####################
# NORMAL RUN-3 of 3 #
#####################
#################
# START TRADING #
#################

import logging
from kiteconnect import KiteTicker
logging.basicConfig(level=logging.DEBUG)
# kws = KiteTicker("w19o0chuo929jxkp", "eA5B5OJQNOtZ0OYihkBmYw7Ke3B9pmCC")

def on_ticks(ws, ticks):
    global pivot_price, moving_pivot_on
    checkpivot(ticks[0]['last_price'])

    if(moving_pivot_on):
        pivot_price =round(base * round(ticks[0]['average_price']/base),4) # Moving Trade Average Line : 1
        calc_pivots();                                                     # Moving Trade Average Line : 2
        
    #print(hjhj)
    #print('===========================')
    #print()
    #print(type(ticks[0]['last_price']))
    print('==========================')
    print( 'LTP : ' + str(ticks[0]['last_price']))
    #print('==========================')
    #print()

def on_connect(ws, response):
    global inst_token
    ws.subscribe([inst_token])
    ws.set_mode(ws.MODE_FULL, [inst_token])

def on_close(ws, code, reason):
    ws.stop()
    
def on_error(ws, code, reason):
    logging.error("closed connection on error: {} {}".format(code, reason))

def on_noreconnect(ws):
    logging.error("Reconnecting the websocket failed")

def on_reconnect(ws, attempt_count):
    logging.debug("Reconnecting the websocket: {}".format(attempt_count))

#def on_order_update(ws, data):
#    print("order update: ", data)

kws.on_error = on_error
kws.on_noreconnect = on_noreconnect
kws.on_reconnect = on_reconnect
#kws.on_order_update = on_order_update

kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

kws.connect()
