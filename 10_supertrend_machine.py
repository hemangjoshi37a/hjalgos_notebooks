########## auto login ############
import login
from login import *
import pandas as pd
from loguru import logger
logger.add("./logs/SUPERTREND_logs_{time}.log")
# auto_login()

 ################ NEW VERSION ######################################
fullquant = 50
symbol_ip = 'GBPINR20NOVFUT'  # USDINR20OCTFUT / GBPINR20OCTFUT / EURINR20OCTFUT / JPYINR20OCTFUT
inst_token = 289539  # 690691(USD) / 490755(GBP) / 278019(EUR) / 690435(JPY)
order_type = 'MIS'
exchange_type = kite.EXCHANGE_CDS
offset_quantity = 0
away_from_circuit = 0.50
###################################################################
supertrend_period1 = 9
supertrend_multiplier1=2.0
supertrend_period2 = 14
supertrend_multiplier2=3.0
supertrend_period3 = 10
supertrend_multiplier3=3.0
supertrend_period4 = 20
supertrend_multiplier4=5.0
supertrend_period5 = 27
supertrend_multiplier5=6.0
supertrend_period6 = 35
supertrend_multiplier6=8.0
supertrend_period7 = 40
supertrend_multiplier7=9.0
supertrend_period8 = 50
supertrend_multiplier8=11.0
supertrend_period9 = 60
supertrend_multiplier9=13.0
supertrend_period10 = 75
supertrend_multiplier10=16.0
####################################################################

myquote = kite.quote([exchange_type+':'+symbol_ip])
upper_circuit = myquote[exchange_type+':'+symbol_ip]['upper_circuit_limit']
lower_circuit = myquote[exchange_type+':'+symbol_ip]['lower_circuit_limit']

logger.debug("Parameters : ")
logger.debug("fullquant : " + str(fullquant))
logger.debug("symbol_ip : " + str(symbol_ip))
logger.debug("upper_circuit : " + str(upper_circuit))
logger.debug("lower_circuit : " + str(lower_circuit))
logger.debug("inst_token : " + str(inst_token))
logger.debug("order_type : " + str(order_type))
# logger.debug("supertrend_period1 : " + str(supertrend_period1))
# logger.debug("supertrend_multiplier1 : " + str(supertrend_multiplier1))
# logger.debug("supertrend_period2 : " + str(supertrend_period2))
# logger.debug("supertrend_multiplier2 : " + str(supertrend_multiplier2))
# logger.debug("supertrend_period3 : " + str(supertrend_period3))
# logger.debug("supertrend_multiplier3 : " + str(supertrend_multiplier3))
# logger.debug("supertrend_period4 : " + str(supertrend_period4))
# logger.debug("supertrend_multiplier4 : " + str(supertrend_multiplier4))

######## CHECK QTY #############
myquantity = 0
def getquant():
    logger.debug("getquant : ")
    global myquantity, order_type, symbol_ip
    allpos = kite.positions()['net']
    for i in range(len(allpos)):
        if (allpos[i]['tradingsymbol'] == symbol_ip
                and allpos[i]['product'] == order_type):
            myquantity = allpos[i]['quantity']
            print('My Quantity : ' + str(allpos[i]['quantity']))
    logger.debug("My Quantity : " + str(myquantity))
    return myquantity
# getquant()

############### tesating block SUPER-TREND ########################
import datetime
import pandas_ta as ta
candle_interval='minute'
def calc_super():
    old_lst = []
    interval = candle_interval
    todaydt = datetime.date.today()
    hud_ago = todaydt - datetime.timedelta(days=6)
    to_date = datetime.date.isoformat(todaydt)
    from_date = datetime.date.isoformat(hud_ago)
    for i2 in range(1):
        new_lst = kite.historical_data(inst_token,
                                       from_date,
                                       to_date,
                                       interval,
                                       continuous=False)
        old_lst = new_lst + old_lst
        todaydt = todaydt - datetime.timedelta(days=7)
        hud_ago = hud_ago - datetime.timedelta(days=7)
        to_date = datetime.date.isoformat(todaydt)
        from_date = datetime.date.isoformat(hud_ago)
    #         print(len(old_lst))

    mydf = pd.DataFrame(old_lst)
    mydata1 = mydf.ta.supertrend(multiplier=supertrend_multiplier1, length=supertrend_period1)
    mydata2 = mydf.ta.supertrend(multiplier=supertrend_multiplier2, length=supertrend_period2)
    mydata3 = mydf.ta.supertrend(multiplier=supertrend_multiplier3, length=supertrend_period3)
    mydata4 = mydf.ta.supertrend(multiplier=supertrend_multiplier4, length=supertrend_period4)
    mydata5 = mydf.ta.supertrend(multiplier=supertrend_multiplier5, length=supertrend_period5)
    mydata6 = mydf.ta.supertrend(multiplier=supertrend_multiplier6, length=supertrend_period6)
    mydata7 = mydf.ta.supertrend(multiplier=supertrend_multiplier7, length=supertrend_period7)
    mydata8 = mydf.ta.supertrend(multiplier=supertrend_multiplier8, length=supertrend_period8)
    mydata9 = mydf.ta.supertrend(multiplier=supertrend_multiplier9, length=supertrend_period9)
    mydata10 = mydf.ta.supertrend(multiplier=supertrend_multiplier10, length=supertrend_period10)

    st1 = mydata1[-1:]['SUPERTd_9_2.0']
    st2 = mydata2[-1:]['SUPERTd_14_3.0']
    st3 = mydata3[-1:]['SUPERTd_10_3.0']
    st4 = mydata4[-1:]['SUPERTd_20_5.0']
    st5 = mydata5[-1:]['SUPERTd_27_6.0']
    st6 = mydata6[-1:]['SUPERTd_35_8.0']
    st7 = mydata7[-1:]['SUPERTd_40_9.0']
    st8 = mydata8[-1:]['SUPERTd_50_11.0']
    st9 = mydata9[-1:]['SUPERTd_60_13.0']
    st10 = mydata10[-1:]['SUPERTd_75_16.0']
    
    st_total = st1 + st2 + st3 + st4 + st5 + st6 + st7 + st8 + st9 + st10
    
    logger.debug("SUPERTd_9_2.0 : " + str(st1))
    logger.debug("SUPERTd_14_3.0 : " + str(st2))
    logger.debug("SUPERTd_10_3.0 : " + str(st3))
    logger.debug("SUPERTd_20_5.0 : " + str(st4))
    logger.debug("SUPERTd_27_6.0 : " + str(st5))
    logger.debug("SUPERTd_35_8.0 : " + str(st6))
    logger.debug("SUPERTd_40_9.0 : " + str(st7))
    logger.debug("SUPERTd_50_11.0 : " + str(st8))
    logger.debug("SUPERTd_60_13.0 : " + str(st9))
    logger.debug("SUPERTd_75_16.0 : " + str(st10))
    
    logger.debug("st_total : " + str(st_total))
    print('st_total : ' + str(st_total))
    return st_total
    
calc_super()

########### place order ###########
stopbuy = False
stopsell = False

placing_order = False
is_up = False
is_down = False

def placeneworder(quantdiff, price_ip):
    '''placeneworder(quantdiff,price_ip)'''
    logger.debug('placeneworder @ (' + str(quantdiff) + ',' + str(price_ip) +')')
    global stopbuy, stopsell, order_type, symbol_ip, exchange_type,placing_order,is_up,is_down
#     myquantity=getquant()
    if (quantdiff > 0 and stopbuy == False ):
        if(not (is_up == True and is_down == False)):
            try:
                placing_order = True
                order_id = kite.place_order(
                    tradingsymbol=symbol_ip,
                    exchange=exchange_type,
                    transaction_type=kite.TRANSACTION_TYPE_BUY,
                    quantity=abs(quantdiff),
                    price=price_ip,
                    order_type=kite.ORDER_TYPE_MARKET
                    if price_ip == 0 else kite.ORDER_TYPE_LIMIT,
                    variety=kite.VARIETY_REGULAR,
                    product=kite.PRODUCT_MIS
                    if order_type == 'MIS' else kite.PRODUCT_NRML)
                stopsell = False
                logger.debug('Order Successfully Placed @ ' + str(order_type) +' ' + str(quantdiff) + ' ' + str(price_ip))
                is_up = True
                is_down = False

                getquant()
                placing_order = False
                return True
            except Exception as e:
                stopbuy = True
                print(e)
                logger.debug('Order Rejected For @ ' + str(order_type) + ' ' +str(quantdiff) + ' ' + str(price_ip))
                getquant()
                return False
    if (quantdiff < 0 and stopsell == False):
        if(not (is_up == False and is_down == True)):
            try:
                placing_order = True
                order_id = kite.place_order(
                    tradingsymbol=symbol_ip,
                    exchange=exchange_type,
                    transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=abs(quantdiff),
                    price=price_ip,
                    order_type=kite.ORDER_TYPE_MARKET
                    if price_ip == 0 else kite.ORDER_TYPE_LIMIT,
                    variety=kite.VARIETY_REGULAR,
                    product=kite.PRODUCT_MIS
                    if order_type == 'MIS' else kite.PRODUCT_NRML)
                stopbuy = False
                logger.debug('Order Successfully Placed @ ' + str(order_type) + ' ' + str(quantdiff) + ' ' + str(price_ip))
                is_up = False
                is_down = True
                
                getquant()
                placing_order = False
                return True
            except Exception as e:
                stopsell = True
                print(e)
                logger.debug('Order Rejected For @ ' + str(order_type) + ' ' + str(quantdiff) + ' ' + str(price_ip))
                getquant()
                return False


def ckqnt(orderprice, orderquant):
    logger.debug('ckqnt @ (' + str(orderquant))
    global myquantity,placing_order
    quantdiff = orderquant - myquantity
    logger.debug(str(quantdiff) + ' = ' +str(orderquant) + '-' + str(myquantity))
    is_done = False
    if(placing_order==False):
        is_done = placeneworder(quantdiff, orderprice)

    return is_done

last_super_val= 0
update_status_counter = 0
last_ten = 0
def calc_and_update():
    global last_super_val,update_status_counter,last_ten

    logger.debug('calc_and_update : ')
    super_val=0
    try:
        super_val = calc_super()
    except:
        super_val=NA
        print('calc_rsi failed')
    super_val = int(super_val)
    last_super_val = int(last_super_val)
    update_status_counter = int(update_status_counter)
    if((last_super_val!=super_val) and (super_val==10 or super_val==-10) 
       and update_status_counter>0 and (last_ten!=super_val)):
        update_status(super_val)

    last_super_val = super_val
    if(super_val==10 or super_val==-10):
        last_ten = super_val
    update_status_counter = update_status_counter+1
    
    ################ update status #################
getquant()

def update_status(super_val):
    global fullquant,myquantity
    myquantity = getquant()
    super_val = int(super_val)
    myquantity = int(myquantity)
    fullquant = int(fullquant)
    if(super_val == 10):
        logger.debug('super_val = ' + str(super_val))
        is_done = ckqnt(0, fullquant+offset_quantity)
        if(is_done):
            logger.debug("Order Executed @ "+ 'super_val = ' + str(super_val)+' for quant '+str(fullquant))
        else:
            logger.debug("Order Rejected @ "+ 'super_val = ' + str(super_val)+' for quant '+str(fullquant))

    if(super_val == -10):
        logger.debug('super_val = ' + str(super_val))
        is_done = ckqnt(0, -fullquant+offset_quantity)
        if(is_done):
            logger.debug("Order Executed @ "+ 'super_val = ' + str(super_val)+' for quant '+str(fullquant))
        else:
            logger.debug("Order Rejected @ "+ 'super_val = ' + str(super_val)+' for quant '+str(fullquant))

# update_status()

############# start trading ############
import threading
import logging
from kiteconnect import KiteTicker
logging.basicConfig(level=logging.DEBUG)
# kws = KiteTicker("w19o0chuo929jxkp", "eA5B5OJQNOtZ0OYihkBmYw7Ke3B9pmCC")
last_min = 0
def on_ticks(ws, ticks):
    global pivot_price, moving_pivot_on, limits, away_from_circuit, lower_circuit, upper_circuit,last_min
    this_min = datetime.datetime.now().minute
        
    if(((upper_circuit-away_from_circuit)>ticks[0]['last_price']) 
       and ((lower_circuit+away_from_circuit)<ticks[0]['last_price'])):
        if (last_min!=this_min):
            last_min = this_min
            print('in the thread : ')
            x = threading.Thread(target=calc_and_update, args=())
            x.start()

        print('=====================================')
        print('LTP : ' + str(ticks[0]['last_price']))

    else:
        ckqnt(0,0)
        if((upper_circuit-away_from_circuit)<=ticks[0]['last_price']):
            logger.debug('Upper Circuit Hit @ '+str(ticks[0]['last_price']))
        if((lower_circuit+away_from_circuit)>=ticks[0]['last_price']):
            logger.debug('Lower Circuit Hit @ '+str(ticks[0]['last_price']))

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

def on_order_update(ws, data):
    logger.debug('on_order_update : ')
    global baseprice, symbol_ip, myquantity, init_qty, current_point
    print("order update: ", data)

kws.on_error = on_error
kws.on_noreconnect = on_noreconnect
kws.on_reconnect = on_reconnect
kws.on_order_update = on_order_update

kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

try:
    kws.connect()
except Exception as e:
    logger.debug('Auto relogging in...')
    auto_login()
    kws.stop()
    kws.close()
    kws.connect()

# logger.debug("supertrend_period5 : " + str(supertrend_period5))
# logger.debug("supertrend_multiplier5 : " + str(supertrend_multiplier5))
