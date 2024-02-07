# %%
####################### AUTO LOGIN #####################
import os
os.system('clear')
print('Starting Strategy...')

import login
from login import *
import pandas as pd
import joblib
import ta
import numpy as np
# pd.set_option('plotting.backend', 'pandas_bokeh')
# import pandas_bokeh
# pandas_bokeh.output_notebook()
import h2o
hh=h2o.init(ip='127.0.0.1',port='12349')

# %%
order_type = 'MIS' 
# full_qty = 10
interval='5minute'
daysdiff = 5
bkg = 0.0
isneg = 1

# %%
nfifty_tokens_list = [3861249,60417,1510401,4267265,4268801,81153,2714625,134657,140033,
                      177665,5215745,2800641,225537,232961,1207553,315393,1850625,340481,
                      341249,119553,345089,348929,356865,1270529,1346049,408065,415745,424961,
                      3001089,492033,2939649,519937,2815745,4598529,2977281,633601,3834113,
                      738561,5582849,779521,794369,857857,884737,895745,2953217,3465729,
                      897537,2952193,2889473,969473]
most_active_by_volume = [884737,3861249,3677697,54273,3518721,1207553,2730497,3014145,2863105,4343041]

# %%
combined_df = pd.DataFrame()


todaydt=datetime.date.today()
hud_ago=todaydt-datetime.timedelta(days=daysdiff)
to_date=datetime.date.isoformat(todaydt)
from_date=datetime.date.isoformat(hud_ago)

saved_model = h2o.load_model('./model1')
saved_model2 = h2o.load_model('./model2')
saved_model3 = h2o.load_model('./model3')

counter1 = 1

for iii in nfifty_tokens_list:
    print('===================================================')
    print('Analyzing Token : '+str(iii)+ ' '+str(counter1)+'/50')
    print('===================================================')
    print()
    counter1 = counter1+1
    old_lst = kite.historical_data(iii, from_date, to_date, interval,continuous=False)
    mydf = pd.DataFrame(old_lst)
    mydf['next_close'] = mydf['close'].shift(periods=-1)
    mydf = mydf.fillna(0)
    mydf = ta.add_all_ta_features(mydf, open="open", high="high", low="low", close="close", volume="volume");
    mydf = mydf.fillna(0)[150:]
    
    last_close = mydf.iloc[[-1]]['close'].to_numpy()[0]

    hf = h2o.H2OFrame(mydf)
    prd = saved_model.predict(hf);
    prd2 = saved_model2.predict(hf);

    df123 = prd2.concat([prd,hf])
    prd3 = saved_model3.predict(df123);
    df1234 = prd3.concat([df123])
    os.system('clear')
    data_as_df = h2o.as_list(df1234)

    sel_data = data_as_df[['predict','predict0','predict0.1','date','close','next_close']]

    last_isbullbear = 0

    total_gained_pt = []
    total_gained_pt_total = 0
    total_gained_pt_lst = []

    for ind in sel_data.index:

        is_bullbear = 0
        if(ind<(len(sel_data.index)-1)):
            if(sel_data['close'][ind]<sel_data['close'][ind+1]):
                is_bullbear=1
            elif(sel_data['close'][ind]>sel_data['close'][ind+1]):
                is_bullbear=-1

        pred_true = -1
        if(sel_data['predict'][ind]==is_bullbear):
            pred_true = 1

        gained_pt=0
        if(ind<(len(sel_data.index)-1)):
            gained_pt = pred_true*abs(sel_data['close'][ind]-sel_data['close'][ind+1])

        gained_pt = gained_pt*isneg

        if(ind>0 and (last_isbullbear!=is_bullbear)):
            gained_pt = gained_pt+bkg

        total_gained_pt.append(gained_pt)
        total_gained_pt_total = total_gained_pt_total + gained_pt

        total_gained_pt_lst.append(total_gained_pt_total)

        last_isbullbear = is_bullbear

    globals()['df'+str(iii)] = pd.DataFrame({iii:total_gained_pt_lst})
    
    globals()['df'+str(iii)] = (globals()['df'+str(iii)]/last_close)*100

    combined_df = pd.concat([combined_df,globals()['df'+str(iii)]], axis=1)

#     current_bullbear = data_as_df.iloc[[-1]]['predict'].to_numpy()[0]
#     print('current_bullbear : '+str(current_bullbear))

# combined_df.plot()
os.system('clear')
mynp = combined_df.iloc[[-1]].to_numpy()[0]
mynp = np.nan_to_num(mynp)
print('Nifty50 Mean : '+str(np.mean(mynp)))
print('Nifty50 Sum : '+str(np.sum(mynp)))

insttkn = 0
last_mean = 0

for ij in range(len(nfifty_tokens_list)):
    this_mean = combined_df[nfifty_tokens_list[ij]].to_numpy().mean()
    
    if(this_mean>last_mean):
        insttkn= nfifty_tokens_list[ij]
        last_mean = this_mean
        
n50 = pd.read_csv('n50.csv')
symbol_ip = n50.loc[n50['instrument_token'] == insttkn, 'tradingsymbol'].to_numpy()[0]

print('Success Rate Mean : '+str(last_mean))
print('Selected Token : '+str(insttkn))
print('Selected Symbol : '+str(symbol_ip))

# %%
# one stock predict and place order

todaydt=datetime.date.today()
hud_ago=todaydt-datetime.timedelta(days=daysdiff)
to_date=datetime.date.isoformat(todaydt)
from_date=datetime.date.isoformat(hud_ago)

def one_stk_fun():

    old_lst = kite.historical_data(insttkn, from_date, to_date, interval,continuous=False)

    mydf = pd.DataFrame(old_lst)
    mydf['next_close'] = mydf['close'].shift(periods=-1)
    # mydf = ta.utils.dropna(mydf)
    mydf = mydf.fillna(0)
    mydf = ta.add_all_ta_features(mydf, open="open", high="high", low="low", close="close", volume="volume")
    mydf = mydf.fillna(0)[150:]
    
    hf = h2o.H2OFrame(mydf)
    prd = saved_model.predict(hf)
    prd2 = saved_model2.predict(hf)

    df123 = prd2.concat([prd,hf])
    prd3 = saved_model3.predict(df123)
    df1234 = prd3.concat([df123])
    os.system('clear')
    data_as_df = h2o.as_list(df1234)

    # data_as_df[['predict','predict0','predict0.1','date','close','next_close']].to_csv('./mycsv.csv')
    sel_data = data_as_df[['predict','predict0','predict0.1','date','close','next_close']]

    last_isbullbear = 0

    total_gained_pt = []
    total_gained_pt_total = 0
    total_gained_pt_lst = []


    for ind in sel_data.index:

        is_bullbear = 0
        if(ind<(len(sel_data.index)-1)):
            if(sel_data['close'][ind]<sel_data['close'][ind+1]):
                is_bullbear=1
            elif(sel_data['close'][ind]>sel_data['close'][ind+1]):
                is_bullbear=-1

        pred_true = -1
        if(sel_data['predict'][ind]==is_bullbear):
            pred_true = 1

        gained_pt=0
        if(ind<(len(sel_data.index)-1)):
            gained_pt = pred_true*abs(sel_data['close'][ind]-sel_data['close'][ind+1])

        gained_pt = gained_pt*isneg

        if(ind>0 and (last_isbullbear!=is_bullbear)):
            gained_pt = gained_pt+bkg

        total_gained_pt.append(gained_pt)
        total_gained_pt_total = total_gained_pt_total + gained_pt

        total_gained_pt_lst.append(total_gained_pt_total)

        last_isbullbear = is_bullbear

    gdptdf = pd.DataFrame({'total_gained_pt':total_gained_pt,'total_gained_pt_lst':total_gained_pt_lst})
    # gdptdf.plot()

    current_bullbear = data_as_df.iloc[[-1]]['predict'].to_numpy()[0]
    print('=============================')
    print('current_bullbear : '+str(current_bullbear))
    print(mydf['date'][-1:].to_numpy()[0])
    print('=============================')
    
    ckqnt(0,current_bullbear*full_qty)
    

# %%
# base functions : getquant, placeneworder, ckqnt

myquantity = 0
def getquant():
    global myquantity,order_type,symbol_ip
    allpos = kite.positions()['net']
    for i in range(len(allpos)):
        if(allpos[i]['tradingsymbol']==symbol_ip and  allpos[i]['product']== order_type):
            myquantity = allpos[i]['quantity']
            print('My Quantity : ' + str(allpos[i]['quantity']))

getquant()

stop_trading = False
stopbuy = False
stopsell = False
ex_type = kite.EXCHANGE_NSE

def placeneworder(quantdiff,price_ip):
    global stopbuy,stop_trading,stopsell,order_type,symbol_ip,ex_type
    if(quantdiff>0 and stopbuy == False and stop_trading==False):
        try:
            stop_trading = True
            order_id= kite.place_order(tradingsymbol=symbol_ip,
                        exchange=ex_type,
                        transaction_type=kite.TRANSACTION_TYPE_BUY,
                        quantity=abs(quantdiff),
#                             price=price_ip,
                        order_type=kite.ORDER_TYPE_MARKET,
                        variety = kite.VARIETY_REGULAR,
                        product=kite.PRODUCT_MIS if order_type=='MIS' else kite.PRODUCT_NRML)
            getquant()
            stopsell = False
            stop_trading = False
        except Exception as e:
            stopbuy = False
            print(e)
            getquant()
    if(quantdiff<0 and stopsell == False and stop_trading==False):
        try:
            stop_trading = True
            order_id= kite.place_order(tradingsymbol=symbol_ip,
                        exchange=ex_type,
                        transaction_type=kite.TRANSACTION_TYPE_SELL,
                        quantity=abs(quantdiff),
#                             price=price_ip,
                        order_type=kite.ORDER_TYPE_MARKET,
                        variety = kite.VARIETY_REGULAR,
                        product=kite.PRODUCT_MIS if order_type=='MIS' else kite.PRODUCT_NRML)
            getquant()
            stopbuy = False
            stop_trading = False
        except Exception as e:
            stopsell = True
            print(e)
            getquant()


def ckqnt (orderprice,orderquant):
    global myquantity
    quantdiff = orderquant - myquantity
    placeneworder(quantdiff,orderprice)
    return myquantity

def print_params():
    print('=============================')
    print('Nifty50 Mean : '+str(np.mean(mynp)))
    print('Nifty50 Sum : '+str(np.sum(mynp)))
    print('Success Rate Mean : '+str(last_mean))
    print('Selected Token : '+str(insttkn))
    print('Selected Symbol : '+str(symbol_ip))
    print('=============================')

# %%
# 5 minute time loop with predict/order Fn
import time
last_min = 0
while(1):
    minute=datetime.datetime.now().minute
    if(minute%5==0 and(minute!=last_min)):
        os.system('clear')
        one_stk_fun()
        print_params()
        last_min=minute
        
    time.sleep(0.1)

# %%
# # Positive Mean TKN list : great_list

# great_list = []

# for jj in combined_df.keys():
#     npv = combined_df[jj].to_numpy()
#     meanip = np.mean(npv)
#     if(meanip>0.75):
#         great_list.append(jj)
    
# great_list

# %%
# import py_compile
# py_compile.compile('./mystrat.py')

# %%
# ipynb-py-convert stock_predict_pipe_11_1_2021.ipynb mystrat.py
# python3 -m compileall mystrat.py 

# %%
