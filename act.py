from kiteconnect import KiteConnect
from kiteconnect import KiteTicker
import pandas as pd
import datetime
import pdb

kws = ""
kite = ""

api_k = "w19o0chuo929jxkp"  # api_key
api_s = "gsw8ps17ex7lf3cuji4prfnwb4vlyr4y"  # api_secret


def get_login(api_k="w19o0chuo929jxkp", api_s="gsw8ps17ex7lf3cuji4prfnwb4vlyr4y"):  # log in to zerodha API panel
    global kws, kite
    kite = KiteConnect(api_key=api_k)

    print("[*] Generate access Token : ", kite.login_url())
    request_tkn = input("[*] Enter Your Request Token Here : ")
    data = kite.generate_session(request_tkn, api_secret=api_s)
    kite.set_access_token(data["access_token"])
    kws = KiteTicker(api_k, data["access_token"])
    print(data['access_token'])


get_login(api_k, api_s)
