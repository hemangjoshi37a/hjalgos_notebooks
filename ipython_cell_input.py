
quote_list = []

for i in n50df['tradingsymbol']:
    print(i)
    myquote = kite.quote([exchange_type+':'+i])
    quote_list.append(myquote)
quote_list
#     print(n50df['tradingsymbol'])
