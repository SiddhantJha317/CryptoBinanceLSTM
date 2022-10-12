from keras.models import load_model

#Loading the trained model
model = load_model('Bitcoin_model.h5')

#Summarize the model
model.summary()

api_key = ""
secret_key = ""
client = Client(api_key,secret_key)
check = client.get_klines(symbol='BTCUSDT',interval = Client.KLINE_INTERVAL_1MINUTE)

check[499]

price = np.array([float(check[i][4]) for i in range(500)])
price = price.reshape(500,1)

scaler = StandardScaler()
scaler.fit(price[:374])

price = scaler.transform(price)

index = [496,497,498,499]

candles = scaler.transform(np.array([float(check[i][4]) for i in index]).reshape(-1,1))

model_feed = candles.reshape(1,4,1)
model_feed = model_feed.reshape(-1,1)
scaler.inverse_transform(model.predict(model_feed)[0])[0]

index = [496,497,498,499]

candles = scaler.transform(np.array([float(check[i][4]) for i in index]).reshape(-1,1))

model_feed = candles.reshape(1,4,1)
model_feed = model_feed.reshape(-1,1)
scaler.inverse_transform(model.predict(model_feed)[0])[0]

symbol = "BTCUSDT"   #Code of cryptocurrency
quantity = '0.05'    #quantity to trade

order = False
index = [496,497,498,499]

while True:
    price = client.get_recent_trades(symbol=symbol)
    candle = client.get_klines(symbol=symbol,interval=Client.KLINE_INTERVAL_1MINUTE)
    candles = scaler.transform(np.array([float(candle[i][4]) for i in index]).reshape(-1,1))
    model_feed = candles.reshape(1,4,1)
    
    
    if order == False and float(price[len(price)-1]['price']) < float(scaler.inverse_transform(model.predict(model_feed.reshape(-1,1))[0])[0]):
        
        #client.order_market_buy(symbol=symbol,quantity = quantity)
        order = True
        buy_price = client.get_order_book(symbol=symbol)['asks'][0][0]
        print('Buy @Market Price : ',float(buy_price),'Timestamp : ',str(datetime.now()))
        
    elif order == True and float(price[len(price)-1]['price'])-float(buy_price) >=10:
        
        #client.order_market_sell(symbol=symbol, quantity=quantity)
        order = False
        sell_price = client.get_order_book(symbol=symbol)['bids'][0][0]
        print('Sell @Market Price : ',float(sell_price),'Timestamp : ',str(datetime.now()))
