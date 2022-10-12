
# CryptoBinanceLSTM

This is a LSTM model used to pedict and analyse $BTC data ,from the binance API and make (covert ai) trades , although API live trades won't be executed.

# Theory
Data is extracted from python-binance API , which is then turned into a numoy array ,broken apart in test and train sets which is then passed through a LSTM and then dense layer , with linear layers to train the model, then create a rest API key to execute trades through model , with a predict lower stratergy.

## LSTM
LSTM (Long Term Short Memory), is a more sophisticated verison of RNNs(Recurrent Neural Networks).
Human thinking is  a linear learning mechanism , wherein people construct their thoughs from scratch each second.s you read this essay, you understand each word based on your understanding of previous words. You don’t throw everything away and start thinking from scratch again. Your thoughts have persistence.

Neural Networks can't do this since they can't store their immediate information storage into specific cells due to traditional ANN structures.
![LSTM3-C-line](https://user-images.githubusercontent.com/111745916/195310323-ace5fa2a-322e-4b7e-9d4d-f3a201814ae6.png)
Recurrent neural networks address this issue. They are networks with loops in them, allowing information to persist.

The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at ht−1 and xt, and outputs a number between 0 and 1 for each number in the cell state Ct−1. A 1 represents “completely keep this” while a 0 represents “completely get rid of this.”
These loops make recurrent neural networks seem kind of mysterious. However, if you think a bit more, it turns out that they aren’t all that different than a normal neural network. A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens if we unroll the loop:
![LSTM3-focus-f](https://user-images.githubusercontent.com/111745916/195311083-02bf232a-a78e-4682-b51e-7c42b1bbd813.png)
First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, C~t, that could be added to the state. In the next step, we’ll combine these two to create an update to the state.

![LSTM3-focus-i](https://user-images.githubusercontent.com/111745916/195311317-1a9f1e3d-fed6-4f65-af01-47a57452256b.png)
Using the tanh function to apply the local transformation on entrant variable from the sigmoid function.
![LSTM3-focus-C](https://user-images.githubusercontent.com/111745916/195311342-57df5b57-afc8-412d-9522-16dce732441a.png)
We multiply the old state by ft, forgetting the things we decided to forget earlier. Then we add it∗C~t. This is the new candidate values, scaled by how much we decided to update each state value.
![LSTM3-focus-o](https://user-images.githubusercontent.com/111745916/195311360-ae0bae42-bb7f-4ee6-8ca0-291e56454743.png)
Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.
#### Completed LSTM Model
![LSTM3-chain](https://user-images.githubusercontent.com/111745916/195311523-ec257a8e-c9a9-4925-824f-7587020725e2.png)

# Code 
Code is split into two files , the training file and implementation file.
## Model.py
Importing Libraries for the model
```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from datetime import datetime
from sklearn.metrics import mean_squared_error

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from binance.client import Client
```

Setting API keys for rest extraction to the data

```
api_key = ""
secret_key="t"

client = Client(api_key,secret_key)

candles = client.get_klines(symbol='BTCUSDT',interval = Client.KLINE_INTERVAL_1MINUTE);
```

Check Candles data

```
candles[11]
-----------------------------------------
[1665529440000,
 '19057.53000000',
 '19060.35000000',
 '19052.26000000',
 '19053.55000000',
 '74.45922000',
 1665529499999,
 '1418872.64888260',
 1673,
 '30.84180000',
 '587713.08511030',
 '0']
```
Fetching array , plotting the time step file
```
#Fetching closing prices from candlesticks data

price = np.array([float(candles[i][4]) for i in range(len(candles))])

time = np.array([int(candles[i][0]) for i in range(len(candles))])

#Converting time to HH:MM:SS format
t = np.array([datetime.fromtimestamp(time[i]/1000).strftime('%H:%M:%S') for i in range(len(candles))])

plt.xlabel("Time Step")
plt.ylabel("Bitcoin Price $")
plt.plot(price)
------------------------------------
```
![download](https://user-images.githubusercontent.com/111745916/195309842-aa79e2f9-5806-4d5e-8008-4aca83685f08.png)

Constructing the Dataset
```
timeframe = pd.DataFrame({'Time':t,'Price $BTC':price})
timeframe
--------------------------------------------------------
Time	Price $BTC
0	22:53:00	19050.12
1	22:54:00	19049.21
2	22:55:00	19051.13
3	22:56:00	19052.56
4	22:57:00	19052.73
...	...	...
495	07:08:00	19146.04
496	07:09:00	19147.60
497	07:10:00	19143.02
498	07:11:00	19134.78
499	07:12:00	19135.06
```
Scaling the Data with Standard Scaler
```
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
price = price.reshape(-1,1)

sc.fit(price)
price = sc.transform(price)

price = price.reshape(500,1)
```
Setting the Data in four columns
```
df = pd.DataFrame(price.reshape(100,5),columns = ['First','Second','Third','Fourth','Target'])
df.head(5)
----------------------------------------------------------------------------------------------
First	Second	Third	Fourth	Target
0	-0.891748	-0.917869	-0.862757	-0.821711	-0.816831
1	-0.992212	-1.178788	-1.174483	-1.359336	-1.159844
2	-0.679052	-0.793294	-0.699144	-0.786118	-0.886869
3	-0.857878	-1.154677	-1.077176	-1.211224	-1.404689
4	-1.313410	-1.193140	-1.089806	-0.962647	-1.067709
```
Converting to a numpy array and splitting into test and train tests
```
x_train = df.iloc[:74,:4]
y_train = df.iloc[:74,-1]

x_test = df.iloc[75:99,:4]
y_test = df.iloc[75:99,-1]
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

x_train.shape, x_test.shape
```
Constructing the Model , Sequential under Tensorflow
```
model = Sequential()

model.add(LSTM(20,return_sequences=True,input_shape=(4,1)))
model.dropout(0.5)
model.add(LSTM(40,return_sequences=False))
model.dropout(0.4)
model.add(LSTM(40,return_sequences=False))
model.add(LSTM(30,return_sequences=False))
model.add(LSTM(30,return_sequences=False))
model.add(LSTM(25,return_sequences=False))

model.add(Dense(128,activation='linear'))
model.dropout(0.4)
model.add(Dense(64,activation='linear'))
model.dropout(0.4)
model.add(Dense(64,activation='linear'))
model.dropout(0.4)
model.add(Dense(32,activation='linear'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mse',optimizer='rmsprop')
```
training the model
```
model.fit(x_train,y_train,epochs=100,batch_size=5)
------------------------------------------------------------
Epoch 1/100
15/15 [==============================] - 8s 37ms/step - loss: 0.4159
Epoch 2/100
15/15 [==============================] - 1s 37ms/step - loss: 0.1682
Epoch 3/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0984
Epoch 4/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0775
Epoch 5/100
15/15 [==============================] - 1s 43ms/step - loss: 0.0699
Epoch 6/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0678
Epoch 7/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0696
Epoch 8/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0630
Epoch 9/100
15/15 [==============================] - 1s 37ms/step - loss: 0.0678
Epoch 10/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0656
Epoch 11/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0610
Epoch 12/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0652
Epoch 13/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0634
Epoch 14/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0643
Epoch 15/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0635
Epoch 16/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0628
Epoch 17/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0611
Epoch 18/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0631
Epoch 19/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0604
Epoch 20/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0629
Epoch 21/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0615
Epoch 22/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0626
Epoch 23/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0622
Epoch 24/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0599
Epoch 25/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0614
Epoch 26/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0614
Epoch 27/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0580
Epoch 28/100
15/15 [==============================] - 1s 42ms/step - loss: 0.0601
Epoch 29/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0600
Epoch 30/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0582
Epoch 31/100
15/15 [==============================] - 1s 42ms/step - loss: 0.0578
Epoch 32/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0585
Epoch 33/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0583
Epoch 34/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0601
Epoch 35/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0575
Epoch 36/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0566
Epoch 37/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0550
Epoch 38/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0548
Epoch 39/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0543
Epoch 40/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0590
Epoch 41/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0572
Epoch 42/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0549
Epoch 43/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0549
Epoch 44/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0537
Epoch 45/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0560
Epoch 46/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0537
Epoch 47/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0550
Epoch 48/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0548
Epoch 49/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0523
Epoch 50/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0544
Epoch 51/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0529
Epoch 52/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0526
Epoch 53/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0531
Epoch 54/100
15/15 [==============================] - 1s 37ms/step - loss: 0.0527
Epoch 55/100
15/15 [==============================] - 1s 42ms/step - loss: 0.0512
Epoch 56/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0507
Epoch 57/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0509
Epoch 58/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0502
Epoch 59/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0503
Epoch 60/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0516
Epoch 61/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0501
Epoch 62/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0504
Epoch 63/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0508
Epoch 64/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0498
Epoch 65/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0494
Epoch 66/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0502
Epoch 67/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0502
Epoch 68/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0511
Epoch 69/100
15/15 [==============================] - 1s 66ms/step - loss: 0.0490
Epoch 70/100
15/15 [==============================] - 1s 44ms/step - loss: 0.0487
Epoch 71/100
15/15 [==============================] - 1s 69ms/step - loss: 0.0480
Epoch 72/100
15/15 [==============================] - 1s 78ms/step - loss: 0.0476
Epoch 73/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0485
Epoch 74/100
15/15 [==============================] - 1s 42ms/step - loss: 0.0478
Epoch 75/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0489
Epoch 76/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0466
Epoch 77/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0476
Epoch 78/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0470
Epoch 79/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0458
Epoch 80/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0486
Epoch 81/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0460
Epoch 82/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0477
Epoch 83/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0460
Epoch 84/100
15/15 [==============================] - 1s 69ms/step - loss: 0.0466
Epoch 85/100
15/15 [==============================] - 1s 51ms/step - loss: 0.0455
Epoch 86/100
15/15 [==============================] - 1s 68ms/step - loss: 0.0485
Epoch 87/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0476
Epoch 88/100
15/15 [==============================] - 1s 38ms/step - loss: 0.0461
Epoch 89/100
15/15 [==============================] - 1s 64ms/step - loss: 0.0464
Epoch 90/100
15/15 [==============================] - 1s 91ms/step - loss: 0.0445
Epoch 91/100
15/15 [==============================] - 1s 52ms/step - loss: 0.0438
Epoch 92/100
15/15 [==============================] - 1s 43ms/step - loss: 0.0460
Epoch 93/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0455
Epoch 94/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0431
Epoch 95/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0461
Epoch 96/100
15/15 [==============================] - 1s 41ms/step - loss: 0.0444
Epoch 97/100
15/15 [==============================] - 1s 40ms/step - loss: 0.0435
Epoch 98/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0456
Epoch 99/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0426
Epoch 100/100
15/15 [==============================] - 1s 39ms/step - loss: 0.0425
<keras.callbacks.History at 0x7f0450287750>
```
Plotting the Predicted value with real test data
```
y_pred = model.predict(x_test)
plt.figure(figsize=[16,7])
plt.title('Model Fit')
plt.xlabel('Time Step')
plt.ylabel('Normalized Price')
plt.plot(y_test,label = "True")
plt.plot(y_pred,label="Prediction")
plt.legend()
plt.show()
-------------------------------------------------------------------
```
![download](https://user-images.githubusercontent.com/111745916/195310194-d1531e00-0eec-49e8-93f2-7db98e419c6b.png)

Checking the R^2 accuracy 
```
from sklearn.metrics import r2_score
print('RSquared : ', '{:.2%}'.format(r2_score(y_test,y_pred)))
model.save("Bitcoin_model.h5")
----------------------------------------------------------------------
RSquared :  87.17%
```
## Bot.py

Bot implementation to live data through Binance Rest API

Importing Libraries
```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from keras.models import load_model

from binance.client import Client
```
Setting up the Rest API keys

```
api_key = ""
secret_key = ""
client = Client(api_key,secret_key)
check = client.get_klines(symbol='BTCUSDT',interval = Client.KLINE_INTERVAL_1MINUTE)

check[150]
----------------------------------------------------------------------------------------------
[1665560520000,
 '19101.86000000',
 '19103.25000000',
 '19099.74000000',
 '19102.17000000',
 '10.38972000',
 1665560579999,
 '198463.86867620',
 364,
 '3.68371000',
 '70368.75849190',
 '0']
```
Scaling the data for implementation
```
price = np.array([float(check[i][4]) for i in range(500)])
price = price.reshape(500,1)

scaler = StandardScaler()
scaler.fit(price[:374])

price = scaler.transform(price)
```
### Trading

```
symbol = "BTCUSDT"   #Code of cryptocurrency
quantity = '0.05'    #quantity to trade

order = False
index = [496,497,498,499]

while True:
    price = client.get_recent_trades(symbol=symbol)
    candle = client.get_klines(symbol=symbol,interval=Client.KLINE_INTERVAL_1MINUTE)
    candles = scaler.transform(np.array([float(candle[i][4]) for i in index]).reshape(-1,1))
    model_feed = candles.reshape(1,4,1)
    
    
    if order == False and float(price[len(price)-1]['price']) < float(scaler.inverse_transform(model.predict(model_feed)[0])[0]):
        
        #client.order_market_buy(symbol=symbol,quantity = quantity)
        order = True
        buy_price = client.get_order_book(symbol=symbol)['asks'][0][0]
        print('Buy @Market Price : ',float(buy_price),'Timestamp : ',str(datetime.now()))
        
    elif order == True and float(price[len(price)-1]['price'])-float(buy_price) >=10:
        
        #client.order_market_sell(symbol=symbol, quantity=quantity)
        order = False
        sell_price = client.get_order_book(symbol=symbol)['bids'][0][0]
        print('Sell @Market Price : ',float(sell_price),'Timestamp : ',str(datetime.now()))
-----------------------------------------------------------------------------------------------------------------------------------
Buy @Market Price :  22049.85 Timestamp :  2022-10-12 03:45:59.610620
Sell @Market Price :  22144.58 Timestamp :  2022-10-12 03:47:20.656783
Buy @Market Price :  22129.54 Timestamp :  2022-10-12 03:48:22.990451
Sell @Market Price :  22144.32 Timestamp :  2022-10-12 03:49:32.217690
Buy @Market Price :  22143.61 Timestamp :  2022-10-12 03:49:34.117890
```
# Thanks for reading ,
### Disclaimer : This a project i made to tecah myself binance api plz don't make trading decisons from this model, bitcoin is a volatile asset and is not regulated , the binance API is not official , i shall not be held liable foe losses caused by my algorithm.
