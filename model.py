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

api_key = ""
secret_key="t"

client = Client(api_key,secret_key)

candles = client.get_klines(symbol='BTCUSDT',interval = Client.KLINE_INTERVAL_1MINUTE);

print(candles[11])

#Fetching closing prices from candlesticks data

price = np.array([float(candles[i][4]) for i in range(len(candles))])

time = np.array([int(candles[i][0]) for i in range(len(candles))])

#Converting time to HH:MM:SS format
t = np.array([datetime.fromtimestamp(time[i]/1000).strftime('%H:%M:%S') for i in range(len(candles))])

plt.xlabel("Time Step")
plt.ylabel("Bitcoin Price $")
plt.plot(price)

timeframe = pd.DataFrame({'Time':t,'Price $BTC':price})
timeframe

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
price = price.reshape(-1,1)

sc.fit(price)
price = sc.transform(price)

price = price.reshape(500,1)

df = pd.DataFrame(price.reshape(100,5),columns = ['First','Second','Third','Fourth','Target'])

print(df.head(5))

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

model.summary()

model.fit(x_train,y_train,epochs=100,batch_size=5)

y_pred = model.predict(x_test)

plt.figure(figsize=[16,7])
plt.title('Model Fit')
plt.xlabel('Time Step')
plt.ylabel('Normalized Price')
plt.plot(y_test,label = "True")
plt.plot(y_pred,label="Prediction")
plt.legend()
plt.show()

testScore = np.sqrt(mean_squared_error(sc.inverse_transform(y_test.reshape(-1,1)),sc.inverse_transform(y_pred.reshape(-1,1))))
print('Test Score : %2f RMSE' % (testScore))

from sklearn.metrics import r2_score
print('RSquared : ', '{:.2%}'.format(r2_score(y_test,y_pred)))

model.save("Bitcoin_model.h5")

