import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
 
 
value = sys.argv[1]
 
####import all the packages####
import pymongo
import json 
from pandas import read_csv
from pandas import to_datetime
from datetime import datetime
from pandas import DataFrame
import math
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import  Dense,LSTM,Dropout
from tensorflow.keras import backend  
from tensorflow.keras.models import Sequential
from pandas.tseries.offsets import DateOffset
from datetime import datetime,date

dev_eui= value
if value  != '8cf9574000000012':
    print ('ERROR !!! Wrong sensore, there is no data on this device. please try again later.')   
    ####import and read database csv file####   
 

myclient = pymongo.MongoClient("mongodb://ibti:ibti@iotibti.ddns.net:27017/admin?tls=true")
mydb = myclient["data"]
#dev_eui='8cf9574000000012'
col_data = mydb[dev_eui]

lista_dados = []

lista_tempo= []
tempo = 0

for item in col_data.find():
    if int(item['ts'])- tempo >= 3600*24:
        tempo=int(item['ts'])
        dado=float(item['temp'])
        lista_tempo.append(datetime.utcfromtimestamp(tempo).strftime('%Y-%m-%d'))
        lista_dados.append(dado)
        
lista_tempo.reverse()
lista_dados.reverse()

del lista_tempo[100:]
del lista_dados[100:]

lista_tempo.reverse()
lista_dados.reverse()

dic={'ds':lista_tempo, 'y':lista_dados}
df=pd.DataFrame(dic) 

####select Data column as index####
df["ds"] =pd.to_datetime(df.ds)
df=df.set_index ('ds')
#dataset=dataset.sort_values(by='Data')

####filter a select column####
df= df.replace(',','.', regex=True)
#df = dataset.filter(["Velocidade do vento (m/s)"])
#print(df)



# set datas between 0 and 1 for neural network model  
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(df)
X_replace= df.replace(',','.', regex=True)
df_scaled = scaler.fit_transform(df)
# convert it back to numpy array
X_np = X_replace.values
# set the object type as float
X_fa = X_np.astype(float)
# perdict for seven days
forecast_features_set = []
labels = []
for i in range(7,len(df)):
    forecast_features_set.append(df_scaled[i-7:i, 0])
    labels.append(df_scaled[i, 0])


    
forecast_features_set , labels = np.array(forecast_features_set ), np.array(labels)

forecast_features_set = np.reshape(forecast_features_set, (forecast_features_set.shape[0], forecast_features_set.shape[1], 1))
forecast_features_set.shape

# LSTM Model 
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(forecast_features_set.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

hist = model.fit(forecast_features_set, labels, epochs = 1, batch_size = 100 )

forecast_list=[]

batch=df_scaled[-forecast_features_set.shape[1]:].reshape((1,forecast_features_set.shape[1],1))

for i in range(forecast_features_set.shape[1]):
    forecast_list.append(model.predict(batch)[0])
    batch = np.append(batch[:,1:,:], [[forecast_list[i]]], axis=1)
df_predict=pd.DataFrame(scaler.inverse_transform(forecast_list),index=df[-forecast_features_set.shape[1]:].index,columns=["forecasting"])  
                            

df_predict =pd.concat([df,df_predict],axis=1)


add_dates=[df.index[-1]+DateOffset(days=x) for x in range(0,8)]
future_dates=pd.DataFrame(index=add_dates[1:],columns=df.columns)
#df_forecast=pd.DataFrame(scaler.inverse_transform(forecast_list),index=future_dates[-forecast_features_set.shape[1]:].index) 
                        
df_forecast=pd.DataFrame(scaler.inverse_transform(forecast_list),index=future_dates[-forecast_features_set.shape[1]:].index, 
                        columns=["forecasting"])                           
df_forecast =pd.concat([df,df_forecast],axis=1)
df_forecast=df_forecast.drop(['y'], axis=1)
df_forecast=df_forecast.dropna()
    
df_forecast=df_forecast.reset_index()
#df_forecast.index.name = 'foo' 
#df_forecast.index = ['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5', 'Row_6', 'Row_7']
df_forecast['index'] = pd.to_datetime(df_forecast['index']).dt.date
df_forecast['forecasting']=  df_forecast['forecasting'].apply('{:,.4f}'.format)
    
#df_forecast_new.index.name = 'foo'               

#### save data to CSV file ###s#
#df_forecast.to_csv(r'Forcasting_Velocidade.csv', index = True, header=True)
#df_forecast = df_forecast.set_axis(['dia', 'forecasting'], axis=1, inplace=False)
#print(df_forecast)
df_forecast['forecasting'] = pd.to_numeric(df_forecast['forecasting'])
    
pf=pd.DataFrame(df_forecast)

 
print ('Deep learning neural networks are trained using the stochastic gradient descent optimization algorithm.' )





plt.plot(df_forecast['index'],df_forecast['forecasting'], label='Prediction of  Temperature', alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Predicted Value')
plt.legend()
plt.show()
    
