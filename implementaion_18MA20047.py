import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.layers import Input, Embedding, LSTM, Dense,Concatenate,Lambda
from keras.models import Model
import keras
from keras import backend as K
from keras.layers import Dense, Activation, Flatten,TimeDistributed,Dropout,Bidirectional
#from attention_utils import get_activations, get_data_recurrent
from keras.callbacks import EarlyStopping
import _pickle as cPickle
np.random.seed(123)
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests

class Model:
    def __init__(inert):
        inert.data = None
        inert.model = None
        
    # LSTM architecture
    def model_arch():
        lstm_hidden = 64
        initi = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=123)
        main_input = Input(shape=(10,1), name='main_input')
        lstm_out = LSTM(lstm_hidden,kernel_initializer=initi, recurrent_initializer= initi, bias_initializer=initi)(main_input)
        lstm_out = Dropout(0.45)(lstm_out)
        main_output = (Dense(7, name='main_output')(lstm_out))
        # Adam=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)

        model = Model(inputs=main_input, outputs=main_output)
        model.compile(loss='mean_squared_error',optimizer='adam')
        return model
     
    def compute_allocation(inert, data):
        data_w_ret = np.concatenate([ data.values[1:], data.pct_change().values[1:] ], axis=1)
        data = data.iloc[1:]
        print(data_w_ret.shape)
        inert.data = tf.cast(tf.constant(data), float)
        
        
        if inert.model is None:
            inert.model = inert.alt_model(data_w_ret.shape, len(data.columns))
        
        fit_predict_data = data_w_ret[np.newaxis,:]        
        inert.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=15, shuffle=False)
        return inert.model.predict(fit_predict_data)[0]
      
      
        model = model_arch()
        earlystop=EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        callbacks_list = [earlystop]
        print("model is building")
        bs = 128
        model.fit(batch_size=bs,epochs=15,x=X_train,y=Y_train, verbose=1,validation_split=0.1)
        print("model building done")
        
        
    def alt_model(inert, input_shape, outputs):
        # the model will compute the allocation ratios to optimize the Sharpe Ratio of the portfolio
        model = Sequential([ LSTM(64, input_shape=input_shape),
            Flatten(), Dense(outputs, activation='softmax')
        ])
        
    def sharpe_loss(_, y_pred):
        data = tf.divide(self.data, self.data[0])  
        portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
        
        portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  

        sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)
        
        # the min of a negated function is its max
        return -sharpe
    
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
        
class OptimizedUncoupledAutosequencers(): #QCAlgorithm
        
    def OnData(inert, data):
        if inert.prev_day != inert.Time.day:
            inert.data.Add(data)
        
        inert.prev_day = inert.Time.day

    def Rebalance(inert):
        
        if not inert.data.IsReady:
            return
        
        try:
            data = inert.PandasConverter.GetDataFrame(inert.data).iloc[::-1]
        except:
            return
     
        data = data['close'].unstack(level=0)
        
        if len(data) < inert.data.Count:
            return
        
        tickers = [symbol.split(' ')[0] for symbol in data.columns]
        
        if inert.model is None:
            inert.model = Model()
        
        allocations = inert.model.compute_allocation(data)
        inert.Log(f'Optimal Weights/Protfolio Allocations are : {allocations}')
        
        for ticker, allocation in zip(tickers, allocations):
            inert.SetHoldings(ticker, allocation)

data = pd.read_excel('combined_dataset_18MA20047.xlsx')  
model = Model()
portfolio = model.compute_allocation(data)
print(portfolio)
