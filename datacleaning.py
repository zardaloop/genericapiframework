from keras.layers import Input, Dense
from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
import config
import os
import numpy
import glob
import pandas as pd

dataset_file_name =  config.DATASET['dataset_file_name']
path =               config.DATASET['path']
label =              config.DATASET['label']
label_map =          config.DATASET['label_map']

allFiles = glob.glob(path + dataset_file_name)
print(allFiles)
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
dataset = pd.concat(list_)

dataset = dataset.dropna()

if label_map:
  dataset[label] = dataset[label].map(label_map)

scaler = MinMaxScaler(feature_range=(0, 1))

# integer encode the documents
vocab_size = 1000
max_length = 20

train=[]
for d in dataset.values:    
  temp = one_hot(d[6],vocab_size)
  temp.append(one_hot(d[4],vocab_size)[0])
  temp.append(d[5])
  temp.append(d[7])
  train.append(temp)

dataset = pad_sequences(train, maxlen=max_length, padding='pre')
dataset = scaler.fit_transform(dataset)

# lstm autoencoder recreate sequence
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
# define input sequence
sequence = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# reshape input into [samples, timesteps, features]
dataset = dataset.reshape((dataset.shape[0], dataset.shape[1], 1))
# define model
model = Sequential()
model.add(LSTM(4, activation='relu', input_shape=(dataset.shape[1],1)))
model.add(RepeatVector(dataset.shape[1]))
model.add(LSTM(4, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(dataset, dataset, epochs=300, verbose=0)
#ßplot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
yhat = model.predict(dataset, verbose=0)
print(yhat[0,:,0])