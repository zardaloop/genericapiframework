
import os
import numpy
import glob
import config
import pandas as pd
from keras.layers import LSTM
from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import MinMaxScaler

#model configs
validation_split =   config.MODEL['validation_split']
epochs =             config.MODEL['epochs'] 
model_name =         config.MODEL['model_name']
model_weight =       config.MODEL['model_weight']
model_name_tfjs =    config.MODEL['model_name_tfjs']
loss =               config.MODEL['loss']
optimizer =          config.MODEL['optimizer']
activation =         config.MODEL['activation']
batch_size=          config.MODEL['batch_size']

#dataset configs
dataset_file_name =  config.DATASET['dataset_file_name']
path =               config.DATASET['path']
label =              config.DATASET['label']
label_map =          config.DATASET['label_map']

# integer encode the documents
vocab_size = 1000
max_length = 20


# Load dataset
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

train=[]
for d in dataset.values:    
  temp = one_hot(d[6],vocab_size)
  temp.append(one_hot(d[4],vocab_size)[0])
  temp.append(d[5])
  temp.append(d[7])
  train.append(temp)

train = pad_sequences(train, maxlen=max_length, padding='pre')
train = scaler.fit_transform(train)

#create training and prediction
train_X, train_y = train[:, :-1], train[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

# create Model
model = Sequential()
model.add(LSTM(3,batch_input_shape=(batch_size, 1, train_X.shape[2]), stateful=True))
model.add(Dense(1))
model.add(Activation(activation))

# compile the model
model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

# summarise the model
print(model.summary())

# fit the model
history = model.fit(train_X, train_y,
                    epochs=epochs, verbose=1,
                    batch_size=batch_size, validation_split=validation_split,
                    shuffle=True)

#saving model and weight                    
model_json = model.to_json()
with open(model_name, "w") as json_file:
    json_file.write(model_json)
model.save_weights(model_weight)
