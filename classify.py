import os
import config
import numpy as np
import pandas as pd
from keras.models import load_model, model_from_json
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

#model configs
model_name =         config.MODEL['model_name']
model_weight =       config.MODEL['model_weight']
path =               config.DATASET['path']
label =              config.DATASET['label']
label_map =          config.DATASET['label_map']
validation_split =   config.MODEL['validation_split']
batch_size=          config.MODEL['batch_size']
epochs =             config.MODEL['epochs']
loss =               config.MODEL['loss']
optimizer =          config.MODEL['optimizer']
model_weight =       config.MODEL['model_weight']

vocab_size = 1000
max_length = 19


json_file = open(model_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_weight)


def classify(testObject):
  
  if not model:
    return "model is not trained yet, or it is training"

  # predict
  temp = one_hot(testObject["Info"],vocab_size)
  temp.append(testObject["Length"])
  temp.append(one_hot(testObject["Protocol"],vocab_size)[0])
  encoded_docs=[temp]
  padded_data = pad_sequences(encoded_docs, maxlen=max_length, padding='pre')
  test = padded_data.reshape((padded_data.shape[0], 1, padded_data.shape[1]))
  result = model.predict(test, verbose=0)
  found = [key  for (key, value) in label_map.items() if value == round(result[0][0])]  
  
  #re-trained the model with new predicted value
  temp.append(result)
  encoded_docs=[temp]  
  new_dataset = pad_sequences(encoded_docs, maxlen=max_length + 1, padding='pre')
  scaler = MinMaxScaler(feature_range=(0, 1))
  new_dataset = scaler.fit_transform(new_dataset)
  train_X, train_y = new_dataset[:, :-1], new_dataset[:, -1]
  train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])  
  model.fit(train_X, train_y, epochs=epochs, verbose=1,batch_size=1)
  model.save_weights(model_weight)

  return found[0]