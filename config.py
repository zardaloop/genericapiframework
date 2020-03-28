DATASET = {
    'days': 15,
    'path': './',
    'dataset_file_name': 'data/*.csv',
    'label':'label',
    'label_map':{
      'norm': 0,
      'mirai':1,
      'udp': 2, 
      'dns':3,
      'ack':4,
      'ack2':5
    }
}

MODEL = {
    'model_name': 'model.json',
    'model_weight': 'model.h5',
    'model_name_tfjs': 'model.js',
    'validation_split': 0.3,
    'epochs': 1,
    'loss': 'mse',
    'optimizer': 'Adam',
    'activation': 'elu',
    'batch_size':1
}