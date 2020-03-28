# Generic Api Framework#
A generic API framework designed for quick model development, model retraining to classify and predict real-time data input.

# Keycloak configuration #
Download keycloak from [https://www.keycloak.org/downloads
](https://www.keycloak.org/downloads)
unzip the downloaded file and then run the application following the instruction [here](https://www.keycloak.org/docs/latest/getting_started/index.html).

Create a new client  called "api", then set client Authenticator to "Client Id and Secret".

![](https://github.com/zardaloop/genericapiframework/blob/master/manual/secret.PNG?raw=true)

Create a  new user for the Client.

Then use the client use along side of the secret to get token from keycloak using postman. 

![](https://github.com/zardaloop/genericapiframework/blob/master/manual/token.PNG?raw=true)

then use the token in the header of the call to the API 
![](https://github.com/zardaloop/genericapiframework/blob/master/manual/header.PNG?raw=true)

finally set the body of the post with the data needed to be classified. 

![](https://github.com/zardaloop/genericapiframework/blob/master/manual/body.PNG?raw=true)


# Config #

you can adjust the config file via the config.py file . 
The default config is as below: 
    
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