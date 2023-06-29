import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten, Conv1D, LSTM, Input
from fileUtils import prepareTrain, saveResultsAverage
from deepUtils import comparePreds
from keras import backend as K

def inicializaModelo_CNN(x_train2, forecast_horizon, shift, type):
  """
  Initializes the CNN proposed model.
  :param x_train2: List, contains the training data to check its shape.
  :param forecast_horizon: int, the number of steps in the future that the model will forecast
  """
  if(shift == 0):
    inp = Input(shape=(x_train2.shape[-2:]))
    
    x = Conv1D(64, 5, activation='relu', padding='same')(inp)
    x = Flatten()(x)
    x = Dense(forecast_horizon)(x)
    model = keras.Model(inputs=inp, outputs=x)
      
    model.compile(optimizer='adam', loss='mae')
 
  elif(shift == 24):
    inp = Input(shape=(x_train2.shape[-2:]))
    x = Conv1D(8, 2, activation='relu', padding='same')(inp)

    x = Flatten()(x)
    x = Dense(24, activation='relu')(x)
    x = Dense(forecast_horizon, activation='relu')(x)
    model = keras.Model(inputs=inp, outputs=x)

    model.compile(optimizer='SGD', loss='mae')
    K.set_value(model.optimizer.learning_rate, 0.1)
    
  elif(shift == 48):
    inp = Input(shape=(x_train2.shape[-2:]))
      
    x = Conv1D(128, 1, activation='relu', padding='valid')(inp)
    x = Conv1D(8, 7, activation='sigmoid', padding='same')(x)
    x = Conv1D(128, 7, activation='sigmoid', padding='valid')(x)
    x = Conv1D(8, 7, activation='relu', padding='valid')(x)
    x = Flatten()(x)
    x = Dense(12, activation='relu')(x)
    x = Dense(12, activation='relu')(x)
    x = Dense(forecast_horizon, activation='relu')(x)
    model = keras.Model(inputs=inp, outputs=x)
      
    model.compile(optimizer='adam', loss='mae')
    K.set_value(model.optimizer.learning_rate, 0.0001)
    
  return model

def inicializaModelo_CNN_LSTM(x_train2, forecast_horizon, shift, type):
  """
  Initializes the CNN proposed model.
  :param x_train2: List, contains the training data to check its shape.
  :param forecast_horizon: int, the number of steps in the future that the model will forecast.
  """
  if(shift == 0):
    inp = Input(shape=(x_train2.shape[-2:]))
    x = Conv1D(64, 7, activation='relu', padding='same')(inp)
    x = LSTM(64, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(forecast_horizon)(x)
    model = keras.Model(inputs=inp, outputs=x)
    model.compile(optimizer='adam', loss='mae')

  elif(shift == 24):
    inp = Input(shape=(x_train2.shape[-2:]))
    x = Conv1D(8, 2, activation='relu', padding='same')(inp)
    x = LSTM(32, return_sequences=True)(x)

    x = Flatten()(x)
    x = Dense(24, activation='relu')(x)
    x = Dense(forecast_horizon, activation='relu')(x)
    model = keras.Model(inputs=inp, outputs=x)
      
    model.compile(optimizer='SGD', loss='mae')
    K.set_value(model.optimizer.learning_rate, 0.1)

  elif(shift == 48):
    inp = Input(shape=(x_train2.shape[-2:]))
      
    x = Conv1D(128, 1, activation='relu', padding='valid')(inp)
    x = LSTM(64, return_sequences=True)(x)
    x = Conv1D(8, 7, activation='sigmoid', padding='same')(x)
    x = Conv1D(128, 7, activation='sigmoid', padding='valid')(x)
    x = Conv1D(8, 7, activation='relu', padding='valid')(x)
    x = Flatten()(x)
    x = Dense(12, activation='relu')(x)
    x = Dense(12, activation='relu')(x)
    x = Dense(forecast_horizon, activation='relu')(x)
    model = keras.Model(inputs=inp, outputs=x)
      
    model.compile(optimizer='adam', loss='mae')
    K.set_value(model.optimizer.learning_rate, 0.0001)

  return model

def CNN(all_data, df2, folder_split, cv, epochs, batch_size, train_split, type, 
        forecast_horizon, past_history, shift):
  """
  Allows to run the CNN_univariant experiment. 
  :param all_data: Dataframe object, contains all the data.
  :param df2: Dataframe object, contains only the train and test data.
  :param folder_split: int, that represents the size of a folder.
  :param cv: int, the number of folders that divides the data.
  :param epochs: int, the number of epochs to train our model.
  :param batch_size: int, the size of the batch during training.
  :param train_split: int, represents the point of the folder where 
  the test data start.
  :param type: String, represents type of model, U-> Univariate, M-> Multivariate .
  :param forecast_horizon: int, the number of steps in the future that the model will forecast.
  :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data.
  :param shift: int, Specifies how many entries are between the last entry of 
  the past_history and the first of the forecast_horizon. 
  """
  maeWape, maeWape_esios = [[], []], [[], []]
  realData, forecastedData, esiosForecast = [], [], []
  x_train, y_train, x_test, y_test, norm_params = prepareTrain(
          folder_split, df2, 0, train_split, shift, past_history, forecast_horizon)
  model = inicializaModelo_CNN(x_train, forecast_horizon, shift, type)

  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
            validation_data=(x_test, y_test))
        
  preds = model.predict(x_test)
    
  comparePreds(norm_params, preds, y_test, all_data, 'CNN', forecastedData, realData, 
              train_split, folder_split, esiosForecast, 0, maeWape, maeWape_esios, 
              df2, type, past_history, forecast_horizon, shift)

  for iteration in range(1, cv):
      realData, forecastedData, esiosForecast = [], [], []
      x_train, y_train, x_test, y_test, norm_params = prepareTrain(folder_split, df2, 
      iteration, train_split, shift, past_history, forecast_horizon)
            
      model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                 validation_data=(x_test, y_test))
        
      preds = model.predict(x_test)
        
      comparePreds(norm_params, preds, y_test, all_data, 'CNN', forecastedData, realData, 
                  train_split, folder_split, esiosForecast, iteration, maeWape, 
                  maeWape_esios, df2, type, past_history, forecast_horizon, shift)
  saveResultsAverage(maeWape, maeWape_esios, 'CNN', type, shift)

def CNN_LSTM(all_data, df2, folder_split, cv, epochs, batch_size, train_split, type, 
            forecast_horizon, past_history, shift):
  """
  Allows to run the CNN_LSTM_univariant experiment. 
  :param all_data: Dataframe object, contains all the data.
  :param df2: Dataframe object, contains only the train and test data.
  :param folder_split: int, represents the size of a folder.
  :param cv: int, the number of folders that divides the data.
  :param epochs: int, the number of epochs to train our model.
  :param batch_size: int, the size of the batch during training.
  :param train_split: int, represents the point of the folder where 
  the test data start.
  :param type: String, represents type of model, U-> Univariate, M-> Multivariate 
  :param forecast_horizon: int, the number of steps in the future that the model will forecast.
  :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data.
  :param shift: int, Specifies how many entries are between the last entry of 
  the past_history and the first of the forecast_horizon. 
  
  """
  x_train, y_train, x_test, y_test, norm_params = prepareTrain(folder_split, df2, 0, 
  train_split, shift, past_history, forecast_horizon)
  model = inicializaModelo_CNN_LSTM(x_train, forecast_horizon, shift, type)
  maeWape, maeWape_esios= [[], []], [[], []]

  realData, forecastedData, esiosForecast = [], [], []

  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
            validation_data=(x_test, y_test))
        
  preds = model.predict(x_test)
        
  comparePreds(norm_params, preds, y_test, all_data, 'CNN_LSTM', forecastedData, 
  realData, train_split, folder_split, esiosForecast, 0, maeWape, maeWape_esios, 
  df2, type, past_history, forecast_horizon, shift)

  for iteration in range(1, cv):
    realData, forecastedData, esiosForecast = [], [], []
    x_train, y_train, x_test, y_test, norm_params = prepareTrain(folder_split, df2, 
    iteration, train_split, shift, past_history, forecast_horizon)
            
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
    validation_data=(x_test, y_test))
        
    preds = model.predict(x_test)

    comparePreds(norm_params, preds, y_test, all_data, 'CNN_LSTM', forecastedData, 
    realData, train_split, folder_split, esiosForecast, iteration, maeWape, 
    maeWape_esios, df2, type, past_history, forecast_horizon, shift)

  saveResultsAverage(maeWape, maeWape_esios, 'CNN_LSTM', type, shift)
