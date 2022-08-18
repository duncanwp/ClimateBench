import numpy as np
import xarray as xr

from utils import data_path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Input, Reshape, AveragePooling2D, MaxPooling2D, Conv2DTranspose, TimeDistributed, LSTM, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2

import random
seed = 6
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

len_historical = 165

# Functions for reshaping the data
def input_for_training(X_train_xr, slider=10, skip_historical=False, len_historical=None):
    X_train_np = X_train_xr.to_array().transpose('time', 'latitude', 'longitude', 'variable').data

    time_length = X_train_np.shape[0]
    # If we skip historical data, the first sequence created has as last element the first scenario data point
    if skip_historical:
        X_train_to_return = np.array(
            [X_train_np[i:i + slider] for i in range(len_historical - slider + 1, time_length - slider + 1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        X_train_to_return = np.array([X_train_np[i:i + slider] for i in range(0, time_length - slider + 1)])

    return X_train_to_return


def output_for_training(Y_train_xr, var, slider=10, skip_historical=False, len_historical=None):
    Y_train_np = Y_train_xr[var].data

    time_length = Y_train_np.shape[0]

    # If we skip historical data, the first sequence created has as target element the first scenario data point
    if skip_historical:
        Y_train_to_return = np.array(
            [[Y_train_np[i + slider - 1]] for i in range(len_historical - slider + 1, time_length - slider + 1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        Y_train_to_return = np.array([[Y_train_np[i + slider - 1]] for i in range(0, time_length - slider + 1)])

    return Y_train_to_return


def get_training_data(simus):
    X_train = []
    Y_train = []

    for i, simu in enumerate(simus):

        input_name = 'inputs_' + simu + '.nc'
        output_name = 'outputs_' + simu + '.nc'

        # Just load hist data in these cases 'hist-GHG' and 'hist-aer'
        if 'hist' in simu:
            # load inputs
            input_xr = xr.open_dataset(data_path + input_name)

            # load outputs
            output_xr = xr.open_dataset(data_path + output_name).mean(dim='member')
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                          "pr90": output_xr.pr90 * 86400}).rename({'lon': 'longitude',
                                                                                   'lat': 'latitude'}).transpose('time',
                                                                                                                 'latitude',
                                                                                                                 'longitude').drop(
                ['quantile'])

        # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
        else:
            # load inputs
            input_xr = xr.open_mfdataset([data_path + 'inputs_historical.nc',
                                          data_path + input_name]).compute()

            # load outputs
            output_xr = xr.concat([xr.open_dataset(data_path + 'outputs_historical.nc').mean(dim='member'),
                                   xr.open_dataset(data_path + output_name).mean(dim='member')],
                                  dim='time').compute()
            output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                          "pr90": output_xr.pr90 * 86400}).rename({'lon': 'longitude',
                                                                                   'lat': 'latitude'}).transpose('time',
                                                                                                                 'latitude',
                                                                                                                 'longitude').drop(
                ['quantile'])

        print(input_xr.dims, simu)

        # Append to list
        X_train.append(input_xr)
        Y_train.append(output_xr)


def train(var_to_predict):

    print(var_to_predict)

    # Data
    X_train_all = np.concatenate(
        [input_for_training(X_train_norm[i], skip_historical=(i < 2), len_historical=len_historical) for i in
         range(len(simus))], axis=0)
    Y_train_all = np.concatenate(
        [output_for_training(Y_train[i], var_to_predict, skip_historical=(i < 2), len_historical=len_historical) for
         i in range(len(simus))], axis=0)
    print(X_train_all.shape)
    print(Y_train_all.shape)

    # Model
    keras.backend.clear_session()
    cnn_model = None

    cnn_model = Sequential()
    cnn_model.add(Input(shape=(slider, 96, 144, 4)))
    cnn_model.add(
        TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='relu'), input_shape=(slider, 96, 144, 4)))
    cnn_model.add(TimeDistributed(AveragePooling2D(2)))
    cnn_model.add(TimeDistributed(GlobalAveragePooling2D()))
    cnn_model.add(LSTM(25, activation='relu'))
    cnn_model.add(Dense(1 * 96 * 144))
    cnn_model.add(Activation('linear'))
    cnn_model.add(Reshape((1, 96, 144)))

    cnn_model.compile(optimizer="rmsprop", loss="mse", metrics=["mse"])

    hist = cnn_model.fit(X_train_all,
                         Y_train_all,
                         use_multiprocessing=True,
                         # workers=5,
                         batch_size=16, epochs=30,
                         verbose=1)

    # Make predictions using trained model
    m_pred = cnn_model.predict(X_test_np)
    # Reshape to xarray
    m_pred = m_pred.reshape(m_pred.shape[0], m_pred.shape[2], m_pred.shape[3])
    m_pred = xr.DataArray(m_pred, dims=['time', 'lat', 'lon'],
                          coords=[X_test.time.data[slider - 1:], X_test.latitude.data, X_test.longitude.data])
    xr_prediction = m_pred.transpose('lat', 'lon', 'time').sel(time=slice(2015, 2101)).to_dataset(
        name=var_to_predict)

    if var_to_predict == "pr90" or var_to_predict == "pr":
        xr_prediction = xr_prediction.assign({var_to_predict: xr_prediction[var_to_predict] / 86400})

    # Save test predictions as .nc
    if var_to_predict == 'diurnal_temperature_range':
        xr_prediction.to_netcdf(data_path + 'outputs_ssp245_predict_dtr-aer.nc', 'w')
    else:
        xr_prediction.to_netcdf(data_path + 'outputs_ssp245_predict_{}-aer.nc'.format(var_to_predict), 'w')
    xr_prediction.close()

