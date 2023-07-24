#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : ConvTransformerTS
@ FileName: preprocessing.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 1/20/23 15:31
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from math import sqrt
from pandas import concat, read_csv
from sklearn.preprocessing import MinMaxScaler
from helper import series_to_supervised, stage_series_to_supervised


def ws_preprocessing(n_hours, K, S):
    """
    n_hours: past window length
    K: forecasting lenght
    S: shifting length
    """
    dataset = pd.read_csv('../data/Merged-update_hourly.csv', index_col=0)
    dataset.fillna(0, inplace=True)
    
    # specify the number of lag hours
#     n_hours = 24*3
#     K = 24
#     S = 24
    
    
    # Target time series (4): 'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26'
    stages = dataset[['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
    print("stages.shape:", stages.shape)

    stages_supervised = series_to_supervised(stages, n_hours, K)
    print("stages_supervised.shape:", stages_supervised.shape)
    
    
    # Prior 6 unknown covariates: 'FLOW_S25A', 'HWS_S25A', 'FLOW_S25B', 'HWS_S25B', 'FLOW_S26', 'HWS_S26'
    prior_unknown = dataset[['FLOW_S25A', 'HWS_S25A', 'FLOW_S25B', 'HWS_S25B', 'FLOW_S26', 'HWS_S26' ]]
    print("prior_unknown.shape:", prior_unknown.shape)

    prior_unknown_supervised = series_to_supervised(prior_unknown, n_hours, S)
    print("prior_unknown_supervised.shape:", prior_unknown_supervised.shape)
    
    
    # Prior 9 known covariates: 'WS_S4', 'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'PUMP_S25B', 'GATE_S26_1', 'GATE_S26_2', 'PUMP_S26', 'MEAN_RAIN'
    prior_known = dataset[['WS_S4', 'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'PUMP_S25B', 'GATE_S26_1', \
                           'GATE_S26_2', 'PUMP_S26', 'MEAN_RAIN']]
    print("prior_known.shape:", prior_known.shape)

    prior_known_supervised = series_to_supervised(prior_known, n_hours, S)
    print("prior_known_supervised.shape:", prior_known_supervised.shape)
    
    
    # all past covariates in past w hours, 15 covariates in total
    cov = dataset[['FLOW_S25A', 'HWS_S25A', 'FLOW_S25B', 'HWS_S25B', 'FLOW_S26', 'HWS_S26','WS_S4', 'GATE_S25A', 
                   'GATE_S25B', 'GATE_S25B2', 'PUMP_S25B', 'GATE_S26_1', 'GATE_S26_2', 'PUMP_S26', 'MEAN_RAIN']]

    cov_supervised = series_to_supervised(cov, n_hours, S)    
    past_cov_supervised = cov_supervised.iloc[:, :n_hours*cov.shape[1]]
    
    
    # all target water stages in past w hours, 4 in total
    past_ws_supervised = stages_supervised.iloc[:, :n_hours*stages.shape[1]]
    past_ws_supervised.columns = ['past_ws_supervised_' + i for i in list(past_ws_supervised.columns)]
    
    
    # merge all data (covariates, ws) in past w hours
    columns = []
    for i in range(n_hours):
        columns = columns + past_cov_supervised.columns[i*cov.shape[1]:(i+1)*cov.shape[1]].tolist()
        columns = columns + past_ws_supervised.columns[i*stages.shape[1]:(i+1)*stages.shape[1]].tolist()
    
    
    past_cov_supervised.reset_index(drop=True, inplace=True)
    past_ws_supervised.reset_index(drop=True, inplace=True)

    past_cov_ws_supervised = pd.concat([past_cov_supervised.iloc[:min(past_cov_supervised.shape[0],past_ws_supervised.shape[0]), :], past_ws_supervised.iloc[:min(past_cov_supervised.shape[0], past_ws_supervised.shape[0]), :]], axis=1)

    past_cov_ws_supervised = past_cov_ws_supervised[columns]
    
    
    # shift prior known covariates in future s steps to the past
    shift_prior_known_supervised = prior_known_supervised.iloc[:, S*prior_known.shape[1]:]
    shift_prior_known_supervised.reset_index(drop=True, inplace=True)
    shift_prior_known_supervised.columns = ['shift_prior_known_supervised_' + i for i in list(shift_prior_known_supervised.columns)]
    shift_prior_known_supervised
    
    
    # merge all past data and shifted prior know future covariate
    columns1 = []
    for i in range(n_hours):
        columns1 = columns1 + past_cov_ws_supervised.columns[i*dataset.shape[1]:(i+1)*dataset.shape[1]].tolist()
        columns1 = columns1 + shift_prior_known_supervised.columns[i*prior_known.shape[1]:(i+1)*prior_known.shape[1]].tolist()
       
    
    shift_prior_known_cov_ws_supervised = pd.concat([past_cov_ws_supervised.iloc[:min(past_cov_ws_supervised.shape[0], shift_prior_known_supervised.shape[0]), :], shift_prior_known_supervised.iloc[:min(past_cov_ws_supervised.shape[0], shift_prior_known_supervised.shape[0]), :]], axis=1)

    shift_prior_known_cov_ws_supervised = shift_prior_known_cov_ws_supervised[columns1]
    
    
    # target water stage in future k time steps
    future_ws_supervised = stages_supervised.iloc[:, n_hours*stages.shape[1]:]
    future_ws_supervised.reset_index(drop=True, inplace=True)
    
    
    # Concatenation (input and labels)
    all_data = concat([shift_prior_known_cov_ws_supervised,
                       future_ws_supervised.iloc[:shift_prior_known_cov_ws_supervised.shape[0], :]], axis=1)
    print("all_data.shape:", all_data.shape)
    
    
    # Train & Test set spliting
    all_data = all_data.values
    n_train_hours = int(len(all_data)*0.8)
    print("n_train_hours:", n_train_hours)

    train = all_data[:n_train_hours, :]
    test = all_data[n_train_hours:, :]
   
    
    # split into input and outputs
    all_features = prior_known.shape[1] + dataset.shape[1]
    n_obs = n_hours * all_features
    train_X, train_y = train[:, :n_obs], train[:, n_obs:]
    test_X, test_y = test[:, :n_obs], test[:, n_obs:]
    print("train_X.shape, train_y.shape, test_X.shape, test_y.shape:", train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    train_y = scaler.fit_transform(train_y)
    test_X = scaler.fit_transform(test_X)
    test_y = scaler.fit_transform(test_y)
    
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, all_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, all_features))
    print("train_X.shape, train_y.shape, test_X.shape, test_y.shape:", train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    
    return train_X, train_y, test_X, test_y, scaler



def pm25_preprocessing(n_hours, K):
    """
    n_hours: past window length
    K: forecasting lenght
    """
    dataset = pd.read_csv('../data/pollution.csv', index_col=0)
    dataset.fillna(0, inplace=True)
    print(dataset.columns)
    
    
    # specify the number of lag hours
    n_hours = 24*3
    K = 24
    
    
    # Target time series (1): pollution
    pm25 = dataset[['pollution']]
    print("pm25.shape:", pm25.shape)

    pm25_supervised = series_to_supervised(pm25, n_hours, K)
    print("pm25_supervised.shape:", pm25_supervised.shape)
    
    
    # Prior known covariates (10): 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain', 'NE', 'NW', 'SE', 'cv'
    prior_known = dataset[['dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain', 'NE', 'NW', 'SE', 'cv']]
    print("prior_known.shape:", prior_known.shape)

    prior_known_supervised = series_to_supervised(prior_known, n_hours, K)
    print("prior_known_supervised.shape:", prior_known_supervised.shape)
    
    
    # all past covariates in past w hours, 10 covariates in total
    past_cov = dataset[['dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain', 'NE', 'NW', 'SE', 'cv']]

    past_cov_supervised = series_to_supervised(past_cov, n_hours, K)
    print("past_cov_supervised.shape:", past_cov_supervised.shape)
    
    
    # all target pm25 in past w hours, 1 in total
    past_pm25_supervised = pm25_supervised.iloc[:, :n_hours*pm25.shape[1]]
    past_pm25_supervised.reset_index(drop=True, inplace=True)
    past_pm25_supervised.columns = ['past_pm25_supervised_' + i for i in list(past_pm25_supervised.columns)]
    
    
    # merge all data (covariates, pm25) in past w hours
    columns = []
    for i in range(n_hours):
        columns = columns + past_cov_supervised.columns[i*past_cov.shape[1]:(i+1)*past_cov.shape[1]].tolist()
        columns = columns + past_pm25_supervised.columns[i*pm25.shape[1]:(i+1)*pm25.shape[1]].tolist()
    
    past_cov_supervised.reset_index(drop=True, inplace=True)
    past_pm25_supervised.reset_index(drop=True, inplace=True)

    past_cov_pm25_supervised = pd.concat([past_cov_supervised, past_pm25_supervised], axis=1)
    past_cov_pm25_supervised = past_cov_pm25_supervised[columns]

    
    # shift prior known covariates in future s steps to the past
    shift_prior_known_supervised = prior_known_supervised.iloc[:, K*prior_known.shape[1]:]
    shift_prior_known_supervised.reset_index(drop=True, inplace=True)
    shift_prior_known_supervised.columns = ['shift_prior_known_supervised_' + i for i in list(shift_prior_known_supervised.columns)]
    
    
    # merge all past data and shifted prior know future covariate
    shift_prior_known_past_cov_pm25_supervised = pd.concat([past_cov_pm25_supervised, shift_prior_known_supervised], axis=1)
    
    
    # target pm25 in future k time steps
    future_pm25_supervised = pm25_supervised.iloc[:, n_hours*pm25.shape[1]:]
    future_pm25_supervised.reset_index(drop=True, inplace=True)
    
    
    # Concatenation (input and labels)
    all_data = concat([shift_prior_known_past_cov_pm25_supervised, future_pm25_supervised], axis=1)
    # print("all_data", all_data)
    print("all_data.shape:", all_data.shape)
    
    
    # Train & Test set spliting
    all_data = all_data.values
    n_train_hours = int(len(all_data)*0.8)
    print("n_train_hours:", n_train_hours)


    train = all_data[:n_train_hours, :]
    test = all_data[n_train_hours:, :]
    
    
    # split into input and outputs
    all_features = prior_known.shape[1] + dataset.shape[1]
    n_obs = n_hours * all_features
    train_X, train_y = train[:, :n_obs], train[:, n_obs:]
    test_X, test_y = test[:, :n_obs], test[:, n_obs:]
    print("train_X.shape, train_y.shape, test_X.shape, test_y.shape", train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    train_y = scaler.fit_transform(train_y)
    test_X = scaler.fit_transform(test_X)
    test_y = scaler.fit_transform(test_y)
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, all_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, all_features))
    print("train_X.shape, train_y.shape, test_X.shape, test_y.shape:", train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    return train_X, train_y, test_X, test_y, scaler




def ele_preprocessing(n_hours, K):
    """
    n_hours: past window length
    K: forecasting lenght
    """
    dataset = pd.read_csv('../data/TFT_energy.csv', index_col=0)
    dataset.fillna(0, inplace=True)
    print(dataset.columns)
    
    
    # specify the number of lag hours
    n_hours = 24*3
    K = 24
    
    
    # Target time series (1): pollution
    energy_price = dataset[['price']]
    print("energy_price.shape:", energy_price.shape)

    energy_price_supervised = series_to_supervised(energy_price, n_hours, K)
    print("energy_price_supervised.shape:", energy_price_supervised.shape)
    
    
    # Prior known covariates (25):
    prior_known = dataset[['price_dayahead', 'gen_coal', 'gen_gas', 'load_actual',
       'gen_lig', 'gen_oil', 'gen_oth_renew', 'pressure_Barcelona',
       'pressure_Bilbao', 'gen_waste', 'gen_bio', 'temp_min_Valencia',
       'pressure_Valencia', 'temp_min_Barcelona', 'humidity_Seville',
       'wind_deg_Bilbao', 'clouds_all_Bilbao', 'gen_hyd_river',
       'wind_deg_Seville', 'wind_speed_Barcelona', 'wind_speed_Valencia',
       'wind_speed_Bilbao', 'gen_wind', 'wind_speed_Madrid', 'gen_hyd_pump']]
    print("prior_known.shape:", prior_known.shape)

    prior_known_supervised = series_to_supervised(prior_known, n_hours, K)
    print("prior_known_supervised.shape:", prior_known_supervised.shape)


    # all past covariates in past w hours, 25 covariates in total
    past_cov = dataset[['price_dayahead', 'gen_coal', 'gen_gas', 'load_actual',
       'gen_lig', 'gen_oil', 'gen_oth_renew', 'pressure_Barcelona',
       'pressure_Bilbao', 'gen_waste', 'gen_bio', 'temp_min_Valencia',
       'pressure_Valencia', 'temp_min_Barcelona', 'humidity_Seville',
       'wind_deg_Bilbao', 'clouds_all_Bilbao', 'gen_hyd_river',
       'wind_deg_Seville', 'wind_speed_Barcelona', 'wind_speed_Valencia',
       'wind_speed_Bilbao', 'gen_wind', 'wind_speed_Madrid', 'gen_hyd_pump']]

    past_cov_supervised = series_to_supervised(past_cov, n_hours, K)
    print("past_cov_supervised.shape:", past_cov_supervised.shape)
    
    
    # all target energy_price in past w hours, 1 in total
    past_energy_price_supervised = energy_price_supervised.iloc[:, :n_hours*energy_price.shape[1]]
    past_energy_price_supervised.reset_index(drop=True, inplace=True)
    past_energy_price_supervised.columns = ['past_energy_price_supervised_' + i for i in list(past_energy_price_supervised.columns)]
    
    
    # merge all data (covariates, energy_price) in past w hours
    columns = []
    for i in range(n_hours):
        columns = columns + past_cov_supervised.columns[i*past_cov.shape[1]:(i+1)*past_cov.shape[1]].tolist()
        columns = columns + past_energy_price_supervised.columns[i*energy_price.shape[1]:(i+1)*energy_price.shape[1]].tolist()
        
    past_cov_supervised.reset_index(drop=True, inplace=True)
    past_energy_price_supervised.reset_index(drop=True, inplace=True)

    past_cov_energy_price_supervised = pd.concat([past_cov_supervised, past_energy_price_supervised], axis=1)
    past_cov_energy_price_supervised = past_cov_energy_price_supervised[columns]

    
    # shift prior known covariates in future s steps to the past
    shift_prior_known_supervised = prior_known_supervised.iloc[:, K*prior_known.shape[1]:]
    shift_prior_known_supervised.reset_index(drop=True, inplace=True)
    shift_prior_known_supervised.columns = ['shift_prior_known_supervised_' + i for i in list(shift_prior_known_supervised.columns)]
    
    
    # merge all past data and shifted prior know future covariate
    shift_prior_known_past_cov_energy_price_supervised = pd.concat([past_cov_energy_price_supervised, shift_prior_known_supervised], 
                                                     axis=1)
    
    # target pm25 in future k time steps
    future_energy_price_supervised = energy_price_supervised.iloc[:, n_hours*energy_price.shape[1]:]
    future_energy_price_supervised.reset_index(drop=True, inplace=True)
    
    
    # Concatenation (input and labels)
    all_data = concat([shift_prior_known_past_cov_energy_price_supervised, future_energy_price_supervised], axis=1)
    # print("all_data", all_data)
    print("all_data.shape:", all_data.shape)
    
    
    # Train & Test set spliting
    all_data = all_data.values
    n_train_hours = int(len(all_data)*0.8)
    print("n_train_hours:", n_train_hours)


    train = all_data[:n_train_hours, :]
    test = all_data[n_train_hours:, :]
    
    
    # split into input and outputs
    all_features = prior_known.shape[1] + dataset.shape[1]
    n_obs = n_hours * all_features
    train_X, train_y = train[:, :n_obs], train[:, n_obs:]
    test_X, test_y = test[:, :n_obs], test[:, n_obs:]
    print("train_X.shape, train_y.shape, test_X.shape, test_y.shape", train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    train_y = scaler.fit_transform(train_y)
    test_X = scaler.fit_transform(test_X)
    test_y = scaler.fit_transform(test_y)
    
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, all_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, all_features))
    print("train_X.shape, train_y.shape, test_X.shape, test_y.shape:", train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    return train_X, train_y, test_X, test_y, scaler



if __name__ == "__main__":
    ele_preprocessing(n_hours=72, K=24)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    