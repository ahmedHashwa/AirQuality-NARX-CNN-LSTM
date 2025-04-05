import os
from collections import namedtuple

import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression

import hybrid_helper
import hybrid_preprocess

data_dir = "data"
saving_dir = 'results'

# from helper import Dataset, benchmark_algorithm
# from preprocess import process_dir, ReshapeMethod, ScaleMethod


dataframe_columns_targets_manchester_hourly = {
    "NO": ["NO2", "M_T"],
    "NO2": ["NO", "M_T"],
    "O3": ["M_DIR", "M_SPED", "M_T", "NO2"],
    "M_T": ["M_DIR", "M_SPED", "O3"],
    "M_DIR": ["O3", "M_SPED", "M_T"],
    "M_SPED": ["M_T", "O3", "M_DIR"],
    "PM25": ["NO2", "NO", "O3"]
}


def fix_data_manchester_hourly(dataframe: pd.DataFrame, path, name, remove_invalid=True):
    from numpy import nan
    # fix time issue
    dataframe['End Time'] = dataframe['End Time'].replace('24:00:00', '00:00:00')
    # add index column
    dataframe['Index'] = pd.to_datetime(dataframe['End Date'] + ' ' + dataframe['End Time'], format='%Y-%m-%d %H:%M:%S')
    dataframe['Datetime'] = dataframe['Index'].dt.strftime('%d/%m/%Y %H:%M:%S')
    dataframe.set_index('Index', inplace=True)
    dataframe.drop(columns=['End Date', 'End Time'], inplace=True)
    if remove_invalid:
        # remove invalid data in PM 2.5
        dataframe['PM25'] = dataframe['PM25'].apply(lambda x: x if x >= 0 else nan)
    dir_path = os.path.dirname(path)
    dataframe.corr().to_csv(os.path.join(dir_path, f'../{name}/data_corr_pre_process.csv'))
    return dataframe


# noinspection PyPep8Naming,PyShadowingNames
def impute_data_predictive(dataframe: pd.DataFrame, path, name, regressor, prediction_targets_columns=None,
                           remove_invalid=True, preprocess=None):
    dataframe = preprocess(dataframe, path, name, remove_invalid)
    data = pd.DataFrame(dataframe)
    for (target, cols) in prediction_targets_columns.items():  # items() are the elements of the dictionary
        columns = cols.copy()
        columns.append(target)
        df = data[columns]  # take the data corresponding to the required columns
        # df.reset_index(inplace=True)
        test_df = df[df[target].isnull()]  # get the data where target is missing in order to predict the target
        if test_df.empty:  # if there is no data to be used for prediction then skip as this column cannot be imputed
            continue
        df = df.dropna()  # remove empty data to form the training set
        y_train = df[target]  # get the training column at the target
        X_train = df.drop(target, axis=1)  # remove target from the data set to get the training set
        # to get the data from which you will predict the missing values, remove target
        X_test = test_df.drop(target, axis=1)
        # fill missing data in data used for prediction
        X_test.interpolate(method='time', limit_direction='both', axis=0, inplace=True)
        X_test = X_test.dropna(axis=1, how='all')
        if X_test.empty:
            continue
        regressor.fit(X_train[X_test.columns], y_train)
        y_pred = regressor.predict(X_test)
        # replace the missing values with predicted values
        dataframe.loc[dataframe[target].isnull(), target] = y_pred
    return dataframe


# %% Manchester Processing Helpers


def chunk_filter(chunk):
    end_date_start = '01/01/'
    end_time_start = '01'
    return not (chunk['End Time'].iloc[0][:len(end_time_start)] == end_time_start and
                chunk['End Date'].iloc[0][:len(end_date_start)] == end_date_start and
                chunk['PM10'].iloc[0] is None)


def fix_data_china(dataframe: pd.DataFrame, path, name):
    # add index column
    dataframe['Datetime'] = \
        dataframe['day'].map(str) + '/' + dataframe['month'].map(str) + '/' + dataframe['year'].map(str) + ' ' + \
        dataframe['hour'].map(str) + ":00:00"
    dataframe['Index'] = pd.to_datetime(dataframe['Datetime'], format='%d/%m/%Y %H:%M:%S')

    dataframe.set_index('Index', inplace=True)
    dataframe.drop(columns=['year', 'day', 'month', 'hour', 'No'], inplace=True)
    dir_path = os.path.dirname(path)
    dataframe.corr().to_csv(os.path.join(dir_path, f'../{name}/data_corr_pre_process.csv'))
    return dataframe


# Press the green button in the gutter to run the script.
# %% Other Station
# Start preprocessing
from enum import Enum, auto
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost.sklearn import XGBRFRegressor


class ImputeMethod(Enum):
    ExtractedData = auto()
    ExtractedDataHourly = auto()
    RemovedInvalid = auto()
    RemovedInvalidHourly = auto()
    ImputeMean = auto()
    ImputeTimeInterpolate = auto()
    ImputeLinearInterpolate = auto()
    ImputeLinearRegression = auto()
    ImputeRandomForest = auto()
    ImputeRandomForestHourly = auto()
    ImputeXGBRFHourly = auto()


impute_methods = {

    ImputeMethod.ImputeRandomForestHourly:
        lambda df, p, n: impute_data_predictive(df, p, n, regressor=RandomForestRegressor(),
                                                remove_invalid=True, preprocess=fix_data_manchester_hourly,
                                                prediction_targets_columns=dataframe_columns_targets_manchester_hourly),
    ImputeMethod.RemovedInvalidHourly: lambda df, p, n: fix_data_manchester_hourly(df, p, n),
    ImputeMethod.ExtractedDataHourly: lambda df, p, n: fix_data_manchester_hourly(df, p, n, remove_invalid=False),
    ImputeMethod.ImputeXGBRFHourly:
        lambda df, p, n: impute_data_predictive(df, p, n, regressor=XGBRFRegressor(),
                                                remove_invalid=True, preprocess=fix_data_manchester_hourly,
                                                prediction_targets_columns=dataframe_columns_targets_manchester_hourly)
}
skipped_column_names = ['Status/units', 'Unnamed', 'PM10', 'NOXasNO2', 'NV25', 'V25', 'Status']
t_tuple = namedtuple('t_tuple', 'name func')


# %% Process data and prepare for ML

def set_datetime_index(df: DataFrame):
    dropped_column = 'Unnamed: 0'

    if dropped_column in df.columns:
        df.drop(columns=[dropped_column], inplace=True)
    if 'Datetime' in df.columns:
        df['Index'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M:%S')
        df.drop('Datetime', axis=1, inplace=True)
        df.set_index('Index', inplace=True)
        # noinspection PyUnresolvedReferences
        # df.index = df.index.strftime('%d/%m/%Y %H:%M:%S')


# %% prepare data for usage in ML

def pre_process_data(df: DataFrame, ds_name):
    if 'China' not in ds_name:
        df = df.dropna()
        return df
    df = df[24:]
    # df['pm2.5'].fillna(0, inplace=True)
    # df['pm2.5'].fillna(df['pm2.5'].mean(), inplace=True)
    df = df.dropna()
    return df


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime

randomForestImputeMethod = ImputeMethod.ImputeRandomForest
randomForestImputeMethodHourly = ImputeMethod.ImputeRandomForestHourly
XGBRFImputeMethodHourly = ImputeMethod.ImputeXGBRFHourly
extractedDataImputeMethod = ImputeMethod.ExtractedData
removeInvalidHourlyDataImputeMethod = ImputeMethod.RemovedInvalidHourly
hourlyDataMethod = ImputeMethod.ExtractedDataHourly

manchester_hourly1519_data_XGBRF_imputed = hybrid_preprocess.process_dir(
    f'{data_dir}/Manchester/Piccadilly/AirMeteoHourly1519/',
    skip_rows=4,
    skipped_column_names=skipped_column_names,
    chunk_size=10000, skip_existing=True,
    t_tuple=t_tuple('XGBRFImputeMethodHourly1519',
                    impute_methods[
                        XGBRFImputeMethodHourly]))

china_data_extracted = hybrid_preprocess.process_dir(f'{data_dir}/China/Data', chunk_size=10000, skip_rows=0,
                                                     skip_existing=True,
                                                     t_tuple=t_tuple(ImputeMethod.ExtractedData.name,
                                                                     lambda df, p, n: fix_data_china(df, p, n)))

datasets = [
    hybrid_helper.Dataset(name='CHN_Ext', data=china_data_extracted,
                          feature_columns=['pm2.5', 'Iws', 'Ir', 'Datetime'], target_columns=['pm2.5'],
                          include=True),

    hybrid_helper.Dataset(name='MCR_H_XGBRF_Imputed',
                          feature_columns=['PM25', 'NO', 'NO2', 'M_DIR', 'M_T', 'Datetime'],
                          data=manchester_hourly1519_data_XGBRF_imputed,
                          target_columns=['PM25'], include=True),

]

for ds in datasets:
    if ds.include:
        set_datetime_index(ds.data)
    else:
        del ds.data

import hybrid_algorithms
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.svm import SVR
import fireTS.models

# import fbprophet

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import hybrid_metrics

evaluation_methods = {
    'MSE': mean_squared_error,
    'RMSE': hybrid_metrics.rmse,
    'NRMSE': hybrid_metrics.nrmse,
    'MAE': mean_absolute_error,
    'R2': r2_score,
    'IA': hybrid_metrics.index_agreement,
    'Pearson R': lambda observed, predicted: pearsonr(observed, predicted)[0]
}

session_start_datetime = datetime.now()

regressors = {}


def f_list(lst):
    return f'[{", ".join(f"{x:02d}" for x in lst)}]'


def key(run_index, **kwargs):
    from box import Box
    global algorithm_index
    global regressors
    regressor_name = kwargs["regressor_name"]

    if regressor_name not in regressors.keys():
        regressors[regressor_name] = algorithm_index
        algorithm_index = algorithm_index + 1
    regressor_name = f'{regressors[regressor_name]:02d}_{regressor_name}'

    if any(n in regressor_name for n in ["NARX", "DAR"]):
        kwargs["regressor_name"] = f'{regressor_name}_ao_{look_back:02d}_ed_{f_list(e_delay)}_eo_{f_list(e_order)}'
    else:
        kwargs['regressor_name'] = regressor_name

    kwargs['overwrite_folder'] = overwrite_folder
    kwargs['max_limit'] = max_limit
    kwargs['look_back'] = look_back if not any(n in regressor_name for n in ["NARX", "DAR"]) else 0
    kwargs['n_estimators'] = n_estimators
    kwargs['session_start_datetime'] = session_start_datetime
    kwargs['dropout_rate'] = dropout_rate
    kwargs['n_lstm_nodes'] = n_lstm_nodes
    kwargs['activation'] = activation
    kwargs['n_dense_nodes'] = n_dense_nodes
    kwargs['batch_size'] = batch_size
    kwargs['evaluation_methods'] = evaluation_methods
    kwargs['index'] = run_index
    kwargs['dataset_name'] = ds.name
    kwargs['dataset'] = ds
    kwargs['results_dir'] = saving_dir
    kwargs['round_all_results'] = round_all_results

    kwargs['target_column'] = target_column
    kwargs['datetime'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    requested_key = Box(kwargs, frozen_box=True)
    print(
        f'Training and Predicting {requested_key.regressor_name} on '
        f'{requested_key.dataset_name} Data {ds.data.shape} '
        f'[trn {X_train.shape},{y_train.shape}] [tst {X_test.shape},{y_test.shape}] \n'
        f'targeting {target_column} iteration '
        f'NO: {requested_key.index} / {iterations_count} '
        f'on {requested_key.datetime}')
    return requested_key


last_folder_name = 'ts_10_MCR_1519_XGBRF_Imp_NO_GPU'
overwrite_folder = False

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

algorithm_index = 1
batch_size = 72
epochs = 25
activation = 'tanh'
# activation = 'relu'
round_all_results = True
# ###########################################
enable_CNNLSTM = True
enable_LSTM = True
enable_ET = True
enable_XGBRF = True
# -------------------------------------------
enable_LSTM_dropout = False
enable_RF = False
enable_SVR = False
enable_GB = False
enable_XGB = False
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
enable_XGBRF_DART = False
# ###########################################
enable_NARX_CNNLSTM = True
enable_NARX_LSTM = True
enable_NARX_ET = True
enable_NARX_XGBRF = True
# -------------------------------------------
enable_NARX_LSTM_dropout = False
enable_NARX_RF = False
enable_NARX_SVR = False
enable_NARX_GB = False
enable_NARX_XGB = False
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
enable_NARX_XGBRF_DART = False
enable_DAR_XGBRF = False
enable_DAR_CNNLSTM = False
enable_DAR_LSTM = False

n_estimators = 100
regressor_parameters = {'n_estimators': n_estimators, 'n_jobs': -1, 'verbose': 3}
iterations_count = 10
n_lstm_nodes = 128
n_dense_nodes = 50
dropout_rate = 0.01
look_back = 24
n_subsequences = 4
scaler_method = hybrid_preprocess.ScaleMethod.NoScaler
lstm_scaler_method = hybrid_preprocess.ScaleMethod.StandardScaler
# lstm_scaler_method = hybrid_preprocess.ScaleMethod.NoScaler
scale_target = False
lstm_scale_target = True
max_limit = 4279
data_split_mode = hybrid_preprocess.SplitMode.KFoldTimeSeries


# data_split_mode = hybrid_preprocess.SplitMode.KFold


# %% Run Code
def fit_process(m, x, y):
    res = m.fit(x, y, epochs=epochs, batch_size=batch_size)
    return res


def predict_process(m, x, _=None, __=None):
    if m is (hybrid_algorithms.CNNLSTMModel or hybrid_algorithms.LSTMModel):
        predicted = m.predict(x, batch_size=batch_size)
    else:
        predicted = m.predict(x)

    return predicted


i: int = 0

for ds in [d for d in datasets if d.include]:
    for target_column in ds.target_columns:

        for X_train, X_test, y_train, y_test, num_features, i in \
                hybrid_preprocess.split_data(target_column=target_column, ds=ds, look_back=look_back,
                                             split_mode=data_split_mode,
                                             iterations_count=iterations_count,
                                             pre_process_data=pre_process_data):

            if enable_CNNLSTM:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=hybrid_algorithms.CNNLSTMModel(
                                                      n_lstm_nodes=n_lstm_nodes,
                                                      n_dense_nodes=n_dense_nodes,
                                                      dropout_rate=0,
                                                      activation=activation), num_features=num_features,
                                                  n_subsequences=n_subsequences, scale_target=lstm_scale_target,
                                                  fit_process=fit_process,
                                                  predict_process=predict_process,
                                                  scale_features_method=lstm_scaler_method,
                                                  reshape_features_method=hybrid_preprocess.ReshapeMethod.FourDShape,
                                                  method_key=key(i, regressor_name='CNNLSTM'),
                                                  last_folder_name=last_folder_name)
            if enable_LSTM:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=hybrid_algorithms.LSTMModel(
                                                      n_lstm_nodes=n_lstm_nodes,
                                                      n_dense_nodes=n_dense_nodes,
                                                      dropout_rate=0,
                                                      activation=activation), num_features=num_features,
                                                  n_subsequences=n_subsequences, scale_target=lstm_scale_target,
                                                  fit_process=fit_process,
                                                  predict_process=predict_process,
                                                  scale_features_method=lstm_scaler_method,
                                                  reshape_features_method=hybrid_preprocess.ReshapeMethod.ThreeDShape,
                                                  method_key=key(i, regressor_name='LSTM'),
                                                  last_folder_name=last_folder_name)
            if enable_LSTM_dropout:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=hybrid_algorithms.LSTMModel(
                                                      n_lstm_nodes=n_lstm_nodes,
                                                      n_dense_nodes=n_dense_nodes,
                                                      dropout_rate=dropout_rate,
                                                      activation=activation), num_features=num_features,
                                                  n_subsequences=n_subsequences, scale_target=lstm_scale_target,
                                                  fit_process=fit_process,
                                                  predict_process=predict_process,
                                                  scale_features_method=lstm_scaler_method,
                                                  reshape_features_method=hybrid_preprocess.ReshapeMethod.ThreeDShape,
                                                  method_key=key(i, regressor_name='LSTM_dropout'),
                                                  last_folder_name=last_folder_name)
            if enable_RF:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=RandomForestRegressor(
                                                      **regressor_parameters), scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='RandomForest'),
                                                  last_folder_name=last_folder_name)
            if enable_ET:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=ExtraTreesRegressor(
                                                      **regressor_parameters), scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='ExtraTrees'),
                                                  last_folder_name=last_folder_name)
            if enable_XGB:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=XGBRegressor(
                                                      n_estimators=n_estimators, verbosity=3),
                                                  scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='XGB'),
                                                  last_folder_name=last_folder_name)
            if enable_XGBRF:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=XGBRFRegressor(
                                                      n_estimators=n_estimators, verbosity=3),
                                                  scale_target=scale_target,
                                                  predict_process=predict_process,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='XGBRF'),
                                                  last_folder_name=last_folder_name)
            if enable_XGBRF_DART:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=XGBRFRegressor(
                                                      n_estimators=n_estimators, verbosity=3,
                                                      booster='dart'), scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='XGBRF_dart'),
                                                  last_folder_name=last_folder_name)
            if enable_GB:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=GradientBoostingRegressor(
                                                      n_estimators=n_estimators,
                                                      verbose=3), scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='GradientBoosting'),
                                                  last_folder_name=last_folder_name)
            if enable_SVR:
                hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                  predictor_object=SVR(C=2.0, epsilon=0.1,
                                                                       kernel='rbf', gamma=0.5,
                                                                       tol=0.001, verbose=True,
                                                                       shrinking=True,
                                                                       max_iter=10000), scale_target=scale_target,
                                                  scale_features_method=scaler_method,
                                                  method_key=key(i, regressor_name='SVR'),
                                                  last_folder_name=last_folder_name)


# noinspection PyPep8Naming
def NARX_predict_process(reg: fireTS.models.NARX, x, y):
    return reg.predict(x, y)


# noinspection PyPep8Naming
def DAR_predict_process(reg, x, y):
    return reg.predict(x, y.reshape(-1))


# 1, 4, 6, 8, 12, 24, 48 / 1,6
# exog_order_values = [1, 4, look_back]
# exog_delay_values = [0, 4]

NARX_exo_o_d = [
    # [[0], [0]],
    [[1], [0]],
    [[4], [0]],
    [[1], [8]],
    [[24], [0]]
]


# noinspection PyPep8Naming
def NARX_predictor(regressor, exog_order, exog_delay):
    res = fireTS.models.NARX(regressor,
                             auto_order=look_back,
                             exog_order=exog_order,
                             exog_delay=exog_delay)
    return res


# NARX Algorithms
# has to be run alone as graphics dependent algorithms has to be run consecutively
# import itertools
import numpy as np

for ds in [d for d in datasets if d.include]:
    for target_column in ds.target_columns:
        # split the data
        for X_train, X_test, y_train, y_test, num_features, i in \
                hybrid_preprocess.split_data(target_column=target_column, ds=ds, look_back=0,
                                             split_mode=data_split_mode,
                                             iterations_count=iterations_count,
                                             pre_process_data=pre_process_data):

            for e_order, e_delay in ((list(np.stack(o * X_train.shape[1])), list(np.stack(d * X_train.shape[1]))) for
                                     o, d in NARX_exo_o_d):

                # for e_order in [list(x) for x in list(itertools.product(exog_order_values, repeat=num_features))]:
                #     for e_delay in [list(x) for x in
                #                     list(itertools.product(exog_delay_values, repeat=num_features))]:
                # Normalize the data if required by the algorithms
                # Pass the data to the algorithm
                if enable_NARX_LSTM:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=NARX_predictor(
                                                          hybrid_algorithms.LSTMModel(n_lstm_nodes=n_lstm_nodes,
                                                                                      n_dense_nodes=n_dense_nodes,
                                                                                      dropout_rate=0,
                                                                                      activation=activation),

                                                          exog_order=e_order,
                                                          exog_delay=e_delay),
                                                      scale_target=lstm_scale_target,
                                                      fit_process=fit_process,
                                                      predict_process=NARX_predict_process,
                                                      scale_features_method=lstm_scaler_method,
                                                      method_key=key(i, regressor_name='NARX_LSTM'),
                                                      last_folder_name=last_folder_name)
                if enable_NARX_CNNLSTM:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=NARX_predictor(
                                                          hybrid_algorithms.CNNLSTMModel(
                                                              n_lstm_nodes=n_lstm_nodes,
                                                              n_dense_nodes=n_dense_nodes,
                                                              dropout_rate=0,
                                                              activation=activation,
                                                              num_features=num_features,
                                                              n_subsequences=n_subsequences,
                                                              look_back=look_back),
                                                          exog_order=e_order,
                                                          exog_delay=e_delay),
                                                      scale_target=lstm_scale_target,
                                                      fit_process=fit_process,
                                                      predict_process=NARX_predict_process,
                                                      scale_features_method=lstm_scaler_method,
                                                      method_key=key(i, regressor_name='NARX_CNNLSTM'),
                                                      last_folder_name=last_folder_name)
                if enable_NARX_LSTM_dropout:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=NARX_predictor(
                                                          hybrid_algorithms.LSTMModel(n_lstm_nodes=n_lstm_nodes,
                                                                                      n_dense_nodes=n_dense_nodes,
                                                                                      dropout_rate=dropout_rate,
                                                                                      activation=activation),
                                                          exog_order=e_order,
                                                          exog_delay=e_delay),
                                                      scale_target=lstm_scale_target,
                                                      fit_process=fit_process, predict_process=NARX_predict_process,
                                                      scale_features_method=lstm_scaler_method,
                                                      method_key=key(i, regressor_name='NARX_LSTM_dropout'),
                                                      last_folder_name=last_folder_name)
                if enable_NARX_RF:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=NARX_predictor(
                                                          RandomForestRegressor(**regressor_parameters),
                                                          exog_order=e_order,
                                                          exog_delay=e_delay),
                                                      scale_target=scale_target,
                                                      predict_process=NARX_predict_process,
                                                      scale_features_method=scaler_method,
                                                      method_key=key(i, regressor_name='NARX_RandomForest'),
                                                      last_folder_name=last_folder_name)
                if enable_NARX_ET:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=NARX_predictor(
                                                          ExtraTreesRegressor(**regressor_parameters),
                                                          exog_order=e_order,
                                                          exog_delay=e_delay),
                                                      scale_target=scale_target,
                                                      predict_process=NARX_predict_process,
                                                      scale_features_method=scaler_method,
                                                      method_key=key(i, regressor_name='NARX_ExtraTrees'),
                                                      last_folder_name=last_folder_name)
                if enable_NARX_XGB:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=NARX_predictor(
                                                          XGBRegressor(n_estimators=100, n_jobs=-1,
                                                                       verbosity=3),

                                                          exog_order=e_order,
                                                          exog_delay=e_delay),
                                                      scale_target=scale_target,
                                                      predict_process=NARX_predict_process,
                                                      scale_features_method=scaler_method,
                                                      method_key=key(i, regressor_name='NARX_XGB'),
                                                      last_folder_name=last_folder_name)
                if enable_NARX_XGBRF:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=NARX_predictor(
                                                          XGBRFRegressor(n_estimators=100, n_jobs=-1,
                                                                         verbosity=3),

                                                          exog_order=e_order,
                                                          exog_delay=e_delay),
                                                      scale_target=scale_target,
                                                      predict_process=NARX_predict_process,
                                                      scale_features_method=scaler_method,
                                                      method_key=key(i, regressor_name='NARX_XGBRF'),
                                                      last_folder_name=last_folder_name)
                if enable_NARX_GB:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=NARX_predictor(GradientBoostingRegressor(
                                                          n_estimators=n_estimators, verbose=3),
                                                          exog_order=e_order, exog_delay=e_delay),
                                                      scale_target=scale_target,
                                                      predict_process=NARX_predict_process,
                                                      scale_features_method=scaler_method,
                                                      method_key=key(i, regressor_name='NARX_GradientBoosting'),
                                                      last_folder_name=last_folder_name)
                if enable_NARX_SVR:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=NARX_predictor(
                                                          SVR(C=2.0, epsilon=0.1, kernel='rbf',
                                                              gamma=0.5,
                                                              tol=0.001, verbose=True,
                                                              shrinking=True,
                                                              max_iter=10000),

                                                          exog_order=e_order,
                                                          exog_delay=e_delay),
                                                      scale_target=scale_target,
                                                      predict_process=NARX_predict_process,
                                                      scale_features_method=scaler_method,
                                                      method_key=key(i, regressor_name='NARX_SVR'),
                                                      last_folder_name=last_folder_name)
                if enable_NARX_XGBRF_DART:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=NARX_predictor(
                                                          XGBRFRegressor(n_estimators=n_estimators,
                                                                         verbosity=3,
                                                                         booster='dart'),

                                                          exog_order=e_order,
                                                          exog_delay=e_delay),
                                                      scale_target=scale_target,
                                                      predict_process=NARX_predict_process,
                                                      scale_features_method=scaler_method,
                                                      method_key=key(i, regressor_name='NARX_XGBRF_dart'),
                                                      last_folder_name=last_folder_name)
                if enable_DAR_XGBRF:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=fireTS.models.DirectAutoRegressor(
                                                          XGBRFRegressor(n_estimators=n_estimators,
                                                                         n_jobs=-1, verbosity=3),
                                                          auto_order=look_back,
                                                          exog_order=e_order,
                                                          exog_delay=e_delay,
                                                          pred_step=1),
                                                      scale_target=scale_target,
                                                      predict_process=lambda reg, x, y: reg.predict(x,
                                                                                                    y.reshape(
                                                                                                        -1)),
                                                      scale_features_method=scaler_method,
                                                      method_key=key(i, regressor_name='DAR_XGBRF'),
                                                      last_folder_name=last_folder_name)

                if enable_DAR_CNNLSTM:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=fireTS.models.DirectAutoRegressor(
                                                          hybrid_algorithms.CNNLSTMModel(
                                                              n_lstm_nodes=n_lstm_nodes,
                                                              n_dense_nodes=n_dense_nodes,
                                                              dropout_rate=0,
                                                              activation=activation,
                                                              num_features=num_features,
                                                              n_subsequences=n_subsequences,
                                                              look_back=look_back),
                                                          auto_order=look_back,
                                                          exog_order=e_order,
                                                          exog_delay=e_delay,
                                                          pred_step=1),
                                                      scale_target=lstm_scale_target,
                                                      fit_process=fit_process,
                                                      predict_process=DAR_predict_process,
                                                      scale_features_method=lstm_scaler_method,
                                                      method_key=key(i, regressor_name='DAR_CNNLSTM'),
                                                      last_folder_name=last_folder_name)
                if enable_DAR_LSTM:
                    hybrid_helper.benchmark_algorithm(X_train=X_train, y_train=y_train, X_test=X_test,
                                                      y_test=y_test,
                                                      predictor_object=fireTS.models.DirectAutoRegressor(
                                                          hybrid_algorithms.LSTMModel(n_lstm_nodes=n_lstm_nodes,
                                                                                      n_dense_nodes=n_dense_nodes,
                                                                                      dropout_rate=0,
                                                                                      activation=activation),
                                                          auto_order=look_back,
                                                          exog_order=e_order,
                                                          exog_delay=e_delay,
                                                          pred_step=1),
                                                      scale_target=lstm_scale_target,
                                                      fit_process=fit_process,
                                                      predict_process=DAR_predict_process,
                                                      scale_features_method=lstm_scaler_method,
                                                      method_key=key(i, regressor_name='DAR_LSTM'),
                                                      last_folder_name=last_folder_name)

# %% Joining results
from os import listdir
from os.path import join
import pandas as pd

formatted_datetime = last_folder_name if last_folder_name != '' else session_start_datetime.strftime(
    "%d-%m-%Y_%H_%M_%S")
limit = 10

pd.set_option('display.max_columns', 500)
data_dir = f'{saving_dir}/{formatted_datetime}'
data_result_save_dir = f'{saving_dir}/{formatted_datetime}'
for ds in [d for d in datasets if d.include]:
    for target in ds.target_columns:
        results_dir = f'{data_dir}'
        for i in range(1, iterations_count + 1):
            first_file = None
            for file in sorted(listdir(results_dir), reverse=True):
                file = join(results_dir, file)
                if f'i_{i:02d}_predictions' in file and f'{ds.name}' in file and f'_{target}_' in file:
                    # noinspection DuplicatedCode
                    cols = list(pd.read_csv(file, nrows=1))
                    print(f'Reading {file}')

                    if first_file is None:
                        first_file = pd.read_csv(file, index_col=0,
                                                 usecols=[c for c in cols if 'aqi' not in c])
                    else:

                        import math

                        read_file = pd.read_csv(file, index_col=0,
                                                usecols=[c for c in cols if 'aqi' not in c])
                        temp = read_file[f"{target}_Real"].apply(lambda x: math.ceil(x)).rolling(
                            window=limit).apply(
                            lambda x: (x == first_file[f"{target}_Real"][0:limit].apply(
                                lambda y: math.ceil(y)).values).all(),
                            raw=True).mask(lambda x: x == 0).bfill(limit=limit).dropna()
                        cutOffIndex = 0 if temp.empty else temp.index[0] + 1
                        read_file = read_file if cutOffIndex == 0 or cutOffIndex == 1 else read_file.iloc[
                                                                                           cutOffIndex:]
                        read_file.reset_index(inplace=True)
                        read_file.drop(columns=['index', f'{target}_Real'], inplace=True)
                        first_file = first_file.join(read_file)
            first_file.to_csv(join(results_dir, f'run_{ds.name}_{target}_i{i:02d}'
                                                f'_data_{formatted_datetime}.csv'))
# %% Save Summary
summary_columns = ["Regressor Name", "Training Period Seconds", "Prediction Period Seconds", "MSE", "RMSE", "NRMSE",
                   "MAE", "R2", "IA", "Pearson R", "Step"]

for ds in [d for d in datasets if d.include]:
    for target in ds.target_columns:
        all_metrics_results = None
        results_dir = f'{data_dir}'
        for i in range(1, iterations_count + 1):
            for file in listdir(results_dir):
                file = join(results_dir, file)
                if f'i_{i:02d}_metrics' in file and f'{ds.name}' in file and f'_{target}_' in file:
                    if all_metrics_results is None:
                        all_metrics_results = pd.read_csv(file, index_col=0)
                    else:
                        metrics = pd.read_csv(file, index_col=0)
                        all_metrics_results = pd.concat([all_metrics_results, metrics])
        all_metrics_results.to_csv(
            join(results_dir, f'all_metrics_{ds.name}_{target}_data_{formatted_datetime}.csv'))
        grouped_results_mean = all_metrics_results[summary_columns].groupby('Regressor Name').mean()
        grouped_results_mean.to_csv(
            join(results_dir, f'all_metrics_mean_{ds.name}_{target}_data_{formatted_datetime}.csv'))

        print(grouped_results_mean)

for ds in [d for d in datasets if d.include]:
    for target in ds.target_columns:
        all_mean_metrics_results = None
        results_dir = f'{data_dir}'
        for i in range(1, iterations_count + 1):
            for file in listdir(results_dir):
                file = join(results_dir, file)
                if f'all_metrics_mean_' in file and f'{ds.name}' in file and f'_{target}_' in file:
                    if all_mean_metrics_results is None:
                        all_mean_metrics_results = pd.read_csv(file, index_col=0)
                    else:
                        metrics = pd.read_csv(file, index_col=0)
                        all_mean_metrics_results = pd.concat([all_mean_metrics_results, metrics])
        all_mean_metrics_results.to_csv(
            join(data_dir, f'all_steps_mean_metrics_{ds.name}_{target}_data_{formatted_datetime}.csv'))
        grouped_results_mean = all_mean_metrics_results.groupby('Regressor Name').mean()
        grouped_results_mean.to_csv(
            join(data_dir, f'all_mean_metrics_all_steps_{ds.name}_{target}_data_{formatted_datetime}.csv'))

        print(f"Printing Mean Results for all steps: ")
        print(grouped_results_mean)
