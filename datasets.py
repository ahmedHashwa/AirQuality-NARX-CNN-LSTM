import os
from collections import namedtuple
from enum import Enum, auto

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRFRegressor

import hybrid_helper
import hybrid_preprocess


dataframe_columns_targets_manchester_hourly = {
    "NO": ["NO2", "M_T"],
    "NO2": ["NO", "M_T"],
    "O3": ["M_DIR", "M_SPED", "M_T", "NO2"],
    "M_T": ["M_DIR", "M_SPED", "O3"],
    "M_DIR": ["O3", "M_SPED", "M_T"],
    "M_SPED": ["M_T", "O3", "M_DIR"],
    "PM25": ["NO2", "NO", "O3"],
}


def fix_data_manchester_hourly(dataframe: pd.DataFrame, path, name, remove_invalid=True):
    from numpy import nan
    dataframe['End Time'] = dataframe['End Time'].replace('24:00:00', '00:00:00')
    dataframe['Index'] = pd.to_datetime(
        dataframe['End Date'] + ' ' + dataframe['End Time'],
        format='%Y-%m-%d %H:%M:%S'
    )
    dataframe['Datetime'] = dataframe['Index'].dt.strftime('%d/%m/%Y %H:%M:%S')
    dataframe.set_index('Index', inplace=True)
    dataframe.drop(columns=['End Date', 'End Time'], inplace=True)
    if remove_invalid:
        dataframe['PM25'] = dataframe['PM25'].apply(lambda x: x if x >= 0 else nan)
    dir_path = os.path.dirname(path)
    dataframe.corr().to_csv(os.path.join(dir_path, f'../{name}/data_corr_pre_process.csv'))
    return dataframe


def impute_data_predictive(dataframe: pd.DataFrame, path, name, regressor, prediction_targets_columns=None,
                           remove_invalid=True, preprocess=None):
    dataframe = preprocess(dataframe, path, name, remove_invalid)
    data = pd.DataFrame(dataframe)
    for target, cols in prediction_targets_columns.items():
        columns = cols.copy()
        columns.append(target)
        df = data[columns]
        test_df = df[df[target].isnull()]
        if test_df.empty:
            continue
        df = df.dropna()
        y_train = df[target]
        X_train = df.drop(target, axis=1)
        X_test = test_df.drop(target, axis=1)
        X_test.interpolate(method='time', limit_direction='both', axis=0, inplace=True)
        X_test = X_test.dropna(axis=1, how='all')
        if X_test.empty:
            continue
        regressor.fit(X_train[X_test.columns], y_train)
        y_pred = regressor.predict(X_test)
        dataframe.loc[dataframe[target].isnull(), target] = y_pred
    return dataframe


def chunk_filter(chunk):
    end_date_start = '01/01/'
    end_time_start = '01'
    return not (
        chunk['End Time'].iloc[0][:len(end_time_start)] == end_time_start and
        chunk['End Date'].iloc[0][:len(end_date_start)] == end_date_start and
        chunk['PM10'].iloc[0] is None
    )


def fix_data_china(dataframe: pd.DataFrame, path, name):
    dataframe['Datetime'] = (
        dataframe['day'].map(str) + '/' + dataframe['month'].map(str) + '/' + dataframe['year'].map(str) +
        ' ' + dataframe['hour'].map(str) + ":00:00"
    )
    dataframe['Index'] = pd.to_datetime(dataframe['Datetime'], format='%d/%m/%Y %H:%M:%S')
    dataframe.set_index('Index', inplace=True)
    dataframe.drop(columns=['year', 'day', 'month', 'hour', 'No'], inplace=True)
    dir_path = os.path.dirname(path)
    dataframe.corr().to_csv(os.path.join(dir_path, f'../{name}/data_corr_pre_process.csv'))
    return dataframe


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


tuple_type = namedtuple('t_tuple', 'name func')

impute_methods = {
    ImputeMethod.ImputeRandomForestHourly:
        lambda df, p, n: impute_data_predictive(
            df, p, n,
            regressor=RandomForestRegressor(),
            remove_invalid=True,
            preprocess=fix_data_manchester_hourly,
            prediction_targets_columns=dataframe_columns_targets_manchester_hourly,
        ),
    ImputeMethod.RemovedInvalidHourly: lambda df, p, n: fix_data_manchester_hourly(df, p, n),
    ImputeMethod.ExtractedDataHourly: lambda df, p, n: fix_data_manchester_hourly(df, p, n, remove_invalid=False),
    ImputeMethod.ImputeXGBRFHourly:
        lambda df, p, n: impute_data_predictive(
            df, p, n,
            regressor=XGBRFRegressor(),
            remove_invalid=True,
            preprocess=fix_data_manchester_hourly,
            prediction_targets_columns=dataframe_columns_targets_manchester_hourly,
        ),
}

skipped_column_names = ['Status/units', 'Unnamed', 'PM10', 'NOXasNO2', 'NV25', 'V25', 'Status']

def set_datetime_index(df: DataFrame) -> DataFrame:
    """Ensure the dataframe uses a datetime index.

    A copy of ``df`` is returned to avoid ``SettingWithCopyWarning`` when ``df``
    originates from slicing operations.
    """

    df = df.copy()
    dropped_column = 'Unnamed: 0'
    if dropped_column in df.columns:
        df.drop(columns=[dropped_column], inplace=True)
    if 'Datetime' in df.columns:
        df.loc[:, 'Index'] = pd.to_datetime(
            df['Datetime'], format='%d/%m/%Y %H:%M:%S')
        df.drop('Datetime', axis=1, inplace=True)
        df.set_index('Index', inplace=True)
    return df


def pre_process_data(df: DataFrame, ds_name):
    if 'China' not in ds_name:
        df = df.dropna()
        return df
    df = df[24:]
    df = df.dropna()
    return df


def load_datasets(data_dir):
    manchester_hourly1519_data_XGBRF_imputed = hybrid_preprocess.process_dir(
        f'{data_dir}/Manchester/Piccadilly/AirMeteoHourly1519/',
        skip_rows=4,
        skipped_column_names=skipped_column_names,
        chunk_size=10000,
        skip_existing=True,
        t_tuple=tuple_type('XGBRFImputeMethodHourly1519', impute_methods[ImputeMethod.ImputeXGBRFHourly])
    )

    china_data_extracted = hybrid_preprocess.process_dir(
        f'{data_dir}/China/Data',
        chunk_size=10000,
        skip_rows=0,
        skip_existing=True,
        t_tuple=tuple_type(ImputeMethod.ExtractedData.name, lambda df, p, n: fix_data_china(df, p, n))
    )

    datasets = [
        hybrid_helper.Dataset(
            name='CHN_Ext',
            data=china_data_extracted,
            feature_columns=['pm2.5', 'Iws', 'Ir', 'Datetime'],
            target_columns=['pm2.5'],
            include=True,
        ),
        hybrid_helper.Dataset(
            name='MCR_H_XGBRF_Imputed',
            feature_columns=['PM25', 'NO', 'NO2', 'M_DIR', 'M_T', 'Datetime'],
            data=manchester_hourly1519_data_XGBRF_imputed,
            target_columns=['PM25'],
            include=True,
        ),
    ]

    for ds in datasets:
        if ds.include:
            ds.data = set_datetime_index(ds.data)
        else:
            del ds.data
    return datasets
