from abc import ABC

from sklearn.model_selection._split import _BaseKFold


def process_dir(data_dir, t_tuple=None, skip_existing=True, save_statistics=False,
                save_year=False, save_data=True, save_data_corr=True, chunk_filter=None, chunk_size=24, skip_rows=5,
                skipped_column_names=None):
    from os import listdir, mkdir
    from os.path import isfile, join, exists
    import pandas as pd
    print(f'Starting {t_tuple.name} ...')
    if t_tuple is None:
        data_result_dir = join(data_dir, '../')
    else:
        data_result_dir = join(data_dir, f'../{t_tuple.name}')
        if not exists(data_result_dir):
            mkdir(data_result_dir)
    all_data_file = join(data_result_dir, 'all_data.csv')
    if skip_existing and exists(all_data_file):
        print(f'Data already exists in {all_data_file}. skipping ...')
        saved_data = pd.read_csv(all_data_file)
        return saved_data
    data_files = [f for f in listdir(data_dir)
                  if isfile(join(data_dir, f))]
    all_data = pd.DataFrame()
    for f in data_files:

        data = extract_data(join(data_dir, f), t_tuple, chunk_filter=chunk_filter, chunk_size=chunk_size,
                            skip_rows=skip_rows, skipped_column_names=skipped_column_names)
        print(data.head(5))
        if save_statistics:
            data.describe().to_csv(join(data_result_dir, f'../statistics_{f}'))
        if save_year:
            data.to_csv(join(data_result_dir, '../processed_data_' + f))
        all_data = all_data.append(data, ignore_index=False)

    if save_statistics:
        all_data.describe().to_csv(join(data_result_dir, 'all_data_statistics.csv'))
    if save_data_corr:
        all_data.corr().to_csv(join(data_result_dir, 'corr.csv'))
    if save_data:
        all_data.to_csv(all_data_file)
    return all_data


def extract_data(path: str, file_data_transformer=None, chunk_filter=None, chunk_size=24, skip_rows=5,
                 skipped_column_names=None):
    """
    :param skip_rows:
    :param chunk_size:
    :param chunk_filter:
    :param skipped_column_names:
    :param file_data_transformer: the function required to transform the data in a file corresponding to a year
    :param path: the directory containing the file downloaded from UK pollution website


    """
    import itertools as it
    import pandas as pd
    # Skip introductory lines in datafile

    chunks = pd.read_csv(path, engine='python', chunksize=chunk_size, skiprows=skip_rows)
    chunks = it.takewhile(chunk_filter if chunk_filter is not None else lambda chunk: True, chunks)
    data = pd.concat(chunks)

    # Remove units columns and other non required columns

    columns = data.columns if skipped_column_names is None else \
        [c for c in data.columns if all(
            c.lower()[:len(s)] != s.lower() for s in skipped_column_names)]
    data = data[columns]
    if file_data_transformer is not None and file_data_transformer.func:
        data = file_data_transformer.func(data, path, file_data_transformer.name)
    return data


from enum import Enum, auto


class ScaleMethod(Enum):
    NoScaler = auto()
    MinMaxScaler = auto()
    StandardScaler = auto()


class ReshapeMethod(Enum):
    NoReshape = auto()
    TwoDShape = auto()
    ThreeDShape = auto()
    FourDShape = auto()


class SplitMode(Enum):
    KFold = auto()
    KFoldTimeSeries = auto()
    BlockingTimeSeriesSplit = auto()
    MonthsIntervals = auto()


def encode_non_numeric_columns(df):
    for column in df.columns:
        if df[column].dtype.name == object.__name__:
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
        df[column] = df[column].values.astype('float32')


def relocate_target(df, target_column):
    # insert target column as the first column in the dataframe
    df.insert(0, 'target', df[target_column])
    df.drop(columns=[target_column], inplace=True)  # Drop the original shifted_target column


def shift_dataset(data, look_back, remove_original_data=True):
    import pandas as pd
    ds = pd.DataFrame(data)
    if remove_original_data:
        del data
    if look_back == 0:
        return ds.iloc[:, 1:], ds.iloc[:, 0]
        # values = ds.values.astype('float32')
        # return values[:, 1:], values[:, 0]
    n_features = len(ds.columns) - 1
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(look_back, 0, -1):
        cols.append(ds.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, 1):
        cols.append(ds.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # agg = agg.values.astype('float32')
    agg = agg.iloc[look_back:, :-1 * n_features]
    return agg.iloc[:, :-1], agg.iloc[:, -1]


# noinspection PyPep8Naming
def scale_data(X_train, y_train, X_test, y_test, scale_target=False,
               scale_features_method: ScaleMethod = ScaleMethod.NoScaler, dropna=True):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    train_feature_scaler = None
    train_target_scaler = None
    test_feature_scaler = None
    test_target_scaler = None
    X_train_out, y_train_out, X_test_out, y_test_out = X_train, y_train, X_test, y_test
    if scale_features_method is not ScaleMethod.NoScaler:
        if scale_features_method is ScaleMethod.StandardScaler:
            train_feature_scaler = StandardScaler(with_mean=True, with_std=True)
            train_target_scaler = StandardScaler(with_mean=True, with_std=True)
            test_feature_scaler = StandardScaler(with_mean=True, with_std=True)
            test_target_scaler = StandardScaler(with_mean=True, with_std=True)
        elif scale_features_method is ScaleMethod.MinMaxScaler:
            train_feature_scaler = MinMaxScaler(feature_range=(0, 1))
            train_target_scaler = MinMaxScaler(feature_range=(0, 1))
            test_feature_scaler = MinMaxScaler(feature_range=(0, 1))
            test_target_scaler = MinMaxScaler(feature_range=(0, 1))

        X_train_out = train_feature_scaler.fit_transform(X_train)
        X_test_out = test_feature_scaler.fit_transform(X_test)

        if scale_target:
            y_train_out = y_train.T.reshape(-1, 1)
            y_train_out = train_target_scaler.fit_transform(y_train_out)
            y_test_out = y_test.T.reshape(-1, 1)
            y_test_out = test_target_scaler.fit_transform(y_test_out)
    scalers = [train_feature_scaler, test_feature_scaler, train_target_scaler, test_target_scaler]
    if dropna:
        train = pd.DataFrame(np.append(X_train_out, y_train_out.reshape(-1, 1), axis=1)).dropna().values
        test = pd.DataFrame(np.append(X_test_out, y_test_out.reshape(-1, 1), axis=1)).dropna().values
        X_train_out = train[:, :-1]
        y_train_out = train[:, -1].reshape(-1, 1)
        X_test_out = test[:, :-1]
        y_test_out = test[:, -1].reshape(-1, 1)
    return X_train_out, y_train_out, X_test_out, y_test_out, scalers


# noinspection PyPep8Naming
def reshape_data(X_train, X_test, num_features, reshape_features_method: ReshapeMethod = ReshapeMethod.NoReshape,
                 n_subsequences=4, look_back=5):
    X_train = reshape_features(features=X_train, num_features=num_features,
                               reshape_features_method=reshape_features_method,
                               n_subsequences=n_subsequences, look_back=look_back)
    X_test = reshape_features(features=X_test, num_features=num_features,
                              reshape_features_method=reshape_features_method,
                              n_subsequences=n_subsequences, look_back=look_back)
    return X_train, X_test


# noinspection PyPep8Naming
def reshape_features(features, num_features=None, reshape_features_method: ReshapeMethod = ReshapeMethod.NoReshape,
                     n_subsequences=None, look_back=None, reshape_tuple_func=None):
    train_samples_count = features.shape[0]
    if reshape_features_method is ReshapeMethod.ThreeDShape:
        features = features.reshape((train_samples_count, 1, features.shape[1]))

    if reshape_features_method is ReshapeMethod.FourDShape and reshape_tuple_func is not None:
        features = features.reshape(reshape_tuple_func(train_samples_count))

    if reshape_features_method is ReshapeMethod.FourDShape and n_subsequences is not None:
        # n_steps = int(look_back / n_subsequences)
        # features = features.reshape((train_samples_count, n_subsequences, n_steps, num_features))
        train_columns_count = features.shape[1]
        # Best
        # features = features.reshape((train_samples_count, 1, train_columns_count, 1))
        features = features.reshape((train_samples_count, 1, train_columns_count//2, 2))
        # features = features.reshape((1, 1, train_samples_count, train_columns_count))

    return features


# noinspection PyPep8Naming
def split_data(target_column, look_back, pre_process_data, ds, iterations_count,
               split_mode: SplitMode = SplitMode.KFold, test_months=1):
    features, target, num_features = \
        prepare_for_split(ds, look_back=look_back,
                          target_column=target_column,
                          pre_process_func=lambda x: pre_process_data(x, ds.name))
    if split_mode is SplitMode.KFold or SplitMode.KFoldTimeSeries or SplitMode.BlockingTimeSeriesSplit:

        from sklearn.model_selection import KFold, TimeSeriesSplit
        kf = KFold(n_splits=iterations_count, random_state=None, shuffle=False) if split_mode == SplitMode.KFold \
            else TimeSeriesSplit(n_splits=iterations_count) if split_mode == SplitMode.KFoldTimeSeries \
            else BlockingTimeSeriesSplit(n_splits=iterations_count)

        i = 0
        for train_index, test_index in kf.split(features):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]

            i = i + 1
            yield X_train, X_test, y_train, y_test, num_features, i

    else:
        import math
        min_date = ds.data.index.min()
        max_date = ds.data.index.max()
        num_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)
        interval = math.ceil(num_months / iterations_count)

        for index in range(iterations_count):
            train_start = add_months(min_date, index * interval)
            train_end = add_months(min_date, ((index + 1) * interval) - test_months - 1)
            test_start = add_months(min_date, ((index + 1) * interval) - test_months)
            test_end = add_months(min_date, ((index + 1) * interval) - 1)
            X_train, X_test = \
                features.loc[f'{train_start.month}-{train_start.year}':f'{train_end.month}-{train_end.year}'], \
                features.loc[f'{test_start.month}-{test_start.year}':f'{test_end.month}-{test_end.year}']
            y_train, y_test = \
                target.loc[f'{train_start.month}-{train_start.year}':f'{train_end.month}-{train_end.year}'], \
                target.loc[f'{test_start.month}-{test_start.year}':f'{test_end.month}-{test_end.year}']

            yield X_train, X_test, y_train, y_test, num_features, index + 1


def add_months(d, x):
    import datetime
    new_month = (((d.month - 1) + x) % 12) + 1
    new_year = int(d.year + (((d.month - 1) + x) / 12))
    return datetime.date(new_year, new_month, d.day)


# noinspection PyPep8Naming
def prepare_for_split(df_in, look_back, target_column='target', pre_process_func=None):
    import pandas as pd
    import hybrid_helper

    if type(df_in) is hybrid_helper.Dataset:
        df = pd.DataFrame(df_in.data)
    else:
        df = pd.DataFrame(df_in)
    if pre_process_func is not None:
        df = pre_process_func(df)

    relocate_target(df, target_column)
    encode_non_numeric_columns(df)
    # Reshape the arrays
    num_features = df.shape[1] - 1  # All columns before the shifted_target column are shifted_features
    shifted_features, shifted_target = shift_dataset(df, look_back)
    return shifted_features, shifted_target, num_features


# noinspection PyPep8Naming
def df_to_ml_format(df_in, train_size, look_back=5, target_column='target', dropna=False,
                    scale_features_method: ScaleMethod = ScaleMethod.NoScaler,
                    n_subsequences=4,
                    scale_target=False, pre_process_func=None):
    """

    Input is a Pandas DataFrame.
    Output is a np array in the format of (samples, timesteps, shifted_features).
    Currently this function only accepts one shifted_target variable.

    Usage example:

    # variables
    df = data # should be a pandas dataframe
    test_size = 0.5 # percentage to use for training
    target_column = 'c' # shifted_target column name, all other columns are taken as shifted_features
    scale_X = True
    look_back = 5 # Amount of previous X values to look at when predicting the current y value
    """

    shifted_features, shifted_target, num_features = prepare_for_split(df_in, look_back, target_column,
                                                                       pre_process_func)
    data_samples_count = shifted_features.shape[0]
    # the index at which to split df into train and test
    split_index = int(data_samples_count * train_size) if type(train_size) is float else train_size

    # ...train
    X_train = shifted_features[:split_index, :]
    y_train = shifted_target[:split_index]

    # ...test
    X_test = shifted_features[split_index:, :]  # original is split_index:-1
    y_test = shifted_target[split_index:]  # original is split_index:-1
    X_train, y_train, X_test, y_test, scalers = scale_data(X_train, y_train, X_test, y_test, scale_target=scale_target,
                                                           scale_features_method=scale_features_method, dropna=dropna)
    X_train, X_test = reshape_data(X_train, X_test, num_features, n_subsequences=n_subsequences, look_back=look_back)
    return X_train, y_train, X_test, y_test, scalers


import numpy as np


class BlockingTimeSeriesSplit(_BaseKFold, ABC):
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
