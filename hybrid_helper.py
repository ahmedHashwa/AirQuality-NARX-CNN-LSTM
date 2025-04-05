import os.path

from fireTS.models import NARX, DirectAutoRegressor

import hybrid_preprocess
import aqi


# from preprocess import ScaleMethod, ReshapeMethod


class Dataset:
    def __init__(self, name, data, target_columns, include=True, notes=None, feature_columns=None):
        self.name = name
        self.data = data[feature_columns] if feature_columns is not None else data
        self.target_columns = target_columns
        self.include = include
        self.notes = notes
        self.feature_columns = feature_columns

    def __str__(self):
        return f'{self.name} [{self.feature_columns}]->{self.target_columns}' if self.feature_columns is not None \
            else f'{self.name}->{self.target_columns}'

    # noinspection PyPep8Naming


def evaluate_results(real, prediction, method_key, regressor_name, print_results=True,
                     predicted_result=None):
    results = {}
    if print_results:
        print(
            f'Results for run No {method_key.index} Prediction Method is : '
            f'{regressor_name} on Dataset {method_key.dataset_name}'
            f' at {method_key.datetime}')
        print(f'Duration is {predicted_result.training_period_seconds}')
    import platform

    results['Regressor Name'] = regressor_name
    results['Platform'] = platform.platform()
    results['Running No.'] = method_key.index
    results['Dataset Name'] = method_key.dataset_name
    results['Datetime'] = method_key.datetime
    results['Session Datetime'] = method_key.session_start_datetime.strftime("%d-%m-%Y_%H_%M_%S")
    results['Target Column'] = method_key.target_column
    results['Training Period Seconds'] = predicted_result.training_period_seconds
    results['Prediction Period Seconds'] = predicted_result.prediction_period_seconds
    results['Scale Features Method'] = predicted_result.scale_features_method.name \
        if predicted_result.scale_features_method is not None else None
    results['Look Back'] = method_key.look_back
    results['No. Subsequences'] = predicted_result.n_subsequences
    results['Reshape Features Method'] = predicted_result.reshape_features_method.name \
        if predicted_result.reshape_features_method is not None else None
    results['Input Train Shape'] = predicted_result.input_train_shape
    results['Train Shape'] = predicted_result.train_shape
    results['Input Test Shape'] = predicted_result.input_test_shape
    results['Test Shape'] = predicted_result.test_shape
    results['Min Dataset Date'] = method_key.dataset.data.index.min()
    results['Max Dataset Date'] = method_key.dataset.data.index.max()
    results['Min Train DateTime'] = predicted_result.min_train_index.strftime('%d/%m/%Y %H:%M:%S')
    results['Max Train DateTime'] = predicted_result.max_train_index.strftime('%d/%m/%Y %H:%M:%S')
    results['Min Test DateTime'] = predicted_result.min_test_index.strftime('%d/%m/%Y %H:%M:%S')
    results['Max Test DateTime'] = predicted_result.max_test_index.strftime('%d/%m/%Y %H:%M:%S')

    results['Epochs'] = len(predicted_result.fit_result.epoch) if hasattr(predicted_result.fit_result,
                                                                          'epoch') else None
    results['Dropout Rate'] = \
        method_key.dropout_rate if hasattr(predicted_result.fit_result,
                                           'epoch') and method_key.regressor_name != '2_LSTM' else None
    results['No. LSTM Nodes'] = method_key.n_lstm_nodes if hasattr(predicted_result.fit_result, 'epoch') else None
    results['Batch Size'] = method_key.batch_size if hasattr(predicted_result.fit_result, 'epoch') else None
    results['No. Dense Nodes'] = method_key.n_dense_nodes if hasattr(predicted_result.fit_result, 'epoch') else None
    results['Activation Function'] = method_key.activation if hasattr(predicted_result.fit_result, 'epoch') else None
    results['No. Estimators'] = method_key.n_estimators if not hasattr(predicted_result.fit_result,
                                                                       'epoch') else None
    results['Auto Order'] = predicted_result.auto_order
    results['Exog Order'] = predicted_result.exog_order
    results['Exog Delay'] = predicted_result.exog_delay

    results['Step'] = predicted_result.step if  hasattr(predicted_result,
                                                                       'step') else None

    for m in method_key.evaluation_methods:
        results[m] = method_key.evaluation_methods[m](real, prediction)
    if print_results:
        for res in results:
            print(f'{res}:{results[res]}')

    return results


def add_dataframe_row(df, row):
    df.loc[0] = row
    df.index = df.index + 1
    return df.sort_index()


pollutants_map = {
    'PM25': aqi.POLLUTANT_PM25,
    'pm2.5': aqi.POLLUTANT_PM25,
    'NO2': aqi.POLLUTANT_NO2_1H,
    'O3': aqi.POLLUTANT_O3_8H,
    'SO2': aqi.POLLUTANT_SO2_1H
}


def get_aqi(pollutant, data):
    res = []

    for x in data.reshape(-1):
        # print(f'x={x} ug/m3')
        if x < 0:
            x = 0
            # print(f'x={x} ug/m3')
        elif pollutant == 'PM25' or pollutant == 'pm2.5':
            if x > 500:
                x = 500
        elif pollutant == 'O3':
            x = ((x / 1000) * 24.45) / 48
            # print(f'x={x} ppm')
        elif pollutant == 'NO2':
            x = (x * 24.45) / 46.01
            # print(f'x={x} ppb')
        elif pollutant == 'SO2':
            x = (x * 24.45) / 64.06
            # print(f'x={x} ppb')
        aqi_res = aqi.to_iaqi(pollutants_map[pollutant], f'{x}', algo=aqi.ALGO_EPA)
        res.append(aqi_res)
    return res


def get_aqi_level(aqi_values):
    res = []
    for aqi_value in aqi_values:
        if 0 <= aqi_value <= 50:
            res.append("Good")
        elif 51 <= aqi_value <= 100:
            res.append("Moderate")
        elif 101 <= aqi_value <= 150:
            res.append("Unhealthy for Sensitive Groups")
        elif 201 <= aqi_value <= 300:
            res.append("Unhealthy")
        else:
            res.append("Hazardous")
    return res


def save_results(observed, predicted, fit_result, metric_result, model, session_start_datetime, target_column,
                 dataset_name, index,
                 regressor_name, max_limit, results_dir, last_folder_name=''):
    from matplotlib import pyplot
    import os
    import pandas as pd
    index = f'i_{index:02d}'

    folder_name = last_folder_name if last_folder_name != '' else session_start_datetime.strftime(
        "%d-%m-%Y_%H_%M_%S")
    out_dir = f'{results_dir}/{folder_name}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    predictions = pd.DataFrame()
    col_index = 0
    print(f'Adding real values for {target_column}')
    predictions.insert(loc=0, column=f"{target_column}_Real",
                       value=[item for sublist in [observed[:max_limit]] for item in
                              sublist],
                       allow_duplicates=True)
    col_index = col_index + 1
    if target_column != 'NO':
        predictions.insert(loc=col_index, column=f"{target_column}_aqi_Real",
                           value=[item for sublist in [get_aqi(target_column, observed[:max_limit])]
                                  for
                                  item
                                  in
                                  sublist],
                           allow_duplicates=True)
        col_index = col_index + 1
        predictions.insert(loc=col_index, column=f"{target_column}_aqi_level_Real",
                           value=[item for sublist in
                                  [get_aqi_level(get_aqi(target_column, observed[:max_limit]))]
                                  for
                                  item in sublist],
                           allow_duplicates=True)
        col_index = col_index + 1
    predictions.insert(loc=col_index, column=f'{target_column}_{regressor_name}_{index}',
                       value=[item for sublist in [predicted[:max_limit]] for item in
                              sublist],
                       allow_duplicates=True)
    col_index = col_index + 1
    if target_column != 'NO':
        predictions.insert(loc=col_index, column=f'{target_column}_aqi_{regressor_name}_{index}',
                           value=[item for sublist in [get_aqi(target_column, predicted[:max_limit])]
                                  for
                                  item in
                                  sublist],
                           allow_duplicates=True)
        col_index = col_index + 1
        predictions.insert(loc=col_index, column=f'{target_column}_aqi_level_{regressor_name}_{index}',
                           value=[item for sublist in
                                  [get_aqi_level(get_aqi(target_column, predicted[:max_limit]))] for
                                  item
                                  in sublist],
                           allow_duplicates=True)
    if hasattr(model, 'base_model') or (
            hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'base_model')):
        filename_template = f'{out_dir}/{dataset_name}_{folder_name}_{regressor_name}'
        filename = f'{filename_template}_{index}_model_summary.txt'
        image_filename = f'{filename_template}_{index}_model.png'
        if fit_result is not None:
            fig = pyplot.figure()
            pyplot.plot(fit_result.history['loss'], label='train')
            pyplot.legend()
            pyplot.savefig(f'{filename_template}_{target_column}_{index}_training_loss.png', format='png',
                           dpi=600,
                           transparent=True)
            pyplot.close(fig)

        if not os.path.exists(filename):
            # Open the file
            with open(filename, 'w') as fh:
                requested_model = model.base_model if hasattr(model, 'base_model') else model.base_estimator.base_model
                # Pass the file handle in as a lambda function to make it callable
                requested_model.summary(print_fn=lambda x: fh.write(x + '\n'))
                from keras.utils import plot_model
                plot_model(requested_model, to_file=image_filename, show_shapes=True, expand_nested=True)

    predictions.to_csv(
        f'{out_dir}/{dataset_name}_{regressor_name}_{target_column}_{index}_predictions_'
        f'{folder_name}.csv')
    metrics_columns = list(metric_result.keys())

    metrics = pd.DataFrame(columns=metrics_columns)

    add_dataframe_row(metrics, list(metric_result.values()))
    print(f'saving metrics for {dataset_name}_{regressor_name}_{target_column}_{index}')
    metrics.to_csv(
        f'{out_dir}/{dataset_name}_{regressor_name}_{target_column}_{index}_metrics_'
        f'{folder_name}.csv')


# noinspection PyPep8Naming
def benchmark_algorithm(X_train, y_train, X_test, y_test, predictor_object,
                        num_features=None, n_subsequences=None,
                        scale_target=False,
                        fit_process=None,
                        predict_process=None,
                        scale_features_method: hybrid_preprocess.ScaleMethod =
                        hybrid_preprocess.ScaleMethod.NoScaler,
                        reshape_features_method: hybrid_preprocess.ReshapeMethod =
                        hybrid_preprocess.ReshapeMethod.NoReshape,
                        method_key=None, last_folder_name=''):
    import numpy as np

    index = f'i_{method_key.index:02d}'

    out_dir = f'{method_key.results_dir}/{last_folder_name}'
    metrics_file_name = \
        f'{out_dir}/{method_key.dataset_name}_{method_key.regressor_name}_{method_key.target_column}' \
        f'_{index}_metrics_' \
        f'{last_folder_name}.csv'
    if os.path.exists(metrics_file_name) and not method_key.overwrite_folder:
        print(f'Metrics file: {metrics_file_name} already exists. Skipping ... ✔')
        return
    import time
    min_train_index = X_train.index.min()
    max_train_index = X_train.index.max()
    min_test_index = X_test.index.min()
    max_test_index = X_test.index.max()
    X_train_out, y_train_out, X_test_out, y_test_out, scalers = \
        hybrid_preprocess.scale_data(X_train.values, y_train.values, X_test.values, y_test.values,
                                     scale_target=scale_target,
                                     scale_features_method=scale_features_method, dropna=False)
    X_train_out, X_test_out = \
        hybrid_preprocess.reshape_data(X_train=X_train_out, X_test=X_test_out, num_features=num_features,
                                       reshape_features_method=reshape_features_method,
                                       n_subsequences=n_subsequences, look_back=method_key.look_back)
    auto_order = predictor_object.auto_order if isinstance(predictor_object, (NARX, DirectAutoRegressor)) else None
    exog_delay = predictor_object.exog_delay if isinstance(predictor_object, (NARX, DirectAutoRegressor)) else None
    exog_order = predictor_object.exog_order if isinstance(predictor_object, (NARX, DirectAutoRegressor)) else None
    fit_start = time.time()

    if fit_process is not None:
        fit_result = fit_process(predictor_object, X_train_out, y_train_out.reshape(-1))
    else:
        fit_result = predictor_object.fit(X_train_out, y_train_out.reshape(-1))
    fit_end_predict_start = time.time()
    if predict_process is not None:
        y_test_out = y_test_out.reshape(-1)
    if predict_process is not None:
        predicted = predict_process(predictor_object, X_test_out, y_test_out)
    else:
        predicted = predictor_object.predict(X_test_out)
    predict_end = time.time()
    mask = np.isnan(y_test_out) | np.isnan(predicted)
    predicted_masked, y_test_out_masked = predicted[~mask], y_test_out[~mask]

    if scale_features_method != hybrid_preprocess.ScaleMethod.NoScaler and scalers[3] is not None:
        y_test_out_final = scalers[3].inverse_transform(y_test_out_masked.reshape(-1, 1))
        predicted_out = scalers[3].inverse_transform(predicted_masked.reshape(-1, 1))
    else:
        y_test_out_final = y_test_out_masked
        predicted_out = predicted_masked

    predicted = predicted_out.reshape(-1)
    observed = y_test_out_final.reshape(-1)
    if method_key.round_all_results:
        predicted = predicted.round()
        observed = observed.round()
    from box import Box
    result = Box(
        training_period_seconds=fit_end_predict_start - fit_start,
        prediction_period_seconds=predict_end - fit_end_predict_start,
        scale_features_method=scale_features_method,
        reshape_features_method=reshape_features_method,
        input_train_shape=f'{X_train.shape}',
        input_test_shape=f'{X_test.shape}',
        fit_result=fit_result,
        n_subsequences=n_subsequences,
        train_shape=f'{X_train_out.shape}',
        test_shape=f'{X_test_out.shape}',
        auto_order=auto_order,
        exog_delay=exog_delay,
        exog_order=exog_order,
        min_train_index=min_train_index,
        max_train_index=max_train_index,
        min_test_index=min_test_index,
        max_test_index=max_test_index,
    )
    regressor_name = method_key.regressor_name

    metrics_results = evaluate_results(real=observed, prediction=predicted, method_key=method_key,
                                       print_results=True,
                                       predicted_result=result, regressor_name=regressor_name)

    save_results(observed=observed, predicted=predicted, fit_result=fit_result, metric_result=metrics_results,
                 model=predictor_object,
                 target_column=method_key.target_column,
                 session_start_datetime=method_key.session_start_datetime,
                 max_limit=method_key.max_limit, results_dir=method_key.results_dir,
                 dataset_name=method_key.dataset_name, regressor_name=method_key.regressor_name,
                 index=method_key.index,
                 last_folder_name=last_folder_name)
    # import pickle
    # print('saving model...')
    # filename =
    # f'model_{method_key.regressor_name}_{method_key.dataset_name}_i{method_key.index}_{method_key.target_column}.sav'
    # pickle.dump(predictor_object, open(filename, 'wb'))
    # Cleanup everything from memory
    import gc

    if hasattr(predictor_object, 'base_estimator'):
        if hasattr(predictor_object.base_estimator, 'base_model'):
            del predictor_object.base_estimator.base_model
        del predictor_object.base_estimator

    if hasattr(predictor_object, 'base_model'):
        del predictor_object.base_model
    del predictor_object
    gc.collect()
    from keras import backend as K
    K.clear_session()
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.enable_eager_execution()
