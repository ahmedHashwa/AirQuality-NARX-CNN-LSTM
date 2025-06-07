import os
import datasets
from datetime import datetime
from box import Box

import fireTS.models  # noqa: F401
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRFRegressor, XGBRegressor
from scipy.stats import pearsonr
import pandas as pd
import math

import hybrid_algorithms
import hybrid_helper
import hybrid_metrics
import hybrid_preprocess


def fit_process(model, x, y, epochs, batch_size):
    res = model.fit(x, y, epochs=epochs, batch_size=batch_size)
    return res


def predict_process(model, x, batch_size):
    if model is (hybrid_algorithms.CNNLSTMModel or hybrid_algorithms.LSTMModel):
        return model.predict(x, batch_size=batch_size)
    return model.predict(x)


def train(datasets_list, cfg):
    session_start_datetime = datetime.now()
    regressors = {}
    algorithm_index = 1
    last_folder_name = 'ts_10_MCR_1519_XGBRF_Imp_NO_GPU'
    overwrite_folder = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    evaluation_methods = {
        'MSE': mean_squared_error,
        'RMSE': hybrid_metrics.rmse,
        'NRMSE': hybrid_metrics.nrmse,
        'MAE': mean_absolute_error,
        'R2': r2_score,
        'IA': hybrid_metrics.index_agreement,
        'Pearson R': lambda observed, predicted: pearsonr(observed, predicted)[0]
    }

    def f_list(lst):
        return f"[{', '.join(f'{x:02d}' for x in lst)}]"

    def key(run_index, **kwargs):
        nonlocal algorithm_index
        e_delay = kwargs.get("e_delay", [])
        e_order = kwargs.get("e_order", [])
        regressor_name = kwargs["regressor_name"]
        if regressor_name not in regressors.keys():
            regressors[regressor_name] = algorithm_index
            algorithm_index += 1
        regressor_name = f'{regressors[regressor_name]:02d}_{regressor_name}'
        if any(n in regressor_name for n in ["NARX", "DAR"]):
            kwargs["regressor_name"] = f"{regressor_name}_ao_{cfg.look_back:02d}_ed_{f_list(e_delay)}_eo_{f_list(e_order)}"
        else:
            kwargs["regressor_name"] = regressor_name
        kwargs['overwrite_folder'] = overwrite_folder
        kwargs['max_limit'] = cfg.max_limit
        kwargs['look_back'] = cfg.look_back if not any(n in regressor_name for n in ["NARX", "DAR"]) else 0
        kwargs['n_estimators'] = cfg.n_estimators
        kwargs['session_start_datetime'] = session_start_datetime
        kwargs['dropout_rate'] = cfg.dropout_rate
        kwargs['n_lstm_nodes'] = cfg.n_lstm_nodes
        kwargs['activation'] = cfg.activation
        kwargs['n_dense_nodes'] = cfg.n_dense_nodes
        kwargs['batch_size'] = cfg.batch_size
        kwargs['evaluation_methods'] = evaluation_methods
        kwargs['index'] = run_index
        kwargs['dataset_name'] = ds.name
        kwargs['dataset'] = ds
        kwargs['results_dir'] = cfg.saving_dir
        kwargs['round_all_results'] = cfg.round_all_results
        kwargs['target_column'] = target_column
        kwargs['datetime'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        return Box(kwargs, frozen_box=True)

    for ds in [d for d in datasets_list if d.include]:
        for target_column in ds.target_columns:
            for X_train, X_test, y_train, y_test, num_features, i in hybrid_preprocess.split_data(
                    target_column=target_column,
                    ds=ds,
                    look_back=cfg.look_back,
                    split_mode=cfg.data_split_mode,
                    iterations_count=cfg.iterations_count,
                    pre_process_data=datasets.pre_process_data):

                if cfg.enable_CNNLSTM:
                    hybrid_helper.benchmark_algorithm(
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        predictor_object=hybrid_algorithms.CNNLSTMModel(
                            n_lstm_nodes=cfg.n_lstm_nodes,
                            n_dense_nodes=cfg.n_dense_nodes,
                            dropout_rate=0,
                            activation=cfg.activation),
                        num_features=num_features,
                        n_subsequences=cfg.n_subsequences,
                        scale_target=cfg.lstm_scale_target,
                        fit_process=lambda m, x, y: fit_process(m, x, y, cfg.epochs, cfg.batch_size),
                        predict_process=lambda m, x, y=None: predict_process(m, x, cfg.batch_size),
                        scale_features_method=cfg.lstm_scaler_method,
                        reshape_features_method=hybrid_preprocess.ReshapeMethod.FourDShape,
                        method_key=key(i, regressor_name='CNNLSTM'),
                        last_folder_name=last_folder_name)
                if cfg.enable_LSTM:
                    hybrid_helper.benchmark_algorithm(
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        predictor_object=hybrid_algorithms.LSTMModel(
                            n_lstm_nodes=cfg.n_lstm_nodes,
                            n_dense_nodes=cfg.n_dense_nodes,
                            dropout_rate=0,
                            activation=cfg.activation),
                        num_features=num_features,
                        n_subsequences=cfg.n_subsequences,
                        scale_target=cfg.lstm_scale_target,
                        fit_process=lambda m, x, y: fit_process(m, x, y, cfg.epochs, cfg.batch_size),
                        predict_process=lambda m, x, y=None: predict_process(m, x, cfg.batch_size),
                        scale_features_method=cfg.lstm_scaler_method,
                        reshape_features_method=hybrid_preprocess.ReshapeMethod.ThreeDShape,
                        method_key=key(i, regressor_name='LSTM'),
                        last_folder_name=last_folder_name)
                if cfg.enable_ET:
                    hybrid_helper.benchmark_algorithm(
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        predictor_object=ExtraTreesRegressor(n_estimators=cfg.n_estimators, n_jobs=-1, verbose=3),
                        scale_target=cfg.scale_target,
                        scale_features_method=cfg.scaler_method,
                        method_key=key(i, regressor_name='ExtraTrees'),
                        last_folder_name=last_folder_name)
                if cfg.enable_XGBRF:
                    hybrid_helper.benchmark_algorithm(
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        predictor_object=XGBRFRegressor(n_estimators=cfg.n_estimators, verbosity=3),
                        scale_target=cfg.scale_target,
                        predict_process=lambda m, x, y=None: predict_process(m, x, cfg.batch_size),
                        scale_features_method=cfg.scaler_method,
                        method_key=key(i, regressor_name='XGBRF'),
                        last_folder_name=last_folder_name)
                if cfg.enable_XGB:
                    hybrid_helper.benchmark_algorithm(
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        predictor_object=XGBRegressor(n_estimators=cfg.n_estimators, verbosity=3),
                        scale_target=cfg.scale_target,
                        scale_features_method=cfg.scaler_method,
                        method_key=key(i, regressor_name='XGB'),
                        last_folder_name=last_folder_name)
                if cfg.enable_RF:
                    hybrid_helper.benchmark_algorithm(
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        predictor_object=RandomForestRegressor(n_estimators=cfg.n_estimators, n_jobs=-1, verbose=3),
                        scale_target=cfg.scale_target,
                        scale_features_method=cfg.scaler_method,
                        method_key=key(i, regressor_name='RandomForest'),
                        last_folder_name=last_folder_name)
                if cfg.enable_LSTM_dropout:
                    hybrid_helper.benchmark_algorithm(
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        predictor_object=hybrid_algorithms.LSTMModel(
                            n_lstm_nodes=cfg.n_lstm_nodes,
                            n_dense_nodes=cfg.n_dense_nodes,
                            dropout_rate=cfg.dropout_rate,
                            activation=cfg.activation),
                        num_features=num_features,
                        n_subsequences=cfg.n_subsequences,
                        scale_target=cfg.lstm_scale_target,
                        fit_process=lambda m, x, y: fit_process(m, x, y, cfg.epochs, cfg.batch_size),
                        predict_process=lambda m, x, y=None: predict_process(m, x, cfg.batch_size),
                        scale_features_method=cfg.lstm_scaler_method,
                        reshape_features_method=hybrid_preprocess.ReshapeMethod.ThreeDShape,
                        method_key=key(i, regressor_name='LSTM_dropout'),
                        last_folder_name=last_folder_name)
                if cfg.enable_GB:
                    hybrid_helper.benchmark_algorithm(
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        predictor_object=GradientBoostingRegressor(n_estimators=cfg.n_estimators, verbose=3),
                        scale_target=cfg.scale_target,
                        scale_features_method=cfg.scaler_method,
                        method_key=key(i, regressor_name='GradientBoosting'),
                        last_folder_name=last_folder_name)
                if cfg.enable_SVR:
                    hybrid_helper.benchmark_algorithm(
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        predictor_object=SVR(C=2.0, epsilon=0.1, kernel='rbf', gamma=0.5, tol=0.001,
                                            verbose=True, shrinking=True, max_iter=10000),
                        scale_target=cfg.scale_target,
                        scale_features_method=cfg.scaler_method,
                        method_key=key(i, regressor_name='SVR'),
                        last_folder_name=last_folder_name)
                if cfg.enable_XGBRF_DART:
                    hybrid_helper.benchmark_algorithm(
                        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        predictor_object=XGBRFRegressor(n_estimators=cfg.n_estimators, verbosity=3, booster='dart'),
                        scale_target=cfg.scale_target,
                        scale_features_method=cfg.scaler_method,
                        method_key=key(i, regressor_name='XGBRF_dart'),
                        last_folder_name=last_folder_name)

    # join and summarize results
    formatted_datetime = last_folder_name if last_folder_name else session_start_datetime.strftime("%d-%m-%Y_%H_%M_%S")
    limit = 10
    os.makedirs(f"{cfg.saving_dir}/{formatted_datetime}", exist_ok=True)
    data_dir = f'{cfg.saving_dir}/{formatted_datetime}'
    for ds in [d for d in datasets_list if d.include]:
        for target in ds.target_columns:
            results_dir = f'{data_dir}'
            for i in range(1, cfg.iterations_count + 1):
                first_file = None
                for file in sorted(os.listdir(results_dir), reverse=True):
                    file = os.path.join(results_dir, file)
                    if f'i_{i:02d}_predictions' in file and f'{ds.name}' in file and f'_{target}_' in file:
                        cols = list(pd.read_csv(file, nrows=1))
                        if first_file is None:
                            first_file = pd.read_csv(file, index_col=0, usecols=[c for c in cols if 'aqi' not in c])
                        else:
                            read_file = pd.read_csv(file, index_col=0, usecols=[c for c in cols if 'aqi' not in c])
                            temp = read_file[f"{target}_Real"].apply(lambda x: math.ceil(x)).rolling(window=limit).apply(
                                lambda x: (x == first_file[f"{target}_Real"][0:limit].apply(lambda y: math.ceil(y)).values).all(),
                                raw=True).mask(lambda x: x == 0).bfill(limit=limit).dropna()
                            cut_off = 0 if temp.empty else temp.index[0] + 1
                            read_file = read_file if cut_off in (0, 1) else read_file.iloc[cut_off:]
                            read_file.reset_index(inplace=True)
                            read_file.drop(columns=['index', f'{target}_Real'], inplace=True)
                            first_file = first_file.join(read_file)
                if first_file is not None:
                    first_file.to_csv(os.path.join(results_dir, f'run_{ds.name}_{target}_i{i:02d}_data_{formatted_datetime}.csv'))

    summary_columns = ["Regressor Name", "Training Period Seconds", "Prediction Period Seconds", "MSE", "RMSE", "NRMSE",
                       "MAE", "R2", "IA", "Pearson R", "Step"]
    for ds in [d for d in datasets_list if d.include]:
        for target in ds.target_columns:
            all_metrics_results = None
            results_dir = f'{data_dir}'
            for i in range(1, cfg.iterations_count + 1):
                for file in os.listdir(results_dir):
                    file = os.path.join(results_dir, file)
                    if f'i_{i:02d}_metrics' in file and f'{ds.name}' in file and f'_{target}_' in file:
                        if all_metrics_results is None:
                            all_metrics_results = pd.read_csv(file, index_col=0)
                        else:
                            metrics = pd.read_csv(file, index_col=0)
                            all_metrics_results = pd.concat([all_metrics_results, metrics])
            if all_metrics_results is not None:
                all_metrics_results.to_csv(os.path.join(results_dir, f'all_metrics_{ds.name}_{target}_data_{formatted_datetime}.csv'))
                grouped_results_mean = all_metrics_results[summary_columns].groupby('Regressor Name').mean()
                grouped_results_mean.to_csv(os.path.join(results_dir, f'all_metrics_mean_{ds.name}_{target}_data_{formatted_datetime}.csv'))

    for ds in [d for d in datasets_list if d.include]:
        for target in ds.target_columns:
            all_mean_metrics_results = None
            results_dir = f'{data_dir}'
            for i in range(1, cfg.iterations_count + 1):
                for file in os.listdir(results_dir):
                    file = os.path.join(results_dir, file)
                    if 'all_metrics_mean_' in file and f'{ds.name}' in file and f'_{target}_' in file:
                        if all_mean_metrics_results is None:
                            all_mean_metrics_results = pd.read_csv(file, index_col=0)
                        else:
                            metrics = pd.read_csv(file, index_col=0)
                            all_mean_metrics_results = pd.concat([all_mean_metrics_results, metrics])
            if all_mean_metrics_results is not None:
                all_mean_metrics_results.to_csv(os.path.join(data_dir, f'all_steps_mean_metrics_{ds.name}_{target}_data_{formatted_datetime}.csv'))
                grouped_results_mean = all_mean_metrics_results.groupby('Regressor Name').mean()
                grouped_results_mean.to_csv(os.path.join(data_dir, f'all_mean_metrics_all_steps_{ds.name}_{target}_data_{formatted_datetime}.csv'))

