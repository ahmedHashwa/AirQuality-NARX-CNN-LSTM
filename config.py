from dataclasses import dataclass
from hybrid_preprocess import ScaleMethod, SplitMode

@dataclass
class Config:
    data_dir: str = "data"
    saving_dir: str = 'results'
    batch_size: int = 72
    epochs: int = 25
    activation: str = 'tanh'
    round_all_results: bool = True
    enable_CNNLSTM: bool = True
    enable_LSTM: bool = True
    enable_ET: bool = True
    enable_XGBRF: bool = True
    enable_LSTM_dropout: bool = False
    enable_RF: bool = False
    enable_SVR: bool = False
    enable_GB: bool = False
    enable_XGB: bool = False
    enable_XGBRF_DART: bool = False
    enable_NARX_CNNLSTM: bool = True
    enable_NARX_LSTM: bool = True
    enable_NARX_ET: bool = True
    enable_NARX_XGBRF: bool = True
    enable_NARX_LSTM_dropout: bool = False
    enable_NARX_RF: bool = False
    enable_NARX_SVR: bool = False
    enable_NARX_GB: bool = False
    enable_NARX_XGB: bool = False
    enable_NARX_XGBRF_DART: bool = False
    enable_DAR_XGBRF: bool = False
    enable_DAR_CNNLSTM: bool = False
    enable_DAR_LSTM: bool = False
    n_estimators: int = 100
    iterations_count: int = 10
    n_lstm_nodes: int = 128
    n_dense_nodes: int = 50
    dropout_rate: float = 0.01
    look_back: int = 24
    n_subsequences: int = 4
    scaler_method: ScaleMethod = ScaleMethod.NoScaler
    lstm_scaler_method: ScaleMethod = ScaleMethod.StandardScaler
    scale_target: bool = False
    lstm_scale_target: bool = True
    max_limit: int = 4279
    data_split_mode: SplitMode = SplitMode.KFoldTimeSeries

CONFIG = Config()
