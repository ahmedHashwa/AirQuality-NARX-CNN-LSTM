from dataclasses import dataclass
from .enums import ScaleMethod, SplitMode


@dataclass
class Config:
    """Holds all experiment parameters used across the project."""

    # ------------------------------------------------------------------
    # Directory configuration
    # ------------------------------------------------------------------
    data_dir: str = "data"  # location of raw data files
    saving_dir: str = "results"  # directory where results will be stored

    # ------------------------------------------------------------------
    # General training configuration
    # ------------------------------------------------------------------
    batch_size: int = 72  # number of samples per training batch
    epochs: int = 25  # number of training epochs for neural networks
    activation: str = "tanh"  # activation function used in LSTM based models
    round_all_results: bool = True  # round metrics before saving them

    # ------------------------------------------------------------------
    # Classical machine learning algorithms
    # ------------------------------------------------------------------
    enable_ET: bool = True  # ExtraTrees regressor
    enable_XGBRF: bool = True  # XGBoost random forest variant
    enable_RF: bool = False  # Random forest regressor
    enable_SVR: bool = False  # Support vector regressor
    enable_GB: bool = False  # Gradient boosting regressor
    enable_XGB: bool = False  # Standard XGBoost regressor
    enable_XGBRF_DART: bool = False  # XGBRF with DART booster

    # ------------------------------------------------------------------
    # Deep learning algorithms
    # ------------------------------------------------------------------
    enable_CNNLSTM: bool = True  # Combined CNN-LSTM model
    enable_LSTM: bool = True  # Plain LSTM model
    enable_LSTM_dropout: bool = False  # LSTM model with dropout layers

    # ------------------------------------------------------------------
    # NARX based variants
    # ------------------------------------------------------------------
    enable_NARX_CNNLSTM: bool = True  # NARX version of CNN-LSTM
    enable_NARX_LSTM: bool = True  # NARX version of LSTM
    enable_NARX_ET: bool = True  # NARX version of ExtraTrees
    enable_NARX_XGBRF: bool = True  # NARX version of XGBRF
    enable_NARX_LSTM_dropout: bool = False  # NARX LSTM with dropout
    enable_NARX_RF: bool = False  # NARX Random Forest
    enable_NARX_SVR: bool = False  # NARX Support Vector Regression
    enable_NARX_GB: bool = False  # NARX Gradient Boosting
    enable_NARX_XGB: bool = False  # NARX standard XGBoost
    enable_NARX_XGBRF_DART: bool = False  # NARX XGBRF with DART booster

    # ------------------------------------------------------------------
    # DAR based variants
    # ------------------------------------------------------------------
    enable_DAR_XGBRF: bool = False  # DAR XGBRF model
    enable_DAR_CNNLSTM: bool = False  # DAR CNN-LSTM model
    enable_DAR_LSTM: bool = False  # DAR LSTM model

    # ------------------------------------------------------------------
    # Model hyperparameters
    # ------------------------------------------------------------------
    n_estimators: int = 100  # number of trees for ensemble algorithms
    iterations_count: int = 10  # folds/iterations for cross validation
    n_lstm_nodes: int = 128  # hidden units in LSTM layers
    n_dense_nodes: int = 50  # units in dense layers following LSTM
    dropout_rate: float = 0.01  # dropout rate for dropout-enabled models
    look_back: int = 24  # time steps used as input sequence length
    n_subsequences: int = 4  # subsequences for CNN-LSTM reshaping

    # ------------------------------------------------------------------
    # Pre-processing and scaling
    # ------------------------------------------------------------------
    scaler_method: ScaleMethod = ScaleMethod.NoScaler  # scaling for classical algorithms
    lstm_scaler_method: ScaleMethod = ScaleMethod.StandardScaler  # scaling for LSTM based models
    scale_target: bool = False  # whether target variable is scaled for classical models
    lstm_scale_target: bool = True  # whether target variable is scaled for LSTM models

    # ------------------------------------------------------------------
    # Miscellaneous options
    # ------------------------------------------------------------------
    max_limit: int = 4279  # limit on number of predictions stored in CSVs
    data_split_mode: SplitMode = SplitMode.KFoldTimeSeries  # strategy for train/test splitting


CONFIG = Config()
