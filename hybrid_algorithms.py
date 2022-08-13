from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential


class CNNLSTMModel:
    def __init__(self, n_lstm_nodes, n_dense_nodes, dropout_rate, activation='tanh', look_back=None,
                 n_subsequences=None, num_features=None):  #
        # adjust values to your needs
        self.look_back = look_back
        self.n_subsequences = n_subsequences
        self.num_features = num_features
        self.kwargs = None
        self.base_model = Sequential()

        self.base_model.add(TimeDistributed(
            Conv1D(dilation_rate=6, groups=2, filters=4, kernel_size=2)))

        self.base_model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        # # The extracted features are then flattened and provided to the LSTM model to read
        self.base_model.add(TimeDistributed(Flatten()))
        if dropout_rate != 0:
            # Use Dropout layer to reduce overfitting of the model to the training data
            self.base_model.add(Dropout(dropout_rate))
        # Add LSTM Model
        self.base_model.add(LSTM(n_lstm_nodes, activation=activation))

        # Add one hidden layer with 100 default nodes with relu activation function
        self.base_model.add(Dense(n_dense_nodes, activation=activation))
        self.base_model.add(Dense(1))
        # Compile model
        self.base_model.compile(loss='mae', optimizer='adam')

    def fit(self, x_train, y_train, epochs=100, batch_size=24, verbose=1):
        if len(x_train.shape) != 4:
            import hybrid_preprocess
            x_train = \
                hybrid_preprocess.reshape_features(x_train,
                                                   reshape_features_method=hybrid_preprocess.ReshapeMethod.FourDShape,
                                                   num_features=self.num_features,
                                                   n_subsequences=self.n_subsequences,
                                                   look_back=self.look_back)

        res = self.base_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                                  use_multiprocessing=True)
        return res

    def predict(self, x_test, batch_size=24):
        if len(x_test.shape) != 4:
            import hybrid_preprocess
            x_test = \
                hybrid_preprocess.reshape_features(x_test,
                                                   reshape_features_method=hybrid_preprocess.ReshapeMethod.FourDShape,
                                                   num_features=self.num_features,
                                                   n_subsequences=self.n_subsequences,
                                                   look_back=self.look_back)

        result = self.base_model.predict(x=x_test, batch_size=batch_size)
        return result.reshape(-1)

    def set_params(self, **params):
        """Set the parameters of this estimator.  Modification of the sklearn method to
        allow unknown kwargs. This allows using the full range of xgboost
        parameters that are not defined as member variables in sklearn grid
        search.

        Returns
        -------
        self

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        # this concatenates kwargs into parameters, enabling `get_params` for
        # obtaining parameters from keyword parameters.
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        return self


class LSTMModel:
    def __init__(self, n_lstm_nodes, n_dense_nodes, dropout_rate, activation='tanh'):  #
        self.kwargs = None
        self.base_model = Sequential()

        # Add LSTM Model
        if dropout_rate != 0:
            self.base_model.add(Dropout(dropout_rate))
        self.base_model.add(LSTM(n_lstm_nodes, activation=activation))
        self.base_model.add(Dense(n_dense_nodes, activation=activation))
        self.base_model.add(Dense(1))
        # Compile model
        self.base_model.compile(loss='mae', optimizer='adam')

    def fit(self, x_train, y_train, epochs=100, batch_size=24, verbose=1):
        if len(x_train.shape) != 3:
            import hybrid_preprocess
            x_train = \
                hybrid_preprocess.reshape_features(x_train,
                                                   reshape_features_method=hybrid_preprocess.ReshapeMethod.ThreeDShape)
        res = self.base_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size,
                                  verbose=verbose,
                                  use_multiprocessing=True)
        return res

    def predict(self, x_test, batch_size=24):
        if len(x_test.shape) != 3:
            import hybrid_preprocess
            x_test = \
                hybrid_preprocess.reshape_features(x_test,
                                                   reshape_features_method=hybrid_preprocess.ReshapeMethod.ThreeDShape)
        result = self.base_model.predict(x=x_test, batch_size=batch_size)
        return result.reshape(-1)

    def set_params(self, **params):
        """Set the parameters of this estimator.  Modification of the sklearn method to
        allow unknown kwargs. This allows using the full range of xgboost
        parameters that are not defined as member variables in sklearn grid
        search.

        Returns
        -------
        self

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        # this concatenates kwargs into parameters, enabling `get_params` for
        # obtaining parameters from keyword parameters.
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        return self
