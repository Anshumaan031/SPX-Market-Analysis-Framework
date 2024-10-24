import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.api import OLS
import statsmodels.api as sm
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform, randint
from tqdm import tqdm

class DataCollector:
    @staticmethod
    def get_spx_data(start_date, end_date):
        spx = yf.Ticker("^GSPC")
        df = spx.history(start=start_date, end=end_date)
        return df[['Close']].rename(columns={'Close': 'SPX'})

    @staticmethod
    def get_economic_indicators(start_date, end_date):
        indicators = ['UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'INDPRO', 'T10Y2Y', 'VIXCLS']
        df = pdr.get_data_fred(indicators, start=start_date, end=end_date)
        return df
    
class CustomDataLoader:
    @staticmethod
    def load_csv_data(file_path, start_date='1989-01-01'):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
        df.set_index('date', inplace=True)
        df.rename(columns={'PX_LAST': 'SPX'}, inplace=True)
        # Filter data from 1989 onwards
        df = df[df.index >= start_date]
        return df

class FeatureEngineer:
    @staticmethod
    def create_features(df):
        df['SPX_Returns'] = df['SPX'].pct_change()
        df['SPX_MA50'] = df['SPX'].rolling(window=50).mean()
        df['SPX_MA200'] = df['SPX'].rolling(window=200).mean()
        df['SPX_Volatility'] = df['SPX_Returns'].rolling(window=20).std()
        df['SPX_Returns_Lag1'] = df['SPX_Returns'].shift(1)
        df['SPX_Returns_Lag2'] = df['SPX_Returns'].shift(2)
        df['SPX_RSI'] = FeatureEngineer.calculate_rsi(df['SPX'])
        df['SPX_MACD'] = FeatureEngineer.calculate_macd(df['SPX'])
        return df.dropna()

    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line


# a different iteration of feature engineering to include external features 
# class FeatureEngineer:
#     @staticmethod
#     def create_features(df, start_date, end_date):
#         """
#         Create features from SPX data and incorporate external economic indicators.

#         Parameters
#         ----------
#         df : pandas.DataFrame
#             DataFrame containing the SPX data.
#         start_date : str
#             Start date for fetching economic indicators.
#         end_date : str
#             End date for fetching economic indicators.

#         Returns
#         -------
#         pandas.DataFrame
#             DataFrame with the added features and economic indicators.
#         """
#         # Original SPX features
#         df['SPX_Returns'] = df['SPX'].pct_change()
#         df['SPX_MA50'] = df['SPX'].rolling(window=50).mean()
#         df['SPX_MA200'] = df['SPX'].rolling(window=200).mean()
#         df['SPX_Volatility'] = df['SPX_Returns'].rolling(window=20).std()
#         df['SPX_Returns_Lag1'] = df['SPX_Returns'].shift(1)
#         df['SPX_Returns_Lag2'] = df['SPX_Returns'].shift(2)
#         df['SPX_RSI'] = FeatureEngineer.calculate_rsi(df['SPX'])
#         df['SPX_MACD'] = FeatureEngineer.calculate_macd(df['SPX'])

#         # Fetch and merge external economic indicators
#         economic_indicators = FeatureEngineer.get_economic_indicators(start_date, end_date)
#         df = df.join(economic_indicators)

#         return df.dropna()

#     @staticmethod
#     def get_economic_indicators(start_date, end_date):
#         """
#         Fetch economic indicators from FRED.

#         Parameters
#         ----------
#         start_date : str
#             Start date for fetching data.
#         end_date : str
#             End date for fetching data.

#         Returns
#         -------
#         pandas.DataFrame
#             DataFrame containing economic indicators.
#         """
#         indicators = ['UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'INDPRO', 'T10Y2Y', 'VIXCLS']
#         df = pdr.get_data_fred(indicators, start=start_date, end=end_date)
        
#         # Fill missing values (economic indicators are often reported monthly)
#         df = df.resample('D').ffill()
        
#         return df

#     @staticmethod
#     def calculate_rsi(prices, period=14):
#         # RSI calculation (unchanged)
#         delta = prices.diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
#         rs = gain / loss
#         return 100 - (100 / (1 + rs))

#     @staticmethod
#     def calculate_macd(prices, fast=12, slow=26, signal=9):
#         # MACD calculation (unchanged)
#         ema_fast = prices.ewm(span=fast, adjust=False).mean()
#         ema_slow = prices.ewm(span=slow, adjust=False).mean()
#         macd = ema_fast - ema_slow
#         signal_line = macd.ewm(span=signal, adjust=False).mean()
#         return macd - signal_line

class ModelTrainer:
    def __init__(self):
        self.models = {
            'OLS': None,
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            'LSTM': None,
            'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        }
        self.scaler = StandardScaler()

    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(30, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_models(self, X_train, y_train):
        """
        Train each model in self.models using the given X_train and y_train.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        None
        """
       
        X_scaled = self.scaler.fit_transform(X_train)
        
        for name, model in self.models.items():
            if name == 'OLS':
                X_with_const = sm.add_constant(X_scaled)
                self.models[name] = sm.OLS(y_train, X_with_const).fit()
            elif name == 'ElasticNet':
                # Use different alpha and l1_ratio for ElasticNet
                self.models[name] = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=42).fit(X_scaled, y_train)
            elif name == 'RandomForest':
                # Adjust RandomForest parameters
                self.models[name] = RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_split=5, random_state=42).fit(X_scaled, y_train)
            elif name == 'LSTM':
                X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                self.models[name] = self.create_lstm_model((1, X_scaled.shape[1]))
                self.models[name].fit(X_reshaped, y_train, epochs=50, batch_size=32, verbose=0)
            elif name == 'XGBoost':
                # Adjust XGBoost parameters
                self.models[name] = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42).fit(X_scaled, y_train)

    def predict(self, X):
        """
        Predict SPX returns for given features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns
        -------
        predictions : dict of shape (n_models, n_samples)
            Predicted returns for each model.
        """
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name, model in self.models.items():
            if model is not None:
                if name == 'OLS':
                    X_with_const = sm.add_constant(X_scaled)
                    predictions[name] = model.predict(X_with_const)
                elif name == 'LSTM':
                    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                    predictions[name] = model.predict(X_reshaped).flatten()
                else:
                    predictions[name] = model.predict(X_scaled)
            else:
                print(f"Warning: {name} model is not trained.")
        
        return predictions

class PortfolioManager:
    def __init__(self, max_transactions_per_year=20):
        self.max_transactions = max_transactions_per_year
        self.current_position = 0  # 0: out of market, 1: in market
        self.transactions = 0
        self.returns = []
        self.decisions = []
        self.prediction_buffer = []
        self.prediction_threshold = 0.001  # 0.1% threshold for significant moves

    def make_decision(self, prediction, actual_return):
        """
        Make a decision based on a prediction and actual return.

        Parameters
        ----------
        prediction : float
            Model prediction
        actual_return : float
            Actual return of the stock

        Notes
        -----
        This function implements a decision-making logic based on the prediction
        and actual return. The logic is as follows:

        * If the absolute prediction is greater than a threshold, the decision
          is to buy if the prediction is positive and sell if the prediction is
          negative.
        * If the absolute prediction is less than or equal to the threshold, the
          decision is to hold.
        * The threshold is adjusted based on the remaining transactions and
          prediction volatility.

        Returns
        -------
        None
        """
        self.prediction_buffer.append(prediction)
        if len(self.prediction_buffer) > 5:
            self.prediction_buffer.pop(0)

        if self.transactions >= self.max_transactions:
            self.decisions.append('hold')
            self.returns.append(actual_return if self.current_position == 1 else 0)
            return

        # Calculate average prediction and volatility
        avg_prediction = np.mean(self.prediction_buffer)
        pred_volatility = np.std(self.prediction_buffer)

        # Adjust threshold based on remaining transactions and prediction volatility
        remaining_transactions = self.max_transactions - self.transactions
        adjusted_threshold = self.prediction_threshold * (1 + pred_volatility) * (self.max_transactions / (remaining_transactions + 1))

        if abs(prediction) > adjusted_threshold:
            if prediction > 0 and self.current_position == 0:
                self.current_position = 1
                self.transactions += 1
                self.returns.append(actual_return)
                self.decisions.append('buy')
            elif prediction <= 0 and self.current_position == 1:
                self.current_position = 0
                self.transactions += 1
                self.returns.append(0)
                self.decisions.append('sell')
            else:
                self.returns.append(actual_return if self.current_position == 1 else 0)
                self.decisions.append('hold')
        else:
            self.returns.append(actual_return if self.current_position == 1 else 0)
            self.decisions.append('hold')

    def get_portfolio_return(self):
        return np.prod(1 + np.array(self.returns)) - 1

    def get_decision_summary(self):
        return pd.Series(self.decisions).value_counts()

def rolling_window_backtest(train_data, test_data, features, target, window_size=1260, step_size=252):
    """
    Runs a rolling window backtest on the given data and returns the results
    for each model.

    Parameters
    ----------
    train_data : pandas.DataFrame
        The data to be used for training the models.
    test_data : pandas.DataFrame
        The data to be used for testing the models.
    features : list of str
        The features to be used for training the models.
    target : str
        The target variable to be used for training the models.
    window_size : int, optional
        The size of the rolling window. Defaults to 1260.
    step_size : int, optional
        The step size of the rolling window. Defaults to 252.

    Returns
    -------
    dict
        A dictionary with the results for each model. The keys are the model
        names and the values are lists of tuples containing the portfolio return,
        number of transactions, and decision summary for each window.
    """
    model_trainer = ModelTrainer()
    results = {model: [] for model in model_trainer.models.keys()}
    
    for i in range(0, len(train_data) - window_size, step_size):
        window_data = train_data.iloc[i:i+window_size]
        
        X_train, y_train = window_data[features], window_data[target]
        X_test, y_test = test_data[features], test_data[target]
        
        model_trainer.train_models(X_train, y_train)
        backtest_results = BacktestEngine.run_backtest(X_test, y_test, model_trainer)
        
        for model, (portfolio_return, transactions, decision_summary) in backtest_results.items():
            results[model].append((portfolio_return, transactions, decision_summary))
    
    return results

class BacktestEngine:
    @staticmethod
    def run_backtest(X_test, y_test, model_trainer):
        """
        Run a backtest on the given data using the given model trainer.

        Parameters
        ----------
        X_test : pandas.DataFrame
            The features to be used for testing the models.
        y_test : pandas.Series
            The target variable to be used for testing the models.
        model_trainer : ModelTrainer
            The model trainer to be used for training the models.

        Returns
        -------
        dict
            A dictionary with the results for each model. The keys are the model
            names and the values are tuples containing the portfolio return,
            number of transactions, and decision summary.
        """
        predictions = model_trainer.predict(X_test)
        
        results = {}
        for name, preds in predictions.items():
            pm = PortfolioManager()
            for pred, actual in zip(preds, y_test):
                pm.make_decision(pred, actual)
            results[name] = (pm.get_portfolio_return(), pm.transactions, pm.get_decision_summary())
        
        return results

def main(csv_file_path):
    data = CustomDataLoader.load_csv_data(csv_file_path)
    data = FeatureEngineer.create_features(data)

    features = ['SPX_Returns', 'SPX_MA50', 'SPX_MA200', 'SPX_Volatility', 'SPX_Returns_Lag1', 'SPX_Returns_Lag2', 'SPX_RSI', 'SPX_MACD']
    target = 'SPX_Returns'

    train_data = data[data.index < '2019-01-01']
    test_data = data[data.index >= '2019-01-01']

    # Run rolling window backtest
    train_results = rolling_window_backtest(train_data, test_data, features, target)

    # Print rolling window results
    for model, model_results in train_results.items():
        avg_return = np.mean([r[0] for r in model_results])
        avg_transactions = np.mean([r[1] for r in model_results])
        print(f"Model: {model}")
        print(f"Average Rolling Window Portfolio Return: {avg_return:.2%}")
        print(f"Average Rolling Window Number of Transactions: {avg_transactions:.2f}")
        print("--------------------")

    # Final training and testing
    model_trainer = ModelTrainer()
    model_trainer.train_models(train_data[features], train_data[target])
    test_predictions = model_trainer.predict(test_data[features])

    # Evaluate final performance on 2019-2023 test data
    for name, preds in test_predictions.items():
        pm = PortfolioManager()
        for pred, actual in zip(preds, test_data[target]):
            pm.make_decision(pred, actual)
        print(f"Model {name} final performance on 2019-2023 test data:")
        print(f"Portfolio Return: {pm.get_portfolio_return():.2%}")
        print(f"Number of Transactions: {pm.transactions}")
        print("--------------------")

    # Calculate and print benchmark return for 2019-2023
    spx_return = (test_data['SPX'].iloc[-1] / test_data['SPX'].iloc[0]) - 1
    print(f"Benchmark (Buy and Hold SPX) 2019-2023: {spx_return:.2%}")

if __name__ == "__main__":
    csv_file_path = "SPX_index.csv"  # Replace with the actual path to your CSV file
    main(csv_file_path)