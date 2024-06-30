import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from scipy.stats import randint
import warnings

warnings.filterwarnings('ignore')

nInst = 50
currentPos = np.zeros(nInst)

class AdvancedMLStrategy:
    def __init__(self, n_instruments):
        self.n_instruments = n_instruments
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def preprocess_data(self, prices):
        all_features = []
        for i in range(self.n_instruments):
            df = pd.DataFrame(prices[i], columns=['close'])
            df['returns'] = df['close'].pct_change()
            
            # Technical indicators
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_30'] = df['close'].rolling(window=30).mean()
            df['rsi'] = self.calculate_rsi(df['close'])
            df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
            df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Momentum
            df['momentum'] = df['close'] - df['close'].shift(10)
            
            all_features.append(df)
        
        # Cross-sectional features
        for t in range(len(all_features[0])):
            closes = np.array([df['close'].iloc[t] for df in all_features])
            returns = np.array([df['returns'].iloc[t] for df in all_features])
            
            for i, df in enumerate(all_features):
                df.loc[df.index[t], 'rel_strength'] = (df['close'].iloc[t] / np.mean(closes)) - 1
                df.loc[df.index[t], 'rel_return'] = df['returns'].iloc[t] - np.mean(returns)
                df.loc[df.index[t], 'return_rank'] = pd.Series(returns).rank().iloc[i] / len(returns)
        
        # Calculate correlations
        correlation_matrix = pd.DataFrame([df['returns'] for df in all_features]).T.corr()
        
        # Add correlation features
        for i, df in enumerate(all_features):
            df['avg_correlation'] = correlation_matrix.iloc[i].mean()
            df['max_correlation'] = correlation_matrix.iloc[i].max()
            df['min_correlation'] = correlation_matrix.iloc[i].min()
        
        # Combine all features
        combined_features = pd.concat([df.add_suffix(f'_{i}') for i, df in enumerate(all_features)], axis=1)
        combined_features.dropna(inplace=True)
        
        return combined_features

    @staticmethod
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def train(self, prices):
        features = self.preprocess_data(prices)
        X = features.drop([col for col in features.columns if 'returns' in col], axis=1)
        y = features[[col for col in features.columns if 'returns' in col]].shift(-1)  # Predict next day's returns
        
        # Remove the last row as it won't have a target value
        X = X.iloc[:-1]
        y = y.iloc[:-1]
        
        # Define model and hyperparameter space
        model = RandomForestRegressor(random_state=42)
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, 
            n_iter=20, cv=tscv, scoring='neg_mean_squared_error', 
            random_state=42, n_jobs=-1
        )
        
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('model', random_search)
        ])
        
        pipeline.fit(X, y)
        
        self.model = pipeline
        self.is_trained = True
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best score: {-random_search.best_score_}")

    def predict(self, prices):
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        features = self.preprocess_data(prices)
        X = features.drop([col for col in features.columns if 'returns' in col], axis=1).iloc[-1].values.reshape(1, -1)
        predictions = self.model.predict(X)[0]
        return predictions

model = AdvancedMLStrategy(nInst)

def getMyPosition(prcSoFar):
    global currentPos, model
    (nins, nt) = prcSoFar.shape
    
    if nt < 100:  # Increased minimum data points for training
        return np.zeros(nins)
    
    if not model.is_trained and nt >= 200:  # Increased training data requirement
        model.train(prcSoFar)
    
    if model.is_trained:
        predictions = model.predict(prcSoFar)
        
        # Convert predictions to desired positions
        desired_positions = predictions * 5000  # Reduced position sizing
        
        # Apply position limits ($10k per instrument)
        max_positions = 10000 / prcSoFar[:, -1]
        desired_positions = np.clip(desired_positions, -max_positions, max_positions)
        
        # Risk management: reduce exposure in high volatility regime
        returns = np.diff(np.log(prcSoFar[:, -21:]), axis=1)
        volatility = np.std(returns, axis=1)
        volatility_factor = 1 / (1 + np.exp(10 * (volatility - np.mean(volatility))))  # More aggressive volatility adjustment
        desired_positions *= volatility_factor
        
        # Convert to integer positions
        desired_positions = np.round(desired_positions).astype(int)
        
        # Calculate the change in positions
        position_changes = desired_positions - currentPos
        
        # Apply a maximum change of 10% of the maximum allowed position
        max_change = 0.1 * max_positions
        position_changes = np.clip(position_changes, -max_change, max_change)
        
        # Update current positions
        currentPos += position_changes.astype(int)
    
    return currentPos