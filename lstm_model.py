import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class FinancialLSTM:
    def __init__(self, lookback=60, lstm_units=50, dropout_rate=0.2):
        """
        Initialise le modèle LSTM pour prédictions financières
        
        Args:
            lookback: nombre de pas de temps à regarder en arrière
            lstm_units: nombre d'unités dans les couches LSTM
            dropout_rate: taux de dropout pour régularisation
        """
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def prepare_data(self, df, target_col, feature_cols=None):
        """
        Prépare les données pour l'entraînement du LSTM
        
        Args:
            df: DataFrame pandas avec les données
            target_col: nom de la colonne à prédire
            feature_cols: liste des colonnes explicatives (None = toutes sauf target)
        """
        # Sélectionner les features
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Extraire X et y
        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)
        
        # Normaliser les données
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Créer les séquences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        return X_seq, y_seq
    
    def _create_sequences(self, X, y):
        """Crée des séquences temporelles pour le LSTM"""
        X_seq, y_seq = [], []
        
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i-self.lookback:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape):
        """Construit l'architecture du LSTM"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Entraîne le modèle LSTM"""
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Fait des prédictions et inverse la normalisation"""
        predictions_scaled = self.model.predict(X)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        return predictions
    
    def plot_predictions(self, y_true, y_pred, title="Prédictions vs Valeurs Réelles"):
        """Visualise les prédictions"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Valeurs Réelles', alpha=0.7)
        plt.plot(y_pred, label='Prédictions', alpha=0.7)
        plt.title(title)
        plt.xlabel('Temps')
        plt.ylabel('Valeur')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Exemple d'utilisation
if __name__ == "__main__":
    # 1. Charger tes données
    # df = pd.read_csv('ton_fichier.csv')
    
    # Exemple avec données simulées
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'prix': np.cumsum(np.random.randn(1000)) + 100,
        'volume': np.random.randint(1000, 10000, 1000),
        'ma_7': np.random.randn(1000),
        'ma_30': np.random.randn(1000),
        'volatilite': np.abs(np.random.randn(1000)),
        'rsi': np.random.uniform(0, 100, 1000)
    })
    
    # 2. Initialiser le modèle
    lstm_model = FinancialLSTM(lookback=60, lstm_units=50, dropout_rate=0.2)
    
    # 3. Préparer les données
    target_column = 'prix'
    feature_columns = ['volume', 'ma_7', 'ma_30', 'volatilite', 'rsi']
    
    X, y = lstm_model.prepare_data(df, target_column, feature_columns)
    
    # 4. Split train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, shuffle=False
    )
    
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape y_train: {y_train.shape}")
    
    # 5. Construire et entraîner
    lstm_model.build_model((X_train.shape[1], X_train.shape[2]))
    print("\nArchitecture du modèle:")
    lstm_model.model.summary()
    
    print("\nEntraînement du modèle...")
    history = lstm_model.train(X_train, y_train, X_val, y_val, epochs=50)
    
    # 6. Évaluer sur le test set
    predictions = lstm_model.predict(X_test)
    y_test_original = lstm_model.scaler_y.inverse_transform(y_test)
    
    # Métriques
    mse = np.mean((y_test_original - predictions)**2)
    mae = np.mean(np.abs(y_test_original - predictions))
    
    print(f"\n=== Résultats sur le Test Set ===")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    
    # 7. Visualiser
    lstm_model.plot_predictions(y_test_original, predictions)
    
    # 8. Prédire les prochaines valeurs
    # Utilise les dernières séquences pour prédire le futur
    last_sequence = X[-1:]
    next_prediction = lstm_model.predict(last_sequence)
    print(f"\nProchaine valeur prédite: {next_prediction[0][0]:.2f}")