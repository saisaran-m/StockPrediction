import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)

class SupervisedStockPredictor:
    """
    Manages training, cross-validation, and inference for Supervised Machine Learning algorithms.
    Supports both Regression (Close Price prediction) and Classification (Directional Up/Down prediction).
    """
    def __init__(self, target_type: str = 'regression', test_size: float = 0.2):
        self.target_type = target_type
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.model_results = {}
        self.best_model_name = None
        self.best_model = None

    def _get_models(self) -> Dict[str, Any]:
        """Returns dictionary of initialized supervised learning models."""
        if self.target_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
                'Support Vector Regressor': SVR(C=10.0, epsilon=0.1)
            }
            if HAS_XGBOOST:
                models['XGBoost Regressor'] = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        else:
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
                'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
                'Support Vector Classifier': SVC(C=1.0, kernel='rbf', probability=True)
            }
            if HAS_XGBOOST:
                models['XGBoost Classifier'] = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric='logloss')
        return models

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Splits data chronologically (Time-Series Split) into Train & Test sets,
        fits supervised models, and evaluates out-of-sample performance metrics.
        """
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Fit scaler on training set only to prevent lookahead bias
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = self._get_models()
        results = []
        
        best_score = -float('inf') if self.target_type == 'regression' else -1.0
        
        for name, model in models.items():
            logger.info(f"Training supervised model: {name}...")
            
            # Linear & SVR models benefit from scaled data; tree models work with either
            if 'Linear' in name or 'Ridge' in name or 'Support Vector' in name or 'Logistic' in name:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
            self.trained_models[name] = model
            
            if self.target_type == 'regression':
                mae = mean_absolute_error(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, preds)
                
                metrics = {
                    'Model': name,
                    'MAE': round(mae, 4),
                    'RMSE': round(rmse, 4),
                    'R2_Score': round(r2, 4),
                    'predictions': preds,
                    'y_test': y_test
                }
                
                # Selection score: R2 score (higher is better)
                score = r2
            else:
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)
                
                metrics = {
                    'Model': name,
                    'Accuracy': round(acc, 4),
                    'Precision': round(prec, 4),
                    'Recall': round(rec, 4),
                    'F1_Score': round(f1, 4),
                    'predictions': preds,
                    'y_test': y_test
                }
                score = acc
                
            self.model_results[name] = metrics
            results.append(metrics)
            
            if score > best_score:
                best_score = score
                self.best_model_name = name
                self.best_model = model
                
        logger.info(f"Training Complete! Best model: {self.best_model_name}")
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['predictions', 'y_test']} for r in results])
        return self.model_results, results_df

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Extracts feature importances or coefficients from the best performing model."""
        if self.best_model is None:
            return pd.DataFrame()
            
        importances = None
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_).ravel()
            
        if importances is not None and len(importances) == len(feature_names):
            df_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
            return df_imp
        return pd.DataFrame()

if __name__ == "__main__":
    from data_loader import fetch_stock_data
    from features import build_supervised_dataset
    
    df = fetch_stock_data("AAPL", period="2y")
    X, y, _ = build_supervised_dataset(df, target_type='regression')
    predictor = SupervisedStockPredictor(target_type='regression')
    results, summary_df = predictor.train_and_evaluate(X, y)
    print("Model Evaluation Summary:\n", summary_df)
