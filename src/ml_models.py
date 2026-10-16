"""
Modelos de Machine Learning para pronóstico de la brecha cambiaria y evolución del BCV.
Utiliza HistGradientBoostingRegressor con estimación puntual y por cuantiles (conos de incertidumbre P10 - P90)
y validación cruzada temporal (TimeSeriesSplit).
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_connector import ExchangeRateConnector
from src.feature_engineering import build_features, get_feature_columns

logger = logging.getLogger(__name__)


class BrechaForecaster:
    """Entrena y ejecuta pronósticos de brecha y tasas para horizontes de crédito comercial."""

    def __init__(self, models_dir: Path = config.MODELS_DIR):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.feature_cols: List[str] = []
        self.is_trained: bool = False

    def train_models(self, df_features: pd.DataFrame, horizons: List[int] = config.HORIZONS_DAYS) -> Dict[str, Any]:
        """
        Entrena modelos de HistGradientBoosting (Punto medio, P10 y P90) para cada horizonte.
        """
        self.feature_cols = get_feature_columns(df_features)
        results = {}

        for h in horizons:
            target_brecha = f"target_brecha_{h}d"
            target_bcv = f"target_bcv_growth_{h}d"

            # Filtrar filas válidas donde el target no sea NaN (excluir los últimos h días)
            valid_mask = df_features[target_brecha].notna() & df_features[target_bcv].notna()
            train_df = df_features[valid_mask].copy()

            if len(train_df) < 50:
                logger.warning(f"Insuficientes datos para entrenar horizonte {h}d ({len(train_df)} filas).")
                continue

            X = train_df[self.feature_cols]
            y_brecha = train_df[target_brecha]
            y_bcv = train_df[target_bcv]

            # Filtrar columnas constantes (sin varianza) para evitar errores en binning
            nunique = X.nunique()
            active_cols = nunique[nunique > 1].index.tolist()
            if not active_cols:
                logger.warning(f"No hay características con varianza para el horizonte {h}d.")
                continue

            X_active = X[active_cols]

            # 1. Validación Cruzada Temporal para Brecha
            n_splits = min(config.CV_SPLITS, max(2, len(X_active) // 30))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_mae_list = []
            cv_rmse_list = []

            for train_idx, val_idx in tscv.split(X_active):
                X_tr, X_val = X_active.iloc[train_idx], X_active.iloc[val_idx]
                y_tr, y_val = y_brecha.iloc[train_idx], y_brecha.iloc[val_idx]

                # Filtrar columnas constantes dentro del fold de entrenamiento
                fold_cols = [c for c in active_cols if X_tr[c].nunique() > 1]
                if not fold_cols:
                    continue

                m_cv = HistGradientBoostingRegressor(
                    random_state=config.RANDOM_STATE,
                    max_iter=100,
                    min_samples_leaf=min(20, max(2, len(X_tr) // 10))
                )
                m_cv.fit(X_tr[fold_cols], y_tr)
                preds = m_cv.predict(X_val[fold_cols])
                cv_mae_list.append(mean_absolute_error(y_val, preds))
                cv_rmse_list.append(root_mean_squared_error(y_val, preds))

            # 2. Entrenar Modelo de Brecha Medio (Squared Error)
            model_brecha_mean = HistGradientBoostingRegressor(
                loss="squared_error",
                random_state=config.RANDOM_STATE,
                max_iter=200,
                learning_rate=0.05,
                min_samples_leaf=min(20, max(2, len(X_active) // 10))
            )
            model_brecha_mean.fit(X_active, y_brecha)

            # 3. Entrenar Cuantiles para Conos de Incertidumbre (P10 y P90)
            model_brecha_p10 = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.10,
                random_state=config.RANDOM_STATE,
                max_iter=150,
                learning_rate=0.05,
                min_samples_leaf=min(20, max(2, len(X_active) // 10))
            )
            model_brecha_p10.fit(X_active, y_brecha)

            model_brecha_p90 = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.90,
                random_state=config.RANDOM_STATE,
                max_iter=150,
                learning_rate=0.05,
                min_samples_leaf=min(20, max(2, len(X_active) // 10))
            )
            model_brecha_p90.fit(X_active, y_brecha)

            # 4. Entrenar Modelo para Crecimiento Acumulado del BCV
            model_bcv = HistGradientBoostingRegressor(
                loss="squared_error",
                random_state=config.RANDOM_STATE,
                max_iter=150,
                learning_rate=0.05,
                min_samples_leaf=min(20, max(2, len(X_active) // 10))
            )
            model_bcv.fit(X_active, y_bcv)

            self.models[f"active_cols_{h}d"] = active_cols

            # Guardar en memoria
            self.models[f"brecha_mean_{h}d"] = model_brecha_mean
            self.models[f"brecha_p10_{h}d"] = model_brecha_p10
            self.models[f"brecha_p90_{h}d"] = model_brecha_p90
            self.models[f"bcv_growth_{h}d"] = model_bcv

            metrics_h = {
                "cv_mae": round(float(np.mean(cv_mae_list)), 2),
                "cv_rmse": round(float(np.mean(cv_rmse_list)), 2),
                "train_samples": len(train_df)
            }
            results[f"{h}d"] = metrics_h

        self.metrics = results
        self.is_trained = True
        logger.info(f"Modelos entrenados exitosamente para horizontes: {list(results.keys())}")
        return results

    def predict_horizons(self, latest_features_df: pd.DataFrame, current_rates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera pronósticos para 7, 15 y 30 días usando la última fila de características.
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")

        last_row = latest_features_df[self.feature_cols].tail(1)
        current_bcv = current_rates.get("bcv") or float(latest_features_df["bcv"].iloc[-1])
        current_usdt = current_rates.get("tasa_usdt_referencia") or float(latest_features_df["paralelo"].iloc[-1])
        current_gap = current_rates.get("brecha_pct") or float(latest_features_df["brecha_pct"].iloc[-1])

        forecasts = {
            "current_state": {
                "bcv": round(current_bcv, 2),
                "usdt": round(current_usdt, 2),
                "brecha_pct": round(current_gap, 2)
            },
            "horizons": {}
        }

        for h in config.HORIZONS_DAYS:
            key_mean = f"brecha_mean_{h}d"
            if key_mean not in self.models:
                continue

            active_cols = self.models.get(f"active_cols_{h}d", self.feature_cols)
            row_h = last_row[active_cols]

            pred_gap_mean = float(self.models[key_mean].predict(row_h)[0])
            pred_gap_p10 = float(self.models[f"brecha_p10_{h}d"].predict(row_h)[0])
            pred_gap_p90 = float(self.models[f"brecha_p90_{h}d"].predict(row_h)[0])
            pred_bcv_growth = float(self.models[f"bcv_growth_{h}d"].predict(row_h)[0])

            # Prevenir valores de brecha física/económicamente absurdos (ej. brecha negativa extrema en Venezuela)
            pred_gap_mean = max(pred_gap_mean, 2.0)
            pred_gap_p10 = max(pred_gap_p10, 0.5)
            pred_gap_p90 = max(pred_gap_p90, pred_gap_mean)

            # Estimar tasas absolutas futuras
            projected_bcv = current_bcv * (1.0 + pred_bcv_growth / 100.0)
            projected_usdt_mean = projected_bcv * (1.0 + pred_gap_mean / 100.0)
            projected_usdt_p90 = projected_bcv * (1.0 + pred_gap_p90 / 100.0)

            forecasts["horizons"][h] = {
                "dias": h,
                "brecha_esperada_pct": round(pred_gap_mean, 2),
                "brecha_min_p10_pct": round(pred_gap_p10, 2),
                "brecha_max_p90_pct": round(pred_gap_p90, 2),
                "bcv_crecimiento_estimado_pct": round(pred_bcv_growth, 2),
                "tasa_bcv_proyectada": round(projected_bcv, 2),
                "tasa_usdt_proyectada": round(projected_usdt_mean, 2),
                "tendencia_brecha": "EXPANSIÓN" if pred_gap_mean > (current_gap + 0.5) else ("COMPRESIÓN" if pred_gap_mean < (current_gap - 0.5) else "ESTABLE")
            }

        return forecasts

    def interpolate_horizon(self, days: int, multi_forecast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpola el pronóstico para cualquier día arbitrario (ej. 10 días, 21 días, 45 días).
        """
        horizons = sorted(multi_forecast["horizons"].keys())
        if days in multi_forecast["horizons"]:
            return multi_forecast["horizons"][days]

        # Si es menor al mínimo o mayor al máximo
        if days <= horizons[0]:
            base = multi_forecast["horizons"][horizons[0]]
            ratio = max(days / horizons[0], 0.1)
            gap = multi_forecast["current_state"]["brecha_pct"] + (base["brecha_esperada_pct"] - multi_forecast["current_state"]["brecha_pct"]) * ratio
            bcv_growth = base["bcv_crecimiento_estimado_pct"] * ratio
        elif days >= horizons[-1]:
            base = multi_forecast["horizons"][horizons[-1]]
            extrap_ratio = days / horizons[-1]
            gap = base["brecha_esperada_pct"]
            bcv_growth = base["bcv_crecimiento_estimado_pct"] * extrap_ratio
        else:
            # Encontrar los dos horizontes circundantes
            h_low = [h for h in horizons if h < days][-1]
            h_high = [h for h in horizons if h > days][0]
            weight = (days - h_low) / (h_high - h_low)
            b_low = multi_forecast["horizons"][h_low]
            b_high = multi_forecast["horizons"][h_high]

            gap = b_low["brecha_esperada_pct"] + weight * (b_high["brecha_esperada_pct"] - b_low["brecha_esperada_pct"])
            bcv_growth = b_low["bcv_crecimiento_estimado_pct"] + weight * (b_high["bcv_crecimiento_estimado_pct"] - b_low["bcv_crecimiento_estimado_pct"])

        current_bcv = multi_forecast["current_state"]["bcv"]
        projected_bcv = current_bcv * (1.0 + bcv_growth / 100.0)
        projected_usdt = projected_bcv * (1.0 + gap / 100.0)

        current_gap = multi_forecast["current_state"]["brecha_pct"]
        tendencia = "EXPANSIÓN" if gap > (current_gap + 0.5) else ("COMPRESIÓN" if gap < (current_gap - 0.5) else "ESTABLE")

        return {
            "dias": days,
            "brecha_esperada_pct": round(gap, 2),
            "brecha_min_p10_pct": round(gap * 0.85, 2),
            "brecha_max_p90_pct": round(gap * 1.20, 2),
            "bcv_crecimiento_estimado_pct": round(bcv_growth, 2),
            "tasa_bcv_proyectada": round(projected_bcv, 2),
            "tasa_usdt_proyectada": round(projected_usdt, 2),
            "tendencia_brecha": tendencia
        }


if __name__ == "__main__":
    conn = ExchangeRateConnector()
    df_raw = conn.get_historical_data()
    feats = build_features(df_raw)
    live = conn.get_live_rates()

    forecaster = BrechaForecaster()
    metrics = forecaster.train_models(feats)
    print("Métricas de validación cruzada:")
    print(metrics)

    preds = forecaster.predict_horizons(feats, live)
    print("\nPronósticos de Brecha y Tasas por Horizonte:")
    import pprint
    pprint.pprint(preds)
