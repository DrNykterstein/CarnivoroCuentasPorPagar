"""
Ingeniería de Características (Feature Engineering) para series de tiempo de tasas y brecha cambiaria.
Genera variables técnicas, volatilidad, momentos y estacionalidad de pagos comerciales (quincenas, fin de mes).
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calcula el Relative Strength Index (RSI) para una serie temporal."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def build_features(df_raw: pd.DataFrame, horizons: List[int] = [7, 15, 30]) -> pd.DataFrame:
    """
    Toma el dataframe con ['bcv', 'paralelo', 'brecha_pct'] y calcula todas las variables predictivas y objetivos.
    """
    df = df_raw.copy()
    df.sort_index(inplace=True)

    # 1. Retornos diarios y spreads
    df["bcv_ret_1d"] = df["bcv"].pct_change() * 100.0
    df["paralelo_ret_1d"] = df["paralelo"].pct_change() * 100.0
    df["brecha_diff_1d"] = df["brecha_pct"].diff()

    # 2. Medias Móviles (Tendencias)
    for w in [7, 14, 30, 60]:
        df[f"brecha_ma_{w}"] = df["brecha_pct"].rolling(window=w).mean()
        df[f"bcv_ret_{w}d"] = (df["bcv"] / df["bcv"].shift(w) - 1.0) * 100.0
        df[f"paralelo_ret_{w}d"] = (df["paralelo"] / df["paralelo"].shift(w) - 1.0) * 100.0

    # 3. Volatilidad (Desviación estándar móvil de cambios diarios)
    for w in [7, 14, 30]:
        df[f"brecha_vol_{w}d"] = df["brecha_diff_1d"].rolling(window=w).std()
        df[f"bcv_vol_{w}d"] = df["bcv_ret_1d"].rolling(window=w).std()

    # 4. Indicadores Técnicos de Momento de la Brecha
    df["brecha_rsi_14"] = calculate_rsi(df["brecha_pct"], period=14)
    # MACD de la brecha: MA rápida (7d) menos MA lenta (30d)
    df["brecha_macd"] = df["brecha_ma_7"] - df["brecha_ma_30"]
    # Aceleración de la brecha (segunda derivada)
    df["brecha_accel"] = df["brecha_diff_1d"].diff()

    # 5. Lags autorregresivos
    for lag in [1, 2, 3, 7, 14]:
        df[f"brecha_lag_{lag}"] = df["brecha_pct"].shift(lag)
        df[f"bcv_ret_lag_{lag}"] = df["bcv_ret_1d"].shift(lag)

    # 6. Factores de Calendario y Ciclos de Liquidez en Venezuela
    # Quincenas (12 al 16) y Fin de Mes (27 al 31 y 1 al 3) generan fuerte presión de demanda de divisas
    days = df.index.day
    df["day_of_month"] = days
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_quincena"] = ((days >= 12) & (days <= 16)).astype(int)
    df["is_fin_de_mes"] = ((days >= 27) | (days <= 3)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 7. Variables Objetivo (Targets a predecir para horizontes de crédito comercial)
    for h in horizons:
        # Brecha futura en h días
        df[f"target_brecha_{h}d"] = df["brecha_pct"].shift(-h)
        # Devaluación / Crecimiento acumulado del BCV en h días
        df[f"target_bcv_growth_{h}d"] = ((df["bcv"].shift(-h) - df["bcv"]) / df["bcv"]) * 100.0

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Retorna la lista de columnas numéricas predictivas, excluyendo targets y precios absolutos."""
    exclude_prefixes = ("target_",)
    exclude_cols = {"bcv", "paralelo"}
    
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if np.issubdtype(df[c].dtype, np.number):
            cols.append(c)
    return cols


if __name__ == "__main__":
    from data_connector import ExchangeRateConnector
    conn = ExchangeRateConnector()
    df_raw = conn.get_historical_data()
    features_df = build_features(df_raw)
    feat_cols = get_feature_columns(features_df)
    print(f"Dataset con features creado. Filas: {len(features_df)}, Columnas: {len(features_df.columns)}")
    print(f"Variables predictoras ({len(feat_cols)}): {feat_cols[:10]}...")
    print(features_df[feat_cols].tail(5))
