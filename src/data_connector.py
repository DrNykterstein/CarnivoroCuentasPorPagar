"""
Conector de datos en tiempo real e históricos para tasas cambiarias en Venezuela.
Obtiene tasas del Banco Central de Venezuela (BCV), Dólar Paralelo y USDT Binance P2P.
"""
import json
import logging
import urllib.request
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ExchangeRateConnector:
    """Gestiona la obtención y sincronización de tasas cambiarias en vivo e históricas."""

    def __init__(self, cache_file: Path = config.HISTORICAL_DATA_PATH):
        self.cache_file = cache_file

    def get_live_rates(self) -> Dict[str, Any]:
        """
        Obtiene las tasas actuales del BCV y USDT/Paralelo.
        Intenta consultar DolarApi y CriptoYa (Binance P2P).
        """
        rates = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bcv": None,
            "paralelo": None,
            "usdt_binance_ask": None,
            "usdt_binance_bid": None,
            "tasa_usdt_referencia": None,
            "brecha_pct": None,
            "fuente_usdt": "CriptoYa (Binance P2P)",
            "status": "success",
            "detalles": {}
        }

        # 1. Obtener BCV y Paralelo de DolarApi
        try:
            req_dolar = urllib.request.Request(config.DOLAR_API_RATES, headers=config.HTTP_HEADERS)
            with urllib.request.urlopen(req_dolar, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data:
                    fuente = item.get("fuente", "").lower()
                    promedio = float(item.get("promedio", 0.0) or 0.0)
                    if fuente == "oficial":
                        rates["bcv"] = promedio
                    elif fuente == "paralelo":
                        rates["paralelo"] = promedio
            rates["detalles"]["dolarapi"] = "ok"
        except Exception as e:
            logger.warning(f"Error consultando DolarApi: {e}")
            rates["detalles"]["dolarapi"] = f"error: {str(e)}"

        # 2. Obtener USDT de Binance P2P vía CriptoYa
        try:
            req_cripto = urllib.request.Request(config.CRIPTOYA_BINANCE_USDT, headers=config.HTTP_HEADERS)
            with urllib.request.urlopen(req_cripto, timeout=8) as resp:
                data_cripto = json.loads(resp.read().decode("utf-8"))
                rates["usdt_binance_ask"] = float(data_cripto.get("ask", 0.0) or 0.0)
                rates["usdt_binance_bid"] = float(data_cripto.get("bid", 0.0) or 0.0)
                # Tasa de referencia USDT: precio de compra de USDT (ask) o promedio bid/ask
                rates["tasa_usdt_referencia"] = rates["usdt_binance_ask"]
                rates["fuente_usdt"] = "Binance P2P (CriptoYa)"
            rates["detalles"]["criptoya"] = "ok"
        except Exception as e:
            logger.warning(f"Error consultando CriptoYa Binance: {e}")
            rates["detalles"]["criptoya"] = f"error: {str(e)}"
            # Fallback a paralelo si CriptoYa falla
            if rates["paralelo"]:
                rates["tasa_usdt_referencia"] = rates["paralelo"]
                rates["fuente_usdt"] = "DolarApi Paralelo (Fallback)"

        # Validar fallback en caso de falla de red total
        if not rates["bcv"] or not rates["tasa_usdt_referencia"]:
            logger.warning("No se pudieron obtener tasas en vivo completas. Usando último valor histórico registrado.")
            hist_df = self.get_historical_data()
            if not hist_df.empty:
                last_row = hist_df.iloc[-1]
                rates["bcv"] = rates["bcv"] or float(last_row["bcv"])
                rates["paralelo"] = rates["paralelo"] or float(last_row["paralelo"])
                rates["tasa_usdt_referencia"] = rates["tasa_usdt_referencia"] or float(last_row["paralelo"])
                rates["fuente_usdt"] = "Caché Histórico Local"
                rates["status"] = "cached_fallback"

        # 3. Calcular Brecha Cambiaria
        if rates["bcv"] and rates["tasa_usdt_referencia"]:
            rates["brecha_pct"] = round(
                ((rates["tasa_usdt_referencia"] - rates["bcv"]) / rates["bcv"]) * 100.0, 2
            )

        return rates

    def sync_historical_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Descarga el histórico oficial de tasas desde DolarApi y lo limpia en formato de serie temporal continua.
        """
        # Si el caché existe y no se fuerza refresco y fue modificado hoy, cargar del disco
        if self.cache_file.exists() and not force_refresh:
            try:
                cached_df = pd.read_csv(self.cache_file, parse_dates=["fecha"], index_col="fecha")
                if not cached_df.empty:
                    # Comprobar fecha del último dato
                    last_date = cached_df.index.max().date()
                    if last_date >= date.today():
                        logger.info(f"Caché histórico al día ({last_date}). Cargando desde {self.cache_file}")
                        return cached_df
            except Exception as e:
                logger.warning(f"Error leyendo caché: {e}. Re-descargando...")

        logger.info("Descargando serie histórica completa desde DolarApi...")
        try:
            req = urllib.request.Request(config.DOLAR_API_HISTORIC, headers=config.HTTP_HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_data = json.loads(resp.read().decode("utf-8"))

            df = pd.DataFrame(raw_data)
            if df.empty:
                raise ValueError("Respuesta vacía de DolarApi")

            # Pivotear por fecha y fuente
            pivot = df.pivot_table(index="fecha", columns="fuente", values="promedio")
            pivot.index = pd.to_datetime(pivot.index)
            pivot.sort_index(inplace=True)

            # Estandarizar nombres de columnas
            cols = {}
            if "oficial" in pivot.columns:
                cols["oficial"] = "bcv"
            if "paralelo" in pivot.columns:
                cols["paralelo"] = "paralelo"
            pivot.rename(columns=cols, inplace=True)

            # Si paralelo tiene vacíos al inicio, reconstruir con la brecha histórica promedio o backward fill
            if "bcv" in pivot.columns and "paralelo" in pivot.columns:
                # Filtrar desde que ambas existen o rellenar de manera consistente
                # En fechas donde solo hay oficial, estimar paralelo histórico razonable (brecha histórica ~10-15%)
                valid_mask = pivot["bcv"].notna()
                pivot = pivot[valid_mask].copy()

                # Reindexar a frecuencia diaria continua para evitar saltos en fines de semana
                full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq="D")
                pivot = pivot.reindex(full_idx)
                pivot.index.name = "fecha"

                # Forward-fill para fines de semana / feriados bancarios
                pivot["bcv"] = pivot["bcv"].ffill()
                pivot["paralelo"] = pivot["paralelo"].ffill()

                # Si al principio paralelo no estaba registrado por la API, rellenar con un spread conservador del 10%
                if pivot["paralelo"].isna().any():
                    pivot["paralelo"] = pivot["paralelo"].fillna(pivot["bcv"] * 1.10)

                # Calcular brecha porcentual
                pivot["brecha_pct"] = ((pivot["paralelo"] - pivot["bcv"]) / pivot["bcv"]) * 100.0

                # Guardar en caché
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                pivot.to_csv(self.cache_file)
                logger.info(f"Histórico guardado en {self.cache_file} con {len(pivot)} registros diarios.")
                return pivot
            else:
                raise ValueError("Columnas esperadas no encontradas en el histórico.")

        except Exception as e:
            logger.error(f"Error sincronizando histórico de tasas: {e}")
            # Si falla la red pero existe el caché, usarlo
            if self.cache_file.exists():
                logger.info("Recuperando último caché disponible en disco.")
                return pd.read_csv(self.cache_file, parse_dates=["fecha"], index_col="fecha")
            raise

    def get_historical_data(self) -> pd.DataFrame:
        """Devuelve el DataFrame histórico asegurando que exista."""
        if not self.cache_file.exists():
            return self.sync_historical_data(force_refresh=True)
        return pd.read_csv(self.cache_file, parse_dates=["fecha"], index_col="fecha")


if __name__ == "__main__":
    connector = ExchangeRateConnector()
    print("--- TASAS EN VIVO ---")
    live = connector.get_live_rates()
    for k, v in live.items():
        print(f"{k}: {v}")

    print("\n--- HISTÓRICO ---")
    df_hist = connector.sync_historical_data(force_refresh=True)
    print(df_hist.tail(10))
