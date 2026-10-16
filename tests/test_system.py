"""
Pruebas automáticas unitarias y de integración para el sistema de Cuentas por Pagar.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import config
from src.data_connector import ExchangeRateConnector
from src.feature_engineering import build_features, get_feature_columns
from src.ml_models import BrechaForecaster
from src.monte_carlo import MonteCarloSimulator
from src.decision_engine import DecisionEngine


class TestCuentasPorPagar(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Crear un dataframe sintético representativo de tasas
        dates = pd.date_range(start="2026-01-01", periods=120, freq="D")
        bcv_series = np.linspace(50.0, 70.0, len(dates))
        # Brecha fluctuando entre 8% y 20%
        brecha_series = 14.0 + 3.0 * np.sin(np.linspace(0, 10, len(dates)))
        paralelo_series = bcv_series * (1.0 + brecha_series / 100.0)

        cls.df_test = pd.DataFrame({
            "bcv": bcv_series,
            "paralelo": paralelo_series,
            "brecha_pct": brecha_series
        }, index=dates)
        cls.df_test.index.name = "fecha"

    def test_feature_engineering(self):
        """Verifica que build_features genere las columnas predictivas requeridas sin errores."""
        df_feats = build_features(self.df_test)
        self.assertIn("brecha_ma_7", df_feats.columns)
        self.assertIn("brecha_rsi_14", df_feats.columns)
        self.assertIn("is_quincena", df_feats.columns)
        self.assertIn("target_brecha_15d", df_feats.columns)

        cols = get_feature_columns(df_feats)
        self.assertTrue(len(cols) >= 20)
        self.assertNotIn("target_brecha_15d", cols)

    def test_ml_forecaster(self):
        """Verifica el entrenamiento y predicción de los modelos de Machine Learning."""
        df_feats = build_features(self.df_test)
        forecaster = BrechaForecaster()
        metrics = forecaster.train_models(df_feats, horizons=[7, 15])
        self.assertIn("7d", metrics)
        self.assertIn("15d", metrics)

        live_dummy = {
            "bcv": 70.0,
            "tasa_usdt_referencia": 80.5,
            "brecha_pct": 15.0
        }
        preds = forecaster.predict_horizons(df_feats, live_dummy)
        self.assertIn("horizons", preds)
        self.assertIn(15, preds["horizons"])
        self.assertGreater(preds["horizons"][15]["brecha_esperada_pct"], 0)

    def test_monte_carlo_simulator(self):
        """Verifica la ejecución de la simulación estocástica Monte Carlo."""
        sim = MonteCarloSimulator(n_simulations=500)
        res = sim.simulate(
            monto_ves=100000.0,
            monto_usdt_propuesto=100.0,
            dias_credito=15,
            current_bcv=70.0,
            current_usdt=80.5,
            current_gap=15.0,
            target_gap_forecast=15.5
        )
        self.assertEqual(len(res.ahorros_ves_muestras), 500)
        self.assertTrue(0.0 <= res.probabilidad_ahorro_positivo <= 100.0)

    def test_monte_carlo_zero_days_and_nones(self):
        """Verifica que la simulación maneje 0 días de crédito y valores None sin fallar."""
        sim = MonteCarloSimulator(n_simulations=200)
        res_zero = sim.simulate(
            monto_ves=50000.0,
            monto_usdt_propuesto=60.0,
            dias_credito=0,
            current_bcv=None,
            current_usdt=None,
            current_gap=None,
            target_gap_forecast=None
        )
        self.assertEqual(len(res_zero.ahorros_ves_muestras), 200)
        self.assertTrue(res_zero.probabilidad_ahorro_positivo in (0.0, 100.0))
        self.assertFalse(np.isnan(res_zero.ahorro_esperado_ves))

    def test_decision_engine_loss_trap(self):
        """Verifica que el motor detecte un descuento trampa (donde pagar en USDT pierde dinero)."""
        engine = DecisionEngine(mc_simulations=500)
        # Factura de 100,000 Bs. A BCV 70 = $1,428.57. Ofrecen pagar 1,400 USDT (descuento nominal de solo 2%)
        # Con brecha de 15%, 1,400 USDT * 80.5 = 112,700 Bs (PÉRDIDA CLARA)
        live = {"bcv": 70.0, "tasa_usdt_referencia": 80.5, "brecha_pct": 15.0}
        horizon = {"brecha_esperada_pct": 15.0, "tasa_bcv_proyectada": 71.0, "tasa_usdt_proyectada": 81.65, "tendencia_brecha": "ESTABLE"}

        eval_res = engine.evaluate_invoice(
            monto_ves=100000.0,
            monto_usdt_propuesto=1400.0,
            dias_credito=15,
            live_rates=live,
            forecast_horizon=horizon
        )
        self.assertEqual(eval_res.veredicto, "PAGAR EN BOLÍVARES")
        self.assertLess(eval_res.ahorro_esperado_ves, 0)
        self.assertLess(eval_res.descuento_real_hoy_pct, 0)

    def test_decision_engine_genuine_discount(self):
        """Verifica que el motor apruebe un descuento real sustancial."""
        engine = DecisionEngine(mc_simulations=500)
        # Factura de 100,000 Bs. Ofrecen pagar 1,000 USDT (equivale a 80,500 Bs, ahorro de ~20,000 Bs)
        live = {"bcv": 70.0, "tasa_usdt_referencia": 80.5, "brecha_pct": 15.0}
        horizon = {"brecha_esperada_pct": 15.0, "tasa_bcv_proyectada": 71.0, "tasa_usdt_proyectada": 81.65, "tendencia_brecha": "ESTABLE"}

        eval_res = engine.evaluate_invoice(
            monto_ves=100000.0,
            monto_usdt_propuesto=1000.0,
            dias_credito=15,
            live_rates=live,
            forecast_horizon=horizon
        )
        self.assertEqual(eval_res.veredicto, "PAGAR EN USDT")
        self.assertGreater(eval_res.ahorro_esperado_ves, 0)
        self.assertGreater(eval_res.probabilidad_exito_pct, 70.0)


if __name__ == "__main__":
    unittest.main()
