"""
Simulador de Monte Carlo para análisis estocástico de riesgo en cuentas por pagar.
Simula miles de trayectorias posibles de la brecha cambiaria y la tasa BCV a lo largo del plazo de crédito,
arrojando probabilidades de éxito, Value-at-Risk (VaR) y distribuciones de ahorro esperado.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class SimulationResult:
    probabilidad_ahorro_positivo: float
    ahorro_esperado_ves: float
    ahorro_esperado_usdt: float
    ahorro_mediana_ves: float
    ahorro_p10_ves: float  # Escenario pesimista (Percentil 10)
    ahorro_p90_ves: float  # Escenario optimista (Percentil 90)
    var_95_ves: float       # Value at Risk al 95%
    ahorros_ves_muestras: np.ndarray
    brechas_finales_muestras: np.ndarray
    bcv_finales_muestras: np.ndarray


class MonteCarloSimulator:
    """Ejecuta simulaciones estocásticas multivariadas para evaluar el riesgo de pago en USDT vs Bolívares."""

    def __init__(self, n_simulations: int = 2000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed

    def calibrate_parameters(self, df_features: pd.DataFrame) -> Dict[str, float]:
        """
        Calibra la volatilidad y velocidad de reversión a partir de la serie de tiempo reciente (últimos 90 días).
        """
        recent = df_features.tail(90).copy()
        daily_bcv_ret = recent["bcv_ret_1d"].dropna() / 100.0
        daily_gap_diff = recent["brecha_diff_1d"].dropna()

        sigma_bcv = float(daily_bcv_ret.std()) if len(daily_bcv_ret) > 10 else 0.003
        mu_bcv = float(daily_bcv_ret.mean()) if len(daily_bcv_ret) > 10 else 0.0015
        sigma_gap = float(daily_gap_diff.std()) if len(daily_gap_diff) > 10 else 0.8
        
        # Velocidad de reversión a la media empírica (Ornstein-Uhlenbeck)
        theta_gap = 0.08

        return {
            "mu_bcv": max(mu_bcv, 0.0005),
            "sigma_bcv": max(sigma_bcv, 0.001),
            "sigma_gap": max(sigma_gap, 0.4),
            "theta_gap": theta_gap
        }

    def simulate(
        self,
        monto_ves: float,
        monto_usdt_propuesto: float,
        dias_credito: int,
        current_bcv: float,
        current_usdt: float,
        current_gap: float,
        target_gap_forecast: float,
        factura_indexada_bcv: bool = True,
        tasa_rendimiento_mensual_bs: float = 0.02,
        params: Optional[Dict[str, float]] = None
    ) -> SimulationResult:
        """
        Ejecuta la simulación Monte Carlo para un plazo de crédito específico con protecciones robustas.
        """
        np.random.seed(self.random_seed)
        n = self.n_simulations

        # Sanitización robusta contra None y tipos
        current_bcv = float(current_bcv or 880.0)
        current_usdt = float(current_usdt or 1000.0)
        current_gap = float(current_gap or 13.6)
        if target_gap_forecast is None or np.isnan(target_gap_forecast):
            target_gap_forecast = current_gap
        else:
            target_gap_forecast = float(target_gap_forecast)

        monto_ves = float(monto_ves or 0.0)
        monto_usdt_propuesto = float(monto_usdt_propuesto or 0.0)
        dias_credito = int(dias_credito or 0)
        tasa_rendimiento_mensual_bs = float(tasa_rendimiento_mensual_bs or 0.02)

        # Caso especial: Pago de contado inmediato (0 días de crédito)
        if dias_credito <= 0:
            final_bcv = np.full(n, current_bcv)
            final_gap = np.full(n, current_gap)
            costo_usdt_hoy_en_ves = monto_usdt_propuesto * current_usdt
            costo_ves_en_h = np.full(n, monto_ves)
            ahorros_ves = costo_ves_en_h - costo_usdt_hoy_en_ves
            ahorros_usdt = ahorros_ves / (current_usdt + 1e-9)
            single_ahorro = float(ahorros_ves[0])
            prob_pos = 100.0 if single_ahorro > 0 else 0.0

            return SimulationResult(
                probabilidad_ahorro_positivo=prob_pos,
                ahorro_esperado_ves=round(single_ahorro, 2),
                ahorro_esperado_usdt=round(float(ahorros_usdt[0]), 2),
                ahorro_mediana_ves=round(single_ahorro, 2),
                ahorro_p10_ves=round(single_ahorro, 2),
                ahorro_p90_ves=round(single_ahorro, 2),
                var_95_ves=round(single_ahorro, 2),
                ahorros_ves_muestras=ahorros_ves,
                brechas_finales_muestras=final_gap,
                bcv_finales_muestras=final_bcv
            )

        h = dias_credito

        # Si no se pasan parámetros, usar defaults robustos
        if not params:
            params = {
                "mu_bcv": 0.0018,
                "sigma_bcv": 0.004,
                "sigma_gap": 0.85,
                "theta_gap": 0.08
            }

        # Matriz de trayectorias
        bcv_paths = np.zeros((n, h + 1))
        gap_paths = np.zeros((n, h + 1))

        bcv_paths[:, 0] = current_bcv
        gap_paths[:, 0] = current_gap

        dt = 1.0  # Paso diario
        
        # Simulación paso a paso
        for t in range(h):
            # 1. Brecha vía Ornstein-Uhlenbeck hacia el target pronosticado por ML
            z_gap = np.random.normal(0, 1, n)
            drift_gap = params["theta_gap"] * (target_gap_forecast - gap_paths[:, t]) * dt
            diff_gap = params["sigma_gap"] * np.sqrt(dt) * z_gap
            next_gap = gap_paths[:, t] + drift_gap + diff_gap
            # Acotar para evitar brechas negativas absurdas
            gap_paths[:, t + 1] = np.clip(next_gap, 1.0, 60.0)

            # 2. BCV vía Movimiento Browniano Geométrico
            z_bcv = np.random.normal(0, 1, n)
            drift_bcv = (params["mu_bcv"] - 0.5 * (params["sigma_bcv"] ** 2)) * dt
            diff_bcv = params["sigma_bcv"] * np.sqrt(dt) * z_bcv
            bcv_paths[:, t + 1] = bcv_paths[:, t] * np.exp(drift_bcv + diff_bcv)

        final_bcv = bcv_paths[:, -1]
        final_gap = gap_paths[:, -1]
        final_usdt = final_bcv * (1.0 + final_gap / 100.0)

        # Costo de oportunidad de los Bolívares (tasa libre de riesgo o colocación)
        rendimiento_periodo_bs = (1.0 + tasa_rendimiento_mensual_bs) ** (h / 30.0) - 1.0

        # Costo de la opción A: Pagar HOY en USDT
        costo_usdt_hoy_en_ves = monto_usdt_propuesto * current_usdt
        # Llevado a valor futuro en el día H (capitalizando al costo de oportunidad del capital en Bs):
        costo_usdt_vf_en_ves = costo_usdt_hoy_en_ves * (1.0 + rendimiento_periodo_bs)

        # Costo de la opción B: Pagar en el DÍA H en Bolívares
        if factura_indexada_bcv:
            # Si está indexada al BCV: Monto_USD_BCV * Tasa_BCV(H)
            monto_usd_bcv_original = monto_ves / max(current_bcv, 1e-6)
            costo_ves_en_h = monto_usd_bcv_original * final_bcv
        else:
            # Factura fija en bolívares
            costo_ves_en_h = np.full(n, monto_ves)

        # Ahorro neto en Bolívares por haber pagado en USDT vs pagar en Bs a vencimiento:
        ahorros_ves = costo_ves_en_h - costo_usdt_vf_en_ves
        ahorros_ves = np.nan_to_num(ahorros_ves, nan=0.0)
        
        # Ahorro expresado en USDT equivalente al momento de la decisión
        ahorros_usdt = ahorros_ves / max(current_usdt, 1e-6)

        # Métricas estadísticas
        prob_ahorro_pos = float(np.mean(ahorros_ves > 0)) * 100.0
        ahorro_esperado_ves = float(np.mean(ahorros_ves))
        ahorro_esperado_usdt = float(np.mean(ahorros_usdt))
        ahorro_mediana_ves = float(np.median(ahorros_ves))
        ahorro_p10 = float(np.percentile(ahorros_ves, 10))
        ahorro_p90 = float(np.percentile(ahorros_ves, 90))
        var_95 = float(np.percentile(ahorros_ves, 5))  # Peor 5%

        return SimulationResult(
            probabilidad_ahorro_positivo=round(prob_ahorro_pos, 1),
            ahorro_esperado_ves=round(ahorro_esperado_ves, 2),
            ahorro_esperado_usdt=round(ahorro_esperado_usdt, 2),
            ahorro_mediana_ves=round(ahorro_mediana_ves, 2),
            ahorro_p10_ves=round(ahorro_p10, 2),
            ahorro_p90_ves=round(ahorro_p90, 2),
            var_95_ves=round(var_95, 2),
            ahorros_ves_muestras=ahorros_ves,
            brechas_finales_muestras=final_gap,
            bcv_finales_muestras=final_bcv
        )


if __name__ == "__main__":
    sim = MonteCarloSimulator(n_simulations=3000)
    # Ejemplo: Factura de 100,000 Bs, ofrecen pagar 95 USDT con 15 días de crédito
    res = sim.simulate(
        monto_ves=100000.0,
        monto_usdt_propuesto=95.0,
        dias_credito=15,
        current_bcv=880.0,
        current_usdt=1002.0,
        current_gap=13.86,
        target_gap_forecast=14.5,
        factura_indexada_bcv=True
    )
    print("Probabilidad de Ahorro Positivo:", res.probabilidad_ahorro_positivo, "%")
    print("Ahorro Esperado en VES:", res.ahorro_esperado_ves)
    print("Ahorro Esperado en USDT:", res.ahorro_esperado_usdt)
    print("VaR al 95% (Peor 5% en VES):", res.var_95_ves)
