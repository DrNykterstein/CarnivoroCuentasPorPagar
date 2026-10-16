"""
Motor de Decisiones Financieras para Cuentas por Pagar - Carnívoros.
Calcula el descuento real, tasa implícita, break-even en USDT, ahorro esperado y contraoferta sugerida.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.monte_carlo import MonteCarloSimulator, SimulationResult


@dataclass
class InvoiceEvaluation:
    # Datos de Entrada
    monto_ves: float
    monto_usdt_propuesto: float
    dias_credito: int
    factura_indexada_bcv: bool

    # Tasas Actuales
    tasa_bcv_hoy: float
    tasa_usdt_hoy: float
    brecha_actual_pct: float

    # Análisis Nominal vs Real
    monto_usd_bcv_original: float
    descuento_nominal_pct: float
    costo_usdt_hoy_en_ves: float
    descuento_real_hoy_pct: float
    tasa_implicita_proveedor: float

    # Pronóstico ML de Mercado
    brecha_pronosticada_pct: float
    tasa_bcv_pronosticada: float
    tasa_usdt_pronosticada: float
    tendencia_brecha: str

    # Umbrales Break-even y Contraoferta
    monto_usdt_max_breakeven: float
    descuento_nominal_minimo_requerido_pct: float
    contraoferta_usdt_sugerida: float
    ahorro_con_contraoferta_ves: float

    # Resultados Estocásticos (Monte Carlo)
    probabilidad_exito_pct: float
    ahorro_esperado_ves: float
    ahorro_esperado_usdt: float
    ahorro_pesimista_p10_ves: float
    ahorro_optimista_p90_ves: float
    var_95_ves: float

    # Decisión Final
    veredicto: str                  # "PAGAR EN USDT", "PAGAR EN BOLÍVARES", "NEGOCIAR CONTRAOFERTA"
    color_veredicto: str            # "green", "red", "yellow"
    diagnostico: str
    justificacion_detallada: str
    mc_result: Optional[SimulationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class DecisionEngine:
    """Evalúa la conveniencia de pagar una factura en USDT vs Bolívares integrando ML y Monte Carlo."""

    def __init__(self, mc_simulations: int = config.MC_SIMULATIONS):
        self.mc_simulator = MonteCarloSimulator(n_simulations=mc_simulations)

    def evaluate_invoice(
        self,
        monto_ves: float,
        monto_usdt_propuesto: float,
        dias_credito: int,
        live_rates: Dict[str, Any],
        forecast_horizon: Dict[str, Any],
        factura_indexada_bcv: bool = True,
        costo_oportunidad_mensual_bs: float = config.DEFAULT_MONTHLY_TREASURY_RATE
    ) -> InvoiceEvaluation:
        """
        Realiza la evaluación exhaustiva y cuantitativa de la factura de compra.
        """
        # 1. Extracción de tasas actuales
        tasa_bcv_hoy = float(live_rates.get("bcv") or 880.0)
        tasa_usdt_hoy = float(live_rates.get("tasa_usdt_referencia") or live_rates.get("paralelo") or 1000.0)
        brecha_actual_pct = float(live_rates.get("brecha_pct") or 13.6)

        # 2. Extracción de pronósticos ML
        brecha_pronosticada_pct = float(forecast_horizon.get("brecha_esperada_pct") or brecha_actual_pct)
        tasa_bcv_pronosticada = float(forecast_horizon.get("tasa_bcv_proyectada") or tasa_bcv_hoy)
        tasa_usdt_pronosticada = float(forecast_horizon.get("tasa_usdt_proyectada") or tasa_usdt_hoy)
        tendencia_brecha = forecast_horizon.get("tendencia_brecha") or "ESTABLE"

        # 3. Cálculos Base
        monto_usd_bcv_original = monto_ves / tasa_bcv_hoy
        descuento_nominal_pct = (1.0 - (monto_usdt_propuesto / monto_usd_bcv_original)) * 100.0

        # Costo hoy en Bolívares si pagamos en USDT
        costo_usdt_hoy_en_ves = monto_usdt_propuesto * tasa_usdt_hoy
        # Descuento real en poder adquisitivo hoy
        descuento_real_hoy_pct = ((monto_ves - costo_usdt_hoy_en_ves) / monto_ves) * 100.0
        # Tasa de cambio implícita a la que el proveedor está tomando el USDT
        tasa_implicita_proveedor = monto_ves / (monto_usdt_propuesto + 1e-9)

        # 4. Umbral de Indiferencia (Break-even)
        # Factor de costo de oportunidad de mantener los Bolívares durante los días de crédito
        rendimiento_periodo_bs = (1.0 + costo_oportunidad_mensual_bs) ** (dias_credito / 30.0) - 1.0

        if factura_indexada_bcv:
            # A vencimiento pagamos: monto_usd_bcv_original * tasa_bcv_futura
            # Para que pagar en USDT hoy sea indiferente:
            # monto_usdt * tasa_usdt_hoy * (1 + r) = monto_usd_bcv_original * tasa_bcv_futura
            # monto_usdt = (monto_usd_bcv_original * tasa_bcv_futura) / (tasa_usdt_hoy * (1 + r))
            monto_usdt_max_breakeven = (monto_usd_bcv_original * tasa_bcv_pronosticada) / (tasa_usdt_hoy * (1.0 + rendimiento_periodo_bs))
        else:
            # Factura fija en Bolívares: a vencimiento se pagan monto_ves exactamente
            monto_usdt_max_breakeven = monto_ves / (tasa_usdt_hoy * (1.0 + rendimiento_periodo_bs))

        # Descuento nominal mínimo que debe ofrecer el proveedor para alcanzar el break-even
        descuento_nominal_minimo_requerido_pct = (1.0 - (monto_usdt_max_breakeven / monto_usd_bcv_original)) * 100.0

        # Contraoferta sugerida: monto en USDT que asegura a Carnívoros un 5% de ahorro real neto garantizado
        contraoferta_usdt_sugerida = monto_usdt_max_breakeven * 0.95
        costo_contraoferta_ves = contraoferta_usdt_sugerida * tasa_usdt_hoy * (1.0 + rendimiento_periodo_bs)
        costo_futuro_esperado_ves = (monto_usd_bcv_original * tasa_bcv_pronosticada) if factura_indexada_bcv else monto_ves
        ahorro_con_contraoferta_ves = costo_futuro_esperado_ves - costo_contraoferta_ves

        # 5. Ejecutar Simulación Monte Carlo
        mc_res: SimulationResult = self.mc_simulator.simulate(
            monto_ves=monto_ves,
            monto_usdt_propuesto=monto_usdt_propuesto,
            dias_credito=dias_credito,
            current_bcv=tasa_bcv_hoy,
            current_usdt=tasa_usdt_hoy,
            current_gap=brecha_actual_pct,
            target_gap_forecast=brecha_pronosticada_pct,
            factura_indexada_bcv=factura_indexada_bcv,
            tasa_rendimiento_mensual_bs=costo_oportunidad_mensual_bs
        )

        # 6. Lógica de Veredicto y Diagnóstico
        prob = mc_res.probabilidad_ahorro_positivo
        ahorro_esp = mc_res.ahorro_esperado_ves

        if prob >= 70.0 and ahorro_esp > 0:
            veredicto = "PAGAR EN USDT"
            color_veredicto = "green"
            diagnostico = f"¡Excelente oportunidad! El descuento del {descuento_nominal_pct:.1f}% en USDT compensa con creces la brecha cambiaria y el crédito."
            justificacion = (
                f"Al pagar {monto_usdt_propuesto:.2f} USDT hoy, la empresa obtiene un ahorro neto estimado de "
                f"Bs. {ahorro_esp:,.2f} ({mc_res.ahorro_esperado_usdt:.2f} USDT) frente a pagar en Bolívares en {dias_credito} días. "
                f"La tasa implícita ofrecida por el proveedor (Bs. {tasa_implicita_proveedor:,.2f}/USDT) es superior a la tasa de mercado "
                f"(Bs. {tasa_usdt_hoy:,.2f}/USDT), lo que confirma un descuento financiero genuino con {prob:.1f}% de probabilidad de éxito."
            )
        elif prob <= 35.0 or ahorro_esp < 0:
            veredicto = "PAGAR EN BOLÍVARES"
            color_veredicto = "red"
            perdida_est = abs(ahorro_esp)
            diagnostico = f"¡ALERTA DE PÉRDIDA! Pagar en USDT generaría una pérdida de poder adquisitivo de aprox. Bs. {perdida_est:,.2f}."
            justificacion = (
                f"Aunque el proveedor ofrece un descuento nominal de {descuento_nominal_pct:.1f}%, la brecha cambiaria actual del "
                f"{brecha_actual_pct:.1f}% (Binance P2P vs BCV) y su tendencia de {tendencia_brecha} anulan el beneficio. "
                f"Comprar o gastar {monto_usdt_propuesto:.2f} USDT hoy a Bs. {tasa_usdt_hoy:,.2f} cuesta más que pagar la factura en Bolívares. "
                f"Para que sea rentable, el proveedor debería aceptar como máximo {monto_usdt_max_breakeven:.2f} USDT (descuento nominal mínimo de {descuento_nominal_minimo_requerido_pct:.1f}%)."
            )
        else:
            veredicto = "NEGOCIAR CONTRAOFERTA"
            color_veredicto = "yellow"
            diagnostico = f"Margen muy estrecho (Probabilidad {prob:.1f}%). Se recomienda presentar contraoferta en USDT o pagar en Bolívares."
            justificacion = (
                f"El descuento propuesto está muy cerca del punto de equilibrio ({monto_usdt_max_breakeven:.2f} USDT). "
                f"Cualquier movimiento adverso en la brecha o en la devaluación del BCV podría eliminar el beneficio. "
                f"Se sugiere contraofertar al proveedor {contraoferta_usdt_sugerida:.2f} USDT para asegurar un beneficio real neto de Bs. {ahorro_con_contraoferta_ves:,.2f}."
            )

        return InvoiceEvaluation(
            monto_ves=round(monto_ves, 2),
            monto_usdt_propuesto=round(monto_usdt_propuesto, 2),
            dias_credito=dias_credito,
            factura_indexada_bcv=factura_indexada_bcv,
            tasa_bcv_hoy=round(tasa_bcv_hoy, 2),
            tasa_usdt_hoy=round(tasa_usdt_hoy, 2),
            brecha_actual_pct=round(brecha_actual_pct, 2),
            monto_usd_bcv_original=round(monto_usd_bcv_original, 2),
            descuento_nominal_pct=round(descuento_nominal_pct, 2),
            costo_usdt_hoy_en_ves=round(costo_usdt_hoy_en_ves, 2),
            descuento_real_hoy_pct=round(descuento_real_hoy_pct, 2),
            tasa_implicita_proveedor=round(tasa_implicita_proveedor, 2),
            brecha_pronosticada_pct=round(brecha_pronosticada_pct, 2),
            tasa_bcv_pronosticada=round(tasa_bcv_pronosticada, 2),
            tasa_usdt_pronosticada=round(tasa_usdt_pronosticada, 2),
            tendencia_brecha=tendencia_brecha,
            monto_usdt_max_breakeven=round(monto_usdt_max_breakeven, 2),
            descuento_nominal_minimo_requerido_pct=round(descuento_nominal_minimo_requerido_pct, 2),
            contraoferta_usdt_sugerida=round(contraoferta_usdt_sugerida, 2),
            ahorro_con_contraoferta_ves=round(ahorro_con_contraoferta_ves, 2),
            probabilidad_exito_pct=mc_res.probabilidad_ahorro_positivo,
            ahorro_esperado_ves=mc_res.ahorro_esperado_ves,
            ahorro_esperado_usdt=mc_res.ahorro_esperado_usdt,
            ahorro_pesimista_p10_ves=mc_res.ahorro_p10_ves,
            ahorro_optimista_p90_ves=mc_res.ahorro_p90_ves,
            var_95_ves=mc_res.var_95_ves,
            veredicto=veredicto,
            color_veredicto=color_veredicto,
            diagnostico=diagnostico,
            justificacion_detallada=justificacion,
            mc_result=mc_res
        )


if __name__ == "__main__":
    from src.data_connector import ExchangeRateConnector
    from src.feature_engineering import build_features
    from src.ml_models import BrechaForecaster

    conn = ExchangeRateConnector()
    df_raw = conn.get_historical_data()
    feats = build_features(df_raw)
    live = conn.get_live_rates()

    forecaster = BrechaForecaster()
    forecaster.train_models(feats)
    preds = forecaster.predict_horizons(feats, live)
    horizon_15 = forecaster.interpolate_horizon(15, preds)

    engine = DecisionEngine()
    
    # Caso 1: Factura de 100,000 Bs (equivale a ~113.6 USD BCV), ofrecen pagar 108 USDT (un mísero 5% nominal)
    eval1 = engine.evaluate_invoice(
        monto_ves=100000.0,
        monto_usdt_propuesto=108.0,
        dias_credito=15,
        live_rates=live,
        forecast_horizon=horizon_15
    )
    print("=== CASO 1: Descuento trampa de 5% nominal ===")
    print(f"Veredicto: {eval1.veredicto}")
    print(f"Diagnóstico: {eval1.diagnostico}")
    print(f"Ahorro esperado: {eval1.ahorro_esperado_ves} Bs")
    print(f"Break-even USDT: {eval1.monto_usdt_max_breakeven} USDT")

    # Caso 2: Factura de 100,000 Bs, ofrecen pagar 85 USDT (descuento nominal de ~25%)
    eval2 = engine.evaluate_invoice(
        monto_ves=100000.0,
        monto_usdt_propuesto=85.0,
        dias_credito=15,
        live_rates=live,
        forecast_horizon=horizon_15
    )
    print("\n=== CASO 2: Descuento genuino de 25% nominal ===")
    print(f"Veredicto: {eval2.veredicto}")
    print(f"Diagnóstico: {eval2.diagnostico}")
    print(f"Ahorro esperado: {eval2.ahorro_esperado_ves} Bs")
    print(f"Contraoferta sugerida: {eval2.contraoferta_usdt_sugerida} USDT")
