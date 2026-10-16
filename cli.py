"""
Interfaz de Línea de Comandos (CLI) para evaluación de Cuentas por Pagar.
Permite ingresar monto en Bs y USDT para determinar si conviene pagar en USDT con Machine Learning.
"""
import argparse
import sys
from pathlib import Path

# Configurar salida UTF-8 para consola de Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.append(str(Path(__file__).resolve().parent))
import config
from src.data_connector import ExchangeRateConnector
from src.feature_engineering import build_features
from src.ml_models import BrechaForecaster
from src.decision_engine import DecisionEngine


def run_evaluation(monto_ves: float, monto_usdt: float, dias_credito: int, indexada: bool):
    console = Console(highlight=False)
    console.print(Panel.fit("[bold blue]SISTEMA INTELIGENTE DE CUENTAS POR PAGAR - CARNIVOROS[/bold blue]\n"
                            "[dim]Machine Learning & Monte Carlo para Optimizacion Cambiaria (USDT vs VES)[/dim]"))

    with console.status("[bold green]Conectando con APIs de mercado y cargando tasas en tiempo real..."):
        connector = ExchangeRateConnector()
        live_rates = connector.get_live_rates()
        df_hist = connector.get_historical_data()
        df_feats = build_features(df_hist)

    # Mostrar Tasas de Mercado
    table_rates = Table(title="[TASAS DE MERCADO EN TIEMPO REAL]")

    table_rates.add_column("Indicador", style="cyan", no_wrap=True)
    table_rates.add_column("Valor Actual", style="bold magenta")
    table_rates.add_column("Fuente", style="dim")

    table_rates.add_row("Tasa Oficial BCV", f"Bs. {live_rates['bcv']:,.2f}", "Banco Central de Venezuela")
    table_rates.add_row("Tasa USDT (Binance P2P)", f"Bs. {live_rates['tasa_usdt_referencia']:,.2f}", live_rates["fuente_usdt"])
    table_rates.add_row("Brecha Cambiaria Actual", f"{live_rates['brecha_pct']:.2f}%", "(USDT - BCV) / BCV")
    console.print(table_rates)

    with console.status(f"[bold yellow]Ejecutando modelos de Machine Learning y Monte Carlo a {dias_credito} días..."):
        forecaster = BrechaForecaster()
        forecaster.train_models(df_feats)
        all_forecasts = forecaster.predict_horizons(df_feats, live_rates)
        horizon_forecast = forecaster.interpolate_horizon(dias_credito, all_forecasts)

        engine = DecisionEngine()
        result = engine.evaluate_invoice(
            monto_ves=monto_ves,
            monto_usdt_propuesto=monto_usdt,
            dias_credito=dias_credito,
            live_rates=live_rates,
            forecast_horizon=horizon_forecast,
            factura_indexada_bcv=indexada
        )

    # Panel de Pronostico ML
    console.print("\n[bold]Pronostico del Modelo de Machine Learning (HistGradientBoosting):[/bold]")
    table_ml = Table()
    table_ml.add_column("Horizonte", style="cyan")
    table_ml.add_column("Brecha Estimada", style="bold")
    table_ml.add_column("Rango Confianza (P10 - P90)", style="dim")
    table_ml.add_column("Tendencia", style="bold")
    table_ml.add_column("Tasa USDT Proyectada", style="bold")

    table_ml.add_row(
        f"{dias_credito} dias",
        f"{result.brecha_pronosticada_pct:.2f}%",
        f"[{horizon_forecast['brecha_min_p10_pct']:.1f}% - {horizon_forecast['brecha_max_p90_pct']:.1f}%]",
        f"[{'red' if result.tendencia_brecha == 'EXPANSIÓN' else 'green'}]{result.tendencia_brecha}[/]",
        f"Bs. {result.tasa_usdt_pronosticada:,.2f}"
    )
    console.print(table_ml)

    # Panel de Veredicto
    color_map = {"green": "bold green", "red": "bold red", "yellow": "bold yellow"}
    style_veredicto = color_map.get(result.color_veredicto, "bold white")
    
    console.print(Panel(
        f"[{style_veredicto}]VEREDICTO FINAL: {result.veredicto}[/{style_veredicto}]\n\n"
        f"[bold]{result.diagnostico}[/bold]\n\n"
        f"{result.justificacion_detallada}",
        title="Recomendacion Financiera",
        border_style=result.color_veredicto
    ))

    # Metricas Clave de Negocio
    table_metrics = Table(title="[METRICAS COMPARATIVAS DE LA FACTURA]")
    table_metrics.add_column("Métrica", style="cyan")
    table_metrics.add_column("Valor", style="bold")

    table_metrics.add_row("Monto Factura en Bolívares", f"Bs. {result.monto_ves:,.2f}")
    table_metrics.add_row("Equivalente USD al BCV Original", f"${result.monto_usd_bcv_original:,.2f} USD")
    table_metrics.add_row("Monto USDT Ofrecido", f"{result.monto_usdt_propuesto:,.2f} USDT")
    table_metrics.add_row("Descuento Nominal (que dice el proveedor)", f"{result.descuento_nominal_pct:.2f}%")
    table_metrics.add_row("Descuento Real en Poder Adquisitivo Hoy", f"{result.descuento_real_hoy_pct:.2f}%")
    table_metrics.add_row("Tasa Implícita del Proveedor", f"Bs. {result.tasa_implicita_proveedor:,.2f} / USDT")
    table_metrics.add_row("Máximo USDT para Break-even (no perder)", f"{result.monto_usdt_max_breakeven:,.2f} USDT")
    table_metrics.add_row("Descuento Nominal Mínimo Requerido", f"{result.descuento_nominal_minimo_requerido_pct:.2f}%")
    table_metrics.add_row("Ahorro Esperado Neto (Monte Carlo)", f"Bs. {result.ahorro_esperado_ves:,.2f} ({result.ahorro_esperado_usdt:.2f} USDT)")
    table_metrics.add_row("Probabilidad Estadística de Ganancia", f"{result.probabilidad_exito_pct:.1f}%")
    table_metrics.add_row("Contraoferta Recomendada al Proveedor", f"[bold green]{result.contraoferta_usdt_sugerida:,.2f} USDT[/bold green]")

    console.print(table_metrics)


def main():
    parser = argparse.ArgumentParser(description="Evaluador Inteligente de Cuentas por Pagar USDT vs Bolívares")
    parser.add_argument("--monto-ves", type=float, help="Monto de la factura en Bolívares")
    parser.add_argument("--monto-usdt", type=float, help="Monto ofrecido en USDT")
    parser.add_argument("--dias", type=int, default=15, help="Días de crédito comercial (default: 15)")
    parser.add_argument("--no-indexada", action="store_true", help="Factura fija en Bs (no indexada al BCV)")

    args = parser.parse_args()

    if args.monto_ves and args.monto_usdt:
        run_evaluation(args.monto_ves, args.monto_usdt, args.dias, not args.no_indexada)
    else:
        # Modo interactivo
        print("--- Evaluador de Cuentas por Pagar (Modo Interactivo) ---")
        try:
            monto_ves = float(input("Ingrese el monto de la factura en Bolívares (Bs): "))
            monto_usdt = float(input("Ingrese el monto que le ofrecen pagar en USDT: "))
            dias_str = input("Días de crédito comercial que tiene la factura [default: 15]: ").strip()
            dias = int(dias_str) if dias_str else 15
            idx_str = input("¿La factura está indexada a la tasa oficial BCV? (S/N) [default: S]: ").strip().lower()
            indexada = False if idx_str == "n" else True

            run_evaluation(monto_ves, monto_usdt, dias, indexada)
        except KeyboardInterrupt:
            print("\nOperación cancelada por el usuario.")
        except Exception as e:
            print(f"Error en los datos ingresados: {e}")


if __name__ == "__main__":
    main()
