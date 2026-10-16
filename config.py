"""
Configuración general del proyecto de Cuentas por Pagar - Carnívoros.
Módulo de Machine Learning y Ciencia de Datos para evaluación de pagos en USDT vs Bolívares.
"""
from pathlib import Path

# Rutas del Proyecto
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

HISTORICAL_DATA_PATH = DATA_DIR / "historico_tasas.csv"
SAMPLE_INVOICES_PATH = DATA_DIR / "sample_facturas.json"

# Endpoints de Tasas
DOLAR_API_BASE = "https://ve.dolarapi.com/v1"
DOLAR_API_RATES = f"{DOLAR_API_BASE}/dolares"
DOLAR_API_HISTORIC = f"{DOLAR_API_BASE}/historicos/dolares"
CRIPTOYA_BINANCE_USDT = "https://criptoya.com/api/binancep2p/usdt/ves/100"

# User-Agent para solicitudes HTTP
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Parámetros Financieros por Defecto
DEFAULT_MONTHLY_TREASURY_RATE = 0.02  # 2.0% mensual de costo de oportunidad / rendimiento en Bs
HORIZONS_DAYS = [7, 15, 30]           # Horizontes típicos de crédito comercial
DEFAULT_CREDIT_DAYS = 15

# Parámetros de Machine Learning
RANDOM_STATE = 42
CV_SPLITS = 5
MC_SIMULATIONS = 2000                 # Número de iteraciones Monte Carlo
