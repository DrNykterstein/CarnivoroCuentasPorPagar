# 🥩 Carnívoros — Optimizador Inteligente de Cuentas por Pagar (USDT vs Bolívares)

Sistema de **Ciencia de Datos, Machine Learning y Simulación Estocástica** diseñado para optimizar las decisiones de tesorería y cuentas por pagar en Venezuela. Evalúa si realmente es conveniente aceptar ofertas de proveedores para liquidar facturas comerciales en **USDT (Cripto / Dólar Paralelo)** a cambio de un descuento nominal, o si es financieramente superior pagar la factura en **Bolívares (VES)** al vencimiento del crédito.

---

## 🎯 El Problema de Negocio en Venezuela

En el sector cárnico y comercial en Venezuela, es frecuente que los proveedores emitan facturas expresadas en Bolívares o indexadas a la tasa oficial del Banco Central de Venezuela (BCV), y posteriormente ofrezcan un "descuento" si la factura se cancela de inmediato en USDT:

* *Ejemplo típico:* Factura de **Bs. 150.000,00** (equivalente a **$170,43 USD** a tasa BCV de Bs. 880,12).
* El proveedor propone: *"Págame **160,00 USDT** hoy y te liquido la factura (te doy un 6% de descuento nominal)"*.

### ⚠️ La "Trampa" del Descuento Nominal vs. Real:
1. **Brecha Cambiaria (Spread):** En el mercado, 1 USDT en Binance P2P cotiza a **Bs. 1.002,00**, mientras que la tasa oficial BCV está en **Bs. 880,12** (una brecha del **13,85%**).
2. **Costo de Reposición en Bolívares:** Para entregar 160 USDT hoy, la empresa debe desembolsar o valorar:  
   $$160\text{ USDT} \times \text{Bs. } 1.002,00 = \text{Bs. } 160.320,00$$
3. **Resultado:** En lugar de ahorrar dinero, la empresa estaría pagando **Bs. 10.320,00 MÁS** que el monto original de la factura. El descuento nominal del 6% se convierte en una **pérdida real del -6,88%**.
4. **Horizonte de Crédito y Tendencia de la Brecha:** Si además la empresa tiene **15 o 30 días de crédito comercial**, pagar hoy en USDT implica renunciar a la liquidez y al rendimiento del capital en Bolívares. Si el modelo de Machine Learning proyecta que la brecha se va a expandir, pagar en USDT hoy es aún más perjudicial.

---

## 🧠 Arquitectura de Ciencia de Datos y Machine Learning

El proyecto implementa un flujo cuantitativo riguroso:

```
[ APIs en Vivo ] --> [ Ingesta & Features ] --> [ Machine Learning ] --> [ Monte Carlo ] --> [ Motor de Decisión ]
  • DolarApi (BCV)      • 41 Variables Técnicas    • HistGradientBoosting   • 2.000 Trayectorias  • Descuento Real
  • Binance P2P (USDT)   • Medias Móviles 7/14/30d • Conos P10 - P90        • Ornstein-Uhlenbeck • Break-even USDT
  • Histórico Diario     • RSI & Volatilidad       • Crecimiento BCV        • Value-at-Risk 95%   • Contraoferta Sugerida
```

### 1. Ingesta de Datos en Tiempo Real y Serie Histórica
- **Conexión en vivo:** Consulta automática sin credenciales a `ve.dolarapi.com` (Tasa BCV y Paralelo) y `criptoya.com` (Orderbook de Binance P2P USDT/VES).
- **Serie temporal:** Más de 1.380 registros diarios históricos continuos con interpolación de feriados y fines de semana.

### 2. Ingeniería de Características (Feature Engineering)
- **Momento y Osciladores:** Relative Strength Index (RSI a 14 días), MACD de la brecha cambiaria, aceleración de segundo orden ($\Delta^2$).
- **Volatilidad Rodante:** Desviación estándar de retornos a 7, 14 y 30 días.
- **Estacionalidad y Ciclos de Liquidez en Venezuela:**
  - `is_quincena`: Días 12 al 16 (presión por pago de nóminas y compras de divisas).
  - `is_fin_de_mes`: Días 27 al 3 (cierre de mes comercial, liquidación de inventarios y anticipos de ISLR/IVA de SENIAT).

### 3. Modelado Predictivo con Machine Learning
- **Algoritmo:** `HistGradientBoostingRegressor` (Gradient Boosting optimizado para series de tiempo tabulares).
- **Validación Cruzada:** `TimeSeriesSplit` hacia adelante con 5 folds temporales estrictos sin fuga de datos (*data leakage*).
- **Conos de Incertidumbre:** Modelos entrenados con función de pérdida por cuantiles (`quantile=0.10` y `quantile=0.90`) para proyectar bandas de confianza a 7, 15 y 30 días.

### 4. Simulación Estocástica de Monte Carlo
- Modela 2.000 trayectorias del mercado mediante:
  - **Brecha Cambiaria:** Proceso de reversión a la media de **Ornstein-Uhlenbeck** anclado a la proyección de Machine Learning.
  - **Tasa BCV:** Movimiento Browniano Geométrico (GBM) calibrado con el *drift* y volatilidad empírica reciente.
- Calcula la **Probabilidad de Éxito (%)** y el **Value-at-Risk (VaR 95%)** ante shocks cambiarios adversos.

### 5. Motor de Decisión y Contraoferta
- **Tasa Implícita del Proveedor:** Evalúa si la cotización a la que el proveedor toma el USDT está por encima o por debajo de Binance P2P.
- **Break-even USDT:** Determina el monto máximo exacto en USDT que la empresa puede pagar sin perder dinero.
- **Contraoferta Inteligente:** Calcula el monto en USDT que asegura a Carnívoros un **ahorro real garantizado del 5%**.

---

## 💻 Instalación en un Nuevo Equipo (Paso a Paso)

Para llevar este proyecto a cualquier otra computadora con Windows:

1. **Copiar la carpeta del proyecto:**
   - Copia la carpeta `Proyecto_CuentasPorPagar` a la nueva máquina (por pendrive, red o archivo `.zip`).
   - *Nota:* No es necesario copiar la carpeta `.venv` (el instalador la creará limpia y adaptada al nuevo equipo).

2. **Tener Python instalado:**
   - Si el nuevo equipo aún no tiene Python, descárgalo e instálalo desde [python.org](https://www.python.org/downloads/) (versión 3.10 o superior).
   - ⚠️ **Importante:** Al iniciar el instalador de Python, marca la casilla: **`[X] Add python.exe to PATH`** (Agregar Python al PATH).

3. **Instalación 100% Automática:**
   - Simplemente haz doble clic sobre el archivo **`instalar.bat`** (o directamente en **`run_app.bat`**).
   - El instalador se encargará de todo:
     1. Verificará que Python esté listo.
     2. Creará un entorno virtual aislado (`.venv`).
     3. Instalará automáticamente todas las librerías necesarias desde `requirements.txt`.
     4. Ejecutará las pruebas unitarias para validar que todo funcione correctamente.
     5. Te ofrecerá iniciar la aplicación de inmediato.

---

## 🚀 Guía de Uso Rápido

### Opción 1: Aplicación Web Interactiva (Streamlit)
La forma más amigable para el equipo de compras y cuentas por pagar:

1. Simplemente haz doble clic sobre el archivo **`run_app.bat`** en la carpeta del proyecto.
2. O desde PowerShell/Terminal ejecuta:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
3. Se abrirá automáticamente en tu navegador (`http://localhost:8501`) con:
   - Panel de tasas en vivo (BCV, Binance P2P, Brecha actual).
   - Selector de facturas de ejemplo o ingreso manual.
   - Semáforo y tarjeta grande de veredicto (**PAGAR EN USDT**, **PAGAR EN BOLÍVARES**, o **NEGOCIAR**).
   - Conos de proyección de Machine Learning.
   - Histograma de distribución Monte Carlo de ahorros y VaR 95%.
   - Texto listo para copiar y pegar por WhatsApp para enviar la contraoferta al proveedor.

---

### Opción 2: Línea de Comandos (CLI)
Para consultas rápidas por terminal:

* **Modo directo con argumentos:**
  ```powershell
  .\.venv\Scripts\python.exe cli.py --monto-ves 150000 --monto-usdt 130 --dias 15
  ```

* **Modo interactivo (pregunta paso a paso):**
  ```powershell
  .\.venv\Scripts\python.exe cli.py
  ```

---

### Opción 3: Ejecución de Pruebas Unitarias
Para verificar la integridad matemática y algorítmica de todos los módulos:
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_system.py
```

---

## 📁 Estructura del Proyecto

```
Proyecto_CuentasPorPagar/
├── .venv/                         # Entorno virtual de Python con todas las dependencias
├── config.py                      # Configuración general, URLs y parámetros financieros
├── requirements.txt               # Lista de dependencias del proyecto
├── README.md                      # Documentación completa y guía de uso
├── instalar.bat                   # Instalador automático en 1 clic para nuevos equipos
├── run_app.bat                    # Lanzador rápido de la app web en Windows
├── cli.py                         # Herramienta de consola para evaluación rápida
├── app.py                         # Aplicación Web Dashboard interactiva con Streamlit
│
├── src/
│   ├── __init__.py
│   ├── data_connector.py          # Conector a APIs (DolarApi, CriptoYa) y caché local
│   ├── feature_engineering.py     # RSI, medias móviles, volatilidad, quincenas y fin de mes
│   ├── ml_models.py               # HistGradientBoosting, cuantiles P10/P90 y TimeSeriesSplit
│   ├── monte_carlo.py             # Simulación Ornstein-Uhlenbeck y evaluación de riesgo
│   └── decision_engine.py         # Motor de decisión de cuentas por pagar y contraoferta
│
├── data/
│   ├── historico_tasas.csv        # Serie temporal diaria descargada y limpia
│   └── sample_facturas.json       # Facturas de prueba de compras cárnicas e insumos
│
└── tests/
    ├── __init__.py
    └── test_system.py             # Pruebas unitarias automatizadas
```

---

## 🥩 Desarrollado para Carnívoros
*Cuentas por Pagar & Inteligencia Financiera aplicada a Mercados Cambiarios.*
