
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import config
from src.data_connector import ExchangeRateConnector
from src.feature_engineering import build_features
from src.ml_models import BrechaForecaster
from src.decision_engine import DecisionEngine

# Configuración de página minimalista
st.set_page_config(
    page_title="Carnívoros | Cuentas por Pagar USDT vs Bs",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Paleta de Colores Institucional
C_PRIMARY_DARK = "#3a2d32"  # Charcoal / Dark Aubergine
C_ACCENT_SAGE = "#a2ad91"   # Sage / Muted Olive Green
C_ACCENT_ROSE = "#dbb4ad"   # Dusty Rose / Terracotta Blush
C_CANVAS_BG = "#FBF9F8"     # Soft Warm Canvas
C_CARD_BG = "#FFFFFF"       # Clean White

# Configurar fuente Roboto para Matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto', 'DejaVu Sans', 'Arial', 'sans-serif']


def render_html(html_code: str):
    """
    Renderiza HTML puro de forma nativa sin pasar por el motor de Markdown.
    Evita que las etiquetas HTML con sangría o saltos de línea sean interpretadas
    como bloques de código preformateado (<pre><code>).
    """
    if hasattr(st, "html"):
        st.html(html_code)
    else:
        st.markdown(textwrap.dedent(html_code).strip(), unsafe_allow_html=True)


st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    
    html, body, .stApp {{
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: {C_CANVAS_BG} !important;
        color: {C_PRIMARY_DARK};
    }}

    p, span, label, h1, h2, h3, h4, h5, h6, input, button, select, textarea, .stMarkdown {{
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    /* Ocultar barra lateral */
    [data-testid="stSidebar"] {{
        display: none;
    }}

    /* Contenedor central espacioso */
    .block-container {{
        max-width: 1040px;
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
    }}

    /* Header y Branding */
    .brand-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: {C_PRIMARY_DARK};
        color: {C_ACCENT_ROSE};
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        padding: 5px 14px;
        border-radius: 9999px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}
    .hero-title {{
        font-size: 2.2rem;
        font-weight: 900;
        color: {C_PRIMARY_DARK};
        letter-spacing: -0.025em;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }}
    .hero-sub {{
        font-size: 0.98rem;
        color: #6b5c62;
        margin-bottom: 22px;
        font-weight: 400;
    }}

    /* Tarjetas de Mercado en Vivo */
    .rate-card {{
        background: {C_CARD_BG};
        border: 1px solid rgba(58, 45, 50, 0.1);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(58, 45, 50, 0.04);
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .rate-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(58, 45, 50, 0.07);
    }}
    .rate-label {{
        font-size: 0.74rem;
        color: #7b6c72;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }}
    .rate-value {{
        font-size: 1.48rem;
        font-weight: 800;
        color: {C_PRIMARY_DARK};
        letter-spacing: -0.02em;
    }}
    .rate-badge-sage {{
        background-color: rgba(162, 173, 145, 0.22);
        color: #3f4734;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 6px;
    }}
    .rate-badge-rose {{
        background-color: rgba(219, 180, 173, 0.35);
        color: #5c352f;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 6px;
    }}
    .rate-badge-dark {{
        background-color: rgba(58, 45, 50, 0.08);
        color: {C_PRIMARY_DARK};
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 6px;
    }}

    /* Sección de Formulario de Factura */
    .form-container {{
        background: {C_CARD_BG};
        border: 1px solid rgba(58, 45, 50, 0.12);
        border-radius: 16px;
        padding: 22px 26px;
        margin-top: 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(58, 45, 50, 0.03);
    }}
    .form-header {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {C_PRIMARY_DARK};
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Veredictos de Decisión con la Paleta */
    .verdict-box-green {{
        background: linear-gradient(135deg, rgba(162, 173, 145, 0.22) 0%, rgba(162, 173, 145, 0.10) 100%);
        border: 2px solid {C_ACCENT_SAGE};
        border-radius: 16px;
        padding: 24px 28px;
        color: {C_PRIMARY_DARK};
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(162, 173, 145, 0.18);
    }}
    .verdict-box-red {{
        background: linear-gradient(135deg, rgba(219, 180, 173, 0.32) 0%, rgba(219, 180, 173, 0.14) 100%);
        border: 2px solid {C_ACCENT_ROSE};
        border-radius: 16px;
        padding: 24px 28px;
        color: {C_PRIMARY_DARK};
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(219, 180, 173, 0.22);
    }}
    .verdict-box-yellow {{
        background: linear-gradient(135deg, rgba(58, 45, 50, 0.06) 0%, rgba(219, 180, 173, 0.20) 100%);
        border: 2px solid {C_PRIMARY_DARK};
        border-radius: 16px;
        padding: 24px 28px;
        color: {C_PRIMARY_DARK};
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(58, 45, 50, 0.08);
    }}
    .verdict-tag {{
        display: inline-block;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 8px;
    }}
    .verdict-tag-green {{
        background: {C_ACCENT_SAGE};
        color: #242c1b;
    }}
    .verdict-tag-red {{
        background: {C_ACCENT_ROSE};
        color: #4f2520;
    }}
    .verdict-tag-yellow {{
        background: {C_PRIMARY_DARK};
        color: {C_ACCENT_ROSE};
    }}
    .verdict-headline {{
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        color: {C_PRIMARY_DARK};
        letter-spacing: -0.02em;
    }}
    .verdict-summary {{
        font-size: 1.02rem;
        font-weight: 600;
        color: {C_PRIMARY_DARK};
        margin-bottom: 8px;
    }}
    .verdict-body {{
        font-size: 0.94rem;
        line-height: 1.6;
        color: #4a3d42;
    }}

    /* Tarjetas KPI de Métricas */
    .kpi-tile {{
        background: {C_CARD_BG};
        border: 1px solid rgba(58, 45, 50, 0.1);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(58, 45, 50, 0.03);
        position: relative;
        overflow: hidden;
    }}
    .kpi-tile::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }}
    .kpi-tile-sage::before {{
        background-color: {C_ACCENT_SAGE};
    }}
    .kpi-tile-rose::before {{
        background-color: {C_ACCENT_ROSE};
    }}
    .kpi-tile-dark::before {{
        background-color: {C_PRIMARY_DARK};
    }}
    .kpi-heading {{
        font-size: 0.74rem;
        color: #7b6c72;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}
    .kpi-stat {{
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }}
    .kpi-footnote {{
        font-size: 0.76rem;
        margin-top: 4px;
        color: #8c7b82;
        font-weight: 500;
    }}

    /* Estilización de Pestañas Nativas de Streamlit */
    div[data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid rgba(58, 45, 50, 0.08);
        padding-bottom: 4px;
        margin-top: 14px;
        margin-bottom: 18px;
    }}
    div[data-baseweb="tab"] {{
        background-color: transparent !important;
        border: none !important;
        color: #7b6c72 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="tab"][aria-selected="true"] {{
        background-color: rgba(58, 45, 50, 0.08) !important;
        color: {C_PRIMARY_DARK} !important;
        font-weight: 700 !important;
    }}

    /* Tarjetas de Comparación de Opciones (Reemplazo elegante de st.dataframe) */
    .comparison-card {{
        background: #FFFFFF;
        border: 1px solid rgba(58, 45, 50, 0.12);
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 3px 10px rgba(58, 45, 50, 0.04);
        margin-bottom: 16px;
    }}
    .comparison-card-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 12px;
        margin-bottom: 14px;
        border-bottom: 2px solid rgba(58, 45, 50, 0.1);
    }}
    .comparison-title {{
        font-size: 1.05rem;
        font-weight: 800;
        color: {C_PRIMARY_DARK};
        margin: 0;
    }}
    .custom-table {{
        width: 100%;
        border-collapse: collapse;
    }}
    .custom-table tr {{
        border-bottom: 1px solid rgba(58, 45, 50, 0.07);
    }}
    .custom-table tr:last-child {{
        border-bottom: none;
    }}
    .custom-table td {{
        padding: 10px 4px;
        font-size: 0.92rem;
    }}
    .col-concept {{
        color: #6b5c62;
        font-weight: 500;
    }}
    .col-val {{
        text-align: right;
        color: {C_PRIMARY_DARK};
        font-weight: 700;
    }}
    .badge-modalidad {{
        background: rgba(162, 173, 145, 0.22);
        color: #2f3923;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }}

    /* Tarjeta Ficha Técnica del Modelo ML */
    .ml-model-card {{
        background: #FFFFFF;
        border: 1.5px solid rgba(58, 45, 50, 0.14);
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 14px rgba(58, 45, 50, 0.04);
        margin-bottom: 20px;
    }}
    .badge-ml {{
        background: {C_PRIMARY_DARK};
        color: {C_ACCENT_ROSE};
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}
    .ml-stats-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 14px;
        background: {C_CANVAS_BG};
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(58, 45, 50, 0.08);
    }}
    .ml-stat-item {{
        text-align: center;
    }}
    .ml-stat-label {{
        font-size: 0.72rem;
        color: #7b6c72;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 2px;
    }}
    .ml-stat-val {{
        font-size: 1.25rem;
        font-weight: 800;
        color: {C_PRIMARY_DARK};
    }}
    .ml-stat-sub {{
        font-size: 0.72rem;
        color: #8c7b82;
        margin-top: 2px;
    }}

    /* Tabla de Proyecciones ML */
    .projections-table {{
        width: 100%;
        border-collapse: collapse;
        background: #FFFFFF;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(58, 45, 50, 0.12);
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(58, 45, 50, 0.03);
    }}
    .projections-table th {{
        background: {C_PRIMARY_DARK};
        color: #FFFFFF;
        padding: 12px 14px;
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-align: left;
    }}
    .projections-table td {{
        padding: 11px 14px;
        font-size: 0.88rem;
        border-bottom: 1px solid rgba(58, 45, 50, 0.07);
        color: {C_PRIMARY_DARK};
    }}
    .projections-table tr:hover {{
        background-color: rgba(162, 173, 145, 0.08);
    }}
    .trend-pill-expansion {{
        background: rgba(219, 180, 173, 0.4);
        color: #5c352f;
        font-size: 0.74rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }}
    .trend-pill-compression {{
        background: rgba(162, 173, 145, 0.35);
        color: #242c1b;
        font-size: 0.74rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }}
    .trend-pill-stable {{
        background: rgba(58, 45, 50, 0.12);
        color: {C_PRIMARY_DARK};
        font-size: 0.74rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }}

    /* Estilo de la caja de propuesta */
    .proposal-card {{
        background: {C_CARD_BG};
        border: 1px solid {C_ACCENT_ROSE};
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(219, 180, 173, 0.12);
    }}
    .highlight-usdt {{
        font-size: 2.1rem;
        font-weight: 900;
        color: {C_PRIMARY_DARK};
        margin: 4px 0 8px 0;
        letter-spacing: -0.02em;
    }}
    .badge-margin {{
        background: rgba(162, 173, 145, 0.25);
        color: #373f2e;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_market_data():
    connector = ExchangeRateConnector()
    live_rates = connector.get_live_rates()
    df_hist = connector.get_historical_data()
    df_feats = build_features(df_hist)
    return live_rates, df_hist, df_feats


@st.cache_resource
def get_trained_forecaster(df_feats_hash, _df_feats):
    forecaster = BrechaForecaster()
    forecaster.train_models(_df_feats)
    return forecaster


def main():

    st.markdown('<h1 class="hero-title">Optimizador de Pagos USDT vs. Bolívares</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Evaluación cuantitativa de descuentos comerciales en criptoactivos, brecha cambiaria, Machine Learning y simulación Monte Carlo.</p>', unsafe_allow_html=True)

    # 2. Cargar datos de mercado
    with st.spinner("Sincronizando tasas de mercado y calibrando modelos..."):
        try:
            live_rates, df_hist, df_feats = load_market_data()
            forecaster = get_trained_forecaster(len(df_feats), df_feats)
            forecasts = forecaster.predict_horizons(df_feats, live_rates)
        except Exception as e:
            st.error(f"Error cargando datos de tasas: {e}")
            st.stop()

    bcv_val = float(live_rates.get("bcv") or 880.0)
    usdt_val = float(live_rates.get("tasa_usdt_referencia") or 1000.0)
    brecha_val = float(live_rates.get("brecha_pct") or 13.6)
    pred_15 = forecasts["horizons"].get(15, {})
    tendencia_15 = pred_15.get("tendencia_brecha", "ESTABLE")

    # 3. Barra Central de Tasas de Mercado
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_html(f"""<div class="rate-card">
<div class="rate-label">Oficial BCV</div>
<div class="rate-value">Bs. {bcv_val:,.2f}</div>
<span class="rate-badge-dark">Tasa Oficial</span>
</div>""")
    with c2:
        render_html(f"""<div class="rate-card">
<div class="rate-label">Binance P2P (USDT)</div>
<div class="rate-value">Bs. {usdt_val:,.2f}</div>
<span class="rate-badge-sage">Mercado USDT</span>
</div>""")
    with c3:
        render_html(f"""<div class="rate-card">
<div class="rate-label">Brecha Cambiaria</div>
<div class="rate-value">{brecha_val:.2f}%</div>
<span class="rate-badge-rose">+{usdt_val - bcv_val:,.2f} Bs/USD</span>
</div>""")
    with c4:
        badge_cls = "rate-badge-rose" if tendencia_15 == "EXPANSIÓN" else "rate-badge-sage"
        render_html(f"""<div class="rate-card">
<div class="rate-label">Proyección ML (15d)</div>
<div class="rate-value">{pred_15.get('brecha_esperada_pct', brecha_val):.1f}%</div>
<span class="{badge_cls}">{tendencia_15}</span>
</div>""")

    # Sección Central de Ingreso de Datos
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown('<div class="form-header">📋 Datos de la Factura y Oferta del Proveedor</div>', unsafe_allow_html=True)

    # Cargar facturas de ejemplo
    ejemplos = []
    if config.SAMPLE_INVOICES_PATH.exists():
        try:
            with open(config.SAMPLE_INVOICES_PATH, "r", encoding="utf-8") as f:
                ejemplos = json.load(f)
        except Exception:
            pass

    opciones_ej = ["Ingreso Manual de Factura"] + [f"{item['id']} — {item['proveedor'][:30]} ({item['monto_ves']:,.0f} Bs)" for item in ejemplos]
    sel_factura = st.selectbox("Cargar caso de compra preestablecido:", opciones_ej)

    # Valores por defecto
    def_ves = 150000.0
    def_usdt = 135.0
    def_dias = 15
    def_indexada = True

    if sel_factura != "Ingreso Manual de Factura":
        idx = opciones_ej.index(sel_factura) - 1
        item = ejemplos[idx]
        def_ves = float(item["monto_ves"])
        def_usdt = float(item["monto_usdt_propuesto"])
        def_dias = int(item["dias_credito"])
        def_indexada = bool(item["factura_indexada_bcv"])
        st.caption(f"📌 **Concepto:** {item.get('descripcion', '')} — *{item.get('nota', '')}*")

    # Cuadrícula de 3 columnas para inputs
    row1_c1, row1_c2, row1_c3 = st.columns([1.2, 1.2, 1])

    with row1_c1:
        monto_ves = st.number_input(
            "Monto de la Factura en Bolívares (Bs):",
            min_value=100.0,
            value=def_ves,
            step=5000.0,
            format="%.2f"
        )
        equiv_bcv_usd = monto_ves / max(bcv_val, 1e-4)
        st.caption(f"Equivalente a tasa oficial BCV: **${equiv_bcv_usd:,.2f} USD**")

    with row1_c2:
        tipo_oferta = st.radio("Formato de la propuesta USDT:", ["Monto en USDT", "% Descuento Nominal"], horizontal=True)
        if tipo_oferta == "Monto en USDT":
            monto_usdt = st.number_input(
                "Monto solicitado en USDT:",
                min_value=1.0,
                value=def_usdt,
                step=5.0,
                format="%.2f"
            )
        else:
            pct_desc = st.slider("% Descuento que afirma dar el proveedor:", 0.0, 50.0, 15.0, 0.5)
            monto_usdt = equiv_bcv_usd * (1.0 - pct_desc / 100.0)
            st.caption(f"Monto resultante: **{monto_usdt:,.2f} USDT**")

    with row1_c3:
        dias_credito = st.number_input(
            "Días de Crédito Comercial:",
            min_value=0,
            max_value=90,
            value=def_dias,
            step=5,
            help="Plazo otorgado para cancelar en Bolívares si no se acepta la oferta en USDT hoy (0 = contado)."
        )
        factura_indexada = st.checkbox("Factura indexada a tasa oficial BCV", value=def_indexada)

    # Opciones Avanzadas de Tesorería (deshabilitadas para simplificar UX)
    # with st.expander("⚙️ Opciones Avanzadas de Tesorería"):
    #     col_adv1, col_adv2 = st.columns(2)
    #     with col_adv1:
    #         costo_oportunidad = st.slider(
    #             "Rendimiento / Costo de oportunidad mensual en Bs (%):",
    #             min_value=0.0, max_value=8.0, value=2.0, step=0.5
    #         ) / 100.0
    #     with col_adv2:
    #         st.caption("Costo de oportunidad del capital en bolívares mantenido en tesorería o colocación durante los días de crédito.")
    costo_oportunidad = config.DEFAULT_MONTHLY_TREASURY_RATE

    st.markdown('</div>', unsafe_allow_html=True)

    # Motor de Decisiones
    horizon_forecast = forecaster.interpolate_horizon(dias_credito, forecasts)
    engine = DecisionEngine()
    result = engine.evaluate_invoice(
        monto_ves=monto_ves,
        monto_usdt_propuesto=monto_usdt,
        dias_credito=dias_credito,
        live_rates=live_rates,
        forecast_horizon=horizon_forecast,
        factura_indexada_bcv=factura_indexada,
        costo_oportunidad_mensual_bs=costo_oportunidad
    )

    # 6. Gran Tarjeta de Veredicto (Estilizada con Paleta)
    if result.veredicto == "PAGAR EN USDT":
        box_class = "verdict-box-green"
        tag_class = "verdict-tag-green"
        icono = ""
    elif result.veredicto == "PAGAR EN BOLÍVARES":
        box_class = "verdict-box-red"
        tag_class = "verdict-tag-red"
        icono = ""
    else:
        box_class = "verdict-box-yellow"
        tag_class = "verdict-tag-yellow"
        icono = ""

    render_html(f"""<div class="{box_class}">
<span class="verdict-tag {tag_class}">{icono} RECOMENDACIÓN DE TESORERÍA</span>
<div class="verdict-headline">{result.veredicto}</div>
<div class="verdict-summary">{result.diagnostico}</div>
<div class="verdict-body">{result.justificacion_detallada}</div>
</div>""")

    # Métricas Clave de Decisión
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        desc_color = C_ACCENT_SAGE if result.descuento_real_hoy_pct > 0 else C_ACCENT_ROSE
        render_html(f"""<div class="kpi-tile kpi-tile-sage">
<div class="kpi-heading">Descuento Real Efectivo</div>
<div class="kpi-stat" style="color: {desc_color};">{result.descuento_real_hoy_pct:.1f}%</div>
<div class="kpi-footnote">Nominal proveedor: {result.descuento_nominal_pct:.1f}%</div>
</div>""")

    with k2:
        ahorro_color = C_ACCENT_SAGE if result.ahorro_esperado_ves > 0 else C_ACCENT_ROSE
        render_html(f"""<div class="kpi-tile kpi-tile-dark">
<div class="kpi-heading">Ahorro Neto Estimado</div>
<div class="kpi-stat" style="color: {ahorro_color};">Bs. {result.ahorro_esperado_ves:,.0f}</div>
<div class="kpi-footnote">{result.ahorro_esperado_usdt:,.2f} USDT equivalentes</div>
</div>""")

    with k3:
        render_html(f"""<div class="kpi-tile kpi-tile-rose">
<div class="kpi-heading">Break-Even Máximo USDT</div>
<div class="kpi-stat" style="color: {C_PRIMARY_DARK};">{result.monto_usdt_max_breakeven:,.2f}</div>
<div class="kpi-footnote">Desc. mínimo req: {result.descuento_nominal_minimo_requerido_pct:.1f}%</div>
</div>""")

    with k4:
        prob_color = C_ACCENT_SAGE if result.probabilidad_exito_pct >= 70 else (C_ACCENT_ROSE if result.probabilidad_exito_pct <= 35 else C_PRIMARY_DARK)
        render_html(f"""<div class="kpi-tile kpi-tile-sage">
<div class="kpi-heading">Probabilidad de Éxito</div>
<div class="kpi-stat" style="color: {prob_color};">{result.probabilidad_exito_pct:.1f}%</div>
<div class="kpi-footnote">2,000 iteraciones estocásticas</div>
</div>""")

    st.write("")

    # Para Negociaciar
    tab_options, tab_montecarlo, tab_ml, tab_history = st.tabs([
        "🤝 Comparación y Contraoferta",
        "🎲 Simulación Monte Carlo",
        "🧠 Machine Learning",
        "📈 Histórico Cambiario"
    ])

    # COMPARACIÓN Y CONTRAOFERTA
    with tab_options:
        st.markdown("#### Comparativa de Desembolso Financiero")
        col_c1, col_c2 = st.columns(2)

        rendimiento_periodo_bs = (1.0 + costo_oportunidad) ** (dias_credito / 30.0) - 1.0
        costo_total_usdt_vf = result.costo_usdt_hoy_en_ves * (1.0 + rendimiento_periodo_bs)
        costo_futuro_bcv = (result.monto_usd_bcv_original * result.tasa_bcv_pronosticada) if factura_indexada else result.monto_ves
        modalidad_txt = "Indexada a Tasa Oficial BCV" if factura_indexada else "Monto Fijo en Bolívares"

        with col_c1:
            card_a_html = (
                '<div class="comparison-card">\n'
                f'<div class="comparison-card-header" style="border-bottom-color: {C_ACCENT_SAGE};">\n'
                '<span class="comparison-title"> Opción A: Desembolsar HOY en USDT</span>\n'
                '<span class="badge-modalidad" style="background: rgba(162, 173, 145, 0.25);">Pago Inmediato</span>\n'
                '</div>\n'
                '<table class="custom-table">\n'
                '<tr>\n'
                '<td class="col-concept">Desembolso en Criptoactivo</td>\n'
                f'<td class="col-val"><strong>{result.monto_usdt_propuesto:,.2f} USDT</strong></td>\n'
                '</tr>\n'
                '<tr>\n'
                '<td class="col-concept">Tasa Reposición (Binance P2P)</td>\n'
                f'<td class="col-val">Bs. {result.tasa_usdt_hoy:,.2f}</td>\n'
                '</tr>\n'
                '<tr>\n'
                '<td class="col-concept">Costo Efectivo HOY en Bolívares</td>\n'
                f'<td class="col-val">Bs. {result.costo_usdt_hoy_en_ves:,.2f}</td>\n'
                '</tr>\n'
                '<tr>\n'
                f'<td class="col-concept">Costo Financiero con Oportunidad ({dias_credito}d)</td>\n'
                f'<td class="col-val" style="color: {C_PRIMARY_DARK}; font-weight: 800;">Bs. {costo_total_usdt_vf:,.2f}</td>\n'
                '</tr>\n'
                '</table>\n'
                '</div>'
            )
            render_html(card_a_html)

        with col_c2:
            card_b_html = (
                '<div class="comparison-card">\n'
                f'<div class="comparison-card-header" style="border-bottom-color: {C_ACCENT_ROSE};">\n'
                f'<span class="comparison-title"> Opción B: Cancelar en {dias_credito} días en Bolívares</span>\n'
                f'<span class="badge-modalidad" style="background: rgba(219, 180, 173, 0.3); color: #5c352f;">Crédito {dias_credito}d</span>\n'
                '</div>\n'
                '<table class="custom-table">\n'
                '<tr>\n'
                '<td class="col-concept">Modalidad Contractual</td>\n'
                f'<td class="col-val"><span class="badge-modalidad">{modalidad_txt}</span></td>\n'
                '</tr>\n'
                '<tr>\n'
                '<td class="col-concept">Monto Original Facturado</td>\n'
                f'<td class="col-val">Bs. {result.monto_ves:,.2f}</td>\n'
                '</tr>\n'
                '<tr>\n'
                '<td class="col-concept">Tasa Oficial BCV Proyectada a Vencimiento</td>\n'
                f'<td class="col-val">Bs. {result.tasa_bcv_pronosticada:,.2f}</td>\n'
                '</tr>\n'
                '<tr>\n'
                '<td class="col-concept">Pago Esperado a Vencimiento en Bolívares</td>\n'
                f'<td class="col-val" style="color: {C_PRIMARY_DARK}; font-weight: 800;">Bs. {costo_futuro_bcv:,.2f}</td>\n'
                '</tr>\n'
                '</table>\n'
                '</div>'
            )
            render_html(card_b_html)

        st.markdown("---")
        st.markdown("#### Asistente de Contraoferta y Negociación")

        col_p1, col_p2 = st.columns([1, 1.2])

        with col_p1:
            card_prop_html = (
                '<div class="proposal-card">\n'
                '<span class="badge-margin">Margen Neto Garantizado: 5%</span>\n'
                '<div style="font-size: 0.85rem; color: #7b6c72; margin-top: 8px;">Contraoferta sugerida para Carnívoros:</div>\n'
                f'<div class="highlight-usdt">{result.contraoferta_usdt_sugerida:,.2f} USDT</div>\n'
                '<div style="font-size: 0.85rem; color: #4a3d42;">\n'
                f'Ahorro neto asegurado: <strong>Bs. {result.ahorro_con_contraoferta_ves:,.2f}</strong><br>\n'
                f'Tasa implícita reconocida: <strong>Bs. {monto_ves / max(result.contraoferta_usdt_sugerida, 1e-4):,.2f} / USDT</strong>\n'
                '</div>\n'
                '</div>'
            )
            render_html(card_prop_html)

        with col_p2:
            mensaje_wa = (
                f"Estimados, en relación con la factura por Bs. {monto_ves:,.2f}, considerando la tasa de reposición "
                f"en Binance P2P y el plazo otorgado, les podemos realizar el pago de contado hoy por "
                f"{result.contraoferta_usdt_sugerida:.2f} USDT. Quedamos atentos para procesar la transacción."
            )
            st.text_area("Mensaje listo para copiar y enviar al proveedor (WhatsApp / Correo):", mensaje_wa, height=120)

    # MONTE CARLO PROTEGIDO
    with tab_montecarlo:
        st.markdown("#### Simulación Estocástica de Monte Carlo, Juan Carlos es un pajuo (2,000 Escenarios)")
        st.caption("Distribución de probabilidad del ahorro neto en Bolívares simulando fluctuaciones de la brecha y el BCV.")

        try:
            mc_res = result.mc_result
            if mc_res is None or len(mc_res.ahorros_ves_muestras) == 0:
                from src.monte_carlo import MonteCarloSimulator
                sim_fallback = MonteCarloSimulator(n_simulations=2000)
                mc_res = sim_fallback.simulate(
                    monto_ves=monto_ves,
                    monto_usdt_propuesto=monto_usdt,
                    dias_credito=dias_credito,
                    current_bcv=bcv_val,
                    current_usdt=usdt_val,
                    current_gap=brecha_val,
                    target_gap_forecast=result.brecha_pronosticada_pct,
                    factura_indexada_bcv=factura_indexada,
                    tasa_rendimiento_mensual_bs=costo_oportunidad
                )

            ahorros = mc_res.ahorros_ves_muestras

            col_mc_plot, col_mc_stat = st.columns([1.5, 1])

            with col_mc_plot:
                fig_mc, ax_mc = plt.subplots(figsize=(9, 4.4))
                fig_mc.patch.set_facecolor(C_CARD_BG)
                ax_mc.set_facecolor(C_CANVAS_BG)

                n_uniq = len(np.unique(ahorros))
                num_bins = min(32, max(8, n_uniq)) if n_uniq < 32 else 32

                # Histograma con la paleta: Barras en #a2ad91 con borde en #3a2d32
                ax_mc.hist(
                    ahorros,
                    bins=num_bins,
                    color=C_ACCENT_SAGE,
                    edgecolor=C_PRIMARY_DARK,
                    linewidth=0.7,
                    alpha=0.85
                )

                # Líneas de referencia con la paleta
                ax_mc.axvline(x=0, color=C_PRIMARY_DARK, linestyle="-", linewidth=1.6, label="Break-Even (0)")
                ax_mc.axvline(x=mc_res.ahorro_esperado_ves, color="#373f2e", linestyle="--", linewidth=2, label=f"Media: Bs. {mc_res.ahorro_esperado_ves:,.0f}")
                ax_mc.axvline(x=mc_res.var_95_ves, color=C_ACCENT_ROSE, linestyle=":", linewidth=2.2, label=f"VaR 95%: Bs. {mc_res.var_95_ves:,.0f}")

                ax_mc.set_title("Distribución de Probabilidad del Ahorro Neto (VES)", fontsize=10.5, fontweight="bold", color=C_PRIMARY_DARK, pad=12)
                ax_mc.set_xlabel("Ahorro Neto en Bolívares (Positivo = Ganancia al pagar en USDT)", fontsize=8.5, color=C_PRIMARY_DARK)
                ax_mc.set_ylabel("Frecuencia (Simulaciones)", fontsize=8.5, color=C_PRIMARY_DARK)
                ax_mc.tick_params(colors=C_PRIMARY_DARK, labelsize=8)
                ax_mc.grid(True, linestyle=":", alpha=0.25, color=C_PRIMARY_DARK)
                ax_mc.legend(fontsize=8, loc="upper right", framealpha=0.9)
                for spine in ax_mc.spines.values():
                    spine.set_color(C_PRIMARY_DARK)
                    spine.set_alpha(0.2)

                plt.tight_layout()
                st.pyplot(fig_mc)
                plt.close(fig_mc)

            with col_mc_stat:
                st.markdown("**Métricas Cuantitativas de Riesgo:**")
                st.write(f"• **Probabilidad de Éxito:** `{mc_res.probabilidad_ahorro_positivo}%`")
                st.write(f"• **Ahorro Medio Esperado:** `Bs. {mc_res.ahorro_esperado_ves:,.2f}`")
                st.write(f"• **Escenario Favorable (P90):** `Bs. {mc_res.ahorro_p90_ves:,.2f}`")
                st.write(f"• **Escenario Conservador (P10):** `Bs. {mc_res.ahorro_p10_ves:,.2f}`")
                st.write(f"• **Value-at-Risk 95% (Peor 5%):** `Bs. {mc_res.var_95_ves:,.2f}`")

                if mc_res.var_95_ves < 0:
                    st.warning(f"⚠️ En el 5% de los peores escenarios estocásticos, pagar en USDT podría representar una pérdida de hasta Bs. {abs(mc_res.var_95_ves):,.2f}.")
                else:
                    st.success("🛡️ Riesgo bajo: Incluso en el 95% de los escenarios adversos, el pago en USDT mantiene saldo positivo a favor de Carnívoros.")

        except Exception as err_mc:
            st.error(f"Error procesando simulación de Monte Carlo: {err_mc}")

    # ---------------- TAB 3: MACHINE LEARNING (FICHA DEL MODELO Y TABLA ROBUSTA) ----------------
    with tab_ml:
        # Ficha Técnica Destacada del Modelo
        mae_15 = forecaster.metrics.get("15d", {}).get("cv_mae", 3.35)
        rmse_15 = forecaster.metrics.get("15d", {}).get("cv_rmse", 5.32)
        mae_7 = forecaster.metrics.get("7d", {}).get("cv_mae", 2.15)
        mae_30 = forecaster.metrics.get("30d", {}).get("cv_mae", 3.22)

        card_model_html = (
            f'<div class="ml-model-card">\n'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">\n'
            f'<div>\n'
            f'<div style="font-size: 0.74rem; font-weight: 800; color: {C_ACCENT_SAGE}; text-transform: uppercase; letter-spacing: 0.08em;">Algoritmo Predictivo Central</div>\n'
            f'<div style="font-size: 1.45rem; font-weight: 900; color: {C_PRIMARY_DARK}; letter-spacing: -0.02em;">HistGradientBoostingRegressor</div>\n'
            f'</div>\n'
            f'<span class="badge-ml">Scikit-Learn • TimeSeriesSplit</span>\n'
            f'</div>\n'
            f'<p style="font-size: 0.92rem; color: #5a4b51; margin-bottom: 14px; line-height: 1.55;">\n'
            f'Modelo basado en histogramas de ensamble por gradiente (Histogram-based Gradient Boosting). Entrenado con más de 1.380 registros diarios continuos del mercado venezolano para predecir la trayectoria de la brecha cambiaria y la devaluación oficial, generando conos de incertidumbre cuantitativos (cuantiles P10 y P90).\n'
            f'</p>\n'
            f'<div class="ml-stats-grid">\n'
            f'<div class="ml-stat-item">\n'
            f'<div class="ml-stat-label">Validación Temporal</div>\n'
            f'<div class="ml-stat-val">5 Folds</div>\n'
            f'<div class="ml-stat-sub">Sin fuga de datos hacia adelante</div>\n'
            f'</div>\n'
            f'<div class="ml-stat-item">\n'
            f'<div class="ml-stat-label">Error MAE (15 días)</div>\n'
            f'<div class="ml-stat-val">±{mae_15:.2f}%</div>\n'
            f'<div class="ml-stat-sub">RMSE: ±{rmse_15:.2f}%</div>\n'
            f'</div>\n'
            f'<div class="ml-stat-item">\n'
            f'<div class="ml-stat-label">Variables Activas</div>\n'
            f'<div class="ml-stat-val">41 Features</div>\n'
            f'<div class="ml-stat-sub">Momentum, RSI, Volatilidad</div>\n'
            f'</div>\n'
            f'<div class="ml-stat-item">\n'
            f'<div class="ml-stat-label">Estacionalidad</div>\n'
            f'<div class="ml-stat-val">Quincenas</div>\n'
            f'<div class="ml-stat-sub">Cierres de mes y SENIAT</div>\n'
            f'</div>\n'
            f'</div>\n'
            f'</div>'
        )
        render_html(card_model_html)

        st.markdown("#### Proyecciones por Plazo de Crédito Comercial")

        # Construir Filas de Tabla de Proyecciones
        horizontes_orden = [7, 15, 30]
        if dias_credito not in horizontes_orden:
            horizontes_orden.append(dias_credito)
            horizontes_orden.sort()

        filas_html = ""
        for h in horizontes_orden:
            if h == 0:
                h_data = {
                    "dias": 0,
                    "brecha_esperada_pct": brecha_val,
                    "brecha_min_p10_pct": brecha_val,
                    "brecha_max_p90_pct": brecha_val,
                    "bcv_crecimiento_estimado_pct": 0.0,
                    "tasa_bcv_proyectada": bcv_val,
                    "tasa_usdt_proyectada": float(live_rates.get("tasa_usdt_referencia") or live_rates.get("paralelo") or 1000.0),
                    "tendencia_brecha": "ESTABLE"
                }
            elif h in forecasts["horizons"]:
                h_data = forecasts["horizons"][h]
            else:
                h_data = forecaster.interpolate_horizon(h, forecasts)

            t_val = str(h_data.get("tendencia_brecha", "")).upper()
            if "EXPAN" in t_val:
                pill_html = '<span class="trend-pill-expansion">EXPANSIÓN</span>'
            elif "COMPRE" in t_val:
                pill_html = '<span class="trend-pill-compression">COMPRESIÓN</span>'
            else:
                pill_html = '<span class="trend-pill-stable">ESTABLE</span>'

            es_factura_actual = (h == dias_credito)
            row_style = ' style="background-color: rgba(162, 173, 145, 0.14); font-weight: 700;"' if es_factura_actual else ""
            tag_factura = ' <span style="font-size: 0.72rem; color: #a2ad91; font-weight: 800;">(Tu Factura)</span>' if es_factura_actual else ""
            plazo_label = "0 días (Contado)" if h == 0 else f"{h} días"
            deval_sign = "+" if h_data['bcv_crecimiento_estimado_pct'] >= 0 else ""

            filas_html += (
                f'<tr{row_style}>'
                f'<td><strong>{plazo_label}</strong>{tag_factura}</td>'
                f'<td><strong>{h_data["brecha_esperada_pct"]:.2f}%</strong></td>'
                f'<td>[{h_data["brecha_min_p10_pct"]:.1f}% — {h_data["brecha_max_p90_pct"]:.1f}%]</td>'
                f'<td>{deval_sign}{h_data["bcv_crecimiento_estimado_pct"]:.2f}%</td>'
                f'<td>Bs. {h_data["tasa_bcv_proyectada"]:,.2f}</td>'
                f'<td>Bs. {h_data["tasa_usdt_proyectada"]:,.2f}</td>'
                f'<td>{pill_html}</td>'
                f'</tr>\n'
            )

        tabla_completa_html = (
            '<table class="projections-table">\n'
            '<thead>\n'
            '<tr>\n'
            '<th>Plazo Crédito</th>\n'
            '<th>Brecha Estimada</th>\n'
            '<th>Rango Confianza (P10 — P90)</th>\n'
            '<th>Devaluación BCV</th>\n'
            '<th>Tasa BCV Proyectada</th>\n'
            '<th>Tasa USDT Proyectada</th>\n'
            '<th>Tendencia</th>\n'
            '</tr>\n'
            '</thead>\n'
            f'<tbody>\n{filas_html}</tbody>\n'
            '</table>'
        )
        render_html(tabla_completa_html)

        # Gráfico Cono de Incertidumbre con la Paleta
        try:
            dias_x = [0, 7, 15, 30]
            gaps_mean = [brecha_val] + [forecasts["horizons"].get(h, {}).get("brecha_esperada_pct", brecha_val) for h in [7, 15, 30]]
            gaps_p10 = [brecha_val] + [forecasts["horizons"].get(h, {}).get("brecha_min_p10_pct", brecha_val * 0.85) for h in [7, 15, 30]]
            gaps_p90 = [brecha_val] + [forecasts["horizons"].get(h, {}).get("brecha_max_p90_pct", brecha_val * 1.20) for h in [7, 15, 30]]

            fig_ml, ax_ml = plt.subplots(figsize=(9, 4))
            fig_ml.patch.set_facecolor(C_CARD_BG)
            ax_ml.set_facecolor(C_CANVAS_BG)

            # Curva esperada en #3a2d32, cono en #a2ad91
            ax_ml.plot(dias_x, gaps_mean, marker="o", color=C_PRIMARY_DARK, linewidth=2.4, label="Brecha Esperada (ML)")
            ax_ml.fill_between(dias_x, gaps_p10, gaps_p90, color=C_ACCENT_SAGE, alpha=0.35, label="Cono de Incertidumbre (P10 - P90)")
            ax_ml.axvline(x=dias_credito, color=C_ACCENT_ROSE, linestyle="--", linewidth=1.8, label=f"Vencimiento Factura ({dias_credito}d)")
            ax_ml.scatter([dias_credito], [result.brecha_pronosticada_pct], color=C_ACCENT_ROSE, s=90, zorder=5)

            ax_ml.set_xlabel("Días hacia el futuro", fontsize=8.5, color=C_PRIMARY_DARK)
            ax_ml.set_ylabel("Brecha Cambiaria (%)", fontsize=8.5, color=C_PRIMARY_DARK)
            ax_ml.set_title("Trayectoria Proyectada de la Brecha Cambiaria", fontsize=10.5, fontweight="bold", color=C_PRIMARY_DARK, pad=12)
            ax_ml.tick_params(colors=C_PRIMARY_DARK, labelsize=8)
            ax_ml.grid(True, linestyle=":", alpha=0.25, color=C_PRIMARY_DARK)
            ax_ml.legend(fontsize=8, loc="upper left", framealpha=0.9)
            for spine in ax_ml.spines.values():
                spine.set_color(C_PRIMARY_DARK)
                spine.set_alpha(0.2)

            plt.tight_layout()
            st.pyplot(fig_ml)
            plt.close(fig_ml)
        except Exception as e_ml:
            st.error(f"Error generando gráfico de Machine Learning: {e_ml}")

    # HISTÓRICO DE TASAS (PALETA APLICADA)
    with tab_history:
        st.markdown("#### Histórico de Cotizaciones y Brecha en Venezuela")
        ventana = st.radio("Ventana temporal de análisis:", ["Últimos 6 meses", "Último año", "Todo el Histórico"], horizontal=True)

        df_sub = df_hist.copy()
        if ventana == "Últimos 6 meses":
            df_sub = df_sub.tail(180)
        elif ventana == "Último año":
            df_sub = df_sub.tail(365)

        try:
            fig_h, (ax_h1, ax_h2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            fig_h.patch.set_facecolor(C_CARD_BG)
            ax_h1.set_facecolor(C_CANVAS_BG)
            ax_h2.set_facecolor(C_CANVAS_BG)

            # Subplot 1: BCV (#a2ad91) vs Paralelo/USDT (#3a2d32)
            ax_h1.plot(df_sub.index, df_sub["bcv"], label="Oficial BCV", color=C_ACCENT_SAGE, linewidth=2.0)
            ax_h1.plot(df_sub.index, df_sub["paralelo"], label="Paralelo / USDT", color=C_PRIMARY_DARK, linewidth=2.0)
            ax_h1.set_ylabel("Bs / USD", fontsize=8.5, color=C_PRIMARY_DARK)
            ax_h1.set_title("Evolución de Cotizaciones Oficial BCV vs. Paralelo", fontsize=10.5, fontweight="bold", color=C_PRIMARY_DARK)
            ax_h1.tick_params(colors=C_PRIMARY_DARK, labelsize=8)
            ax_h1.grid(True, linestyle=":", alpha=0.25, color=C_PRIMARY_DARK)
            ax_h1.legend(fontsize=8)
            for spine in ax_h1.spines.values():
                spine.set_color(C_PRIMARY_DARK)
                spine.set_alpha(0.2)

            # Subplot 2: Brecha % en #dbb4ad
            ax_h2.plot(df_sub.index, df_sub["brecha_pct"], label="Brecha Cambiaria (%)", color=C_ACCENT_ROSE, linewidth=1.8)
            ax_h2.axhline(y=df_sub["brecha_pct"].mean(), color=C_PRIMARY_DARK, linestyle="--", linewidth=1.2, label=f"Media ({df_sub['brecha_pct'].mean():.1f}%)")
            ax_h2.set_ylabel("Brecha (%)", fontsize=8.5, color=C_PRIMARY_DARK)
            ax_h2.set_xlabel("Fecha", fontsize=8.5, color=C_PRIMARY_DARK)
            ax_h2.tick_params(colors=C_PRIMARY_DARK, labelsize=8)
            ax_h2.grid(True, linestyle=":", alpha=0.25, color=C_PRIMARY_DARK)
            ax_h2.legend(fontsize=8)
            for spine in ax_h2.spines.values():
                spine.set_color(C_PRIMARY_DARK)
                spine.set_alpha(0.2)

            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)
        except Exception as e_h:
            st.error(f"Error generando gráfico histórico: {e_h}")


if __name__ == "__main__":
    main()
