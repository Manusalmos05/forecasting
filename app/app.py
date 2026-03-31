import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="📊 Forecasting Ventas · Noviembre 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Estilos CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] { background-color: #f7f8fc; }
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.6rem 2.2rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1.4rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(102,126,234,0.35);
        }
        .main-header h1 { color: white !important; margin: 0; font-size: 1.9rem; font-weight: 800; }
        .main-header p  { color: rgba(255,255,255,0.88); margin: 0.4rem 0 0 0; font-size: 0.95rem; }
        .sub-info {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.75rem 1.3rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.8rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .section-header {
            font-size: 1rem;
            font-weight: 700;
            color: #4a5568;
            padding: 0.35rem 0.9rem;
            background: linear-gradient(90deg, #667eea18 0%, transparent 100%);
            border-left: 4px solid #667eea;
            border-radius: 0 8px 8px 0;
            margin: 1.2rem 0 0.9rem 0;
        }
        div[data-testid="metric-container"] {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .esc-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .esc-card h4 { margin: 0 0 0.5rem 0; font-size: 0.88rem; color: #718096; font-weight: 600; }
        .esc-card .esc-big  { font-size: 1.7rem; font-weight: 800; }
        .esc-card .esc-sub  { font-size: 0.9rem; color: #718096; margin-top: 0.25rem; }
        .info-banner {
            background: linear-gradient(135deg, #667eea12 0%, #764ba212 100%);
            border: 1px solid #667eea30;
            border-radius: 14px;
            padding: 2.5rem;
            text-align: center;
            margin: 3rem auto;
            max-width: 580px;
        }
        .info-banner h2 { color: #667eea; margin-bottom: 0.6rem; font-size: 1.4rem; }
        .info-banner p  { color: #4a5568; font-size: 0.95rem; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_final.joblib"
DATA_PATH  = BASE_DIR / "data" / "processed" / "inferencia_df_transformado.csv"

# ─── Nombres de días en español (evita problemas de codificación) ─────────────
_DIAS_ES = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
            4: "Viernes", 5: "Sábado", 6: "Domingo"}

# ─── Carga de recursos ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])
    # Recompute nombre_dia to avoid encoding issues
    df["nombre_dia"] = df["fecha"].dt.weekday.map(_DIAS_ES)
    return df


# ─── Predicción recursiva ─────────────────────────────────────────────────────
def predict_recursive(df_prod, model, ajuste_descuento, factor_competencia):
    """Predicción recursiva día a día: actualiza lags y mm7 tras cada predicción."""
    df_work = (
        df_prod.copy()
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    # — Ajuste de descuento
    descuento_nuevo = (df_work["descuento_porcentaje"] + ajuste_descuento).clip(-99, 99)
    df_work["descuento_porcentaje"] = descuento_nuevo
    df_work["precio_venta"] = (df_work["precio_base"] * (1 - descuento_nuevo / 100)).clip(lower=0.01)

    # — Ajuste de competencia
    df_work["precio_competencia"] = df_work["precio_competencia"] * (1 + factor_competencia / 100)
    df_work["ratio_precio_competencia"] = np.where(
        df_work["precio_competencia"] > 0,
        df_work["precio_venta"] / df_work["precio_competencia"],
        1.0,
    )

    feature_cols = list(model.feature_names_in_)

    # — Historial inicial: lags del día 1 de más antiguo a más reciente
    row0 = df_work.loc[0]
    history = [
        float(row0["unidades_vendidas_lag7"]),
        float(row0["unidades_vendidas_lag6"]),
        float(row0["unidades_vendidas_lag5"]),
        float(row0["unidades_vendidas_lag4"]),
        float(row0["unidades_vendidas_lag3"]),
        float(row0["unidades_vendidas_lag2"]),
        float(row0["unidades_vendidas_lag1"]),
    ]

    predictions = []

    for i in range(len(df_work)):
        # A partir del día 2 actualizamos los lags con el historial reciente
        if i > 0:
            df_work.at[i, "unidades_vendidas_lag1"] = history[-1]
            df_work.at[i, "unidades_vendidas_lag2"] = history[-2]
            df_work.at[i, "unidades_vendidas_lag3"] = history[-3]
            df_work.at[i, "unidades_vendidas_lag4"] = history[-4]
            df_work.at[i, "unidades_vendidas_lag5"] = history[-5]
            df_work.at[i, "unidades_vendidas_lag6"] = history[-6]
            df_work.at[i, "unidades_vendidas_lag7"] = history[-7]
            df_work.at[i, "unidades_vendidas_mm7"]  = float(np.mean(history[-7:]))

        X    = df_work.loc[[i], feature_cols]
        pred = float(max(0.0, model.predict(X)[0]))
        predictions.append(pred)
        history.append(pred)

    df_work["unidades_predichas"] = predictions
    df_work["ingresos_predichos"] = df_work["unidades_predichas"] * df_work["precio_venta"]
    return df_work


# ─── Gráfico de predicción diaria ─────────────────────────────────────────────
def plot_prediccion(df_result, producto):
    sns.set_theme(style="whitegrid", font_scale=0.95)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor("#f7f8fc")
    ax.set_facecolor("#f7f8fc")

    dias  = df_result["dia_mes"].values
    preds = df_result["unidades_predichas"].values
    rango = preds.max() - preds.min() if preds.max() != preds.min() else 1.0

    # Línea principal
    ax.plot(
        dias, preds,
        color="#667eea", linewidth=2.5,
        marker="o", markersize=4.5,
        markerfacecolor="white", markeredgewidth=1.8, markeredgecolor="#667eea",
        zorder=3,
    )
    ax.fill_between(dias, preds, alpha=0.13, color="#667eea")

    # Black Friday
    bf_mask = df_result["es_black_friday"] == 1
    if bf_mask.any():
        bf_idx    = df_result[bf_mask].index[0]
        bf_dia    = int(df_result.loc[bf_idx, "dia_mes"])
        bf_units  = float(df_result.loc[bf_idx, "unidades_predichas"])

        ax.axvline(x=bf_dia, color="#e53e3e", linestyle="--", linewidth=2, alpha=0.75, zorder=2)
        ax.scatter([bf_dia], [bf_units], color="#e53e3e", s=140, zorder=5, marker="*")

        # Anotación a la izquierda para no salir del gráfico
        offset_x = -5.5
        offset_y =  rango * 0.25 + 0.5
        ax.annotate(
            f"⬆ BLACK FRIDAY\n{bf_units:.0f} uds",
            xy=(bf_dia, bf_units),
            xytext=(bf_dia + offset_x, bf_units + offset_y),
            fontsize=9, fontweight="bold", color="#e53e3e",
            arrowprops=dict(arrowstyle="->", color="#e53e3e", lw=1.6),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5",
                      edgecolor="#e53e3e", alpha=0.9),
        )

    ax.set_xlabel("Día de Noviembre 2025", fontsize=10, color="#4a5568", labelpad=6)
    ax.set_ylabel("Unidades Vendidas",    fontsize=10, color="#4a5568", labelpad=6)
    ax.set_title(
        f"Predicción Diaria · {producto}",
        fontsize=12, fontweight="bold", color="#2d3748", pad=12,
    )
    ax.set_xticks(range(1, 31))
    ax.tick_params(axis="both", labelsize=8, colors="#718096")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.xaxis.grid(True, alpha=0.35, linestyle="--")
    ax.yaxis.grid(True, alpha=0.35)

    plt.tight_layout()
    return fig


# ─── Tabla detallada ──────────────────────────────────────────────────────────
def build_table(df_result):
    cols = [
        "fecha", "dia_mes", "nombre_dia",
        "precio_venta", "precio_competencia",
        "descuento_porcentaje", "unidades_predichas", "ingresos_predichos",
    ]
    tabla = df_result[cols].copy()
    tabla = tabla.rename(columns={
        "fecha":               "Fecha",
        "dia_mes":             "Día",
        "nombre_dia":          "Día Semana",
        "precio_venta":        "P. Venta (€)",
        "precio_competencia":  "P. Comp. (€)",
        "descuento_porcentaje":"Descuento (%)",
        "unidades_predichas":  "Unidades",
        "ingresos_predichos":  "Ingresos (€)",
    })

    tabla["Fecha"]         = pd.to_datetime(tabla["Fecha"]).dt.strftime("%d/%m/%Y")
    tabla["P. Venta (€)"]  = tabla["P. Venta (€)"].round(2)
    tabla["P. Comp. (€)"]  = tabla["P. Comp. (€)"].round(2)
    tabla["Descuento (%)"] = tabla["Descuento (%)"].round(1)
    tabla["Unidades"]      = tabla["Unidades"].round(0).astype(int)
    tabla["Ingresos (€)"]  = tabla["Ingresos (€)"].round(2)

    # Marcar Black Friday
    bf_loc = df_result["es_black_friday"] == 1
    if bf_loc.any():
        bf_global_idx = df_result[bf_loc].index[0]
        # Reindex since tabla index matches df_result after reset_index
        tabla_bf_idx = tabla.index[tabla.index == bf_global_idx]
        if len(tabla_bf_idx) > 0:
            tabla.loc[tabla_bf_idx[0], "Día Semana"] = "🏷️ BLACK FRIDAY"

    return tabla


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
data_loaded = False
with st.sidebar:
    st.markdown("## ⚙️ Controles de Simulación")
    st.markdown("---")

    try:
        df_raw = load_data()
        model  = load_model()
        productos = sorted(df_raw["nombre"].unique().tolist())

        producto_sel = st.selectbox(
            "🛒 Producto",
            options=productos,
            index=0,
            help="Selecciona el producto a simular",
        )

        st.markdown("---")

        ajuste_descuento = st.slider(
            "🏷️ Ajuste de Descuento (%)",
            min_value=-50, max_value=50, value=0, step=5,
            help="Ajuste adicional sobre el descuento base. Un valor positivo aumenta el descuento.",
        )

        st.markdown("---")

        escenario_map = {
            "Actual (0%)":      0,
            "Competencia -5%": -5,
            "Competencia +5%":  5,
        }
        escenario_sel = st.radio(
            "🏪 Escenario Competencia",
            options=list(escenario_map.keys()),
            index=0,
            help="Variación aplicada al precio de la competencia.",
        )
        factor_competencia = escenario_map[escenario_sel]

        st.markdown("---")

        simular = st.button(
            "🚀 Simular Ventas",
            use_container_width=True,
            type="primary",
        )

        st.markdown("---")
        st.markdown(
            """
            <div style='font-size:0.76rem;color:#718096;text-align:center;line-height:1.8;'>
            📅 Datos de <strong>noviembre 2025</strong><br>
            🤖 Modelo: <strong>HistGradientBoosting</strong><br>
            🔁 Predicción <strong>recursiva</strong> día a día<br>
            📦 <strong>24 productos</strong> disponibles
            </div>
            """,
            unsafe_allow_html=True,
        )

        data_loaded = True

    except Exception as e:
        st.error(f"❌ Error cargando recursos:\n{e}")
        simular = False

# ─── ZONA PRINCIPAL ───────────────────────────────────────────────────────────
if not data_loaded:
    st.error("⚠️ No se pudo cargar el modelo o los datos. Verifica las rutas.")
    st.stop()

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>📊 Dashboard de Simulación de Ventas</h1>
        <p>Predicciones para <strong>Noviembre 2025</strong> · Modelo HistGradientBoostingRegressor · Predicción Recursiva Diaria</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Estado de sesión
if "resultados" not in st.session_state:
    st.session_state["resultados"] = None

# Ejecutar simulación al pulsar el botón
if simular:
    df_prod = df_raw[df_raw["nombre"] == producto_sel].copy()

    with st.spinner(f"🔄 Ejecutando predicciones recursivas para **{producto_sel}**..."):
        try:
            df_result      = predict_recursive(df_prod, model, ajuste_descuento, factor_competencia)
            result_actual  = predict_recursive(df_prod, model, ajuste_descuento,  0)
            result_minus5  = predict_recursive(df_prod, model, ajuste_descuento, -5)
            result_plus5   = predict_recursive(df_prod, model, ajuste_descuento,  5)

            st.session_state["resultados"] = {
                "df_result":       df_result,
                "producto":        producto_sel,
                "ajuste_descuento": ajuste_descuento,
                "escenario":       escenario_sel,
                "escenarios": {
                    "Actual (0%)":     result_actual,
                    "Competencia -5%": result_minus5,
                    "Competencia +5%": result_plus5,
                },
            }
        except Exception as e:
            st.error(f"❌ Error al ejecutar predicciones: {e}")

# ─── Mostrar resultados ───────────────────────────────────────────────────────
if st.session_state["resultados"] is None:
    st.markdown(
        """
        <div class="info-banner">
            <h2>👈 Configura y simula</h2>
            <p>
                Selecciona un <strong>producto</strong>, ajusta el
                <strong>descuento</strong> y el <strong>escenario de competencia</strong>
                en el panel lateral y pulsa<br><br>
                <strong>🚀 Simular Ventas</strong><br><br>
                para ver el dashboard completo con predicciones recursivas día a día.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    res  = st.session_state["resultados"]
    df_r = res["df_result"]
    prod = res["producto"]

    # ── Sub-cabecera del producto ──────────────────────────────────────────────
    precio_base_prod = float(df_r["precio_base"].iloc[0])
    st.markdown(
        f"""
        <div class="sub-info">
            <span style='font-size:1.4rem;'>🛒</span>
            <div>
                <div style='font-weight:700;color:#2d3748;font-size:1.05rem;'>{prod}</div>
                <div style='color:#718096;font-size:0.82rem;'>
                    Precio base: <strong>{precio_base_prod:.2f} €</strong> &nbsp;·&nbsp;
                    Ajuste descuento: <strong>{res['ajuste_descuento']:+d}%</strong> &nbsp;·&nbsp;
                    Escenario: <strong>{res['escenario']}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Principales Métricas — Noviembre 2025</div>', unsafe_allow_html=True)

    total_unidades   = df_r["unidades_predichas"].sum()
    total_ingresos   = df_r["ingresos_predichos"].sum()
    precio_promedio  = df_r["precio_venta"].mean()
    descuento_prom   = df_r["descuento_porcentaje"].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("📦 Unidades Totales",     f"{total_unidades:,.0f} uds")
    with k2:
        st.metric("💰 Ingresos Proyectados", f"{total_ingresos:,.2f} €")
    with k3:
        st.metric("🏷️ Precio Promedio",      f"{precio_promedio:.2f} €")
    with k4:
        st.metric("📉 Descuento Promedio",   f"{descuento_prom:.1f} %")

    st.divider()

    # ── GRÁFICO ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Predicción de Ventas Diarias · Noviembre 2025</div>', unsafe_allow_html=True)

    fig = plot_prediccion(df_r, prod)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    # ── TABLA DETALLADA ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Detalle Diario · Todos los días de Noviembre</div>', unsafe_allow_html=True)
    st.caption("💡 La fila del Black Friday (28 nov) está marcada con 🏷️")

    tabla = build_table(df_r)
    st.dataframe(tabla, use_container_width=True, hide_index=True, height=490)

    st.divider()

    # ── COMPARATIVA DE ESCENARIOS ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🔀 Comparativa de Escenarios de Competencia</div>', unsafe_allow_html=True)
    st.caption(
        f"Mismo producto y descuento ({res['ajuste_descuento']:+d}%) en los 3 escenarios — "
        f"solo varía el precio de la competencia."
    )

    escenarios = res["escenarios"]
    esc_config = {
        "Actual (0%)":     {"icon": "⚖️", "color": "#667eea"},
        "Competencia -5%": {"icon": "📉", "color": "#38a169"},
        "Competencia +5%": {"icon": "📈", "color": "#e53e3e"},
    }

    cols_esc = st.columns(3)
    uds_base = escenarios["Actual (0%)"]["unidades_predichas"].sum()
    ing_base = escenarios["Actual (0%)"]["ingresos_predichos"].sum()

    for col, (nombre_esc, df_esc) in zip(cols_esc, escenarios.items()):
        cfg     = esc_config[nombre_esc]
        uds_esc = df_esc["unidades_predichas"].sum()
        ing_esc = df_esc["ingresos_predichos"].sum()

        delta_uds = uds_esc - uds_base
        delta_ing = ing_esc - ing_base
        delta_uds_str = (
            f'+{delta_uds:,.0f}' if delta_uds >= 0 else f'{delta_uds:,.0f}'
        ) if nombre_esc != "Actual (0%)" else ""
        delta_ing_str = (
            f'+{delta_ing:,.2f} €' if delta_ing >= 0 else f'{delta_ing:,.2f} €'
        ) if nombre_esc != "Actual (0%)" else ""

        delta_color = "#38a169" if delta_uds >= 0 else "#e53e3e"

        with col:
            delta_html = (
                f'<div style="font-size:0.78rem;color:{delta_color};margin-top:0.3rem;font-weight:600;">'
                f'vs actual: {delta_uds_str} uds · {delta_ing_str}'
                f'</div>'
            ) if nombre_esc != "Actual (0%)" else ""

            st.markdown(
                f"""
                <div class="esc-card" style="border-top:4px solid {cfg['color']};">
                    <h4>{cfg['icon']} {nombre_esc}</h4>
                    <div class="esc-big" style="color:{cfg['color']};">{uds_esc:,.0f}</div>
                    <div style="font-size:0.73rem;color:#718096;">unidades totales</div>
                    <div class="esc-sub">💰 {ing_esc:,.2f} €</div>
                    {delta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
