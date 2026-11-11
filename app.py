"""
Simulador de predicción de abandono de clientes — Netflix Customer Churn
Avance del proyecto final · Corporación Universitaria Antonio José de Sucre

Carga los seis modelos de clasificación ya entrenados (carpeta modelos/) y permite
cambiar las variables independientes para predecir la variable dependiente
Abandono (0 = el cliente permanece, 1 = el cliente abandona).
"""
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_MODELOS = os.path.join(BASE, "modelos")

ROJO = "#E50914"
VERDE = "#2BB673"

st.set_page_config(page_title="Simulador de abandono · Netflix", page_icon="🎬", layout="wide")

# --------------------------------------------------------------------------------------
# Textos en español para las variables y sus categorías (el modelo usa los valores originales)
# --------------------------------------------------------------------------------------
ETIQUETAS = {
    "Edad": "Edad (años)",
    "Género": "Género",
    "Tipo_de_suscripción": "Tipo de suscripción",
    "Horas_vistas": "Horas vistas",
    "Días_desde_último_acceso": "Días desde el último acceso",
    "Región": "Región",
    "Dispositivo": "Dispositivo",
    "Método_de_pago": "Método de pago",
    "Número_de_perfiles": "Número de perfiles",
    "Género_favorito": "Género favorito",
}
TRADUCCION = {
    "Female": "Femenino", "Male": "Masculino", "Other": "Otro",
    "Basic": "Básico ($8.99)", "Standard": "Estándar ($13.99)", "Premium": "Premium ($17.99)",
    "Africa": "África", "Asia": "Asia", "Europe": "Europa", "North America": "Norteamérica",
    "Oceania": "Oceanía", "South America": "Suramérica",
    "Desktop": "Computador de escritorio", "Laptop": "Portátil", "Mobile": "Celular",
    "TV": "Televisor", "Tablet": "Tableta",
    "Credit Card": "Tarjeta de crédito", "Debit Card": "Tarjeta débito", "Crypto": "Criptomonedas",
    "Gift Card": "Tarjeta de regalo", "PayPal": "PayPal",
    "Action": "Acción", "Comedy": "Comedia", "Documentary": "Documental", "Drama": "Drama",
    "Horror": "Terror", "Romance": "Romance", "Sci-Fi": "Ciencia ficción",
}
ICONOS = {
    "Regresión logística simple": "📈",
    "Regresión logística múltiple": "📊",
    "Regresión logística polinomial": "〰️",
    "Árbol de decisión": "🌳",
    "Bosques aleatorios": "🌲",
    "Máquina de vectores de soporte": "🧭",
}
METRICAS = ["Exactitud (Accuracy)", "Precisión (Precision)", "Sensibilidad (Recall)", "Puntaje F1", "AUC ROC"]


# --------------------------------------------------------------------------------------
# Carga de recursos
# --------------------------------------------------------------------------------------
@st.cache_data
def cargar_metadata():
    with open(os.path.join(CARPETA_MODELOS, "metadata.json"), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_prueba():
    return pd.read_csv(os.path.join(CARPETA_MODELOS, "prueba.csv"))


def _reentrenar(nombre, info, meta):
    """Respaldo: si un .joblib no se puede abrir (versiones distintas de scikit-learn),
    se vuelve a entrenar con el mismo preprocesamiento y los mejores hiperparámetros."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv(os.path.join(BASE, "netflix_customer_churn.csv"))
    df = df.drop(columns=["customer_id", "monthly_fee", "avg_watch_time_per_day"]).rename(columns={
        "age": "Edad", "gender": "Género", "subscription_type": "Tipo_de_suscripción",
        "watch_hours": "Horas_vistas", "last_login_days": "Días_desde_último_acceso", "region": "Región",
        "device": "Dispositivo", "churned": "Abandono", "payment_method": "Método_de_pago",
        "number_of_profiles": "Número_de_perfiles", "favorite_genre": "Género_favorito"})
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    limpio = df.copy()
    for col in meta["columnas_numericas"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        limpio = limpio[(limpio[col] > q1 - 1.5 * iqr) & (limpio[col] < q3 + 1.5 * iqr)]
    limpio = limpio.reset_index(drop=True)
    for col, mapa in meta["equivalencias"].items():
        limpio[col] = limpio[col].map(mapa)
    X, y = limpio[meta["variables"]], limpio["Abandono"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    hp = info["hiperparametros"]
    if nombre == "Regresión logística simple":
        modelo = make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=1000))
    elif nombre == "Regresión logística múltiple":
        modelo = make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=1000))
    elif nombre == "Regresión logística polinomial":
        modelo = make_pipeline(MinMaxScaler(), PolynomialFeatures(include_bias=False), LogisticRegression(max_iter=5000))
    elif nombre == "Árbol de decisión":
        modelo = DecisionTreeClassifier(random_state=42)
    elif nombre == "Bosques aleatorios":
        modelo = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        modelo = make_pipeline(MinMaxScaler(), SVC(probability=True, random_state=42))
    modelo.set_params(**hp)
    modelo.fit(X_train[info["variables"]], y_train)
    return modelo


@st.cache_resource(show_spinner="Cargando los modelos entrenados…")
def cargar_modelos():
    meta = cargar_metadata()
    modelos, reentrenados = {}, []
    for nombre, info in meta["modelos"].items():
        try:
            modelos[nombre] = joblib.load(os.path.join(CARPETA_MODELOS, info["archivo"]))
        except Exception:
            modelos[nombre] = _reentrenar(nombre, info, meta)
            reentrenados.append(nombre)
    return modelos, reentrenados


meta = cargar_metadata()
modelos, reentrenados = cargar_modelos()
prueba = cargar_prueba()
nombres_modelos = list(meta["modelos"].keys())
inverso = {col: {v: k for k, v in mapa.items()} for col, mapa in meta["equivalencias"].items()}


# --------------------------------------------------------------------------------------
# Estado de los controles (permite cargar clientes reales y restablecer valores)
# --------------------------------------------------------------------------------------
def valores_por_defecto():
    valores = {}
    for col in meta["columnas_numericas"]:
        r = meta["rangos"][col]
        valores[col] = int(r["mediana"]) if col != "Horas_vistas" else float(r["mediana"])
    for col in meta["columnas_categoricas"]:
        valores[col] = list(meta["equivalencias"][col].keys())[0]
    return valores


def aplicar_valores(valores):
    for col, val in valores.items():
        st.session_state["in_" + col] = val


def cargar_cliente_real():
    fila = prueba.iloc[random.randrange(len(prueba))]
    valores = {}
    for col in meta["variables"]:
        if col in meta["columnas_categoricas"]:
            valores[col] = inverso[col][int(fila[col])]
        elif col == "Horas_vistas":
            valores[col] = float(fila[col])
        else:
            valores[col] = int(fila[col])
    aplicar_valores(valores)
    st.session_state["cliente_real"] = {"valores": valores, "abandono": int(fila["Abandono"])}


def restablecer():
    aplicar_valores(valores_por_defecto())
    st.session_state["cliente_real"] = None
    st.session_state["resultado"] = None


if "iniciado" not in st.session_state:
    aplicar_valores(valores_por_defecto())
    st.session_state["cliente_real"] = None
    st.session_state["resultado"] = None
    st.session_state["iniciado"] = True


# --------------------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 Simulador de abandono")
    st.caption("Netflix Customer Churn · 6 modelos de clasificación")
    st.divider()
    st.markdown("**Datos del cliente**")
    st.button("🎲 Cargar un cliente real (prueba)", on_click=cargar_cliente_real, width="stretch",
              help="Llena el formulario con un cliente real del conjunto de prueba. Luego presiona Calcular.")
    st.button("↺ Restablecer valores", on_click=restablecer, width="stretch")
    st.caption(f"Clientes reales disponibles: {len(prueba)} del conjunto de prueba (20 %).")
    if reentrenados:
        st.warning("Se reentrenaron por compatibilidad: " + ", ".join(reentrenados))


# --------------------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
    <h1 style="margin-bottom:0">Simulador de predicción de abandono de clientes</h1>
    <p style="opacity:.75;margin-top:4px">Selecciona un modelo, escribe las <b>variables independientes</b> del cliente
    y presiona <b>Calcular</b> para obtener la <b>variable dependiente</b> <code>Abandono</code>
    (<span style="color:{VERDE}">0 = permanece</span> · <span style="color:{ROJO}">1 = abandona</span>).</p>
    """,
    unsafe_allow_html=True,
)

tab_sim, tab_comp, tab_met, tab_info = st.tabs(
    ["🔮 Simulador", "⚖️ Comparar los 6 modelos", "📊 Métricas", "ℹ️ Acerca del proyecto"]
)


# --------------------------------------------------------------------------------------
# Funciones de apoyo
# --------------------------------------------------------------------------------------
def rango_texto(col):
    r = meta["rangos"][col]
    if col == "Horas_vistas":
        return f"Entre {r['min']:.2f} y {r['max']:.2f}"
    return f"Entre {int(r['min'])} y {int(r['max'])}"


def campos_numericos():
    r = meta["rangos"]
    for col in ["Edad", "Días_desde_último_acceso", "Número_de_perfiles"]:
        st.number_input(f"{ETIQUETAS[col]} · {rango_texto(col).lower()}", min_value=int(r[col]["min"]),
                        max_value=int(r[col]["max"]), step=1, key="in_" + col)
    st.number_input(f"{ETIQUETAS['Horas_vistas']} · {rango_texto('Horas_vistas').lower()}",
                    min_value=float(r["Horas_vistas"]["min"]), max_value=float(r["Horas_vistas"]["max"]),
                    step=0.01, format="%.2f", key="in_Horas_vistas")


def campos_categoricos(columnas):
    for col in columnas:
        st.selectbox(ETIQUETAS[col], list(meta["equivalencias"][col]), format_func=TRADUCCION.get, key="in_" + col)


def valores_actuales():
    return {col: st.session_state["in_" + col] for col in meta["variables"]}


def codificar(valores):
    """Convierte los valores del cliente a la codificación numérica usada en el entrenamiento."""
    fila = {col: (meta["equivalencias"][col][v] if col in meta["columnas_categoricas"] else v)
            for col, v in valores.items()}
    return pd.DataFrame([fila], columns=meta["variables"])


def predecir(nombre, X):
    variables = meta["modelos"][nombre]["variables"]
    modelo = modelos[nombre]
    prob = float(modelo.predict_proba(X[variables])[0, 1])
    pred = int(modelo.predict(X[variables])[0])
    return pred, prob


def medidor(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": " %", "font": {"size": 44}},
        title={"text": "Probabilidad de abandono", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": ROJO if prob >= 0.5 else VERDE, "thickness": 0.3},
            "steps": [{"range": [0, 50], "color": "rgba(43,182,115,.15)"},
                      {"range": [50, 100], "color": "rgba(229,9,20,.15)"}],
            "threshold": {"line": {"color": "#888", "width": 3}, "thickness": 0.9, "value": 50},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def tarjeta(pred):
    color = ROJO if pred == 1 else VERDE
    texto = "ABANDONA" if pred == 1 else "PERMANECE"
    icono = "🚪" if pred == 1 else "✅"
    st.markdown(
        f"""
        <div style="border:2px solid {color};border-radius:14px;padding:18px 20px;text-align:center;
                    background:{color}14">
          <div style="font-size:.9rem;opacity:.8">Predicción · Abandono = <b>{pred}</b></div>
          <div style="font-size:2.2rem;font-weight:800;color:{color};letter-spacing:.04em">{icono} {texto}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Pestaña 1: simulador
# --------------------------------------------------------------------------------------
with tab_sim:
    izq, der = st.columns([1.35, 1], gap="large")

    with izq:
        # Paso 1: elegir el modelo.
        st.markdown("#### 1. Selecciona el modelo")
        modelo_sel = st.selectbox(
            "Modelo de clasificación", nombres_modelos, index=None, placeholder="Elige un modelo…",
            format_func=lambda n: f"{ICONOS.get(n, '')} {n}", key="modelo_sel",
        )
        if modelo_sel:
            m = meta["modelos"][modelo_sel]["metricas"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Exactitud", f"{m['Exactitud (Accuracy)']:.2%}")
            c2.metric("Precisión", f"{m['Precisión (Precision)']:.2%}")
            c3.metric("F1", f"{m['Puntaje F1']:.4f}")
            c4.metric("AUC ROC", f"{m['AUC ROC']:.4f}")
            if meta["modelos"][modelo_sel]["variables"] != meta["variables"]:
                var = meta["modelos"][modelo_sel]["variables"][0]
                st.caption(f"ℹ️ Este modelo solo usa una variable independiente: **{ETIQUETAS[var]}**. "
                           "Las demás no cambian su predicción.")

        # Paso 2: escribir los datos del cliente.
        st.markdown("#### 2. Ingresa los datos del cliente")
        with st.form("form_cliente", border=True):
            a, b = st.columns(2)
            with a:
                campos_numericos()
                campos_categoricos(["Género"])
            with b:
                campos_categoricos(["Tipo_de_suscripción", "Método_de_pago", "Región", "Dispositivo", "Género_favorito"])
            calcular = st.form_submit_button("🔮 Calcular", type="primary", width="stretch")

        if calcular:
            if not modelo_sel:
                st.warning("Primero selecciona un modelo en el paso 1.")
            else:
                valores = valores_actuales()
                X_cliente = codificar(valores)
                pred, prob = predecir(modelo_sel, X_cliente)
                real = st.session_state.get("cliente_real")
                st.session_state["resultado"] = {
                    "modelo": modelo_sel, "pred": pred, "prob": prob, "valores": valores,
                    "real": real["abandono"] if real and real["valores"] == valores else None,
                }

    # Paso 3: resultado.
    with der:
        st.markdown("#### 3. Resultado")
        res = st.session_state.get("resultado")
        if not res:
            st.info("Selecciona un modelo, escribe los datos del cliente y presiona **Calcular**.")
        else:
            st.markdown(f"Calculado con **{ICONOS.get(res['modelo'], '')} {res['modelo']}**")
            tarjeta(res["pred"])
            st.plotly_chart(medidor(res["prob"]), width="stretch", config={"displayModeBar": False})
            if res["real"] is not None:
                acierto = res["real"] == res["pred"]
                st.info(
                    f"Cliente real del conjunto de prueba · valor real de Abandono = **{res['real']}** "
                    f"({'abandonó' if res['real'] == 1 else 'permaneció'}). "
                    + ("✅ El modelo **acertó**." if acierto else "❌ El modelo **se equivocó**.")
                )
            if modelo_sel and modelo_sel != res["modelo"]:
                st.caption("⚠️ Cambiaste de modelo: presiona **Calcular** para obtener su resultado.")
            if valores_actuales() != res["valores"]:
                st.caption("⚠️ Los datos del formulario cambiaron: presiona **Calcular** para actualizar.")
            with st.expander("Ver los datos usados en el cálculo"):
                legible = {ETIQUETAS[k]: (TRADUCCION.get(v, v) if isinstance(v, str)
                                          else (f"{v:.2f}" if isinstance(v, float) else str(v)))
                           for k, v in res["valores"].items()}
                st.dataframe(pd.DataFrame({"Variable": list(legible), "Valor": list(legible.values())}),
                             hide_index=True, width="stretch")
                st.caption("Codificación numérica que recibe el modelo:")
                st.dataframe(codificar(res["valores"]), hide_index=True, width="stretch")


# --------------------------------------------------------------------------------------
# Pestaña 2: comparación de los seis modelos con el mismo cliente
# --------------------------------------------------------------------------------------
with tab_comp:
    st.markdown("#### Mismo cliente, seis modelos")
    res = st.session_state.get("resultado")
    if not res:
        st.info("Primero calcula un cliente en la pestaña **Simulador**; aquí se evaluará con los seis modelos.")
    else:
        st.caption("Se usan los datos del último cálculo hecho en la pestaña **Simulador**.")
        X_cliente = codificar(res["valores"])
        filas = []
        for nombre in nombres_modelos:
            p, pr = predecir(nombre, X_cliente)
            filas.append({"Modelo": f"{ICONOS.get(nombre, '')} {nombre}", "Predicción (Abandono)": p,
                          "Resultado": "Abandona" if p == 1 else "Permanece",
                          "Probabilidad de abandono": pr,
                          "Exactitud del modelo": meta["modelos"][nombre]["metricas"]["Exactitud (Accuracy)"]})
        comp = pd.DataFrame(filas)

        votos = int(comp["Predicción (Abandono)"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelos que predicen abandono", f"{votos} de 6")
        c2.metric("Probabilidad promedio", f"{comp['Probabilidad de abandono'].mean():.1%}")
        c3.metric("Decisión por mayoría", "Abandona" if votos > 3 else ("Empate" if votos == 3 else "Permanece"))

        orden = comp.sort_values("Probabilidad de abandono")
        fig = go.Figure(go.Bar(
            x=orden["Probabilidad de abandono"] * 100, y=orden["Modelo"], orientation="h",
            marker_color=[ROJO if v >= 0.5 else VERDE for v in orden["Probabilidad de abandono"]],
            text=[f"{v:.1%}" for v in orden["Probabilidad de abandono"]], textposition="outside",
        ))
        fig.add_vline(x=50, line_dash="dash", line_color="#888", annotation_text="Umbral 50 %")
        fig.update_layout(height=340, xaxis=dict(range=[0, 110], title="Probabilidad de abandono (%)"),
                          margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.dataframe(
            comp, hide_index=True, width="stretch",
            column_config={
                "Probabilidad de abandono": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Exactitud del modelo": st.column_config.NumberColumn(format="%.4f"),
            },
        )


# --------------------------------------------------------------------------------------
# Pestaña 3: métricas
# --------------------------------------------------------------------------------------
with tab_met:
    st.markdown("#### Tabla de métricas de los modelos de clasificación")
    st.caption(f"Calculadas sobre el mismo conjunto de prueba ({meta['datos']['prueba']} clientes, 20 % del dataset). "
               "En verde, el mejor valor de cada métrica.")
    tabla = pd.DataFrame({n: meta["modelos"][n]["metricas"] for n in nombres_modelos}).T
    tabla.index.name = "Modelo"
    st.dataframe(
        tabla[METRICAS].style.format("{:.4f}").highlight_max(axis=0, color="#2BB67355"),
        width="stretch",
    )

    fig = go.Figure()
    for metrica in METRICAS:
        fig.add_bar(name=metrica, x=tabla.index, y=tabla[metrica])
    fig.update_layout(barmode="group", height=420, yaxis=dict(range=[0.6, 1.01], title="Valor (0 a 1)"),
                      legend=dict(orientation="h", y=-0.25), margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Sobreentrenamiento (exactitud en entrenamiento vs. prueba)")
    sobre = pd.DataFrame({
        "Entrenamiento": tabla["Exactitud entrenamiento"],
        "Prueba": tabla["Exactitud (Accuracy)"],
        "Diferencia": tabla["Exactitud entrenamiento"] - tabla["Exactitud (Accuracy)"],
    })
    st.dataframe(sobre.style.format("{:.4f}"), width="stretch")


# --------------------------------------------------------------------------------------
# Pestaña 4: acerca del proyecto
# --------------------------------------------------------------------------------------
with tab_info:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(
            """
            #### Avance del proyecto final
            **Corporación Universitaria Antonio José de Sucre** · Ingeniería de Sistemas

            **Estudiantes**
            - Arnovis David Osorio Castillo
            - Diego Alfredo Pérez Corpas
            - Camilo Andrés Ricardo Hoyos
            - Simón Antonio Santana Santana

            **Profesor:** Sergio Antonio Sánchez Hernández
            """
        )
        st.markdown(
            f"""
            #### Preprocesamiento
            1. Se eliminan `customer_id`, `monthly_fee` (redundante con el tipo de suscripción) y `avg_watch_time_per_day`.
            2. Se eliminan nulos y duplicados.
            3. Se eliminan valores atípicos con el rango intercuartílico → **{meta['datos']['renglones_limpios']} clientes**.
            4. Las variables cualitativas se codifican con `LabelEncoder`.
            5. División estratificada 80/20 → {meta['datos']['entrenamiento']} de entrenamiento y {meta['datos']['prueba']} de prueba.
            6. Hiperparámetros con `GridSearchCV` (validación cruzada de 5 particiones, métrica F1).
            """
        )
    with b:
        st.markdown("#### Equivalencias de la codificación")
        for col, mapa in meta["equivalencias"].items():
            st.markdown(f"**{ETIQUETAS[col]}:** " + " · ".join(f"{TRADUCCION.get(k, k)} = {v}" for k, v in mapa.items()))
        st.markdown("#### Versiones usadas en el entrenamiento")
        st.json(meta["versiones"], expanded=False)
