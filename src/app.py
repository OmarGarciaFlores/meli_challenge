import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(
    page_title="Dashboard SHAP - Mercado Libre",
    layout="wide",  
    page_icon="src/mercado-libre.png"
)

# ========================
# Cargar modelo y datos
# ========================
@st.cache_resource
def cargar_modelo():
    return joblib.load("src/modelo_xgb_v2.pkl")

@st.cache_data
def cargar_datos():
    X_test = pd.read_csv("src/X_test_v2.csv")
    y_test = pd.read_csv("src/y_test_v2.csv")  # Opcional, si quieres mostrar la clase real
    return X_test, y_test

modelo = cargar_modelo()
X_test, y_test = cargar_datos()

# ========================
# Calcular SHAP
# ========================
st.sidebar.image("src/mercado-libre.png", width=200)
st.sidebar.markdown("<h2 style='text-align: center; color: #fdd835;'>Panel de Configuración</h2>", unsafe_allow_html=True)

st.sidebar.markdown("#### Selecciona un cliente:")
#cliente_idx = st.sidebar.slider("", 0, len(X_test) - 1)
cliente_idx = st.sidebar.slider("", 0, len(X_test) - 1, value=7205)
st.sidebar.markdown(f"Cliente seleccionado: **{cliente_idx}**")


@st.cache_resource
def cargar_explainer(_modelo):
    explainer = shap.TreeExplainer(_modelo)
    return explainer

explainer = cargar_explainer(_modelo=modelo)
shap_values_raw = explainer.shap_values(X_test)

# ========================
# Construir objeto Explanation
# ========================
shap_explanation = shap.Explanation(
    values=shap_values_raw[cliente_idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[cliente_idx],
    feature_names=X_test.columns
)

## ========================
# Visualización
# ========================
st.title("Modelo de clasificación de artículos nuevos y usados")

st.markdown("""
Este modelo utiliza un algoritmo *xgboost* para predecir si un artículo es **nuevo** o **usado** 
basado en 14 características.

La intención de este dashboard es ayudar a los usuarios a entender cómo funciona el modelo y cómo se llega a la predicción.

Para ello, se muestra la probabilidad de que el artículo sea nuevo o usado y cómo cada variable contribuye a la decisión final.

""")

st.markdown(f"### Cliente seleccionado: {cliente_idx}")


# ========================
# Métricas horizontales alineadas
# ========================
st.markdown("### Predicción del modelo para el cliente seleccionado")

col1, col2, col3 = st.columns(3)

with col1:
    pred_prob = modelo.predict_proba(X_test)[cliente_idx, 1]
    pred_clase = modelo.predict(X_test)[cliente_idx]
    st.metric("Probabilidad de que el artículo sea nuevo", f"{pred_prob:.2%}")

with col2:
    st.metric("Probabilidad de que el artículo sea usado", f"{1 - pred_prob:.2%}")

with col3:
    st.metric("Predicción", "Nuevo" if pred_clase == 1 else "Usado")


# ========================
# Convertir log-odds a probabilidad
# ========================
def log_odds_to_proba(log_odds):
    return 1 / (1 + np.exp(-log_odds))

fx_logodds = shap_explanation.values.sum() + explainer.expected_value
fx_prob = log_odds_to_proba(fx_logodds)
expected_prob = log_odds_to_proba(explainer.expected_value)


# ========================
# Explicación y gráfica individual en términos probabilísticos
# ========================
st.write("#### ¿Cómo interpretar esta gráfica?")
st.markdown(f"""
- **f(x)** representa la salida del modelo en escala log-odds para este cliente.
  En este caso, `f(x)` = {fx_logodds:.3f}, equivale a una probabilidad de **{fx_prob:.2%}** de que el artículo sea **nuevo**.
- **Esperanza** (*Expected Value*): es el valor promedio de salida del modelo antes de ver cualquier variable. En este caso: {explainer.expected_value:.3f}, equivalente a una probabilidad de **{expected_prob:.2%}**.
- 🔴 **Rojo**: la variable aumenta la probabilidad de que el artículo sea **nuevo**  
- 🔵 **Azul**: la variable reduce la probabilidad de que sea nuevo (más probable que sea **usado**)  
- Esta gráfica muestra cómo se llega desde la esperanza hasta la predicción individual `f(x)`.

Los valores SHAP en esta gráfica están en log-odds (no directamente en probabilidad), debajo de esta gráfica se muestra la tabla de variables con su impacto en la probabilidad de que el artículo sea nuevo.
""")


# Graficar estilo waterfall con SHAP
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 6

fig, ax = plt.subplots(figsize=(3, 3))  # tamaño ajustado aquí
fig.set_facecolor('#f9f9f9')            # fondo general
ax.set_facecolor('white')               # fondo del área de gráfica

# Título centrado y con estilo
plt.title("¿Por qué el modelo clasificó este artículo como NUEVO o USADO?",
          fontsize=15, pad=10, fontweight='bold', color='dodgerblue',
          fontstyle='italic', loc='center')

# Gráfica SHAP tipo waterfall
shap.plots.waterfall(shap_explanation, max_display=11, show=False)

# Cambiar tamaño de fuente de las etiquetas del eje Y (variables a la izquierda)
for label in ax.get_yticklabels():
    label.set_fontsize(10)  # o un valor menor si deseas aún más pequeño

plt.tight_layout(pad=1)
st.pyplot(fig)


# ========================
# Mostrar tabla con impacto en probabilidad
# ========================
st.markdown("#### Conversión de SHAP a probabilidad")

# Transformar valores SHAP a contribuciones en probabilidad
log_odds = explainer.expected_value + shap_explanation.values.sum()
final_prob = 1 / (1 + np.exp(-log_odds))
gradient = final_prob * (1 - final_prob)
shap_prob_values = shap_explanation.values * gradient

# Tabla ordenada
df_prob = pd.DataFrame({
    'Variable': shap_explanation.feature_names,
    'Impacto en probabilidad (%)': shap_prob_values * 100
}).sort_values(by='Impacto en probabilidad (%)', key=abs, ascending=False)

df_prob['Impacto en probabilidad (%)'] = df_prob['Impacto en probabilidad (%)'].map(lambda x: f"{x:+.2f}%")

st.dataframe(df_prob.reset_index(drop=True), use_container_width=True)




