# 🎬 Simulador de predicción de abandono — Netflix Customer Churn

Aplicación web en **Streamlit** que reúne los seis modelos de clasificación del avance del proyecto final ya entrenados. Puedes cambiar las **variables independientes** de un cliente y los modelos calculan la **variable dependiente** `Abandono` (0 = permanece, 1 = abandona).

**Corporación Universitaria Antonio José de Sucre** · Ingeniería de Sistemas
**Estudiantes:** Arnovis David Osorio Castillo · Diego Alfredo Pérez Corpas · Camilo Andrés Ricardo Hoyos · Simón Antonio Santana Santana
**Profesor:** Sergio Antonio Sánchez Hernández

## Modelos

| Modelo | Exactitud | Precisión | Sensibilidad | F1 | AUC ROC |
|---|---|---|---|---|---|
| Regresión logística simple | 0.7513 | 0.7639 | 0.7654 | 0.7646 | 0.8026 |
| Regresión logística múltiple | 0.8688 | 0.8735 | 0.8787 | 0.8761 | 0.9488 |
| Regresión logística polinomial | 0.8793 | 0.8819 | 0.8907 | 0.8863 | 0.9503 |
| Árbol de decisión | **0.9622** | 0.9465 | **0.9841** | **0.9649** | 0.9907 |
| Bosques aleatorios | 0.9612 | **0.9736** | 0.9523 | 0.9628 | **0.9948** |
| Máquina de vectores de soporte | 0.8657 | 0.8641 | 0.8847 | 0.8743 | 0.9485 |

Las métricas se calcularon sobre el mismo conjunto de prueba: 953 clientes, el 20 % del dataset limpio.

## Qué hace la app
- **🔮 Simulador:** eliges un modelo, cambias edad, horas vistas, días desde el último acceso, número de perfiles, género, tipo de suscripción, método de pago, región, dispositivo y género favorito, y ves la predicción con su probabilidad de abandono.
- **🎲 Cliente real:** carga al azar un cliente del conjunto de prueba y muestra si el modelo acertó.
- **⚖️ Comparar los 6 modelos:** el mismo cliente evaluado por los seis modelos, con la decisión por mayoría.
- **📊 Métricas:** la tabla comparativa y la revisión de sobreentrenamiento.

## Estructura
```
app.py                          ← aplicación de Streamlit
requirements.txt                ← versiones exactas usadas al entrenar
netflix_customer_churn.csv      ← conjunto de datos original
modelos/                        ← modelos entrenados (.joblib), metadata.json y prueba.csv
cuaderno/Simulador_Modelos_Netflix.ipynb  ← cuaderno de Colab que entrena y exporta los 6 modelos
```

## Volver a entrenar los modelos
1. Abre `cuaderno/Simulador_Modelos_Netflix.ipynb` en Google Colab y ejecuta todas las celdas. Cuando te lo pida, sube `netflix_customer_churn.csv`.
2. Al final se descarga `modelos_streamlit.zip`.
3. Reemplaza en el repositorio la carpeta `modelos/` y el archivo `requirements.txt` con los del zip.

Si algún `.joblib` no se puede abrir por una diferencia de versiones, la app vuelve a entrenar ese modelo con los mismos hiperparámetros.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
