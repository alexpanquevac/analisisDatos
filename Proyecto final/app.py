import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Eficiencia de Combustible",
    page_icon="🚗",
    layout="wide"
)

# Cargar el modelo y los transformadores
@st.cache_resource
def load_model():
    model = load("modelo_prediccion_kilometraje.joblib")
    scaler = load("scaler.joblib")
    encoders = load("encoders.joblib")
    return model, scaler, encoders

def main():
    # Título y descripción
    st.title("🚗 Predictor de Eficiencia de Combustible")
    st.markdown("""
    Esta aplicación predice la eficiencia de combustible de un vehículo basándose en sus características.
    Complete los siguientes campos para obtener una predicción.
    """)
    
    try:
        model, scaler, encoders = load_model()
        
        # Crear columnas para organizar los inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Características Técnicas")
            modelo = st.number_input("Modelo (Año)", 
                                   min_value=1990, 
                                   max_value=2024, 
                                   value=2023)
            
            cilindros = st.number_input("Número de Cilindros",
                                      min_value=2,
                                      max_value=12,
                                      value=4)
            
            potencia = st.number_input("Potencia (HP)",
                                     min_value=50,
                                     max_value=1000,
                                     value=150)
            
            tamanio_motor = st.number_input("Tamaño del Motor (L)",
                                          min_value=0.5,
                                          max_value=8.0,
                                          value=2.0)
        
        with col2:
            st.subheader("Características Generales")
            transmision = st.selectbox(
                "Transmisión",
                options=list(encoders['Transmisión'].classes_)
            )
            
            combustible = st.selectbox(
                "Tipo de Combustible",
                options=list(encoders['Combustible'].classes_)
            )
            
            categoria = st.selectbox(
                "Categoría del Vehículo",
                options=list(encoders['Categoría'].classes_)
            )
        
        # Botón de predicción
        if st.button("Calcular Eficiencia"):
            # Preparar datos para la predicción
            input_data = {
                'Modelo': modelo,
                'Cilindros': cilindros,
                'Potencia': potencia,
                'Tamaño_motor': tamanio_motor,
                'Transmisión': transmision,
                'Combustible': combustible,
                'Categoría': categoria
            }
            
            # Convertir a DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Procesar variables categóricas
            for feature, encoder in encoders.items():
                if feature in input_df.columns:
                    input_df[feature] = encoder.transform(input_df[feature].astype(str))
            
            # Procesar variables numéricas
            numerical_features = ['Modelo', 'Cilindros', 'Potencia', 'Tamaño_motor']
            input_df[numerical_features] = scaler.transform(input_df[numerical_features])
            
            # Realizar predicción
            prediction = model.predict(input_df)
            
            # Mostrar resultado
            st.success(f"Eficiencia de combustible predicha: {prediction[0]:.2f} km/l")
            
            # Visualización adicional
            st.subheader("Comparación con valores típicos")
            
            # Crear datos de ejemplo para comparación
            comparison_data = pd.DataFrame({
                'Tipo': ['Tu Vehículo', 'Promedio Sedan', 'Promedio SUV', 'Promedio Deportivo'],
                'Eficiencia': [prediction[0], 12.5, 10.8, 9.2]
            })
            
            fig = px.bar(comparison_data, 
                        x='Tipo', 
                        y='Eficiencia',
                        title='Comparación de Eficiencia de Combustible',
                        labels={'Eficiencia': 'km/l'})
            
            st.plotly_chart(fig)
            
            # Añadir recomendaciones
            st.subheader("Recomendaciones")
            if prediction[0] > 15:
                st.info("👍 Este vehículo tiene una excelente eficiencia de combustible!")
            elif prediction[0] > 12:
                st.info("✅ Este vehículo tiene una buena eficiencia de combustible.")
            else:
                st.warning("⚠️ Considere opciones más eficientes si el consumo de combustible es una prioridad.")
                
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.info("Por favor, asegúrese de que los archivos del modelo estén disponibles.")

if __name__ == "__main__":
    main()