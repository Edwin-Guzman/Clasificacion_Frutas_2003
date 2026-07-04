import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración estructural de la aplicación web
st.set_page_config(page_title="Predicción Fruits-360 IA", layout="centered")

# --- ENCABEZADO PERSONALIZADO PARA TU EVALUACIÓN ---
st.title("Clasificador Inteligente de Frutas (Dataset Fruits-360)")
st.markdown("### **Asignatura:** IS-701 - Inteligencia Artificial (Campus Comayagua)")
st.markdown("### **Estudiante:** Edwin Eduardo Guzmán Ramos")
st.markdown("### **Número de Cuenta:** 20211930058")
st.write("Cargue una imagen de una fruta para identificar su categoría mediante Deep Learning.")
st.write("---")

IMG_SIZE = (224, 224)

# --- CONFIGURACIÓN DE ACCESO AL MODELO DE FRUTAS ---
# Apuntamos a la carpeta de tu repositorio donde guardarás los pesos del modelo
MODEL_DIR = Path("modelo_frutas")

# Apuntamos a los nombres exactos de tus archivos según las capturas
CLASS_PATH = MODEL_DIR / "fruits_class_names"  # Sin el .json si se guardó así en tu PC
MODEL_PATHS = [
    MODEL_DIR / "fruits_mobile_net.keras", 
    MODEL_DIR / "fruits_mobile_net.h5"
]

# --- DICCIONARIO CON LAS 10 CATEGORÍAS REALES ENTRENADAS ---
LABELS_ES = {
    "Apple 10": "Manzana Tipo 10",
    "Apple 11": "Manzana Tipo 11",
    "Apple 12": "Manzana Tipo 12",
    "Banana 1": "Banano (Maduro)",
    "Banana 3": "Banano (Variante)",
    "Blueberry 1": "Arándano Azul",
    "Orange 1": "Naranja Común",
    "Orange 2": "Naranja Variante",
    "Onion Red 1": "Cebolla Morada",
    "Onion White 1": "Cebolla Blanca"
}

@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception:
                continue
    st.error(f"No se pudo encontrar el modelo de frutas. Verifique que exista la carpeta '{MODEL_DIR}' con los archivos de pesos.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        try:
            with open(CLASS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Lista de respaldo por si el JSON no se lee correctamente
    return ["Apple 10", "Apple 11", "Apple 12", "Banana 1", "Banana 3", "Blueberry 1", "Orange 1", "Orange 2", "Onion Red 1", "Onion White 1"]

def preparar_imagen(img):
    # Convertimos a RGB y redimensionamos a las dimensiones de MobileNetV2
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # Normalización idéntica al ImageDataGenerator de tu entrenamiento en Colab
    arr = arr / 255.0  
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    # Extraemos el Top 3 de probabilidades
    num_clases = min(3, len(clases))
    top_indices = np.argsort(preds)[-num_clases:][::-1]
    
    return [
        (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in top_indices
    ]

# Carga e inicialización automatizada de la Inteligencia Artificial
modelo = cargar_modelo()
clases = cargar_clases()

# Interfaz interactiva de carga de archivos para el usuario
archivo = st.file_uploader("Seleccione la fotografía de la fruta a clasificar", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen analizada en tiempo real", use_container_width=True)

    with st.spinner("Procesando patrones y texturas por MobileNetV2..."):
        resultados = predecir(imagen)
        
    st.subheader("Análisis de Predicción")
    st.success(f"Fruta detectada principalmente: **{resultados[0][0]}** ({resultados[0][1]:.2f}%)")

    st.write("Top probabilidades encontradas:")
    for clase, prob in resultados:
        st.write(f"- **{clase}**: {prob:.2f}%")
else:
    st.info("Por favor, suba una imagen de las siguientes frutas válidas: Manzana (10, 11, 12), Banano (1, 3), Arándano, Naranja (1, 2) o Cebolla (Morada, Blanca).")
