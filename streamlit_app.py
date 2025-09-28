import streamlit as st
from PIL import Image
import numpy as np
import io
import os
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Detect", layout="wide")
st.title("Banco de Imágenes de Prueba con YOLO 🧠📦")

MODEL_PATH = "weights/best.pt"
TEST_DIR = "test_images"   # carpeta con tus imágenes de prueba

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

model = load_model(MODEL_PATH)
st.caption(f"Modelo cargado: {MODEL_PATH}")

# Configuración en barra lateral
with st.sidebar:
    conf = st.slider("Confianza mínima", 0.1, 0.9, 0.25, 0.05)
    imgsz = st.selectbox("Tamaño de imagen (modelo)", [320, 416, 512, 640, 768, 960], index=3)

# Listar imágenes disponibles en TEST_DIR
if os.path.exists(TEST_DIR):
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
else:
    files = []

if not files:
    st.warning(f"No encontré imágenes en `{TEST_DIR}`. Agrega tus tests ahí.")
else:
    choice = st.selectbox("Selecciona una imagen de prueba", files)
    path = os.path.join(TEST_DIR, choice)

    if st.button("Detectar"):
        img = Image.open(path).convert("RGB")
        results = model.predict(img, conf=conf, imgsz=imgsz)
        res = results[0]

        annotated_bgr = res.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img, use_container_width=True)
        with col2:
            st.subheader("Detectado")
            st.image(annotated_rgb, use_container_width=True)

        if res.boxes is not None and len(res.boxes) > 0:
            st.subheader("Objetos detectados")
            names = model.names
            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                label = names.get(cls_id, str(cls_id))
                score = float(b.conf[0].item())
                st.write(f"- {label}: {score:.2f}")

        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        st.download_button("Descargar imagen anotada", buf.getvalue(), f"deteccion_{choice}", "image/png")
