import streamlit as st
from PIL import Image
import numpy as np
import io
import os
from datetime import datetime
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Detect", layout="wide")
st.title("Banco de Imágenes de Prueba con YOLO 🧠📦")

MODEL_PATH = "weights/best.pt"
TEST_DIR = "test_images"   # carpeta con tus imágenes de prueba
ALLOWED_EXTS = (".jpg", ".jpeg", ".png")

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

model = load_model(MODEL_PATH)
st.caption(f"Modelo cargado: {MODEL_PATH}")

# -------------------------------------------------
# Configuración en barra lateral (sin imgsz)
# -------------------------------------------------
with st.sidebar:
    st.subheader("Parámetros de inferencia")
    conf = st.slider("Confianza mínima", 0.1, 0.9, 0.25, 0.05)

# -------------------------------------------------
# Fuente de la imagen: banco o subida
# -------------------------------------------------
st.markdown("### Fuente de la imagen")
source = st.radio("Elige cómo cargar la imagen:",
                  ["Banco de pruebas", "Subir imagen"], horizontal=True)

# objeto PIL listo para detectar + nombre lógico para descarga
img = None
display_name = None

if source == "Banco de pruebas":
    # Listar imágenes disponibles en TEST_DIR
    if os.path.exists(TEST_DIR):
        files = sorted([f for f in os.listdir(TEST_DIR)
                        if f.lower().endswith(ALLOWED_EXTS)])
    else:
        files = []

    if not files:
        st.warning(f"No encontré imágenes en `{TEST_DIR}`. Agrega tus tests ahí.")
    else:
        choice = st.selectbox("Selecciona una imagen de prueba", files, index=0)
        path = os.path.join(TEST_DIR, choice)
        try:
            img = Image.open(path).convert("RGB")
            display_name = choice
        except Exception as e:
            st.error(f"No pude abrir `{choice}`: {e}")

else:  # Subir imagen
    uploaded = st.file_uploader("Sube tu imagen (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        try:
            img = Image.open(uploaded).convert("RGB")
            # Opción de guardar la subida al banco
            with st.expander("Opciones de guardado", expanded=True):
                save_to_bank = st.checkbox("Guardar en banco (`test_images/`)", value=True)
                suggested = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filename = st.text_input("Nombre de archivo", value=suggested)
                if save_to_bank:
                    if not os.path.exists(TEST_DIR):
                        os.makedirs(TEST_DIR, exist_ok=True)
                    # Asegurar extensión válida
                    root, ext = os.path.splitext(filename)
                    if ext.lower() not in ALLOWED_EXTS:
                        filename = root + ".jpg"
                    save_path = os.path.join(TEST_DIR, filename)
                    try:
                        img.save(save_path)
                        st.info(f"Guardado en: `{save_path}`")
                    except Exception as e:
                        st.warning(f"No pude guardar la imagen: {e}")
            display_name = filename if uploaded else "upload.png"
        except Exception as e:
            st.error(f"No pude leer la imagen subida: {e}")

# -------------------------------------------------
# Inferencia y visualización
# -------------------------------------------------
if img is not None:
    st.subheader("Imagen seleccionada")
    st.image(img, use_container_width=True)

    if st.button("Detectar"):
        # SIN imgsz: el tamaño de inferencia lo gestiona el modelo por defecto
        results = model.predict(img, conf=conf)
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

        # Lista de objetos detectados
        if res.boxes is not None and len(res.boxes) > 0:
            st.subheader("Objetos detectados")
            names = model.names
            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                label = names.get(cls_id, str(cls_id))
                score = float(b.conf[0].item())
                st.write(f"- {label}: {score:.2f}")
        else:
            st.info("No se detectaron objetos con los parámetros actuales.")

        # Botón de descarga
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        out_name = f"deteccion_{display_name or 'imagen'}.png"
        st.download_button("Descargar imagen anotada",
                           buf.getvalue(), out_name, "image/png")
