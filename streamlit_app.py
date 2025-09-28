import streamlit as st
from PIL import Image, ImageOps
import io

st.title("Inputs, imágenes, botones y procesamiento 🔥")

# Input de texto
nombre = st.text_input("Escribe tu nombre:")

# Subida de imagen
imagen = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Botón para saludar
if st.button("Saludar"):
    if nombre:
        st.success(f"¡Hola, {nombre}! Bienvenido a mi app 🎉")
    else:
        st.warning("Escribe tu nombre primero 😉")

# Procesamiento básico
if imagen is not None:
    img = Image.open(imagen).convert("RGB")
    st.subheader("Original")
    st.image(img, use_column_width=True)

    opcion = st.selectbox(
        "Elige un procesamiento:",
        ["Ninguno", "Escala de grises", "Espejo horizontal", "Rotar 90°"]
    )

    procesada = img
    if opcion == "Escala de grises":
        procesada = ImageOps.grayscale(img)
    elif opcion == "Espejo horizontal":
        procesada = ImageOps.mirror(img)
    elif opcion == "Rotar 90°":
        procesada = img.rotate(90, expand=True)

    st.subheader("Procesada")
    st.image(procesada, use_column_width=True)

    # Opción para descargar
    buffer = io.BytesIO()
    procesada.save(buffer, format="PNG")
    st.download_button(
        "Descargar imagen procesada",
        data=buffer.getvalue(),
        file_name="procesada.png",
        mime="image/png"
    )
