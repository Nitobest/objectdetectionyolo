#!/bin/bash

# Activar entorno virtual
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "No existe el entorno virtual 'venv'. Créalo con: python3 -m venv venv"
    exit 1
fi

# Instalar dependencias (por si requirements.txt cambió)
pip install -r requirements.txt

# Lanzar Streamlit
streamlit run streamlit_app.py
