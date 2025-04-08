import requests
import time
import os

URL = os.getenv("RENDER_URL")  # Tu URL de Render

while True:
    try:
        requests.get(URL)
        print(f"Ping a {URL} exitoso")
    except Exception as e:
        print(f"Error en ping: {e}")
    time.sleep(300)  # 5 minutos