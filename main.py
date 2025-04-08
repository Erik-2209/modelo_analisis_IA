import os
import re
import fitz
import whisper
import difflib
import uuid
import json
import datetime
import warnings
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from typing import Optional, List, Dict

# Crear la instancia de la aplicación FastAPI
app = FastAPI(
    title="API de Análisis de Lectura",
    description="API para análisis de audio usando Whisper y Supabase",
    version="1.0"
)

# Configuración de CORS más permisiva para desarrollo
origins = [
    "http://localhost",
    "http://localhost:5173",
    "https://zp1v56uxy8rdx5ypatb0ockcb9tr6a-oci3-ladv5tn2--5173--d69c5f7b.local-credentialless.webcontainer-api.io",
    "https://tu-frontend.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de variables
warnings.filterwarnings("ignore", category=UserWarning)

MODELO_WHISPER = "tiny"
MAX_DURACION_AUDIO = 120
BUCKET_PDF = "pruebas"
BUCKET_AUDIO = "audios"
PDF_TEMP = "temp_ref.pdf"
AUDIO_TEMP = "temp_audio.mp3"

# Definir modelos de datos
class AnalisisRequest(BaseModel):
    paciente_id: str
    archivo_pdf: str
    archivo_audio: str

class HealthCheck(BaseModel):
    status: str
    modelo: str
    supabase_connected: bool
    timestamp: str

@app.on_event("startup")
async def startup_event():
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("Configura SUPABASE_URL y SUPABASE_KEY en Render")

        options = ClientOptions(
            auto_refresh_token=False,
            persist_session=False,
            headers={
                'X-Client-Info': 'my-app/1.0'
            }
        )

        app.state.supabase = create_client(supabase_url, supabase_key, options=options)

        # Probar conexión a Supabase
        data = app.state.supabase.table("audios").select("id").limit(1).execute()
        print("✅ Conexión exitosa a Supabase")

    except Exception as e:
        print(f"⛔ Error conectando a Supabase: {str(e)}")
        raise

    # Cargar modelo Whisper
    try:
        app.state.whisper_model = whisper.load_model(MODELO_WHISPER)
        print(f"✅ Modelo {MODELO_WHISPER} cargado correctamente")
    except Exception as e:
        raise ImportError(f"Error cargando Whisper: {str(e)[:200]}")

@app.post("/programar-analisis")
async def programar_analisis(
    request: AnalisisRequest,
    background_tasks: BackgroundTasks
):
    try:
        if hasattr(app.state, 'analisis_en_curso') and app.state.analisis_en_curso:
            raise HTTPException(
                status_code=429,
                detail="Solo puede procesar un audio a la vez en el plan gratuito"
            )

        app.state.analisis_en_curso = True

        background_tasks.add_task(
            realizar_analisis_completo,
            request.paciente_id,
            request.archivo_pdf,
            request.archivo_audio
        )

        return {
            "status": "success",
            "message": "Análisis en progreso",
            "job_id": str(uuid.uuid4()),
            "limitaciones": {
                "max_duracion": MAX_DURACION_AUDIO,
                "modelo": MODELO_WHISPER
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {
        "status": "OK",
        "modelo": MODELO_WHISPER,
        "supabase_connected": hasattr(app.state, 'supabase'),
        "timestamp": datetime.datetime.now().isoformat()
    }

def descargar_archivo(bucket: str, archivo_remoto: str, archivo_local: str) -> bool:
    try:
        with open(archivo_local, "wb") as f:
            response = app.state.supabase.storage.from_(bucket).download(archivo_remoto)
            f.write(response)
        return True
    except Exception as e:
        print(f"Error descargando {archivo_remoto}: {str(e)[:200]}")
        return False

# Extraer texto del PDF
def extraer_texto_pdf(pdf_path: str) -> str:
    """
    Extrae texto de un archivo PDF
    """
    try:
        doc = fitz.open(pdf_path)
        texto = "\n".join(page.get_text("text") for page in doc)
        return texto.strip()
    except Exception as e:
        print(f"Error extrayendo texto del PDF: {str(e)[:200]}")
        return ""

# Transcribir audio con Whisper
def transcribir_audio(audio_path: str):
    try:
        result = app.state.whisper_model.transcribe(audio_path)
        return result
    except Exception as e:
        print(f"Error al transcribir el audio: {str(e)[:200]}")
        return None

# Normalizar texto
def normalizar_texto(texto: str) -> str:
    return re.sub(r"[^\w\s]", "", texto.lower()).strip()

# Función mejorada para analizar diferencias
def analizar_diferencias(texto_ref: str, texto_audio: str) -> tuple:
    diferencias = []
    ref_words = texto_ref.split()
    audio_words = texto_audio.split()

    matcher = difflib.SequenceMatcher(None, ref_words, audio_words)

    correctas = 0
    incorrectas = 0
    errores_recurrentes = {}

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            correctas += (i2 - i1)
        else:
            incorrectas += max(i2 - i1, j2 - j1)
            original = " ".join(ref_words[i1:i2])
            dicho = " ".join(audio_words[j1:j2])
            
            # Registrar error para la tabla errors_recurrentes
            error_key = (original, dicho)
            errores_recurrentes[error_key] = errores_recurrentes.get(error_key, 0) + 1
            
            diferencias.append({
                "tipo": opcode,
                "palabra_original": original,
                "palabra_dicha": dicho
            })

    total = correctas + incorrectas
    precision = round((correctas / total) * 100, 2) if total > 0 else 0

    return diferencias, correctas, incorrectas, precision, errores_recurrentes

# Función mejorada para calcular fluidez
def calcular_fluidez(transcripcion: dict) -> dict:
    palabras = transcripcion['text'].split()
    n_palabras = len(palabras)
    duracion = transcripcion['segments'][-1]['end'] if transcripcion['segments'] else 1
    palabras_por_minuto = round(n_palabras / (duracion / 60), 2)

    pausas = 0
    for i in range(1, len(transcripcion['segments'])):
        if transcripcion['segments'][i]['start'] - transcripcion['segments'][i-1]['end'] > 1.5:
            pausas += 1
    
    return {
        "palabras_por_minuto": palabras_por_minuto,
        "numero_pausas": pausas,
        "duracion_total_segundos": round(duracion, 2)
    }

# Función para guardar errores recurrentes
def guardar_errores_recurrentes(paciente_id: str, fecha: str, errores: dict):
    records = []
    for (original, dicho), frecuencia in errores.items():
        records.append({
            "paciente_id": paciente_id,
            "fecha": fecha,
            "tipo_error": "pronunciacion",
            "palabra_original": original,
            "palabra_dicha": dicho,
            "frecuencia": frecuencia,
            "contacto": False  # Por defecto
        })
    
    if records:
        app.state.supabase.table("errors_recurrentes").insert(records).execute()

# Función para guardar métricas de fluidez
def guardar_fluidez(paciente_id: str, fecha: str, fluidez: dict):
    app.state.supabase.table("fluidex_lectora").insert([{
        "paciente_id": paciente_id,
        "fecha": fecha,
        "palabras_por_minuto": fluidez["palabras_por_minuto"],
        "numero_passes": fluidez["numero_pausas"],
        "duracion_total_segundos": fluidez["duracion_total_segundos"]
    }]).execute()

# Función para guardar precisión de pronunciación
def guardar_precision(paciente_id: str, fecha: str, correctas: int, incorrectas: int, precision: float):
    app.state.supabase.table("precision_pronunciacion").insert([{
        "paciente_id": paciente_id,
        "fecha": fecha,
        "palabras_correctas": correctas,
        "palabras_incorrectas": incorrectas,
        "porcentaje_predision": precision,
        "observaciones": "Análisis automático"
    }]).execute()

# Función principal de análisis
async def realizar_analisis_completo(paciente_id: str, archivo_pdf: str, archivo_audio: str):
    try:
        # Descargar archivos
        if not descargar_archivo(BUCKET_PDF, archivo_pdf, PDF_TEMP):
            raise Exception("No se pudo descargar el PDF")
        
        if not descargar_archivo(BUCKET_AUDIO, archivo_audio, AUDIO_TEMP):
            raise Exception("No se pudo descargar el audio")

        # Procesar archivos
        texto_ref = normalizar_texto(extraer_texto_pdf(PDF_TEMP))
        transcripcion = transcribir_audio(AUDIO_TEMP)
        
        if not transcripcion:
            raise Exception("Error al transcribir el audio")

        texto_audio = normalizar_texto(transcripcion['text'])
        fecha_actual = datetime.datetime.now().isoformat()
        
        # Analizar resultados
        diferencias, correctas, incorrectas, precision, errores = analizar_diferencias(texto_ref, texto_audio)
        fluidez = calcular_fluidez(transcripcion)

        # Guardar en las tablas específicas
        guardar_errores_recurrentes(paciente_id, fecha_actual, errores)
        guardar_fluidez(paciente_id, fecha_actual, fluidez)
        guardar_precision(paciente_id, fecha_actual, correctas, incorrectas, precision)

        # Limpiar
        os.remove(PDF_TEMP)
        os.remove(AUDIO_TEMP)
        
    except Exception as e:
        print(f"Error en análisis completo: {str(e)}")
        raise
    finally:
        app.state.analisis_en_curso = False