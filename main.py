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
from typing import Optional

# Crear la instancia de la aplicación FastAPI
app = FastAPI(
    title="API de Análisis de Lectura",
    description="API para análisis de audio usando Whisper y Supabase",
    version="1.0"
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Desarrollo local
        "https://tu-frontend.com"  # Tu dominio de producción
    ],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],  # Específica los métodos necesarios
    allow_headers=["Content-Type", "Authorization"],  # Headers explícitos
    expose_headers=["Content-Type"]  # Headers que el frontend puede leer
)

# Añade un manejador explícito para OPTIONS
@app.options("/programar-analisis")
async def options_programar_analisis():
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "https://tu-frontend.com",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "86400"  # Cache preflight por 24 horas
        }
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

@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)[:200]}
        )

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

# Descargar archivo de Supabase
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

# Comparar textos y generar estadísticas
def analizar_diferencias(texto_ref: str, texto_audio: str):
    diferencias = []
    ref_words = texto_ref.split()
    audio_words = texto_audio.split()

    matcher = difflib.SequenceMatcher(None, ref_words, audio_words)

    correctas = 0
    incorrectas = 0

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            correctas += (i2 - i1)
        else:
            incorrectas += max(i2 - i1, j2 - j1)
            diferencias.append({
                "tipo": opcode,
                "palabra_original": " ".join(ref_words[i1:i2]),
                "palabra_dicha": " ".join(audio_words[j1:j2])
            })
    total = correctas + incorrectas
    precision = round((correctas / total) * 100, 2) if total > 0 else 0

    return diferencias, correctas, incorrectas, precision

# Calcular fluidez
def calcular_fluidez(transcripcion):
    palabras = transcripcion['text'].split()
    n_palabras = len(palabras)
    duracion = transcripcion['segments'][-1]['end'] if transcripcion['segments'] else 1
    palabras_por_minuto = round(n_palabras / (duracion / 60), 2)

    for i in range(1, len(transcripcion['segments'])):
        transcripcion['segments'][i]['prev_end'] = transcripcion['segments'][i-1]['end']

    pausas = [seg for seg in transcripcion['segments'] if seg['start'] - seg.get('prev_end', 0) > 1.5]
    
    return {
        "palabras_por_minuto": palabras_por_minuto,
        "numero_pausas": len(pausas),
        "duracion_total_segundos": round(duracion, 2)
    }

# Realizar análisis completo
async def realizar_analisis_completo(paciente_id: str, archivo_pdf: str, archivo_audio: str):
    try:
        if not descargar_archivo(BUCKET_PDF, archivo_pdf, PDF_TEMP):
            raise HTTPException(status_code=500, detail="No se pudo descargar el PDF")

        if not descargar_archivo(BUCKET_AUDIO, archivo_audio, AUDIO_TEMP):
            raise HTTPException(status_code=500, detail="No se pudo descargar el audio")

        texto_ref = normalizar_texto(extraer_texto_pdf(PDF_TEMP))
        transcripcion = transcribir_audio(AUDIO_TEMP)

        if not transcripcion:
            raise HTTPException(status_code=500, detail="Error al transcribir el audio")

        texto_audio = normalizar_texto(transcripcion['text'])
        diferencias, correctas, incorrectas, precision = analizar_diferencias(texto_ref, texto_audio)
        fluidez = calcular_fluidez(transcripcion)

        # Guardar resultados en Supabase
        app.state.supabase.table("resultados").insert([{
            "paciente_id": paciente_id,
            "precision": precision,
            "fluidez": fluidez,
            "diferencias": json.dumps(diferencias),
            "fecha_analisis": datetime.datetime.now().isoformat()
        }]).execute()

        # Finalizar el análisis
        app.state.analisis_en_curso = False

    except Exception as e:
        app.state.analisis_en_curso = False
        print(f"Error en análisis completo: {str(e)}")
        raise

