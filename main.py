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
from pydantic import BaseModel
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from typing import Optional

app = FastAPI(
    title="API de Análisis de Lectura",
    description="API para análisis de audio usando Whisper y Supabase",
    version="1.0"
)

warnings.filterwarnings("ignore", category=UserWarning)

MODELO_WHISPER = "tiny"
MAX_DURACION_AUDIO = 120
BUCKET_PDF = "pruebas"
BUCKET_AUDIO = "audios"
PDF_TEMP = "temp_ref.pdf"
AUDIO_TEMP = "temp_audio.mp3"

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

        # Probar conexión Supabase
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

async def realizar_analisis_completo(paciente_id: str, archivo_pdf: str, archivo_audio: str):
    try:
        if not descargar_archivo(BUCKET_PDF, archivo_pdf, PDF_TEMP):
            raise Exception("Error descargando PDF")

        if not descargar_archivo(BUCKET_AUDIO, archivo_audio, AUDIO_TEMP):
            raise Exception("Error descargando audio")

        texto_ref = extraer_texto_pdf(PDF_TEMP)
        transcripcion = transcribir_audio(AUDIO_TEMP)
        if not transcripcion:
            raise Exception("Error en transcripción")

        texto_audio = normalizar_texto(transcripcion['text'])
        diferencias, correctas, incorrectas, precision = analizar_diferencias(texto_ref, texto_audio)
        fluidez = calcular_fluidez(transcripcion)

        await guardar_resultados(
            app.state.supabase,
            paciente_id,
            diferencias,
            correctas,
            incorrectas,
            precision,
            fluidez
        )

    except Exception as e:
        print(f"Error en análisis: {str(e)[:500]}")
    finally:
        for archivo in [PDF_TEMP, AUDIO_TEMP]:
            try:
                if os.path.exists(archivo):
                    os.remove(archivo)
            except:
                pass
        app.state.analisis_en_curso = False

async def guardar_resultados(supabase: Client, paciente_id: str, diferencias: list, correctas: int, incorrectas: int, precision: float, fluidez: dict):
    now = datetime.datetime.now().isoformat()
    try:
        datos_precision = {
            "paciente_id": paciente_id,
            "fecha": now,
            "palabras_correctas": correctas,
            "palabras_incorrectas": incorrectas,
            "porcentaje_precision": precision
        }

        datos_fluidez = {
            "paciente_id": paciente_id,
            "fecha": now,
            "palabras_por_minuto": fluidez["palabras_por_minuto"],
            "numero_pausas": fluidez["numero_pausas"]
        }

        supabase.table("precision_pronunciacion").insert(datos_precision).execute()
        supabase.table("fluidez_lectora").insert(datos_fluidez).execute()

        for error in diferencias[:5]:
            supabase.table("errores_recurrentes").insert({
                "paciente_id": paciente_id,
                "fecha": now,
                "tipo_error": error["tipo"],
                "palabra_original": error["palabra_original"],
                "palabra_dicha": error["palabra_dicha"]
            }).execute()

    except Exception as e:
        print(f"Error guardando resultados: {str(e)[:500]}")
