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
from typing import Optional

# Configuración básica
app = FastAPI(
    title="API de Análisis de Lectura",
    description="API para análisis de audio usando Whisper y Supabase",
    version="1.0"
)

warnings.filterwarnings("ignore", category=UserWarning)

# Configuración para plan gratuito
MODELO_WHISPER = "base"  # Modelo ligero para free tier
MAX_DURACION_AUDIO = 120  # 2 minutos máximo
BUCKET_PDF = "pruebas"
BUCKET_AUDIO = "audios"
PDF_TEMP = "temp_ref.pdf"
AUDIO_TEMP = "temp_audio.mp3"

# Modelos Pydantic
class AnalisisRequest(BaseModel):
    paciente_id: str
    archivo_pdf: str
    archivo_audio: str

class HealthCheck(BaseModel):
    status: str
    modelo: str
    supabase_connected: bool
    timestamp: str

# Middleware para manejar errores
@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)[:200]}  # Limitar longitud del error
        )

# Evento de inicio mejorado
@app.on_event("startup")
async def startup_event():
    try:
        # Validar variables de entorno
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Variables SUPABASE_URL y SUPABASE_KEY requeridas")

        # Configurar Supabase con timeout
        app.state.supabase = create_client(
            supabase_url,
            supabase_key,
            options={
                'postgrest_client_timeout': 10,
                'storage_client_timeout': 10
            }
        )
        
        # Test de conexión
        try:
            app.state.supabase.table("audios").select("id").limit(1).execute()
        except Exception as e:
            raise ConnectionError(f"No se pudo conectar a Supabase: {str(e)[:200]}")

        # Cargar modelo Whisper
        try:
            app.state.whisper_model = whisper.load_model(MODELO_WHISPER)
            print(f"✅ Modelo {MODELO_WHISPER} cargado correctamente")
        except Exception as e:
            raise ImportError(f"Error cargando Whisper: {str(e)[:200]}")

    except Exception as e:
        print(f"⛔ Error en startup: {str(e)}")
        raise  # Esto detendrá la aplicación si hay errores críticos

# Endpoints
@app.post("/programar-analisis")
async def programar_analisis(
    request: AnalisisRequest,
    background_tasks: BackgroundTasks
):
    """Endpoint optimizado para free tier"""
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
    """Endpoint de salud mejorado"""
    status = {
        "status": "OK",
        "modelo": MODELO_WHISPER,
        "supabase_connected": hasattr(app.state, 'supabase'),
        "timestamp": datetime.datetime.now().isoformat()
    }
    return status

# Funciones principales optimizadas
def descargar_archivo(bucket: str, archivo_remoto: str, archivo_local: str) -> bool:
    """Descarga archivos con manejo de errores"""
    try:
        with open(archivo_local, "wb") as f:
            response = app.state.supabase.storage.from_(bucket).download(archivo_remoto)
            f.write(response)
        return True
    except Exception as e:
        print(f"Error descargando {archivo_remoto}: {str(e)[:200]}")
        return False

async def realizar_analisis_completo(paciente_id: str, archivo_pdf: str, archivo_audio: str):
    """Flujo completo de análisis optimizado"""
    try:
        # Descargar archivos
        if not descargar_archivo(BUCKET_PDF, archivo_pdf, PDF_TEMP):
            raise Exception("Error descargando PDF")
        
        if not descargar_archivo(BUCKET_AUDIO, archivo_audio, AUDIO_TEMP):
            raise Exception("Error descargando audio")

        # Procesamiento
        texto_ref = extraer_texto_pdf(PDF_TEMP)
        transcripcion = transcribir_audio(AUDIO_TEMP)
        if not transcripcion:
            raise Exception("Error en transcripción")

        texto_audio = normalizar_texto(transcripcion['text'])
        diferencias, correctas, incorrectas, precision = analizar_diferencias(texto_ref, texto_audio)
        fluidez = calcular_fluidez(transcripcion)

        # Guardar resultados
        await guardar_resultados(paciente_id, diferencias, correctas, incorrectas, precision, fluidez)
        
    except Exception as e:
        print(f"Error en análisis: {str(e)[:500]}")
    finally:
        # Limpieza
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
        # Datos mínimos necesarios
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
        
        # Insertamos en lotes pequeños
        supabase.table("precision_pronunciacion").insert(datos_precision).execute()
        supabase.table("fluidez_lectora").insert(datos_fluidez).execute()
        
        # Solo guardamos los primeros 5 errores
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